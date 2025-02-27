import torch
import numpy as np
import osqp
from scipy.sparse import csc_matrix

import time

from abc import ABC, abstractmethod

torch.set_default_dtype(torch.float64)

class SimpleProblem(ABC):
    """
    minimize_y 1/2 * y^T Q y + p^Ty
    s.t.       Ay =  b
               Gy <= d
    """

    def __init__(self, Q, p, A, G, b, d, valid_frac=0.0833, test_frac=0.0833):
        self._Q = torch.tensor(Q)
        self._p = torch.tensor(p)
        self._A = torch.tensor(A)
        self._G = torch.tensor(G)
        self._b = torch.tensor(b) # equality RHS
        self._d = torch.tensor(d) # inequality RHS

        self._eq_cm = self._A
        self._ineq_cm = self._G
        self._eq_rhs = self._b
        self._ineq_rhs = self._d

        self.A_transpose = self._A.T
        self.G_transpose = self._G.T

        self._valid_frac = valid_frac
        self._test_frac = test_frac

        self._Y = None
        self._ydim = Q.shape[0]

        ### For Pytorch
        self._device = None

        #! Implement in child!
        self._X = None
        self._num = None
        self._neq = None
        self._nineq = None
        self._xdim = None

        # TODO: What to do with this code?
        # det = 0
        # i = 0
        # while abs(det) < 0.0001 and i < 100:
        #     self._partial_vars = np.random.choice(
        #         self._ydim, self._ydim - self._neq, replace=False
        #     )
        #     self._other_vars = np.setdiff1d(np.arange(self._ydim), self._partial_vars)
        #     det = torch.det(self._A[:, self._other_vars])
        #     i += 1
        # if i == 100:
        #     raise Exception
        # else:
        #     self._A_partial = self._A[:, self._partial_vars]
        #     self._A_other_inv = torch.inverse(self._A[:, self._other_vars])


    ##### ABSTRACT METHODS #####

    @abstractmethod
    def eq_resid(self, X, Y):
        raise NotImplementedError
    
    @abstractmethod
    def ineq_resid(self, X, Y):
        raise NotImplementedError
    
    @abstractmethod
    def opt_solve(self, X, solver_type="osqp", tol=1e-4):
        raise NotImplementedError

    @abstractmethod
    def calc_Y(self):
        raise NotImplementedError


    def __str__(self):
        return "SimpleProblem-{}-{}-{}-{}".format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )
    
    ##### REG METHODS #####

    @property
    def eq_cm(self):
        return self._eq_cm

    @property
    def ineq_cm(self):
        return self._ineq_cm
    
    @property
    def eq_rhs(self):
        return self._eq_rhs
    
    @property
    def ineq_rhs(self):
        return self._ineq_rhs

    @property
    def Q(self):
        return self._Q

    @property
    def p(self):
        return self._p

    @property
    def A(self):
        return self._A

    @property
    def G(self):
        return self._G

    @property
    def b(self):
        return self._b

    @property
    def d(self):
        return self._d

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def partial_vars(self):
        return self._partial_vars

    @property
    def other_vars(self):
        return self._other_vars

    @property
    def partial_unknown_vars(self):
        return self._partial_vars

    @property
    def Q_np(self):
        return self.Q.detach().cpu().numpy()

    @property
    def p_np(self):
        return self.p.detach().cpu().numpy()

    @property
    def A_np(self):
        return self.A.detach().cpu().numpy()

    @property
    def G_np(self):
        return self.G.detach().cpu().numpy()

    @property
    def b_np(self):
        return self.b.detach().cpu().numpy()

    @property
    def d_np(self):
        return self.d.detach().cpu().numpy()

    @property
    def X_np(self):
        return self.X.detach().cpu().numpy()

    @property
    def Y_np(self):
        return self.Y.detach().cpu().numpy()

    @property
    def xdim(self):
        return self._xdim

    @property
    def ydim(self):
        return self._ydim

    @property
    def num(self):
        return self._num

    @property
    def neq(self):
        return self._neq

    @property
    def nineq(self):
        return self._nineq

    @property
    def nknowns(self):
        return self._nknowns

    @property
    def valid_frac(self):
        return self._valid_frac

    @property
    def test_frac(self):
        return self._test_frac

    @property
    def train_frac(self):
        return 1 - self.valid_frac - self.test_frac
    
    @property
    def train_indices(self):
        return list(range(int(self.num * self.train_frac)))
    
    @property
    def valid_indices(self):
        return list(range(int(self.num * self.train_frac), int(self.num * (self.train_frac + self.valid_frac))))
    
    @property
    def test_indices(self):
        return list(range(int(self.num * (self.train_frac + self.valid_frac)), self.num))

    @property
    def trainX(self):
        return self.X[: int(self.num * self.train_frac)]

    @property
    def validX(self):
        return self.X[
            int(self.num * self.train_frac) : int(
                self.num * (self.train_frac + self.valid_frac)
            )
        ]

    @property
    def testX(self):
        return self.X[int(self.num * (self.train_frac + self.valid_frac)) :]

    @property
    def trainY(self):
        return self.Y[: int(self.num * self.train_frac)]

    @property
    def validY(self):
        return self.Y[
            int(self.num * self.train_frac) : int(
                self.num * (self.train_frac + self.valid_frac)
            )
        ]

    @property
    def testY(self):
        return self.Y[int(self.num * (self.train_frac + self.valid_frac)) :]

    @property
    def device(self):
        return self._device

    def obj_fn(self, Y):
        return (0.5 * (Y @ self.Q) * Y +  self.p * Y).sum(dim=1)

    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)

    def eq_grad(self, X, Y):
        return 2 * (Y @ self.A.T - X) @ self.A

    def ineq_grad(self, X, Y):
        ineq_dist = self.ineq_dist(X, Y)
        return 2 * ineq_dist @ self.G

    def ineq_partial_grad(self, X, Y):
        G_effective = self.G[:, self.partial_vars] - self.G[:, self.other_vars] @ (
            self._A_other_inv @ self._A_partial
        )
        h_effective = self.h - (X @ self._A_other_inv.T) @ self.G[:, self.other_vars].T
        grad = (
            2
            * torch.clamp(Y[:, self.partial_vars] @ G_effective.T - h_effective, 0)
            @ G_effective
        )
        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y[:, self.partial_vars] = grad
        Y[:, self.other_vars] = -(grad @ self._A_partial.T) @ self._A_other_inv.T
        return Y

    # Processes intermediate neural network output
    def process_output(self, X, Y):
        return Y

    # Solves for the full set of variables
    def complete_partial(self, X, Z):
        Y = torch.zeros(X.shape[0], self.ydim, device=self.device)
        Y[:, self.partial_vars] = Z
        Y[:, self.other_vars] = (X - Z @ self._A_partial.T) @ self._A_other_inv.T
        return Y

class OriginalQPProblem(SimpleProblem):
    def __init__(self, Q, p, A, G, b, d, valid_frac=0.1, test_frac=0.1):
        super().__init__(Q, p, A, G, b, d, valid_frac, test_frac)

        self._X = self._b
        self._num = self._X.shape[0]
        self._neq = self._A.shape[0]
        self._nineq = self._G.shape[0]
        self._xdim = self._X.shape[1]


    def eq_resid(self, y, eq_cm, eq_rhs):
        # return torch.matmul(y, self.A.T) - eq_rhs
        return torch.matmul(y, self.A_transpose) - eq_rhs

    def ineq_resid(self, y, ineq_cm, ineq_rhs):
        return torch.matmul(y, self.G_transpose) - ineq_rhs
    
    def ineq_dist(self, y, ineq_cm, ineq_rhs):
        resids = self.ineq_resid(y, ineq_cm, ineq_rhs)
        return torch.clamp(resids, 0)
    
    def opt_solve(self, X, solver_type="osqp", tol=1e-4):
        if solver_type == "osqp":
            print("running osqp")
            Q, p, A, G, d = self.Q_np, self.p_np, self.A_np, self.G_np, self.d_np
            X_np = X.detach().cpu().numpy()
            Y = []
            total_time = 0
            for Xi in X_np:
                solver = osqp.OSQP()
                my_A = np.vstack([A, G])
                my_l = np.hstack([Xi, -np.ones(d.shape[0]) * np.inf])
                my_u = np.hstack([Xi, d])
                solver.setup(
                    P=csc_matrix(Q),
                    q=p,
                    A=csc_matrix(my_A),
                    l=my_l,
                    u=my_u,
                    verbose=False,
                    eps_prim_inf=tol,
                )
                start_time = time.time()
                results = solver.solve()
                end_time = time.time()

                total_time += end_time - start_time
                if results.info.status == "solved":
                    Y.append(results.x)
                else:
                    Y.append(np.ones(self.ydim) * np.nan)

                sols = np.array(Y)
                parallel_time = total_time / len(X_np)
        else:
            raise NotImplementedError

        return sols, total_time, parallel_time

    def calc_Y(self):
        Y = self.opt_solve(self.X)[0]
        feas_mask = ~np.isnan(Y).all(axis=1)
        self._num = feas_mask.sum()
        self._X = self._X[feas_mask]
        self._Y = torch.tensor(Y[feas_mask])
        return Y

class QPProblemVaryingG(SimpleProblem):
    def __init__(self, Q, p, A, G_base, G_varying, b, d, n_varying_rows, valid_frac=0.1, test_frac=0.1):
        super().__init__(Q, p, A, G_varying, b, d, valid_frac, test_frac)
        self.G_base = torch.tensor(G_base)
        self.n_varying_rows = n_varying_rows
        # Take the first n rows of G as input
        self._X = self._G[:, :n_varying_rows, :].flatten(start_dim=1)
        self._num = self._X.shape[0]
        self._neq = self._A.shape[0]
        # G now has num_samples in first dimension, num_constraints in second dimension. Take second dimension!
        self._nineq = self._G.shape[1]
        self._xdim = self._X.shape[1]

    def eq_resid(self, X, Y):
        """RHS of equality constraints now remains constant across problem instances."""
        return self.b - Y @ self.A.T

    def rebuild_G_from_X(self, X):
        # Reshape X to match the first self.n_varying_rows rows of G
        custom_G = X.reshape(X.shape[0], self.n_varying_rows, self._ydim)  # Reshape for the batch size

        # Take only the first sample of G and clone it for modification
        G = self.G_base.clone()  # Shape is (M, P)

        # Repeat G for the batch size to avoid memory overlap
        G = G.unsqueeze(0).repeat(X.shape[0], 1, 1)  # Shape is (batch_size, M, P)

        # Assign custom_G to the first self.n_varying_rows rows of G
        G[:, :self.n_varying_rows, :] = custom_G  # Ensure dimensions match
        return G
    
    def ineq_resid(self, X, Y):
        """
        For the ineq resid, we need to extract the first n rows of the G matrix from it's flattened form X, and plug them into G.
        """

        G = self.rebuild_G_from_X(X)

        # resid = Y @ G.transpose(1, 2) - h
        residual = torch.bmm(Y.unsqueeze(1), G.transpose(1, 2)).squeeze(1) - self.d

        # Compute inequality residual
        return residual
    
    def opt_solve(self, X, solver_type="osqp", tol=1e-4):
        """We change op_solve so that the varying G matrices are taken from the input X.
        """
        if solver_type == "osqp":
            print("running osqp")
            Q, p, b, d = self.Q_np, self.p_np, self.b_np, self.d_np
            G_np = self.rebuild_G_from_X(X).detach().cpu().numpy()
            A = self.A_np
            Y = []
            total_time = 0
            for Gi in G_np:
                solver = osqp.OSQP()
                my_A = np.vstack([A, Gi])
                my_l = np.hstack([b, -np.ones(d.shape[0]) * np.inf])
                my_u = np.hstack([b, d])
                solver.setup(
                    P=csc_matrix(Q),
                    q=p,
                    A=csc_matrix(my_A),
                    l=my_l,
                    u=my_u,
                    verbose=False,
                    eps_prim_inf=tol,
                )
                start_time = time.time()
                results = solver.solve()
                end_time = time.time()

                total_time += end_time - start_time
                if results.info.status == "solved":
                    Y.append(results.x)
                else:
                    Y.append(np.ones(self.ydim) * np.nan)

            sols = np.array(Y)
            parallel_time = total_time / len(X)

        else:
            raise NotImplementedError

        return sols, total_time, parallel_time

    def calc_Y(self):
        Y = self.opt_solve(self.X)[0]
        feas_mask = ~np.isnan(Y).all(axis=1)
        self._num = feas_mask.sum()
        self._X = self._X[feas_mask]
        self._Y = torch.tensor(Y[feas_mask])
        return Y

class QPProblemVaryingGbd(SimpleProblem):
    def __init__(self, Q, p, A, G_base, G_varying, b, d, n_varying_rows, valid_frac=0.1, test_frac=0.1):
        super().__init__(Q, p, A, G_varying, b, d, valid_frac, test_frac)
        self.G_base = torch.tensor(G_base)
        self.n_varying_rows = n_varying_rows
        # Flatten the rows of G that are varying, to be added to the NN input.
        G_flattened = self.G[:, :n_varying_rows, :].flatten(start_dim=1)
        self._X = torch.concat([G_flattened, self.b, self.d], dim=1)
        self._num = self._X.shape[0]
        self._neq = self._A.shape[0]
        # G now has num_samples in first dimension, num_constraints in second dimension. Take second dimension!
        self._nineq = self._G.shape[1]
        self._xdim = self._X.shape[1]

    def eq_resid(self, X, Y):
        """B is now varying, we should extract it from X"""
        G, b, d = self.rebuild_Gbd_from_X(X)
        return b - Y @ self.A.T

    def rebuild_Gbd_from_X(self, X):
        # Reshape X to match the first self.n_varying_rows rows of G
        G_size = self.n_varying_rows*self.ydim
        b_size = self.neq
        flattened_G = X[:, :G_size]
        b = X[:, G_size:G_size+b_size]
        d = X[:, G_size+b_size:]
        custom_G = flattened_G.reshape(X.shape[0], self.n_varying_rows, self._ydim)  # Reshape for the batch size

        # Take only the first sample of G and clone it for modification
        G = self.G_base.clone()  # Shape is (M, P)

        # Repeat G for the batch size to avoid memory overlap
        G = G.unsqueeze(0).repeat(X.shape[0], 1, 1)  # Shape is (batch_size, M, P)

        # Assign custom_G to the first self.n_varying_rows rows of G
        G[:, :self.n_varying_rows, :] = custom_G  # Ensure dimensions match
        return G, b, d
    
    def ineq_resid(self, X, Y):
        """
        For the ineq resid, we need to extract the first n rows of the G matrix from it's flattened form X, and plug them into G.
        """

        G, b, d = self.rebuild_Gbd_from_X(X)

        # resid = Y @ G.transpose(1, 2) - h
        residual = torch.bmm(Y.unsqueeze(1), G.transpose(1, 2)).squeeze(1) - d

        # Compute inequality residual
        return residual
    
    def opt_solve(self, X, solver_type="osqp", tol=1e-4):
        """We change op_solve so that the varying G matrices are taken from the input X.
        """
        if solver_type == "osqp":
            print("running osqp")
            Q, p, b, d = self.Q_np, self.p_np, self.b_np, self.d_np
            G, b, d = self.rebuild_Gbd_from_X(X)
            G_np, b_np, d_np = G.detach().cpu().numpy(), b.detach().cpu().numpy(), d.detach().cpu().numpy()
            A = self.A_np
            Y = []
            total_time = 0
            for idx, Gi in enumerate(G_np):
                solver = osqp.OSQP()
                my_A = np.vstack([A, Gi])
                my_l = np.hstack([b_np[idx], -np.ones(d_np[idx].shape[0]) * np.inf])
                my_u = np.hstack([b_np[idx], d_np[idx]])
                solver.setup(
                    P=csc_matrix(Q),
                    q=p,
                    A=csc_matrix(my_A),
                    l=my_l,
                    u=my_u,
                    verbose=False,
                    eps_prim_inf=tol,
                )
                start_time = time.time()
                results = solver.solve()
                end_time = time.time()

                total_time += end_time - start_time
                if results.info.status == "solved":
                    Y.append(results.x)
                else:
                    Y.append(np.ones(self.ydim) * np.nan)

            sols = np.array(Y)
            parallel_time = total_time / len(X)

        else:
            raise NotImplementedError

        return sols, total_time, parallel_time

    def calc_Y(self):
        Y = self.opt_solve(self.X)[0]
        feas_mask = ~np.isnan(Y).all(axis=1)
        self._num = feas_mask.sum()
        self._X = self._X[feas_mask]
        self._Y = torch.tensor(Y[feas_mask])
        return Y
    

class ScaledLPProblem(SimpleProblem):
    def __init__(self, Q, p, A, G, b, d, obj_coeff, valid_frac=0.1, test_frac=0.1):
        super().__init__(Q, p, A, G, b, d, valid_frac, test_frac)

        self._X = self._b
        self._c = torch.tensor(obj_coeff)
        self._num = self._X.shape[0]
        self._neq = self._A.shape[0]
        self._nineq = self._G.shape[0]
        self._xdim = self._X.shape[1]

    def eq_resid(self, X, Y):
        return X - Y @ self.A.T

    def ineq_resid(self, X, Y):
        return Y @ self.G.T - self.d

    def obj_fn(self, Y):
        return Y @ self._c.T

    
    def opt_solve(self, X, solver_type="osqp", tol=1e-4):
        if solver_type == "osqp":
            print("running osqp")
            Q, p, A, G, d = self.Q_np, self.p_np, self.A_np, self.G_np, self.d_np
            c = self._c.numpy()
            X_np = X.detach().cpu().numpy()
            Y = []
            total_time = 0
            zero_Q = np.zeros((c.shape[0], c.shape[0]))

            for Xi in X_np:
                solver = osqp.OSQP()
                my_A = np.vstack([A, G])
                my_l = np.hstack([Xi, -np.ones(d.shape[0]) * np.inf])
                my_u = np.hstack([Xi, d])
                solver.setup(
                    q=c,
                    A=csc_matrix(my_A),
                    l=my_l,
                    u=my_u,
                    verbose=False,
                    eps_prim_inf=tol,
                )
                start_time = time.time()
                results = solver.solve()
                end_time = time.time()

                total_time += end_time - start_time
                if results.info.status == "solved":
                    Y.append(results.x)
                else:
                    Y.append(np.ones(self.ydim) * np.nan)

                sols = np.array(Y)
                parallel_time = total_time / len(X_np)
        else:
            raise NotImplementedError

        return sols, total_time, parallel_time

    def calc_Y(self):
        Y = self.opt_solve(self.X)[0]
        feas_mask = ~np.isnan(Y).all(axis=1)
        self._num = feas_mask.sum()
        self._X = self._X[feas_mask]
        self._Y = torch.tensor(Y[feas_mask])
        return Y