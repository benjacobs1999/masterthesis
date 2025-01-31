import torch
torch.set_default_dtype(torch.float64)

import numpy as np
import osqp
from qpth.qp import QPFunction
from scipy.sparse import csc_matrix
import time
from abc import ABC, abstractmethod


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
