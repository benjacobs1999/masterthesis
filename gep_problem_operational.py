import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataScaler:
    def __init__(self, X, eq_cm, ineq_cm, eq_rhs, ineq_rhs):
        # Store original data
        self.X_raw = X
        self.eq_rhs_raw = eq_rhs
        self.ineq_rhs_raw = ineq_rhs
        
        # Calculate scaling factor for MW values
        # Use a simple scale factor like 1000 (if values are in MW)
        # or determine dynamically from data
        self.mw_scale = self._determine_scale_factor(X)
        
        # Apply the same scaling to X and all RHS values
        self.X = self._scale(X)
        self.eq_rhs = self._scale(eq_rhs)
        self.ineq_rhs = self._scale(ineq_rhs)
        
        # Don't scale coefficient matrices as they represent relationships
        self.eq_cm = eq_cm
        self.ineq_cm = ineq_cm
    
    def _determine_scale_factor(self, X):
        return self.X_raw.mean()  # Example: scale down by 1000
    
    def _scale(self, values):
        # Simple scaling without mean subtraction
        return values / self.mw_scale
    
    def inverse_transform(self, scaled_values):
        # Convert back to original scale
        return scaled_values * self.mw_scale

class GEPOperationalProblemSet():

    def __init__(self, args, T, N, G, L, pDemand, pGenAva, pVOLL, pWeight, pRamping, pInvCost, pVarCost, pUnitCap, pExpCap, pImpCap, target_path):
        # ! Assume y = [p_{g,t}, f_{l,t}, md_{n,t}, ui_g]
        # Parameter for ui_g
        # self.pUnitInvestment = [4130.05009001755, 11232.550865341998] # BEL/GER, SunPV
        # self.pUnitInvestment = [361.86402118882216, 0.0, 26.82598871780973, 164.8367340122868]

        # if args["device"] == "mps":
        #     self.DTYPE = torch.float32
        #     self.DEVICE = torch.device("mps")
        # else:
        #     self.DTYPE = torch.float64
        #     self.DEVICE = torch.device("cpu")
        
        self.DTYPE = torch.float64
        self.DEVICE = torch.device("cpu")

        torch.set_default_dtype(self.DTYPE)
        torch.set_default_device(self.DEVICE)

        # Whether we will use Benders in compact form or not.
        self.benders_compact_form = args["benders_compact"]

        # Args:
        self.args = args
        # Operation sample is always of length 1 for benders!
        self.sample_duration = 1
        # self.sample_duration = args["sample_duration"]
        self.train = args["train"]
        self.valid = args["valid"]
        self.test = args["test"]

        # Number of samples to generate (through randomly chosen investment variables)
        if args["num_synthetic_samples"] > 0:
            self.n_samples = args["num_synthetic_samples"]
        else:
            self.n_samples = len(T) - 1

        # Input Sets
        self.T = T
        self.G = G
        self.L = L
        self.N = N

        self.num_g = len(G)
        self.num_l = len(L)
        self.num_n = len(N)

        # Input Parameters (Dictionaries)
        self.pDemand = pDemand
        self.pGenAva = pGenAva
        self.pVOLL = pVOLL
        self.pWeight = pWeight
        self.pRamping = pRamping
        self.pInvCost = pInvCost
        self.pVarCost = pVarCost
        self.pUnitCap = pUnitCap
        self.pExpCap = pExpCap
        self.pImpCap = pImpCap

        # Value can be None (we don't perturb) or a 
        if self.args['perturb_operating_costs']:
            self.pVarCost = self.perturb_operating_costs(self.args['perturb_operating_costs'])
        
        # Number of variables per timestep -- per timestep, variables are [p_g, f_l, md_n]
        self.n_var_per_t = self.num_g + self.num_l + self.num_n
        # Number of inequality constraints per timestep -- lower and upper bounds (2*) for p_g, f_l, md_n
        self.n_ineq_per_t = 2 * (self.num_g + self.num_l + self.num_n)
        # Number of equality constraints per timestep
        self.n_eq_per_t = self.num_n

        # Number of timesteps per data sample
        assert (float(len(T)) / self.sample_duration).is_integer() # Total number of timesteps should be divisible by the number of timesteps per data sample.

        # Time slices, each slice makes a different sample.
        self.time_ranges = [range(i, i + self.sample_duration, 1) for i in range(1, len(T), self.sample_duration)]

        self.neq = self.n_eq_per_t * self.sample_duration
        self.nineq = self.n_ineq_per_t * self.sample_duration
        self.n_prod_vars = self.num_g * self.sample_duration
        self.n_line_vars = self.num_l * self.sample_duration
        self.n_md_vars = self.num_n * self.sample_duration
        self.ydim = self.n_prod_vars + self.n_line_vars + self.n_md_vars

        self._opt_targets = self.load_targets(target_path)
        if args["num_synthetic_samples"] > 0:
            self.pUnitInvestment = self.generate_ui_data(num_samples=self.n_samples)
        else:
            self.pUnitInvestment = self._opt_targets["y_investment"].to(self.DTYPE).to(self.DEVICE)
        # self.pUnitInvestment = self.generate_ui_data(num_samples=self.n_samples)

        #! Test with randomized optimal objectives: --> This breaks learning!
        # self.pUnitInvestment = self.pUnitInvestment[torch.randperm(self.pUnitInvestment.shape[0])]

        #! Test if missed demand is the problem: --> This does not break learning.
        # self.pUnitInvestment *= 0.5

        #! Test if max cap is the problem: --> This does break learning.
        self.pUnitInvestment *= 2.0

        # Masks for node balance!
        # Initialize mask
        self.node_to_gen_mask = torch.zeros((len(N), len(G)), dtype=self.DTYPE)

        # Populate mask
        for g_idx, (node, _) in enumerate(G):
            node_idx = N.index(node)
            self.node_to_gen_mask[node_idx, g_idx] = 1
        
        # Initialize mask
        self.lineflow_mask = torch.zeros((len(N), len(L)), dtype=self.DTYPE)

        # Populate mask (directed adjacency matrix), -1 where line starts, 1 where line stops
        for l_idx, (start_node, end_node) in enumerate(L):
            start_idx = N.index(start_node)
            end_idx = N.index(end_node)
            self.lineflow_mask[start_idx, l_idx] = -1
            self.lineflow_mask[end_idx, l_idx] = 1
        
        self.md_indices = self.get_md_nt_indices()
        self.f_lt_indices = self.get_f_lt_indices()

        # Create constraint matrices, rhs and obj_fns
        print("Populating ineq constraints")
        self.ineq_cm, self.ineq_rhs = self.build_ineq_cm_rhs()
        print("Populating eq constraints")
        self.eq_cm, self.eq_rhs = self.build_eq_cm_rhs()
        print("Creating objective coefficients")
        self.obj_coeff, self.obj_coeff_log, self.obj_coeff_train = self.build_obj_coeff()
        self.X = self.build_X(self.eq_rhs, self.ineq_rhs)
        # self.X = self.eq_rhs #! Test whether ineq_rhs gives extra information. --> It does, won't learn without it.
            
        # self.X = self.build_X_alternative()
        # self.X = torch.concat([self.eq_rhs, self.ineq_rhs], dim=1)
        self.xdim = self.X.shape[1]

        # Split x into training, val, test sets.
        self._split_X_in_sets(self.train, self.valid, self.test)

        if args["scale_input"]:
            self.scaler = DataScaler(self.X, self.eq_cm, self.ineq_cm, self.eq_rhs, self.ineq_rhs)
            self.X = self.scaler.X
            self.eq_rhs = self.scaler.eq_rhs
            self.ineq_rhs = self.scaler.ineq_rhs

        if self.args["device"] == 'mps':
            self.obj_coeff = self.obj_coeff.to(torch.float32).to(torch.device('mps'))
            self.obj_coeff_log = self.obj_coeff_log.to(torch.float32).to(torch.device('mps'))
            self.obj_coeff_train = self.obj_coeff_train.to(torch.float32).to(torch.device('mps'))
            self.node_to_gen_mask = self.node_to_gen_mask.to(torch.float32).to(torch.device('mps'))
            self.lineflow_mask = self.lineflow_mask.to(torch.float32).to(torch.device('mps'))
    
    # Split the data
    @property
    def train_indices(self): return self._train_indices
    @property
    def valid_indices(self): return self._valid_indices
    @property
    def test_indices(self): return self._test_indices
    
    @property
    def opt_targets(self): return self._opt_targets

    def load_targets(self, target_path):
        with open(target_path, 'rb') as file:
            return pickle.load(file)

    def generate_ui_data(self, num_samples=10000, max_inv=100000):
        exp_dist = torch.distributions.Exponential(rate=0.01)
        return exp_dist.sample((num_samples, self.num_g))
    
    def perturb_operating_costs(self, noise=0.01):
        """Perturbs the operating costs to remove symmetry in the solution space -- This stagnates the learning of the neural network.

        Args:
            noise (_type_, optional): _description_. Defaults to 1e-6.

        Returns:
            _type_: _description_
        """
        # Group keys by their cost value
        groups = {}
        for key, cost in self.pVarCost.items():
            groups.setdefault(cost, []).append(key)

        perturbed_pVarCost = {}
        for cost, keys in groups.items():
            if len(keys) > 1:
                n = len(keys)
                # Generate evenly spaced noise values between -noise and noise.
                noise_values = [ -noise + (2 * noise * i) / (n - 1) for i in range(n) ]
                for key, noise_val in zip(keys, noise_values):
                    perturbed_pVarCost[key] = cost + noise_val*cost
            else:
                perturbed_pVarCost[keys[0]] = cost

        return perturbed_pVarCost

    def get_md_nt_indices(self):
        indices = []
        for i in range(self.sample_duration):
            # We have p_g and f_l first, then come the md_n.
            offset = i * self.n_var_per_t + self.num_g + self.num_l
            indices_this_t = list(range(offset, offset + self.num_n))
            indices.extend(indices_this_t)
        
        return indices
    
    def get_f_lt_indices(self):
        indices = []
        for i in range(self.sample_duration):
            # We have p_g first, then f_l.
            offset = i * self.n_var_per_t + self.num_g
            indices_this_t = list(range(offset, offset + self.num_l))
            indices.extend(indices_this_t)
        
        return indices

    # TODO: Change for compact form!
    def split_ineq_constraints(self, ineq):
        """Groups the inequality [residuals or RHS] into constraints by constraint type.
            Assume ineq = [3.1h, 3.1b, 3.1d, 3.1e, 3.1i, 3,1j] * T (per sample in batch)

            returns each constraint type in the form [Batchsize, nr_of_constraints, time]

            Returns h, b, d, e, i, j
        """
        batch_size = ineq.shape[0]  # Get batch size

        # Reshape ineq to [batch_size, sample_duration, n_ineq_per_t]
        # This keeps constraints grouped per time step
        ineq = ineq.view(batch_size, self.sample_duration, self.n_ineq_per_t).permute(0, 2, 1)

        # Define segment sizes based on multiple constraints per type
        sizes = [self.num_g, self.num_g, self.num_l, self.num_l, self.num_n, self.num_n]  # Num constraints per type

        # Extract constraint tensors along the first dimension (constraint type)
        h, b, d, e, i, j = torch.split(ineq, sizes, dim=1)

        return h, b, d, e, i, j
    
    # TODO: Change for compact form!
    def split_eq_constraints(self, eq, log=False):
        """Groups the inequality [residuals or RHS] into constraints by constraint type.
            Assume ineq = [3.1c] * T (per sample in batch)

            returns each constraint type in the form [Batchsize, nr_of_constraints, time]

            Returns c
        """
        batch_size = eq.shape[0]
        # If we are using compact form, we have ui_g = UI_G constraints prepended.
        if self.benders_compact_form and not log:
            ui_g = eq[:, :self.num_g]
            ui_g = ui_g.view(batch_size, self.sample_duration, self.num_g).permute(0, 2, 1)
            c = eq[:, self.num_g:]
            c = c.view(batch_size, self.sample_duration, self.n_eq_per_t).permute(0, 2, 1)
        else:
            ui_g = None
            c = eq.view(batch_size, self.sample_duration, self.n_eq_per_t).permute(0, 2, 1)
        return ui_g, c

    # TODO: Change for compact form
    def split_dec_vars_from_Y_raw(self, Y):
        """Groups the decision variables from the NN output BEFORE REPAIRS by type.
            Assume y =  [p_{g,t0}, f_{l,t0}, 
                         p_{g,t1}, f_{l,t1}, ..., 
                         p_{g,tn}, f_{l,tn}] (per sample in batch)

            Returns p_{g,t}, f_{l,t}
        """
        batch_size = Y.shape[0]  # Get batch size

        if self.benders_compact_form:
            ui_g = Y[:, :self.num_g]
            Y = Y[:, self.num_g:]
        else:
            ui_g = None

        # Reshape Y for efficient slicing
        Y = Y.view(batch_size, self.sample_duration, self.n_var_per_t - self.num_n).permute(0, 2, 1)

        # Define segment sizes
        sizes = [self.num_g, self.num_l]

        # Extract constraint tensors
        p_gt, f_lt = torch.split(Y, sizes, dim=1)

        return ui_g, p_gt, f_lt

    # TODO: Change for compact form
    def split_dec_vars_from_Y(self, Y, log=False):
        """Groups the decision variables from the NN output AFTER REPAIRS by type.
            Assume y =  [p_{g,t0}, f_{l,t0}, md_{n,t0}, 
                         p_{g,t1}, f_{l,t1}, md_{n,t1}, ..., 
                         p_{g,tn}, f_{l,tn}, md_{n,tn}]

            Returns p_{g,t}, f_{l,t}, md_{n,t} in the shape [B, (G|L|N), T]
        """
        batch_size = Y.shape[0]  # Get batch size

        # Reshape Y to [batch_size, sample_duration, -1] for efficient slicing
        if self.benders_compact_form and not log:
            Y = Y[:, self.num_g:]

        Y = Y.view(batch_size, self.sample_duration, self.n_var_per_t).permute(0, 2, 1)

        # Define segment sizes
        sizes = [self.num_g, self.num_l, self.num_n]

        # Extract constraint tensors
        p_gt, f_lt, md_nt = torch.split(Y, sizes, dim=1)

        return p_gt, f_lt, md_nt

    def ineq_resid(self, Y, ineq_cm, ineq_rhs):
        return torch.bmm(ineq_cm, Y.unsqueeze(-1)).squeeze(-1) - ineq_rhs

    def ineq_dist(self, Y, ineq_cm, ineq_rhs):
        resids = self.ineq_resid(Y, ineq_cm, ineq_rhs)
        return torch.clamp(resids, 0)

    def eq_resid(self, Y, eq_cm, eq_rhs):
        torch.bmm(eq_cm, Y.unsqueeze(-1)).squeeze(-1)
        return eq_rhs - torch.bmm(eq_cm, Y.unsqueeze(-1)).squeeze(-1)
    
    def dual_ineq_resid(self, mu, lamb):
        """Dual inequality Residual, takes on the form:
            -mu <= 0
        """
        return -mu

    def dual_eq_resid(self, mu, lamb, eq_cm, ineq_cm):
        """Dual equality Residual, takes on the form:
            G^T \mu + H^T \lambda + c = 0"""
        return self.obj_coeff - (torch.bmm(eq_cm.transpose(1, 2), lamb.unsqueeze(-1)).squeeze(-1) - torch.bmm(ineq_cm.transpose(1, 2), mu.unsqueeze(-1)).squeeze(-1))

    def obj_fn(self, Y):
        # obj_coeff does not need batching, objective is the same over different samples.
        
        # Take the absolute value of the missed demand in the obj function to penalize negative missed demand
        Y = Y.clone()
        Y[:, self.md_indices] = Y[:, self.md_indices].abs()
        
        return self.obj_coeff @ Y.T
    
    def obj_fn_train(self, Y):
        # Objective function adjusted for training (different than the actual objective function)
        # Y = Y.clone()
        # Y[:, self.md_indices] = torch.relu(Y[:, self.md_indices])
        # Y[:, self.md_indices] = Y[:, self.md_indices]**2
        # Y[:, self.md_indices] = 0
        # Y[:, self.md_indices] = Y[:, self.md_indices].abs()
        return self.obj_coeff_train @ Y.T
        # reg_term_md = torch.norm(Y[:, self.md_indices], p=1, dim=1) #l1 regularization term
        # reg_term_f = torch.norm(Y[:, self.f_lt_indices], p=1, dim=1) #l1 regularization term
        # return self.obj_coeff @ Y.T + 0.1*(reg_term_md + reg_term_f)
    
    def obj_fn_log(self, Y):
        # obj_coeff does not need batching, objective is the same over different samples.
        return self.obj_coeff_log @ Y.T
    
    def dual_obj_fn(self, eq_rhs, ineq_rhs, mu, lamb):
        # Batched dot product
        ineq_term = torch.sum(mu * ineq_rhs, dim=1)
        
        # Batched dot product
        eq_term = torch.sum(lamb * eq_rhs, dim=1)

        return eq_term - ineq_term
        # return (ineq_term + eq_term)
    
    def build_obj_coeff(self,):
        """ Builds the objective function coefficients (only the operating costs)
            Assume y =  [p_{g,t0}, f_{l,t0}, md_{n,t0}, 
                         p_{g,t1}, f_{l,t1}, md_{n,t1}, ..., 
                         p_{g,tn}, f_{l,tn}, md_{n,tn}] """

        # coeff vector of zero's to be filled
        c = torch.zeros(self.n_var_per_t*self.sample_duration, dtype=self.DTYPE)

        for t in range(self.sample_duration):
            # Sum over G,T (PC_g * p_{g,t}) --> Generation costs
            for idx_g, g in enumerate(self.G):
                # Generator variables are always at first |G| indices per timestep
                coeff_idx = t * self.n_var_per_t + idx_g
                c[coeff_idx] = self.pVarCost[g]

            # Sum over N,T (MDC * md_{n,t}) --> Cost missed demand
            for idx_n in range(self.num_n):
                # Missed demand variables come after generator and lineflow variables
                coeff_idx = t * self.n_var_per_t + self.num_g + self.num_l + idx_n
                c[coeff_idx] = self.pVOLL
        
        if self.benders_compact_form:
            c_compact = torch.cat((torch.zeros(self.num_g), c), 0)
        else:
            c_compact = c.clone()

        # During training, we don't want to penalize missed demand in the objective function, but in the loss function (augmented Lagrangian).
        c_train = c.clone()
        c_train[self.md_indices] = 0

        # ! How does pWeight influence the primal/dual solutions??
        # c_compact *= self.pWeight
        # c *= self.pWeight

        return c_compact, c, c_train

    def build_X(self, eq_rhs, ineq_rhs):
        """Builds a tensor containing the features that vary across problem instances. For now, only supports changes in RHS.
        """
        # For ineq RHS, only 3.1b varies across instances --> first |G| constraints are 3.1h, second |G| constraints are 3.1b.
        ineq_rhs_varying_indices = [t * self.n_ineq_per_t + self.num_g + i for t in range(self.sample_duration) for i in range(self.num_g)]

        # The entire RHS changes for equality constraints
        X = torch.concat([self.eq_rhs, self.ineq_rhs[:, ineq_rhs_varying_indices]], dim=1)

        return X
    
    def build_X_alternative(self,):
        X = []
        for t in range(1, self.n_samples - 1):
            x = []
            for idx, g in enumerate(self.G):
                x.append(self.pUnitInvestment[t, idx])
            for n in self.N:
                x.append(self.pDemand[(n, t)])
            for g in self.G:
                x.append(self.pGenAva.get((*g, t), 1.0))
            X.append(x)
        return torch.tensor(X)
            
                
    def build_ineq_cm_rhs(self,):
        ineq_cms = []
        ineq_rhss = []

        # for sample_idx, time_range in enumerate(self.time_ranges):
        for sample_idx in range(self.n_samples):
            ineq_cm, ineq_rhs = self.build_ineq_cm_rhs_sample(sample_idx=sample_idx)
            ineq_cms.append(ineq_cm)
            ineq_rhss.append(ineq_rhs)
        
        return torch.stack(ineq_cms), torch.stack(ineq_rhss)

    def build_eq_cm_rhs(self,):
        eq_cms = []
        eq_rhss = []

        for idx in range(self.n_samples):
            eq_cm, eq_rhs = self.build_eq_cm_rhs_sample(idx)
            eq_cms.append(eq_cm)
            eq_rhss.append(eq_rhs)
        
        return torch.stack(eq_cms), torch.stack(eq_rhss)
    
    # Bounds are always identity, -1 for lower bounds, +1 for upper bounds.
    def assign_identity_or_scalar(self, matrix, row_start, col_start, size, value=1):
        """ Assign identity matrix if size > 1, otherwise assign a scalar value. """
        if size > 1:
            matrix[row_start:row_start + size, col_start:col_start + size] = value*torch.eye(size)
        else:
            matrix[row_start, col_start] = value

    # TODO: Change for compact form
    def build_ineq_cm_rhs_sample(self, sample_idx):
        """For a single data sample characterised by the time_range, 
        build the full inequality constraint matrix for all timesteps using Kronecker product.
        """

        # TODO: Convert to sparse matrices for efficiency??

        # num_timesteps = len(time_range)
        num_columns_per_t = self.n_var_per_t

        # Compute number of constraint rows per timestep
        num_rows_per_t = 2 * (self.num_g + self.num_l + self.num_n)

        # Create a single block for one timestep
        block_matrix = torch.zeros((num_rows_per_t, num_columns_per_t))

        # Apply constraints
        # Bounds are always identity, -1 for lower bounds, +1 for upper bounds.
        self.assign_identity_or_scalar(block_matrix, 0, 0, self.num_g, -1) # 3.1h: Production lower bound
        self.assign_identity_or_scalar(block_matrix, self.num_g, 0, self.num_g, 1) # 3.1b: Production upper bound
        self.assign_identity_or_scalar(block_matrix, 2*self.num_g, self.num_g, self.num_l, -1) # 3.1d: Lineflow lower bound
        self.assign_identity_or_scalar(block_matrix, 2*self.num_g + self.num_l, self.num_g, self.num_l, 1) # 3.1e: Lineflow upper bound
        self.assign_identity_or_scalar(block_matrix, 2*self.num_g + 2*self.num_l, self.num_g + self.num_l, self.num_n, -1) # 3.1i: Missed demand lower bound
        self.assign_identity_or_scalar(block_matrix, 2*self.num_g + 2*self.num_l + self.num_n, self.num_g + self.num_l, self.num_n, 1)  # 3.1j: Missed demand upper bound

        # Use Kronecker product to copy the block matrix diagonally for all timesteps
        #! Sample duration is always 1
        ineq_cm = torch.kron(torch.eye(self.sample_duration), block_matrix)

        if self.benders_compact_form:
            # Append columns of zeroes (one for each generator), there are no inequality constraints on the investment variables
            zeros = torch.zeros((ineq_cm.shape[0], self.num_g))
            ineq_cm = torch.cat([zeros, ineq_cm], dim=1)

        ineq_rhs = []

        # Create right hand side:
        # for t in time_range:

        # If we have more synthetic samples than timesteps, keep it within bounds through modulo (start counting again from 0 once we go out of bounds)
        t = (sample_idx % len(self.T)) + 1

        # 3.1h: Production lower bound
        ineq_rhs += [0 for _ in range(self.num_g)]
        # 3.1b: Production upper bound
        ineq_rhs += [self.pGenAva.get((*g, t), 1.0) * self.pUnitCap[g] * self.pUnitInvestment[sample_idx, idx] for idx, g in enumerate(self.G)]
        # 3.1d: Lineflow lower bound
        ineq_rhs += [self.pImpCap[l] for l in self.L]
        # 3.1e: Lineflow upper bound
        ineq_rhs += [self.pExpCap[l] for l in self.L]
        # 3.1i: Missed demand lower bound
        ineq_rhs += [0 for _ in range(self.num_n)]
        # 3.1j: Missed demand upper bound
        ineq_rhs += [self.pDemand[(n, t)] for n in self.N]
            
        return ineq_cm, torch.tensor(ineq_rhs)

    def build_eq_cm_rhs_sample(self, sample_idx):
        """Build the constraint matrix for the equality constraints
        """
        # TODO: Change operational problem to remove slack variable.

        # p_{g}, f_in - f_out, md_n
        block_matrix = torch.concat([self.node_to_gen_mask, self.lineflow_mask, torch.eye(self.num_n)], dim=1)

        # Use Kronecker product to copy the block matrix diagonally for all timesteps
        eq_cm = torch.kron(torch.eye(self.sample_duration), block_matrix)

        # If we are using the compact form, add ui_g to the constraint matrix. (for ui_g == UI_g constraints)
        if self.benders_compact_form:
            # Prepend ui_g constraints (top left corner of cm)
            ui_g = torch.eye(self.num_g)
            eq_cm = torch.block_diag(ui_g, eq_cm)

        eq_rhs = []
        # If we are using the compact form, add investment 'parameter' to the RHS.
        # TODO: Is it correct to take RHS (UI_g) from Gurobi optimal solution for ui_g? Or should training data be generated to include various UI_G's (like it will encounter in Benders)
        if self.benders_compact_form:
            eq_rhs += self.opt_targets["y_investment"][sample_idx].tolist()

        # Build right hand side:
        # If we have more synthetic samples than timesteps, keep it within bounds through modulo (start counting again from 0 once we go out of bounds)
        t = (sample_idx % len(self.T)) + 1

        eq_rhs += [self.pDemand[(n, t)] for n in self.N]
                
        return eq_cm, torch.tensor(eq_rhs)

    def _split_X_in_sets(self, train=0.8, valid=0.1, test=0.1):
        # Ensure the split ratios sum to 1
        assert train + valid + test == 1.0

        # Total number of samples
        B = self.X.size(0)
        indices = torch.arange(B)

        # Compute sizes for each set
        train_size = int(train * B)
        valid_size = int(valid * B)

        # Split the indices
        self._train_indices = indices[:train_size]
        self._valid_indices = indices[train_size:train_size+valid_size]
        self._test_indices = indices[train_size+valid_size:]

        # Convert time_ranges to a tensor or use list comprehension
        # time_ranges_tensor = torch.tensor(self.time_ranges)

        # Split time ranges
        # self.train_time_ranges = time_ranges_tensor[self.train_indices].tolist()
        # self.val_time_ranges = time_ranges_tensor[self.valid_indices].tolist()
        # self.test_time_ranges = time_ranges_tensor[self.test_indices].tolist()

        print(f"Size of train set: {train_size}")
        print(f"Size of val set: {valid_size}")
        print(f"Size of test set: {B - train_size - valid_size}")
    
    def total_prod(self, p_gt):
        p_nt = torch.einsum('ng,bgt->bnt', self.node_to_gen_mask.to(dtype=self.DTYPE), p_gt)
        return p_nt
    
    def net_flow(self, f_lt):
        net_flow_nt = torch.einsum('nl,blt->bnt', self.lineflow_mask.to(dtype=self.DTYPE), f_lt)
        return net_flow_nt

    def net_balance(self, p_gt, f_lt):
        p_nt = self.total_prod(p_gt)
        net_flow_nt = self.net_flow(f_lt)

        net_balance = p_nt + net_flow_nt
        return net_balance
    
    def missed_demand(self, p_gt, f_lt, D_nt):
        net_balance = self.net_balance(p_gt, f_lt)
        md_nt = D_nt - net_balance

        return md_nt

    def plot_balance(self, primal_net, dual_net):
        sample = 0
        with torch.no_grad():
            # Predictions for primal variables
            Y = primal_net(self.X[self.train_indices], self.eq_rhs[self.train_indices], self.ineq_rhs[self.train_indices])
            
            # Extract decision variables
            p_gt, f_lt, md_nt = self.split_dec_vars_from_Y(Y)  # [B, (G|L|N), T]
            UI_g, D_nt = self.split_eq_constraints(self.eq_rhs[sample:sample+1])  # [1, N, T]
            
            # Convert to numpy and ensure correct shape
            total_prod = self.total_prod(p_gt[sample:sample+1])[0].cpu().numpy()  # Shape [N, T], always positive
            net_flow = self.net_flow(f_lt[sample:sample+1])[0].cpu().numpy()  # Shape [N, T], can be positive or negative
            missed_demand = md_nt[sample:sample+1][0].cpu().numpy()  # Shape [N, T], can be positive or negative
            demand = D_nt[0].cpu().numpy()  # Shape [N, T], total sum should match this
            
            num_nodes, num_timesteps = total_prod.shape
            
            fig, ax = plt.subplots(num_nodes, 1, figsize=(10, 6 * num_nodes), sharex=True)
            
            if num_nodes == 1:
                ax = [ax]
            
            for node in range(num_nodes):
                # Get data for the current node
                tp = total_prod[node]  # Always positive
                nf = net_flow[node]  # Can be positive or negative
                md = missed_demand[node]  # Can be positive or negative
                d = demand[node]  # Total sum of all components
                
                # Define positions
                timesteps = np.arange(num_timesteps)
                
                # Separate positive and negative components
                md_pos = np.clip(md, 0, None)
                md_neg = np.clip(md, None, 0)
                nf_pos = np.clip(nf, 0, None)
                nf_neg = np.clip(nf, None, 0)
                
                # Compute stacking order
                base_pos = np.zeros(num_timesteps)
                base_neg = np.zeros(num_timesteps)
                
                # Plot positive values above zero
                ax[node].bar(timesteps, tp, bottom=base_pos, label='Total Production', color='blue')
                base_pos += tp
                
                ax[node].bar(timesteps, nf_pos, bottom=base_pos, label='Net Flow (+)', color='green')
                base_pos += nf_pos
                
                ax[node].bar(timesteps, md_pos, bottom=base_pos, label='Missed Demand (+)', color='orange')
                base_pos += md_pos
                
                # Plot negative values below zero
                ax[node].bar(timesteps, md_neg, bottom=base_neg, label='Missed Demand (-)', color='darkorange')
                base_neg += md_neg
                
                ax[node].bar(timesteps, nf_neg, bottom=base_neg, label='Net Flow (-)', color='darkgreen')
                base_neg += nf_neg
                
                # Add demand as red dots
                ax[node].scatter(timesteps, d, color='red', label='Demand', zorder=3)
                
                # Make the 0 line thick
                ax[node].axhline(0, color='black', linewidth=2)
                
                ax[node].set_ylabel('Values')
                ax[node].set_title(f'Node {node}')
                ax[node].legend()
                ax[node].grid(True)
            
            ax[-1].set_xlabel('Timesteps')
            plt.tight_layout()
            plt.show()
        
    def plot_decision_variable_diffs(self, primal_net, dual_net):
        sample = 0
        with torch.no_grad():
            Y = primal_net(self.X[self.train_indices], self.eq_rhs[self.train_indices], self.ineq_rhs[self.train_indices])
            Y_target = self.opt_targets["y_operational"][self.train_indices]

            mu, lamb = dual_net(self.X[self.train_indices], self.eq_cm[self.train_indices])
            mu_target = self.opt_targets["mu_operational"][self.train_indices]
            lamb_target = self.opt_targets["lamb_operational"][self.train_indices]
            
            p_gt, f_lt, md_nt = self.split_dec_vars_from_Y(Y)  
            p_gt_target, f_lt_target, md_nt_target = self.split_dec_vars_from_Y(Y_target, log=True)  

            dual_lb_p, dual_ub_p, dual_lb_f, dual_ub_f, dual_lb_md, dual_ub_md = self.split_ineq_constraints(mu)
            dual_target_lb_p, dual_target_ub_p, dual_target_lb_f, dual_target_ub_f, dual_target_lb_md, dual_target_ub_md = self.split_ineq_constraints(mu_target)

            ui_g, dual_node_balance = self.split_eq_constraints(lamb)
            ui_g, dual_node_balance_target = self.split_eq_constraints(lamb_target, log=True)

            num_gens, num_lines, num_nodes, num_timesteps = p_gt.shape[1], f_lt.shape[1], md_nt.shape[1], p_gt.shape[2]
            
            fig, axes = plt.subplots(2, 4, figsize=(24, 12), sharex=False)
            
            x_pos_g = np.arange(num_timesteps * num_gens)
            x_labels_g = [f"T{t}_G{g}" for g in range(num_gens) for t in range(num_timesteps)]
            x_pos_l = np.arange(num_timesteps * num_lines)
            x_labels_l = [f"T{t}_L{l}" for l in range(num_lines) for t in range(num_timesteps)]
            x_pos_n = np.arange(num_timesteps * num_nodes)
            x_labels_n = [f"T{t}_N{n}" for n in range(num_nodes) for t in range(num_timesteps)]
            
            column_titles = ["Generation Production", "Line Flows", "Missed Demand", "Node Balance Duals"]
            for col, title in enumerate(column_titles):
                axes[0, col].set_title(title, fontsize=14)
            
            def plot_variable(ax, x_pos, labels, pred, target, ylabel, color):
                ax.bar(x_pos, pred, label='Predicted', color=color, alpha=0.5)
                ax.scatter(x_pos, target, color='red', label='Target', zorder=3)
                ax.set_ylabel(ylabel)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(labels, rotation=90)
                ax.legend()
                ax.axhline(0, color='black', linewidth=2)
                ax.grid(True)
            
            plot_variable(axes[0, 0], x_pos_g, x_labels_g, p_gt[sample].reshape(-1), p_gt_target[sample].reshape(-1), 'Primal Generation', 'blue')
            plot_variable(axes[0, 1], x_pos_l, x_labels_l, f_lt[sample].reshape(-1), f_lt_target[sample].reshape(-1), 'Primal Line Flow', 'green')
            plot_variable(axes[0, 2], x_pos_n, x_labels_n, md_nt[sample].reshape(-1), md_nt_target[sample].reshape(-1), 'Primal Missed Demand', 'orange')
            
            def plot_dual_variable(ax, x_pos, labels, lb, ub, lb_target, ub_target, ylabel):
                ax.bar(x_pos, lb, label='Dual LB', color='purple', alpha=0.5)
                ax.bar(x_pos, ub, label='Dual UB', color='cyan', alpha=0.5)
                ax.scatter(x_pos, lb_target, color='purple', marker='o', label='Target LB', zorder=3, edgecolors='black')
                ax.scatter(x_pos, ub_target, color='cyan', marker='o', label='Target UB', zorder=2, edgecolors='black')
                ax.set_ylabel(ylabel)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(labels, rotation=90)
                ax.legend()
                ax.axhline(0, color='black', linewidth=2)
                ax.grid(True)
            
            plot_dual_variable(axes[1, 0], x_pos_g, x_labels_g, dual_lb_p[sample].reshape(-1), dual_ub_p[sample].reshape(-1), dual_target_lb_p[sample].reshape(-1), dual_target_ub_p[sample].reshape(-1), 'Dual (Gen)')
            plot_dual_variable(axes[1, 1], x_pos_l, x_labels_l, dual_lb_f[sample].reshape(-1), dual_ub_f[sample].reshape(-1), dual_target_lb_f[sample].reshape(-1), dual_target_ub_f[sample].reshape(-1), 'Dual (Flow)')
            plot_dual_variable(axes[1, 2], x_pos_n, x_labels_n, dual_lb_md[sample].reshape(-1), dual_ub_md[sample].reshape(-1), dual_target_lb_md[sample].reshape(-1), dual_target_ub_md[sample].reshape(-1), 'Dual (Missed Demand)')
            plot_variable(axes[1, 3], x_pos_n, x_labels_n, dual_node_balance[sample].reshape(-1), dual_node_balance_target[sample].reshape(-1), 'Dual Node Balance', 'brown')
            
            plt.tight_layout()
            plt.show()






            