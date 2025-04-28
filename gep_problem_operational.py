import os
import time
import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt

from data_wrangling import dataframe_to_dict
from gep_config_parser import parse_config
from gep_problem import GEPProblemSet
from scipy.stats import qmc

# TODO: Move scaling to trainer class.
# class DataScaler:
#     def __init__(self, X, eq_cm, ineq_cm, eq_rhs, ineq_rhs):
#         # Store original data
#         self.X_raw = X
#         self.eq_rhs_raw = eq_rhs
#         self.ineq_rhs_raw = ineq_rhs
        
#         # Calculate scaling factor for MW values
#         # Use a simple scale factor like 1000 (if values are in MW)
#         # or determine dynamically from data
#         self.mw_scale = self._determine_scale_factor(X)
        
#         # Apply the same scaling to X and all RHS values
#         self.X = self._scale(X)
#         self.eq_rhs = self._scale(eq_rhs)
#         self.ineq_rhs = self._scale(ineq_rhs)
        
#         # Don't scale coefficient matrices as they represent relationships
#         self.eq_cm = eq_cm
#         self.ineq_cm = ineq_cm
    
#     def _determine_scale_factor(self, X):
#         return self.X_raw.mean()  # Example: scale down by 1000
    
#     def _scale(self, values):
#         # Simple scaling without mean subtraction
#         return values / self.mw_scale
    
#     def inverse_transform(self, scaled_values):
#         # Convert back to original scale
#         return scaled_values * self.mw_scale

class GEPOperationalProblemSet():

    def __init__(self, args, T, N, G, L, pDemand, pGenAva, pVOLL, pWeight, pRamping, pInvCost, pVarCost, pUnitCap, pExpCap, pImpCap):
        
        if args["device"] == "mps":
            self.DTYPE = torch.float32
            self.DEVICE = torch.device("cpu")
        else:
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
        self.train = args["train"]
        self.valid = args["valid"]
        self.test = args["test"]

        # Number of samples to generate (through randomly chosen investment variables)
        self.n_samples = 2 ** args["2n_synthetic_samples"]

        # Input Sets
        self.T = T
        self.G = G
        self.L = L
        self.N = N

        self.num_g = len(G)
        self.num_l = len(L)
        self.num_n = len(N)

        # Input Parameters (Dictionaries)
        self.pDemand = pDemand # MW
        self.pGenAva = pGenAva # [0,1]
        self.pVOLL = pVOLL # $/MW
        self.pWeight = pWeight # Integer
        self.pRamping = pRamping # [0,1]
        self.pInvCost = pInvCost # $/MW
        self.pVarCost = pVarCost # $/MW
        self.pUnitCap = pUnitCap # MW
        self.pExpCap = pExpCap # MW
        self.pImpCap = pImpCap # MW 


        # Value can be None (we don't perturb) or a percentage (we perturb by that percentage)
        if self.args['perturb_operating_costs']:
            self.pVarCost = self.perturb_operating_costs(self.args['perturb_operating_costs'])
        
        # Number variables, variables are [p_g, f_l, md_n]
        self.n_vars = self.num_g + self.num_l + self.num_n

        # Number of timesteps per data sample
        assert (float(len(T)) / self.sample_duration).is_integer() # Total number of timesteps should be divisible by the number of timesteps per data sample.

        # Time slices, each slice makes a different sample.
        self.time_ranges = [range(i, i + self.sample_duration, 1) for i in range(1, len(T), self.sample_duration)]

        self.neq = self.num_n
        self.nineq = 2 * (self.num_g + self.num_l + self.num_n)
        self.n_prod_vars = self.num_g * self.sample_duration
        self.n_line_vars = self.num_l * self.sample_duration
        self.n_md_vars = self.num_n * self.sample_duration
        self.ydim = self.n_prod_vars + self.n_line_vars + self.n_md_vars

        # Generate unit investment data:
        self.pUnitInvestment = self.generate_ui_data(m=args["2n_synthetic_samples"], max_inv=args["max_investment"])

        # else:
        #     self._opt_targets = self.load_targets(target_path)
        #     self.pUnitInvestment = self._opt_targets["y_investment"].to(self.DTYPE).to(self.DEVICE)

        #! Test with randomized optimal objectives: --> This breaks learning!
        # self.pUnitInvestment = self.pUnitInvestment[torch.randperm(self.pUnitInvestment.shape[0])]

        #! Test if missed demand is the problem: --> This does not break learning.
        # self.pUnitInvestment *= 0.5

        #! Test if max cap is the problem: --> This does not break learning.
        # self.pUnitInvestment *= 2.0

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

        self.capacity_ub_indices = self.get_capacity_ub_indices()
        self.missed_demand_ub_indices = self.get_missed_demand_ub_indices()

        # Create constraint matrices, rhs and obj_fns
        print("Populating ineq constraints")
        self.ineq_cm, self.ineq_rhs = self.build_ineq_cm_rhs()
        print("Populating eq constraints")
        self.eq_cm = self.build_eq_cm()
        print("Creating objective coefficients")
        self.obj_coeff, self.cost_vec = self.build_obj_coeff()
        if self.args["synthetic_demand_capacity"]:
            self.X = self.build_synthetic_X()
        else:
            self.X = self.build_X()
        
        if self.args["normalize_input"]:
            self.total_demands = self.X[:, :self.data.num_n].sum(dim=1).unsqueeze(1)
            self.X /= self.total_demands
        else:
            self.total_demands = torch.ones((self.X.shape[0], 1))

        self.xdim = self.X.shape[1]

        if args["opt_targets"]:
            self.opt_targets = self.compute_opt_targets()

        # if self.args["device"] == 'mps':
            # self.obj_coeff = self.obj_coeff.to(torch.float32).to(torch.device('mps'))
            # self.cost_vec = self.cost_vec.to(torch.float32).to(torch.device('mps'))
            # self.node_to_gen_mask = self.node_to_gen_mask.to(torch.float32).to(torch.device('mps'))
            # self.lineflow_mask = self.lineflow_mask.to(torch.float32).to(torch.device('mps'))
            
    def to_mps(self):
        print("Converting to mps")
        self.obj_coeff = self.obj_coeff.to(torch.float32).to(torch.device('mps'))
        self.cost_vec = self.cost_vec.to(torch.float32).to(torch.device('mps'))
        self.node_to_gen_mask = self.node_to_gen_mask.to(torch.float32).to(torch.device('mps'))
        self.lineflow_mask = self.lineflow_mask.to(torch.float32).to(torch.device('mps'))
        self.ineq_rhs = self.ineq_rhs.to(torch.float32).to(torch.device('mps'))
        self.ineq_cm = self.ineq_cm.to(torch.float32).to(torch.device('mps'))
        self.eq_cm = self.eq_cm.to(torch.float32).to(torch.device('mps'))

    def load_targets(self, target_path):
        with open(target_path, 'rb') as file:
            return pickle.load(file)
    
    def save_data(self, save_path):
        with open(save_path, 'wb') as file:
            pickle.dump(self, file)

    def generate_ui_data(self, m=15, max_inv=100000, zero_fraction=0.15):
        dimensions = self.num_g
        sobol_sampler = qmc.Sobol(d=dimensions)
        
        # Generate base Sobol points (2^m points) -- [0, 1]^d
        base_points = sobol_sampler.random_base2(m=m)
        np.random.shuffle(base_points)
        # Scale to max_inv
        if self.args["device"] == "mps":
            points = torch.tensor(np.float32(base_points)) * max_inv
        else:
            points = torch.tensor(np.float64(base_points)) * max_inv
        
       #! TODO: Do we need samples to be exactly 0?

        return points
    
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
            offset = i * self.n_vars + self.num_g + self.num_l
            indices_this_t = list(range(offset, offset + self.num_n))
            indices.extend(indices_this_t)
        
        return indices
    
    def get_f_lt_indices(self):
        indices = []
        for i in range(self.sample_duration):
            # We have p_g first, then f_l.
            offset = i * self.n_vars + self.num_g
            indices_this_t = list(range(offset, offset + self.num_l))
            indices.extend(indices_this_t)
        
        return indices
    
    def get_capacity_ub_indices(self):
        #! Second |G| constraints are capacity upper bounds.
        indices = list(range(self.num_g, 2*self.num_g))
        return indices
    
    def get_missed_demand_ub_indices(self):
        #! Last |N| constraints are demand upper bounds.
        indices = list(range(2*self.num_g + 2*self.num_l + self.num_n, 2*self.num_g + 2*self.num_l + 2*self.num_n))
        return indices
    
    # TODO: Change for compact form!
    def split_ineq_constraints(self, ineq):
        """Groups the inequality [residuals or RHS] into constraints by constraint type.
            Assume ineq = [3.1h, 3.1b, 3.1d, 3.1e, 3.1i, 3,1j] (per sample in batch)

            returns each constraint type in the form [Batchsize, nr_of_constraints]

            Returns h, b, d, e, i, j
        """
        batch_size = ineq.shape[0]  # Get batch size
        
        # Reshape ineq to [batch_size, n_ineq_per_t]
        ineq = ineq.view(batch_size, self.nineq)

        # Define segment sizes based on multiple constraints per type
        sizes = [self.num_g, self.num_g, self.num_l, self.num_l, self.num_n, self.num_n]  # Num constraints per type

        # Extract constraint tensors along the first dimension (constraint type)
        h, b, d, e, i, j = torch.split(ineq, sizes, dim=1)

        return h, b, d, e, i, j
        
    # TODO: Change for compact form!
    def split_eq_constraints(self, eq, log=False):
        """Groups the inequality [residuals or RHS] into constraints by constraint type.
            Assume ineq = [3.1c] (per sample in batch)

            returns each constraint type in the form [Batchsize, nr_of_constraints]

            Returns c
        """
        # If we are using compact form, we have ui_g = UI_G constraints prepended.
        if self.benders_compact_form and not log:
            ui_g = eq[:, :self.num_g]
            c = eq[:, self.num_g:]
        else:
            ui_g = None
            c = eq
        return ui_g, c

    # TODO: Change for compact form
    def split_dec_vars_from_Y_raw(self, Y):
        """Groups the decision variables from the NN output BEFORE REPAIRS by type.
            Assume y =  [p_{g}, f_{l}] (per sample in batch)

            Returns p_{g}, f_{l}
        """
        batch_size = Y.shape[0]  # Get batch size

        if self.benders_compact_form:
            ui_g = Y[:, :self.num_g]
            Y = Y[:, self.num_g:]
        else:
            ui_g = None

        # Reshape Y for efficient slicing
        Y = Y.view(batch_size, self.n_vars - self.num_n)

        # Define segment sizes
        sizes = [self.num_g, self.num_l]

        # Extract constraint tensors
        p_gt, f_lt = torch.split(Y, sizes, dim=1)

        return ui_g, p_gt, f_lt

    # TODO: Change for compact form
    def split_dec_vars_from_Y(self, Y, log=False):
        """Groups the decision variables from the NN output AFTER REPAIRS by type.
            Assume y = [p_g, f_l, md_n]
            Returns p_g, f_l, md_n in the shape [B, G|L|N]
        """
        batch_size = Y.shape[0]  # Get batch size

        # Remove benders compact form handling if needed
        if self.benders_compact_form and not log:
            Y = Y[:, self.num_g:]

        # Define segment sizes
        sizes = [self.num_g, self.num_l, self.num_n]
        # print(Y.shape)

        # Extract constraint tensors
        p_gt, f_lt, md_nt = torch.split(Y, sizes, dim=1)

        return p_gt, f_lt, md_nt

    def ineq_resid(self, X, Y):
        eq_rhs, ineq_rhs = self.split_X(X)
        return Y @ self.ineq_cm.T - ineq_rhs

    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)

    def eq_resid(self, X, Y):
        eq_rhs, ineq_rhs = self.split_X(X)
        return eq_rhs - Y @ self.eq_cm.T
    
    def dual_ineq_resid(self, mu, lamb):
        """Dual inequality Residual, takes on the form:
            -mu <= 0
        """
        return -mu

    def dual_eq_resid(self, mu, lamb):
        """Dual equality Residual, takes on the form:
            G^T \mu + H^T \lambda + c = 0"""
        return self.obj_coeff - (lamb @ self.eq_cm - mu @ self.ineq_cm)

    def obj_fn(self, X, Y):
        # obj_coeff does not need batching, objective is the same over different samples.
        p_gt, f_lt, md_nt = self.split_dec_vars_from_Y(Y)

        p_gt = torch.abs(p_gt)

        cost = self.cost_vec @ p_gt.T

        load_shedding = self.pVOLL * torch.norm(md_nt, p=1, dim=1)
        # load_shedding = self.pVOLL * md_nt.sum(dim=1)

        return  cost + load_shedding
        # return self.obj_coeff @ Y.T
    
    # def obj_fn_train(self, Y):
    #     # Objective function adjusted for training (different than the actual objective function)
    #     # Y = Y.clone()
    #     # Y[:, self.md_indices] = torch.relu(Y[:, self.md_indices])
    #     # Y[:, self.md_indices] = Y[:, self.md_indices]**2
    #     # Y[:, self.md_indices] = torch.max(Y[:, self.md_indices], torch.zeros_like(Y[:, self.md_indices]))
    #     # Y[:, self.md_indices] = Y[:, self.md_indices].abs()
    #     return self.obj_coeff_train @ Y.T
    #     # reg_term_md = torch.norm(Y[:, self.md_indices], p=1, dim=1) #l1 regularization term
    #     # reg_term_f = torch.norm(Y[:, self.f_lt_indices], p=1, dim=1) #l1 regularization term
    #     # return self.obj_coeff @ Y.T + 0.1*(reg_term_md + reg_term_f)
    
    # def obj_fn_log(self, Y):
    #     # obj_coeff does not need batching, objective is the same over different samples.
    #     return self.obj_coeff_log @ Y.T
    
    def dual_obj_fn(self, X, mu, lamb):
        eq_rhs, ineq_rhs = self.split_X(X)
        # Batched dot product
        ineq_term = torch.sum(mu * ineq_rhs, dim=1)
        # ineq_term = mu @ ineq_rhs.T
        # Batched dot product
        # eq_term = lamb @ eq_rhs.T
        eq_term = torch.sum(lamb * eq_rhs, dim=1)
        return eq_term - ineq_term
        # return eq_term + ineq_term
    
    def build_obj_coeff(self,):
        """ Builds the objective function coefficients (only the operating costs)
            Assume y =  [p_{g,t0}, f_{l,t0}, md_{n,t0}, 
                         p_{g,t1}, f_{l,t1}, md_{n,t1}, ..., 
                         p_{g,tn}, f_{l,tn}, md_{n,tn}] """
        # coeff vector of zero's to be filled
        obj_coeff = torch.zeros(self.n_vars * self.sample_duration, dtype=self.DTYPE)
        cost_vec = torch.zeros(self.num_g * self.sample_duration, dtype=self.DTYPE)

        for t in range(self.sample_duration):
            # Sum over G,T (PC_g * p_{g,t}) --> Generation costs
            for idx_g, g in enumerate(self.G):
                # Generator variables are always at first |G| indices per timestep
                coeff_idx = t * self.n_vars + idx_g
                obj_coeff[coeff_idx] = self.pVarCost[g]
                cost_vec[coeff_idx] = self.pVarCost[g]

            # Sum over N,T (MDC * md_{n,t}) --> Cost missed demand
            for idx_n in range(self.num_n):
                # Missed demand variables come after generator and lineflow variables
                coeff_idx = t * self.n_vars + self.num_g + self.num_l + idx_n
                obj_coeff[coeff_idx] = self.pVOLL

        # ! How does pWeight influence the primal/dual solutions??
        # c_compact *= self.pWeight
        # c *= self.pWeight

        return obj_coeff, cost_vec

    def split_X(self, X):
        # demand:
        eq_rhs = X[:, :self.neq]
        capacity_ub = X[:, self.neq:]
        ineq_rhs = self.ineq_rhs.clone().repeat(X.size(0), 1)
        #! Second |G| constraints are capacity upper bounds.
        ineq_rhs[:, self.capacity_ub_indices] = capacity_ub
        #! Last |N| constraints are demand upper bounds.
        ineq_rhs[:, self.missed_demand_ub_indices] = eq_rhs

        return eq_rhs, ineq_rhs

    def build_X(self):
        """Builds a tensor containing the features that vary across problem instances.
        """
        # For ineq RHS, only 3.1b varies across instances --> first |G| constraints are 3.1h, second |G| constraints are 3.1b.
        # ineq_rhs_varying_indices = [t * self.n_ineq_per_t + self.num_g + i for t in range(self.sample_duration) for i in range(self.num_g)]

        # The entire RHS changes for equality constraints
        # X = torch.concat([self.eq_rhs, self.ineq_rhs[:, ineq_rhs_varying_indices]], dim=1)
        X = []
        total_demands = []
        for i in range(self.n_samples):
            t = i % len(self.T) + 1
            Xi = []
            for n in self.N:
                Xi.append(self.pDemand[(n, t)])
            for g_idx, g in enumerate(self.G):
                p_gt_ub = self.pUnitInvestment[i, g_idx] * self.pUnitCap[g] * self.pGenAva.get((*g, t), 1.0)
                Xi.append(p_gt_ub)
            X.append(Xi)
        X = torch.tensor(X)
        return X
    
    def build_synthetic_X(self):
        """Builds synthetic samples, which follow an easy distribution.

        Args:
            num_samples (_type_): _description_
        """
        X = np.random.uniform(0, 1, size=(self.n_samples, self.num_n + self.num_g))

        return torch.tensor(X)

    
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

    
    # Bounds are always identity, -1 for lower bounds, +1 for upper bounds.
    def assign_identity_or_scalar(self, matrix, row_start, col_start, size, value=1):
        """ Assign identity matrix if size > 1, otherwise assign a scalar value. """
        if size > 1:
            matrix[row_start:row_start + size, col_start:col_start + size] = value*torch.eye(size)
        else:
            matrix[row_start, col_start] = value

    # TODO: Change for compact form
    def build_ineq_cm_rhs(self):
        """For a single data sample characterised by the time_range, 
        build the full inequality constraint matrix for all timesteps using Kronecker product.
        """

        # TODO: Convert to sparse matrices for efficiency??

        # num_timesteps = len(time_range)
        num_columns_per_t = self.n_vars

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
        # ineq_cm = torch.kron(torch.eye(self.sample_duration), block_matrix)
        ineq_cm = block_matrix

        ineq_rhs = []

        # If we have more synthetic samples than timesteps, keep it within bounds through modulo (start counting again from 0 once we go out of bounds)
        # t = (sample_idx % len(self.T)) + 1

        # 3.1h: Production lower bound
        ineq_rhs += [0.0 for _ in range(self.num_g)]
        # 3.1b: Production upper bound
        # ineq_rhs += [self.pGenAva.get((*g, t), 1.0) * self.pUnitCap[g] * self.pUnitInvestment[sample_idx, idx] for idx, g in enumerate(self.G)]
        #! This is a placeholder, where we will substitute the capacity upper bound taken from X.
        ineq_rhs += [0.0 for _ in range(self.num_g)]
        # 3.1d: Lineflow lower bound
        ineq_rhs += [self.pImpCap[l] for l in self.L]
        # 3.1e: Lineflow upper bound
        ineq_rhs += [self.pExpCap[l] for l in self.L]
        # 3.1i: Missed demand lower bound
        ineq_rhs += [0.0 for _ in range(self.num_n)]
        # 3.1j: Missed demand upper bound
        # ineq_rhs += [self.pDemand[(n, t)] for n in self.N]
        #! This is a placeholder, where we will substitute the missed demand upper bound (Demand) taken from X.
        ineq_rhs += [0.0 for _ in range(self.num_n)]
        
        return ineq_cm, torch.tensor(ineq_rhs)

    def build_eq_cm(self):
        """Build the constraint matrix for the equality constraints
        """
        # TODO: Change operational problem to remove slack variable.

        # p_{g}, f_in - f_out, md_n
        eq_cm = torch.concat([self.node_to_gen_mask, self.lineflow_mask, torch.eye(self.num_n)], dim=1)

        # Use Kronecker product to copy the block matrix diagonally for all timesteps
        #! Not necessary, sample duration is always 1.
        # eq_cm = torch.kron(torch.eye(self.sample_duration), block_matrix)
                
        return eq_cm

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
        """
        p_gt: [batch_size, num_generators]
        node_to_gen_mask: [num_nodes, num_generators]
        returns: [batch_size, num_nodes]
        """
        p_nt = torch.matmul(p_gt, self.node_to_gen_mask.to(dtype=self.DTYPE).T)
        return p_nt

    def net_flow(self, f_lt):
        """
        f_lt: [batch_size, num_lines]
        lineflow_mask: [num_nodes, num_lines]
        returns: [batch_size, num_nodes]
        """
        net_flow_nt = torch.matmul(f_lt, self.lineflow_mask.to(dtype=self.DTYPE).T)
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
            Y = primal_net(self.X)
            eq_rhs, ineq_rhs = self.split_X(self.X)
            # Extract decision variables
            p_gt, f_lt, md_nt = self.split_dec_vars_from_Y(Y)  # [B, (G|L|N)]
            UI_g, D_nt = self.split_eq_constraints(eq_rhs[sample:sample+1])  # [1, N]
            
            # Convert to numpy and ensure correct shape
            total_prod = self.total_prod(p_gt[sample:sample+1])[0].cpu().numpy()  # Shape [N], always positive
            net_flow = self.net_flow(f_lt[sample:sample+1])[0].cpu().numpy()  # Shape [N], can be positive or negative
            missed_demand = md_nt[sample:sample+1][0].cpu().numpy()  # Shape [N], can be positive or negative
            demand = D_nt[0].cpu().numpy()  # Shape [N], total sum should match this
            num_nodes = total_prod.shape[0]
            
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
                timestep = 0  # Single timestep
                
                # Separate positive and negative components
                md_pos = max(md, 0)
                md_neg = min(md, 0)
                nf_pos = max(nf, 0)
                nf_neg = min(nf, 0)
                
                # Compute stacking order
                base_pos = 0
                base_neg = 0
                
                # Plot positive values above zero
                ax[node].bar(timestep, tp, bottom=base_pos, label='Total Production', color='blue')
                base_pos += tp
                
                ax[node].bar(timestep, nf_pos, bottom=base_pos, label='Net Flow (+)', color='green')
                base_pos += nf_pos
                
                ax[node].bar(timestep, md_pos, bottom=base_pos, label='Missed Demand (+)', color='orange')
                base_pos += md_pos
                
                # Plot negative values below zero
                ax[node].bar(timestep, md_neg, bottom=base_neg, label='Missed Demand (-)', color='darkorange')
                base_neg += md_neg
                
                ax[node].bar(timestep, nf_neg, bottom=base_neg, label='Net Flow (-)', color='darkgreen')
                base_neg += nf_neg
                
                # Add demand as a red dot
                ax[node].scatter(timestep, d, color='red', label='Demand', zorder=3)
                
                # Make the 0 line thick
                ax[node].axhline(0, color='black', linewidth=2)
                
                ax[node].set_ylabel('Values')
                ax[node].set_title(f'Node {node}')
                ax[node].legend()
                ax[node].grid(True)
            
            ax[-1].set_xlabel('Timestep')
            plt.tight_layout()
            plt.show()
        
    def plot_decision_variable_diffs(self, primal_net, dual_net):
        sample = 0
        with torch.no_grad():
            Y = primal_net(self.X)
            Y_target = self.opt_targets["y_operational"]

            mu, lamb = dual_net(self.X)
            mu_target = self.opt_targets["mu_operational"]
            lamb_target = self.opt_targets["lamb_operational"]

            eq_rhs, ineq_rhs = self.split_X(self.X)

            _, p_gt_ub, _, _, _, _ = self.split_ineq_constraints(ineq_rhs)
            
            p_gt, f_lt, md_nt = self.split_dec_vars_from_Y(Y)  
            p_gt_target, f_lt_target, md_nt_target = self.split_dec_vars_from_Y(Y_target, log=True)  

            dual_lb_p, dual_ub_p, dual_lb_f, dual_ub_f, dual_lb_md, dual_ub_md = self.split_ineq_constraints(mu)
            dual_target_lb_p, dual_target_ub_p, dual_target_lb_f, dual_target_ub_f, dual_target_lb_md, dual_target_ub_md = self.split_ineq_constraints(mu_target)

            ui_g, dual_node_balance = self.split_eq_constraints(lamb)
            ui_g, dual_node_balance_target = self.split_eq_constraints(lamb_target, log=True)

            num_gens, num_lines, num_nodes = p_gt.shape[1], f_lt.shape[1], md_nt.shape[1]
            
            fig, axes = plt.subplots(2, 4, figsize=(24, 12), sharex=False)
            
            x_pos_g = np.arange(num_gens)
            x_labels_g = [f"G{g}" for g in range(num_gens)]
            x_pos_l = np.arange(num_lines)
            x_labels_l = [f"L{l}" for l in range(num_lines)]
            x_pos_n = np.arange(num_nodes)
            x_labels_n = [f"N{n}" for n in range(num_nodes)]
            
            column_titles = ["Generation Production", "Line Flows", "Missed Demand", "Node Balance Duals"]
            for col, title in enumerate(column_titles):
                axes[0, col].set_title(title, fontsize=14)
            
            def plot_variable(ax, x_pos, labels, pred, target, ylabel, color, ub=None):
                ax.bar(x_pos, pred, label='Predicted', color=color, alpha=0.5)
                ax.scatter(x_pos, target, color='red', label='Target', zorder=3)
                if ub is not None:
                    ax.scatter(x_pos, ub, color='purple', marker='x', label='Max capacity', zorder=4)
                ax.set_ylabel(ylabel)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(labels, rotation=90)
                ax.legend()
                ax.axhline(0, color='black', linewidth=2)
                ax.grid(True)
            
            plot_variable(axes[0, 0], x_pos_g, x_labels_g, p_gt[sample].reshape(-1), p_gt_target[sample].reshape(-1), 'Primal Generation', 'blue', ub=p_gt_ub[sample].reshape(-1))
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
    
    def compute_opt_targets(self):
        y = []
        mu = []
        lamb = []
        obj = []

        obj_coeff = self.obj_coeff.numpy()
        eq_cm = self.eq_cm.numpy()
        ineq_cm = self.ineq_cm.numpy()
        eq_rhs, ineq_rhs = self.split_X(self.X)
        eq_rhs = eq_rhs.numpy()
        ineq_rhs = ineq_rhs.numpy()

        for i in range(len(self.X)):
            y_operational, obj_operational, dual_eq, dual_ineq = solve_matrix_problem_simple(obj_coeff, eq_cm, ineq_cm, eq_rhs[i], ineq_rhs[i])
            y.append(torch.tensor(y_operational, dtype=self.DTYPE))
            #! Negate duals for the inequality constraints, for some reason these are flipped in Gurobi.
            mu.append(torch.tensor(-dual_ineq, dtype=self.DTYPE))
            lamb.append(torch.tensor(dual_eq, dtype=self.DTYPE))
            obj.append(torch.tensor(obj_operational, dtype=self.DTYPE))
        
        return {
            "y_operational": torch.stack(y),
            "mu_operational": torch.stack(mu), 
            "lamb_operational": torch.stack(lamb),
            "obj": torch.stack(obj)
        }

def solve_matrix_problem_simple(obj_coeff, eq_cm, ineq_cm, eq_rhs, ineq_rhs, verbose=False):
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model
    m = gp.Model("matrix_problem")
    
    # Important: Set method to dual simplex to get dual values
    m.setParam('Method', 1)  # Use dual simplex
    
    # Create variables
    x = m.addMVar(shape=len(obj_coeff), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")
    
    # Set objective
    m.setObjective(obj_coeff @ x, GRB.MINIMIZE)
    
    # Add constraints and store their references
    eq_constrs = m.addConstr(eq_cm @ x == eq_rhs, name="eq")
    ineq_constrs = m.addConstr(ineq_cm @ x <= ineq_rhs, name="ineq")
    
    # Optimize
    m.optimize()
    
    if m.Status == GRB.OPTIMAL:
        # Get dual values after confirming optimal solution
        dual_eq = eq_constrs.Pi
        dual_ineq = ineq_constrs.Pi
        
        if verbose:
            print(f"Optimal objective: {m.ObjVal}")
            print(f"Dual values (eq): {dual_eq}")
            print(f"Dual values (ineq): {dual_ineq}")
            
        return x.X, m.ObjVal, dual_eq, dual_ineq
    else:
        raise RuntimeError(f"Optimization failed with status {m.Status}")
    


if __name__ == "__main__":
    import json

    ## Step 1: parse the input data
    print("Parsing the config file")
    CONFIG_FILE_NAME = "config.toml"
    data = parse_config(CONFIG_FILE_NAME)
    experiment = data["experiment"]
    outputs_config = data["outputs_config"]

    with open("config.json", "r") as file:
        args = json.load(file)
    
    print(args)

    # Train the model:
    for i, experiment_instance in enumerate(experiment["experiments"]):
        # Setup output dataframe
        df_res = pd.DataFrame(columns=["setup_time", "presolve_time", "barrier_time", "crossover_time", "restore_time", "objective_value"])

        for j in range(experiment["repeats"]):
            # Run one experiment for j repeats
            run_name = f"refactored_train:{args['train']}_rho:{args['rho']}_rhomax:{args['rho_max']}_alpha:{args['alpha']}_L:{args['alpha']}"
            save_dir = os.path.join('outputs', 'PDL',
                run_name + "-" + str(time.time()).replace('.', '-'))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(os.path.join(save_dir, 'args.dict'), 'wb') as f:
                pickle.dump(args, f)

            target_path = f"outputs/Gurobi/Operational={args['operational']}_T={args['sample_duration']}_Scale={args['scale_problem']}_{args['G']}_{args['L']}"
            inputs = experiment_instance
            # Prep problem data:
            print("Wrangling the input data")

            # Extract sets
            T = inputs["times"] # [1, 2, 3, ... 8760] ---> 8760
            N = args["N"]
            G = args["G"]
            L = args["L"]

            if not (N or G or L):
                G = inputs["generators"] # [('Country1', 'EnergySource1'), ...] ---> 107
                L = inputs["transmission_lines"] # [('Country1', 'Country2'), ...] ---> 44
                N = inputs["nodes"] # ['Country1', 'Country2', ...] ---> 20
            else:
                # Convert to tuples
                # N = [tuple(pair) for pair in N]
                G = [tuple(pair) for pair in G]
                L = [tuple(pair) for pair in L]

            # Extract time series data
            pDemand = dataframe_to_dict(
                inputs["demand_data"],
                keys=["Country", "Time"],
                value="Demand_MW"
            )
            
            pGenAva = dataframe_to_dict(
                inputs["generation_availability_data"],
                keys=["Country", "Technology", "Time"],
                value="Availability_pu"
            )

            # Extract scalar parameters
            pVOLL = inputs["value_of_lost_load"]

            # WOP
            # Scale inversely proportional to times (T)
            pWeight = inputs["representative_period_weight"] / (args["sample_duration"] / 8760)

            pRamping = inputs["ramping_value"]

            # Extract generator parameters
            pInvCost = dataframe_to_dict(
                inputs["generation_data"],
                keys=["Country", "Technology"],
                value="InvCost_kEUR_MW_year"
            )

            pVarCost = dataframe_to_dict(
                inputs["generation_data"],
                keys=["Country", "Technology"],
                value="VarCost_kEUR_per_MWh"
            )

            pUnitCap = dataframe_to_dict(
                inputs["generation_data"],
                keys=["Country", "Technology"],
                value="UnitCap_MW"
            )

            # Extract line parameters
            pExpCap = dataframe_to_dict(
                inputs["transmission_lines_data"],
                keys=["CountryA", "CountryB"],
                value="ExpCap_MW"
            )

            pImpCap = dataframe_to_dict(
                inputs["transmission_lines_data"],
                keys=["CountryA", "CountryB"],
                value="ImpCap_MW"
            )

            # if args["scale_problem"]:
            #     pDemand = scale_dict(pDemand, SCALE_FACTORS["pDemand"])
            #     pGenAva = scale_dict(pGenAva, SCALE_FACTORS["pGenAva"])
            #     pVOLL *= SCALE_FACTORS["pVOLL"]
            #     pWeight *= SCALE_FACTORS["pWeight"]
            #     pRamping *= SCALE_FACTORS["pRamping"]
            #     pInvCost = scale_dict(pInvCost, SCALE_FACTORS["pInvCost"])
            #     pVarCost = scale_dict(pVarCost, SCALE_FACTORS["pVarCost"])
            #     pUnitCap = scale_dict(pUnitCap, SCALE_FACTORS["pUnitCap"])
            #     pExpCap = scale_dict(pExpCap, SCALE_FACTORS["pExpCap"])
            #     pImpCap = scale_dict(pImpCap, SCALE_FACTORS["pImpCap"])


            # We need to sort the dictionaries for changing to tensors!
            # pDemand = dict(sorted(pDemand.items()))
            # pGenAva = dict(sorted(pGenAva.items()))
            # pInvCost = dict(sorted(pInvCost.items()))
            # pVarCost = dict(sorted(pVarCost.items()))
            # pUnitCap = dict(sorted(pUnitCap.items()))
            # pExpCap = dict(sorted(pExpCap.items()))
            # pImpCap = dict(sorted(pImpCap.items()))

            # if not os.path.exists(target_path):
            #     time_ranges = [range(i, i + args["sample_duration"], 1) for i in range(1, len(T), args["sample_duration"])]
            #     save_opt_targets(args, inputs, target_path, T, N, G, L, pDemand, pGenAva, pVOLL, pWeight, pRamping, pInvCost, pVarCost, pUnitCap, pExpCap, pImpCap, time_ranges)


            print("Creating problem instance")
            if args["operational"]:
                data = GEPOperationalProblemSet(args, T, N, G, L, pDemand, pGenAva, pVOLL, pWeight, pRamping, pInvCost, pVarCost, pUnitCap, pExpCap, pImpCap, target_path=target_path)
            else:
                data = GEPProblemSet(args, T, N, G, L, pDemand, pGenAva, pVOLL, pWeight, pRamping, pInvCost, pVarCost, pUnitCap, pExpCap, pImpCap, target_path=target_path)

            known_objs = data.obj_fn(data.X, data.opt_targets["y_operational"])
            known_dual_objs = data.dual_obj_fn(data.eq_rhs, data.ineq_rhs, data.opt_targets["mu_operational"], data.opt_targets["lamb_operational"])
            # for i in range(len(data.X)):
            for i in range(1):
                x, obj, dual_eq, dual_ineq = solve_matrix_problem_simple(data.obj_coeff.numpy(), data.eq_cm.numpy(), data.ineq_cm.numpy(), data.eq_rhs[i].numpy(), data.ineq_rhs[i].numpy(), False)
                assert abs(obj - known_objs[i]) < 1e-9, f"Objective value mismatch at index {i}: {obj} != {known_objs[i]}"
                # print(known_dual_objs[i:i+1])
                # print(data.dual_obj_fn(data.eq_rhs[i:i+1], data.ineq_rhs[i:i+1], torch.tensor(dual_ineq).unsqueeze(0), torch.tensor(dual_eq).unsqueeze(0)))
                # print(data.obj_coeff)
                # print(dual_eq)
                # print(data.opt_targets["lamb_operational"][i:i+1])
                # print(dual_ineq)
                # print(data.opt_targets["mu_operational"][i:i+1])
                # print(data.eq_rhs[i:i+1])
                # print(data.ineq_rhs[i:i+1])
                # print(5336.3813 * 0.05 + 33061.6701 * 0.15)
                calc_obj_fn = data.dual_obj_fn(data.eq_rhs[i:i+1], data.ineq_rhs[i:i+1], -torch.tensor(dual_ineq).unsqueeze(0), torch.tensor(dual_eq).unsqueeze(0))
                assert abs(known_objs[i] - calc_obj_fn.item()) < 1e-9, f"Dual objective value mismatch at index {i}: {known_objs[i]} != {calc_obj_fn.item()}"