import numpy as np
import torch
import pickle
from sklearn.preprocessing import StandardScaler


class GEPProblemSet():

    def __init__(self, T, N, G, L, pDemand, pGenAva, pVOLL, pWeight, pRamping, pInvCost, pVarCost, pUnitCap, pExpCap, pImpCap, sample_duration=12, train=0.8, valid=0.1, test=0.1):
        # ! Assume y = [p_{g,t}, f_{l,t}, md_{n,t}, ui_g]
        self.DTYPE = torch.float64
        self.DEVICE = torch.device="cpu"
        torch.set_default_dtype(self.DTYPE)

        # Input Sets
        self.T = T
        self.G = G
        self.L = L
        self.N = N
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

        # Number of timesteps per data sample
        assert (float(len(T)) / sample_duration).is_integer() # Total number of timesteps should be divisible by the number of timesteps per data sample.
        self.sample_duration = sample_duration

        # Time slices, each slice makes a different sample.
        self.time_ranges = [range(i, i + sample_duration, 1) for i in range(1, len(T), sample_duration)]

        self.index_f_offset = len(self.G) * self.sample_duration
        self.index_md_offset = len(self.G) * self.sample_duration + len(self.L) * self.sample_duration
        self.index_ui_offset = len(self.G) * self.sample_duration + len(self.L) * self.sample_duration + len(self.N) * self.sample_duration

        # Known optimal values from Gurobi.
        # file_path = f"outputs/Gurobi/OPERATIONAL={False}_GEP_OPT_TARGETS_T={self.sample_duration}"
        #! All generators:
        # file_path = "outputs/Gurobi/ALL_GENERATORS-OPERATIONAL=False-GEP_OPT_TARGETS_T=12"
        #! Sun and gas generators:
        # file_path = "outputs/Gurobi/SUN_AND_GAS-OPERATIONAL=False-GEP_OPT_TARGETS_T=12"
        #! Only BEL, GER and Gas generators
        file_path = "outputs/Gurobi/BEL_GER_GAS-OPERATIONAL=False-GEP_OPT_TARGETS_T=12"
        with open(file_path, 'rb') as file:
            self._opt_targets = pickle.load(file)

        self._generator_node_map = {g: connected_node for connected_node, g in self.G}
        self._line_start_node_map = {start_node: end_node for start_node, end_node in self.L}
        self._line_end_node_map = {end_node: start_node for start_node, end_node in self.L}

        # Masks for node balance!
        # Initialize mask
        self.gen_to_node_mask = np.zeros((len(N), len(G)), dtype=int)

        # Populate mask
        for g_idx, (node, _) in enumerate(G):
            node_idx = N.index(node)
            self.gen_to_node_mask[node_idx, g_idx] = 1
        
        # Initialize masks
        self.incoming_mask = np.zeros((len(N), len(L)), dtype=int)
        self.outgoing_mask = np.zeros((len(N), len(L)), dtype=int)

        # Populate masks
        for l_idx, (start_node, end_node) in enumerate(L):
            start_idx = N.index(start_node)
            end_idx = N.index(end_node)
            self.outgoing_mask[start_idx, l_idx] = 1
            self.incoming_mask[end_idx, l_idx] = 1
        
        # Create constraint matrices, rhs and obj_fns
        print("Populating ineq constraints")
        self.ineq_cm, self.ineq_rhs = self.build_ineq_cm_rhs()
        print("Populating eq constraints")
        self.eq_cm, self.eq_rhs = self.build_eq_cm_rhs()
        print("Creating objective coefficients")
        self.obj_coeff = self.build_obj_coeff()
        print("Creating input for NN: X")
        self.X = self.build_X()

        self.X_scaled = self.scale_X(self.X)
        # self.y_coeff = self.scale_Y_coeff()

        self.neq = self.eq_cm.shape[1]
        self.nineq = self.ineq_cm.shape[1]
        self.ydim = len(self.obj_coeff)
        self.xdim = self.X.shape[1]

        # Variable indices, to extract variables from Y
        self.p_gt_indices = range(0, self.index_f_offset)
        self.f_lt_indices = range(self.index_f_offset, self.index_md_offset)
        self.md_nt_indices = range(self.index_md_offset, self.index_ui_offset)
        self.ui_g_indices = range(self.index_ui_offset, self.ydim)

        # Constraint indices, to extract specific constraint residuals
        # eq constraints: only eq 3.1c
        self.constraint_c_indices = range(self.neq)

        # ineq constraints: b,d,e,f,g,h,i,j,k
        self.constraint_b_offset = 0
        self.constraint_d_offset = self.constraint_b_offset + len(self.G)*self.sample_duration 
        self.constraint_e_offset = self.constraint_d_offset + len(self.L)*self.sample_duration 
        self.constraint_f_offset = self.constraint_e_offset + len(self.L)*self.sample_duration 
        self.constraint_g_offset = self.constraint_f_offset + len(G)*(self.sample_duration - 1)
        self.constraint_h_offset = self.constraint_g_offset + len(G)*(self.sample_duration - 1)
        self.constraint_i_offset = self.constraint_h_offset + len(G)*self.sample_duration 
        self.constraint_j_offset = self.constraint_i_offset + len(N)*self.sample_duration 
        self.constraint_k_offset = self.constraint_j_offset + len(N)*self.sample_duration 

        self.constraint_b_indices = range(self.constraint_b_offset, self.constraint_d_offset)
        self.constraint_d_indices = range(self.constraint_d_offset, self.constraint_e_offset)
        self.constraint_e_indices = range(self.constraint_e_offset, self.constraint_f_offset)
        self.constraint_f_indices = range(self.constraint_f_offset, self.constraint_g_offset)
        self.constraint_g_indices = range(self.constraint_g_offset, self.constraint_h_offset)
        self.constraint_h_indices = range(self.constraint_h_offset, self.constraint_i_offset)
        self.constraint_i_indices = range(self.constraint_i_offset, self.constraint_j_offset)
        self.constraint_j_indices = range(self.constraint_j_offset, self.constraint_k_offset)
        self.constraint_k_indices = range(self.constraint_k_offset, self.nineq)

        # Split x into training, val, test sets.
        self._split_X_in_sets(train, valid, test)
        
    
    # Split the data
    @property
    def train_indices(self): return self._train_indices
    @property
    def valid_indices(self): return self._valid_indices
    @property
    def test_indices(self): return self._test_indices
    
    @property
    def opt_targets(self): return self._opt_targets
    
    def set_train_extractor(self, train_extractor):
        self.train_extractor = train_extractor
    
    def set_valid_extractor(self, valid_extractor):
        self.valid_extractor = valid_extractor

    def ineq_resid(self, Y, ineq_cm, ineq_rhs):
        # Y *= self.y_coeff
        return torch.bmm(ineq_cm, Y.unsqueeze(-1)).squeeze(-1) - ineq_rhs

    def eq_resid(self, Y, eq_cm, eq_rhs):
        # Y *= self.y_coeff
        return torch.bmm(eq_cm, Y.unsqueeze(-1)).squeeze(-1) - eq_rhs

    def obj_fn(self, Y):
        # obj_coeff does not need batching, objective is the same over different samples.
        # Y *= self.y_coeff
        return self.obj_coeff @ Y.T
    
    def dual_obj_fn(self, eq_rhs, ineq_rhs, mu, lamb):
        # Batched dot product
        ineq_term = torch.sum(mu * ineq_rhs, dim=1)
        
        # Batched dot product
        eq_term = torch.sum(lamb * eq_rhs, dim=1)

        return -(ineq_term + eq_term)
    
    def ineq_dist(self, Y, ineq_cm, ineq_rhs):
        resids = self.ineq_resid(Y, ineq_cm, ineq_rhs)
        return torch.clamp(resids, 0)
    
    def build_obj_coeff(self,):
        # ! Assume y = [p_{g,t}, f_{l,t}, md_{n,t}, ui_g]
        c = self._create_empty_row()
        for g_idx, g in enumerate(self.G):
            for t_idx in range(self.sample_duration):
                index_p = self._get_index_p(g_idx, t_idx)
                c[index_p] = self.pWeight * self.pVarCost[g]
        
        for n_idx, n in enumerate(self.N):
            for t_idx in range(self.sample_duration):
                index_md = self._get_index_md(n_idx, t_idx)
                c[index_md] = self.pWeight * self.pVOLL
        
        for g_idx, g in enumerate(self.G):
            index_ui = self._get_index_ui(g_idx)
            c[index_ui] = self.pInvCost[g] * self.pUnitCap[g]
        
        return torch.tensor(c)

    def build_X(self,):
        X = []
        for time_range in self.time_ranges:
            x = []
            for n in self.N:
                for t in time_range:
                    x.append(self.pDemand[(n, t)])
            for g in self.G:
                for t in time_range:
                    x.append(self.pGenAva.get((*g, t), 1.0))
            X.append(x)
        return torch.tensor(X)

    def scale_X(self, X):
        # ! Scaling does not help!
        # Compute sizes for pDemand and pGenAva based on input dimensions
        pDemand_size = len(self.N) * self.sample_duration
        pGenAva_size = len(self.G) * self.sample_duration

        # Split X into demand and generation availability parts
        X_demand = X[:, :pDemand_size].flatten().reshape(-1, 1)  # Flatten to 1D array for scaling
        X_genava = X[:, pDemand_size:].flatten().reshape(-1, 1)

        # Initialize scalers
        scaler_demand = StandardScaler()
        scaler_gen_ava = StandardScaler()

        # Fit and transform each part
        pDemand_scaled = scaler_demand.fit_transform(X_demand)  # Scale demand
        pGenAva_scaled = scaler_gen_ava.fit_transform(X_genava)  # Scale generation availability

        # Reshape scaled data back to original shapes
        pDemand_scaled = pDemand_scaled.reshape(X[:, :pDemand_size].shape)  # Reshape to original demand shape
        pGenAva_scaled = pGenAva_scaled.reshape(X[:, pDemand_size:].shape)  # Reshape to original generation availability shape

        # Combine scaled parts into a single tensor
        X_scaled = torch.cat([
            torch.tensor(pDemand_scaled, dtype=self.DTYPE),
            torch.tensor(pGenAva_scaled, dtype=self.DTYPE)
        ], dim=1)

        return X_scaled

    def scale_Y_coeff(self):
        # TODO: Only for first sample now.
        opt_decision_variables = self.opt_targets[0]["y"]
        p_gt_idx = len(self.G)*self.sample_duration
        f_lt_idx = p_gt_idx + len(self.L)*self.sample_duration
        md_nt_idx = f_lt_idx + len(self.N)*self.sample_duration
        ui_g_idx = md_nt_idx + len(self.G)
        p_gt = opt_decision_variables[:p_gt_idx].unsqueeze(0).reshape(-1,1)
        f_lt = opt_decision_variables[p_gt_idx:f_lt_idx].unsqueeze(0).reshape(-1,1)
        md_nt = opt_decision_variables[f_lt_idx:md_nt_idx].unsqueeze(0).reshape(-1,1)
        ui_g = opt_decision_variables[md_nt_idx:].unsqueeze(0).reshape(-1,1)

        # Scale each distribution separately
        scaler_p = StandardScaler()
        scaler_f = StandardScaler()
        scaler_md = StandardScaler()
        scaler_ui = StandardScaler()

        p_scaled = scaler_p.fit_transform(p_gt)
        f_scaled = scaler_f.fit_transform(f_lt)
        md_scaled = scaler_md.fit_transform(md_nt)
        ui_scaled = scaler_ui.fit_transform(ui_g)

        p_coeff = torch.tensor(scaler_p.inverse_transform(np.ones_like(p_gt)))
        f_coeff = torch.tensor(scaler_f.inverse_transform(np.ones_like(f_lt)))
        md_coeff = torch.tensor(scaler_md.inverse_transform(np.ones_like(md_nt)))
        ui_coeff = torch.tensor(scaler_ui.inverse_transform(np.ones_like(ui_g)))

        Y_coeff = torch.concat([p_coeff, f_coeff, md_coeff, ui_coeff])
        return Y_coeff.squeeze()


    def build_ineq_cm_rhs(self,):
        ineq_cms = []
        ineq_rhss = []

        for time_range in self.time_ranges:
            ineq_cm, ineq_rhs = self.build_ineq_cm_rhs_sample(time_range)
            ineq_cms.append(ineq_cm)
            ineq_rhss.append(ineq_rhs)
        
        return torch.tensor(np.array(ineq_cms)), torch.tensor(np.array(ineq_rhss))

    def build_eq_cm_rhs(self,):
        eq_cms = []
        eq_rhss = []

        for time_range in self.time_ranges:
            eq_cm, eq_rhs = self.build_eq_cm_rhs_sample(time_range)
            eq_cms.append(eq_cm)
            eq_rhss.append(eq_rhs)
        
        return torch.tensor(np.array(eq_cms)), torch.tensor(np.array(eq_rhss))
    
    def build_ineq_cm_rhs_sample(self, time_range):
        """Build the constraint matrix for the inequality constraints
        """

        # Initialize lists for the constraint matrix and RHS
        ineq_cm = []
        ineq_rhs = []

        # Build constraints by calling specific functions for each constraint
        self._add_max_production_constraints(ineq_cm, ineq_rhs, time_range) # 3.1 b
        self._add_line_flow_constraints(ineq_cm, ineq_rhs, time_range) # 3.1 d,e
        self._add_ramping_constraints(ineq_cm, ineq_rhs, time_range) # 3.1 f,g
        self._add_non_negative_constraints(ineq_cm, ineq_rhs, time_range) # 3.1 h
        self._add_missed_demand_constraints(ineq_cm, ineq_rhs, time_range) # 3.1 i, j
        self._add_non_negative_geninv_constraints(ineq_cm, ineq_rhs, time_range) # 3.1 k

        # TODO: Convert to sparse matrix for efficiency??
        ineq_cm = np.array(ineq_cm)
        ineq_rhs = np.array(ineq_rhs)

        return ineq_cm, ineq_rhs
    
    def build_eq_cm_rhs_sample(self, time_range):
        """Build the constraint matrix for the equality constraints
        """
        eq_cm = []
        eq_rhs = []

        self._add_node_balance_constraints(eq_cm=eq_cm, eq_rhs=eq_rhs, time_range=time_range)

        # TODO: Convert to sparse matrix for efficiency??
        eq_cm = eq_cm
        eq_rhs = np.array(eq_rhs)

        return eq_cm, eq_rhs

    def _add_max_production_constraints(self, ineq_cm, rhs, time_range):
        """Add maximum production constraints (3.1b)."""
        for idx_g, g in enumerate(self.G):
            for idx_t, t in enumerate(time_range):
                row = self._create_empty_row()
                # Coefficients for p_{g,t}
                idx_p_gt = self._get_index_p(idx_g, idx_t)
                row[idx_p_gt] = 1

                # Coefficients for ui_g
                idx_ui_g = self._get_index_ui(idx_g)
                row[idx_ui_g] = -self.pGenAva.get((*g, t), 1.0) * self.pUnitCap[g]

                ineq_cm.append(row)
                rhs.append(0)

    def _add_line_flow_constraints(self, ineq_cm, rhs, time_range):
        """Add line flow constraints (3.1d and 3.1e)."""
        lb_cm = []
        ub_cm = []
        lb_rhs = []
        ub_rhs = []

        for idx_l, l in enumerate(self.L):
            for idx_t, t in enumerate(time_range):
                row_lb = self._create_empty_row()
                row_ub = row_lb.copy()

                # Coefficients for f_{l,t}
                idx_f_lt = self._get_index_f(idx_l, idx_t)
                row_lb[idx_f_lt] = -1
                row_ub[idx_f_lt] = 1

                lb_cm.append(row_lb)
                lb_rhs.append(self.pImpCap[l])

                ub_cm.append(row_ub)
                ub_rhs.append(self.pExpCap[l])
        
        # Add after eachother to ensure consistency in order with math formula's
        # 3.1d
        ineq_cm += lb_cm
        rhs += lb_rhs
        # 3.1e
        ineq_cm += ub_cm
        rhs += ub_rhs

    def _add_ramping_constraints(self, ineq_cm, rhs, time_range):
        """Add ramping constraints (3.1f and 3.1g)."""
        down_cm = []
        up_cm = []
        down_rhs = []
        up_rhs = []
        for idx_g, g in enumerate(self.G):
            for idx_t, t in enumerate(time_range[1:], start=1):  # Start indexing from 1
                row_down = self._create_empty_row()
                row_up = row_down.copy()

                # Coefficients for p_{g,t} and p_{g,t-1}
                idx_p_gt = self._get_index_p(idx_g, idx_t)
                idx_p_gt_prev = self._get_index_p(idx_g, idx_t - 1)

                # 3.1f
                row_down[idx_p_gt] = -1
                row_down[idx_p_gt_prev] = 1

                # 3.1g
                row_up[idx_p_gt] = 1
                row_up[idx_p_gt_prev] = -1

                # Coefficients for ui_g
                idx_ui_g = self._get_index_ui(idx_g)
                ramping_term = self.pRamping * self.pUnitCap[g]
                row_down[idx_ui_g] = -ramping_term
                row_up[idx_ui_g] = -ramping_term

                down_cm.append(row_down)
                down_rhs.append(0)

                up_cm.append(row_up)
                up_rhs.append(0)
        # 3.1f
        ineq_cm += down_cm
        rhs += down_rhs
        # 3.1g
        ineq_cm += up_cm
        rhs += up_rhs

    def _add_non_negative_constraints(self, ineq_cm, rhs, time_range):
        """Add non-negativity constraints (3.1h)."""
        for idx_g, g in enumerate(self.G):
            for idx_t, t in enumerate(time_range):
                row = self._create_empty_row()

                # Coefficients for p_{g,t}
                idx_p_gt = self._get_index_p(idx_g, idx_t)
                row[idx_p_gt] = -1

                ineq_cm.append(row)
                rhs.append(0)

    def _add_missed_demand_constraints(self, ineq_cm, rhs, time_range):
        """Add missed demand constraints (3.1i and 3.1j)."""
        lb_cm = []
        lb_rhs = []
        ub_cm = []
        ub_rhs = []
        for idx_n, n in enumerate(self.N):
            for idx_t, t in enumerate(time_range):
                row_lb = self._create_empty_row()
                row_ub = row_lb.copy()

                # Coefficients for md_{n,t}
                idx_md_nt = self._get_index_md(idx_n, idx_t)
                row_lb[idx_md_nt] = -1
                row_ub[idx_md_nt] = 1

                lb_cm.append(row_lb)
                lb_rhs.append(0)

                ub_cm.append(row_ub)
                ub_rhs.append(self.pDemand[(n, t)])
        # 3.1i
        ineq_cm += lb_cm
        rhs += lb_rhs
        # 3.1j
        ineq_cm += ub_cm
        rhs += ub_rhs
    
    # 3.1 k
    def _add_non_negative_geninv_constraints(self, ineq_cm, ineq_rhs, time_range):
        for idx_g, g in enumerate(self.G):
            row = self._create_empty_row()
            index_ui_g = self._get_index_ui(idx_g)
            row[index_ui_g] = -1

            ineq_cm.append(row)
            ineq_rhs.append(0)
    
    def _add_node_balance_constraints(self, eq_cm, eq_rhs, time_range):
        """
        Add node balance constraints using precomputed masks and efficient operations.
        """

        # Precompute sparse row templates for efficiency
        row_template = self._create_empty_row()

        for idx_t, t in enumerate(time_range):
            for idx_n, n in enumerate(self.N):
                row = row_template.copy()  # Start with a blank row

                # Add generator coefficients (p_{g,t})
                generator_indices = np.where(self.gen_to_node_mask[idx_n])[0]
                row[[self._get_index_p(idx_g, idx_t) for idx_g in generator_indices]] = 1

                # Add incoming flow coefficients (f_{l,t})
                incoming_indices = np.where(self.incoming_mask[idx_n])[0]
                row[[self._get_index_f(idx_l_in, idx_t) for idx_l_in in incoming_indices]] = 1

                # Add outgoing flow coefficients (f_{l,t})
                outgoing_indices = np.where(self.outgoing_mask[idx_n])[0]
                row[[self._get_index_f(idx_l_out, idx_t) for idx_l_out in outgoing_indices]] = -1

                # Add missed demand coefficients (md_{n,t})
                idx_md_nt = self._get_index_md(idx_n, idx_t)
                row[idx_md_nt] = 1

                # Append row and RHS to the constraint matrix
                eq_cm.append(row)
                eq_rhs.append(self.pDemand[(n, t)])

    def _get_generators_connected_to_node(self, node):
        """Return a list of generators connected to the given node."""
        return [g for g in self.G if self._generator_node_map[g] == node]

    def _get_lines_starting_at_node(self, node):
        """Return a list of lines starting at the given node."""
        return [l for l in self.L if self._line_start_node_map[l] == node]

    def _get_lines_ending_at_node(self, node):
        """Return a list of lines ending at the given node."""
        return [l for l in self.L if self._line_end_node_map[l] == node]
    
    def _create_empty_row(self,):
        return np.zeros(len(self.G) * self.sample_duration + len(self.L) * self.sample_duration + len(self.N) * self.sample_duration + len(self.G))

    def _get_index_p(self, idx_g, idx_t):
        """Get the index of p_{g,t} in the flattened decision variable vector."""
        return idx_g * self.sample_duration + idx_t

    def _get_index_f(self, idx_l, idx_t):
        """Get the index of f_{l,t} in the flattened decision variable vector."""
        return self.index_f_offset + idx_l * self.sample_duration + idx_t

    def _get_index_md(self, idx_n, idx_t):
        """Get the index of md_{n,t} in the flattened decision variable vector."""
        return self.index_md_offset + idx_n * self.sample_duration + idx_t

    def _get_index_ui(self, idx_g):
        """Get the index of ui_g in the flattened decision variable vector."""
        return self.index_ui_offset + idx_g

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
        time_ranges_tensor = torch.tensor(self.time_ranges)

        # Split time ranges
        self.train_time_ranges = time_ranges_tensor[self.train_indices].tolist()
        self.val_time_ranges = time_ranges_tensor[self.valid_indices].tolist()
        self.test_time_ranges = time_ranges_tensor[self.test_indices].tolist()

        print(f"Size of train set: {train_size}")
        print(f"Size of val set: {valid_size}")
        print(f"Size of test set: {B - train_size - valid_size}")
