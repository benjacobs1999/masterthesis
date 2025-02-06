import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt


class GEPProblemSet():

    def __init__(self, args, T, N, G, L, pDemand, pGenAva, pVOLL, pWeight, pRamping, pInvCost, pVarCost, pUnitCap, pExpCap, pImpCap, target_path):
        self.DTYPE = torch.float64
        self.DEVICE = torch.device="cpu"
        torch.set_default_dtype(self.DTYPE)

        # Args:
        self.args = args
        self.sample_duration = args["sample_duration"]
        self.train = args["train"]
        self.valid = args["valid"]
        self.test = args["test"]

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
        self.n_inv_vars = self.num_g
        self.n_prod_vars = self.num_g * self.sample_duration
        self.n_line_vars = self.num_l * self.sample_duration
        self.n_md_vars = self.num_n * self.sample_duration
        self.ydim = self.n_inv_vars, self.n_prod_vars + self.n_line_vars + self.n_md_vars

        self._opt_targets = self.load_targets(target_path)

        # Masks for node balance!
        # Initialize mask
        self.node_to_gen_mask = torch.zeros((len(N), len(G)), dtype=torch.float64)

        # Populate mask
        for g_idx, (node, _) in enumerate(G):
            node_idx = N.index(node)
            self.node_to_gen_mask[node_idx, g_idx] = 1
        
        # Initialize mask
        self.lineflow_mask = torch.zeros((len(N), len(L)), dtype=torch.float64)

        # Populate mask (directed adjacency matrix), -1 where line starts, 1 where line stops
        for l_idx, (start_node, end_node) in enumerate(L):
            start_idx = N.index(start_node)
            end_idx = N.index(end_node)
            self.lineflow_mask[start_idx, l_idx] = -1
            self.lineflow_mask[end_idx, l_idx] = 1
        
        # Create constraint matrices, rhs and obj_fns
        print("Populating ineq constraints")
        self.ineq_cm, self.ineq_rhs = self.build_ineq_cm_rhs()
        print("Populating eq constraints")
        self.eq_cm, self.eq_rhs = self.build_eq_cm_rhs()
        print("Creating objective coefficients")
        self.obj_coeff = self.build_obj_coeff()
        print("Creating input for NN: X")
        # self.X = self.build_X()
        self.X = torch.concat([self.eq_rhs, self.ineq_rhs], dim=1)
        self.xdim = self.X.shape[1]

        # Split x into training, val, test sets.
        self._split_X_in_sets(self.train, self.valid, self.test)
    
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
    
    def split_ineq_constraints(self, ineq):
        # TODO: Fix for inclusion of 3.1k.
        """Groups the inequality [residuals or RHS] into constraints by constraint type.
            Assume ineq = [3.1k, [3.1h, 3.1b, 3.1d, 3.1e, 3.1i, 3,1j] * T] (per sample in batch)

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
    
    def split_eq_constraints(self, eq):
        """Groups the inequality [residuals or RHS] into constraints by constraint type.
            Assume ineq = [3.1c] * T (per sample in batch)

            returns each constraint type in the form [Batchsize, nr_of_constraints, time]

            Returns c
        """
        batch_size = eq.shape[0]
        c = eq.view(batch_size, self.sample_duration, self.n_eq_per_t).permute(0, 2, 1)
        return c

    def split_dec_vars_from_Y_raw(self, Y):
        # TODO: Fix for inclusion of 3.1k.
        """Groups the decision variables from the NN output BEFORE REPAIRS by type.
            Assume y =  [p_{g,t0}, f_{l,t0}, 
                         p_{g,t1}, f_{l,t1}, ..., 
                         p_{g,tn}, f_{l,tn}] (per sample in batch)

            Returns p_{g,t}, f_{l,t}
        """
        batch_size = Y.shape[0]  # Get batch size

        # Reshape Y for efficient slicing
        Y = Y.view(batch_size, self.sample_duration, self.n_var_per_t - self.num_n).permute(0, 2, 1)

        # Define segment sizes
        sizes = [self.num_g, self.num_l]

        # Extract constraint tensors
        p_gt, f_lt = torch.split(Y, sizes, dim=1)

        return p_gt, f_lt

    def split_dec_vars_from_Y(self, Y):
        # TODO: Fix for inclusion of 3.1k.
        """Groups the decision variables from the NN output AFTER REPAIRS by type.
            Assume y =  [p_{g,t0}, f_{l,t0}, md_{n,t0}, 
                         p_{g,t1}, f_{l,t1}, md_{n,t1}, ..., 
                         p_{g,tn}, f_{l,tn}, md_{n,tn}]

            Returns p_{g,t}, f_{l,t}, md_{n,t}
        """
        batch_size = Y.shape[0]  # Get batch size

        # Reshape Y to [batch_size, sample_duration, -1] for efficient slicing
        Y = Y.view(batch_size, self.sample_duration, self.n_var_per_t).permute(0, 2, 1)

        # Define segment sizes
        sizes = [self.num_g, self.num_l, self.num_n]

        # Extract constraint tensors
        p_gt, f_lt, md_nt = torch.split(Y, sizes, dim=1)

        return p_gt, f_lt, md_nt

    def ineq_resid(self, Y, ineq_cm, ineq_rhs):
        return torch.bmm(ineq_cm, Y.unsqueeze(-1)).squeeze(-1) - ineq_rhs

    def eq_resid(self, Y, eq_cm, eq_rhs):
        torch.bmm(eq_cm, Y.unsqueeze(-1)).squeeze(-1)
        return torch.bmm(eq_cm, Y.unsqueeze(-1)).squeeze(-1) - eq_rhs

    def obj_fn(self, Y):
        # obj_coeff does not need batching, objective is the same over different samples.

        return self.pWeight * self.obj_coeff @ Y.T
    
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
        """ Builds the objective function coefficients (only the operating costs)
            Assume y =  [ui_g0, ui_g1, ..., ui_gn,
                         p_{g,t0}, f_{l,t0}, md_{n,t0}, 
                         p_{g,t1}, f_{l,t1}, md_{n,t1}, ..., 
                         p_{g,tn}, f_{l,tn}, md_{n,tn}] """

        # coeff vector of zero's to be filled
        c = torch.zeros(self.num_g + self.n_var_per_t*self.sample_duration, dtype=torch.float64)

        for g_idx, g in enumerate(self.G):
            c[g_idx] = self.pInvCost[g] * self.pUnitCap[g]

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

        return c

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
    
    # Bounds are always identity, -1 for lower bounds, +1 for upper bounds.
    def assign_identity_or_scalar(self, matrix, row_start, col_start, size, value=1):
        """ Assign identity matrix if size > 1, otherwise assign a scalar value. """
        if size > 1:
            matrix[row_start:row_start + size, col_start:col_start + size] = value*torch.eye(size)
        else:
            matrix[row_start, col_start] = value

    def build_ineq_cm_rhs_sample(self, time_range):
        """For a single data sample characterised by the time_range, 
        build the full inequality constraint matrix for all timesteps using Kronecker product.
        """

        # TODO: Convert to sparse matrices for efficiency??

        num_timesteps = len(time_range)
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
        ineq_cm = torch.kron(torch.eye(num_timesteps), block_matrix)

        # 3.1k, ui_g lower bound
        # prepend ui_g to top left corner. lower bound, so minus.
        ui_g = -torch.eye(self.num_g)
        ineq_cm = torch.block_diag(ui_g, ineq_cm)

        row_offset = self.num_g # offset first g for 3.1k
        row_offset += self.num_g # offset 3.1h
        for t in time_range:
            for idx_g, g in enumerate(self.G):
                row_idx = row_offset + idx_g
                # GA_{g,t} * UCAP_g * ui_g
                ineq_cm[row_idx, idx_g] = -(self.pGenAva.get((*g, t), 1.0) * self.pUnitCap[g])
            row_offset += num_rows_per_t


        ineq_rhs = []
        # Create right hand side:
        # 3.1k: ui_g lower bound
        ineq_rhs += [0 for _ in range(self.num_g)]
        for t in time_range:
            # 3.1h: Production lower bound
            ineq_rhs += [0 for _ in range(self.num_g)]
            # 3.1b: Production upper bound
            ineq_rhs += [0 for _ in range(self.num_g)]
            # 3.1d: Lineflow lower bound
            ineq_rhs += [self.pImpCap[l] for l in self.L]
            # 3.1e: Lineflow upper bound
            ineq_rhs += [self.pExpCap[l] for l in self.L]
            # 3.1i: Missed demand lower bound
            ineq_rhs += [0 for _ in range(self.num_n)]
            # 3.1j: Missed demand upper bound
            ineq_rhs += [self.pDemand[(n, t)] for n in self.N]
            
        return ineq_cm, ineq_rhs

    def build_eq_cm_rhs_sample(self, time_range):
        """Build the constraint matrix for the equality constraints
        """
        # TODO: Convert to sparse matrix for efficiency??
        num_timesteps = len(time_range)

        # p_{g}, f_in - f_out, md_n
        block_matrix = torch.concat([self.node_to_gen_mask, self.lineflow_mask, torch.eye(self.num_n)], dim=1)

        # Use Kronecker product to copy the block matrix diagonally for all timesteps
        eq_cm = torch.kron(torch.eye(num_timesteps), block_matrix)

        # prepend num_g columns of zeros (ui_g not present in equality constraint matrix, so zeros)
        zero_columns = torch.zeros((eq_cm.shape[0], self.num_g))
        # Concatenate zero_columns with eq_cm horizontally
        eq_cm = torch.hstack([zero_columns, eq_cm])

        eq_rhs = []
        # Build right hand side:
        for t in time_range:
            eq_rhs += [self.pDemand[(n, t)] for n in self.N]

        return eq_cm, eq_rhs

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

if __name__ == "__main__":
    import pickle
    import time
    import json

    from gep_config_parser import *

    from gep_primal_dual_main import prep_data

    CONFIG_FILE_NAME        = "config.toml"
    VISUALIZATION_FILE_NAME = "visualization.toml"

    ## Step 1: parse the input data
    print("Parsing the config file")

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
            
            # Prep proble data:
            data = prep_data(experiment_instance, N=args["N"], G=args["G"], L=args["L"], train=args["train"], valid=args["valid"], test=args["test"], scale=args["scale_problem"], sample_duration=args["sample_duration"], constant_gen_inv=args["operational"])
    
    # Create figure and axes
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    # Plot each matrix as a heatmap
    matrices = [data.eq_cm[0], data.eq_rhs[0].unsqueeze(-1), data.ineq_cm[0], data.ineq_rhs[0].unsqueeze(-1), data.obj_coeff.unsqueeze(-1)]
    titles = ["eq_cm", "eq_rhs", "ineq_cm", "ineq_rhs", "obj_coeff"]

    for ax, matrix, title in zip(axes, matrices, titles):
        im = ax.imshow(matrix, cmap='viridis', aspect='auto')
        ax.set_title(title)
        ax.set_xlabel("Columns")
        ax.set_ylabel("Rows")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # Add colorbar

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()