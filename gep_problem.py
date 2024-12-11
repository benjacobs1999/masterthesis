import torch
import numpy as np

class GEPProblem():

    def __init__(self, T, G, L, N, pDemand, pGenAva, pVOLL, pWeight, pRamping, pInvCost, pVarCost, pUnitCap, pExpCap, pImpCap, sample_duration=12, shuffle=False):
        # self.DTYPE = torch.float32
        # self.DEVICE = (
        #     torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        # )

        self.DTYPE = torch.float64
        self.DEVICE = torch.device="cpu"

        torch.set_default_dtype(self.DTYPE)

        # Input Sets
        self.T = T
        self.G = G
        self.L = L
        self.N = N
        # Input Parameters
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
        self.sample_duration = sample_duration
        self.shuffle = shuffle

        self.time_ranges = [range(i, i + sample_duration, 1) for i in range(1, len(T), sample_duration)]

        # Convert dictionaries to tensors
        self.pDemand_tensor = torch.tensor( # (N, T)
            [[pDemand[(n, t)] for t in self.T] for n in self.N],
            device=self.DEVICE,
            dtype=self.DTYPE
        )

        self.pGenAva_tensor = torch.tensor( # (G, T)
            [[pGenAva.get((*g, t), 1.0) for t in self.T] for g in self.G],
            device=self.DEVICE,
            dtype=self.DTYPE
        )

        self.pVOLL_tensor = torch.tensor(pVOLL, device=self.DEVICE, dtype=self.DTYPE)
        self.pWeight_tensor = torch.tensor(pWeight, device=self.DEVICE, dtype=self.DTYPE)
        self.pRamping_tensor = torch.tensor(pRamping, device=self.DEVICE, dtype=self.DTYPE)

        self.pInvCost_tensor = torch.tensor( # (G,)
            [pInvCost[g] for g in self.G],
            device=self.DEVICE,
            dtype=self.DTYPE
        )

        self.pVarCost_tensor = torch.tensor( # (G,)
            [pVarCost[g] for g in self.G],
            device=self.DEVICE,
            dtype=self.DTYPE
        )

        self.pUnitCap_tensor = torch.tensor( # (G,)
            [pUnitCap[g] for g in self.G],
            device=self.DEVICE,
            dtype=self.DTYPE
        )
        self.pExpCap_tensor = torch.tensor( # (L,)
            [pExpCap[l] for l in self.L],
            device=self.DEVICE,
            dtype=self.DTYPE
        )

        self.pImpCap_tensor = torch.tensor( # (L,)
            [pImpCap[l] for l in self.L],
            device=self.DEVICE,
            dtype=self.DTYPE
        )
        
        self.X = self._split_X_in_batches()
        self._trainX, self._validX, self._testX = self._split_X_in_sets(self.X)


        self.vGenProd_size = len(self.G) * self.sample_duration  # Forall g in G, t in T
        self.vLineFlow_size = len(self.L) * self.sample_duration # Forall l in L, t in T
        self.vLossLoad_size = len(self.N) * self.sample_duration # Forall n in N, t in T
        self.vGenInv_size = len(self.G) # Forall g in G

        self.y_size = self.vGenProd_size + self.vLineFlow_size + self.vLossLoad_size + self.vGenInv_size

        self._xdim = self.X.shape[1]
        self._ydim = self.y_size

        self.num_ineq_constraints = (len(self.G) * self.sample_duration + # 3.1b
                                     len(self.L) * self.sample_duration + # 3.1d
                                     len(self.L) * self.sample_duration + # 3.1e
                                     len(self.G) * (self.sample_duration-1) + # 3.1f
                                     len(self.G) * (self.sample_duration-1) + # 3.1g
                                     len(self.G) * self.sample_duration + # 3.1h
                                     len(self.N) * self.sample_duration + # 3.1i
                                     len(self.N) * self.sample_duration + # 3.1j
                                     len(self.G))           # 3.1k
        
        self.num_eq_constraints = len(self.N) * self.sample_duration # 3.1c
        self.num_variables = (len(self.G) + len(self.L) + len(self.N)) * self.sample_duration + len(self.G)
        self.num_inputs = (len(self.N) * self.sample_duration) + (len(self.G) * self.sample_duration)

        print(f"Size of mu: {self.num_ineq_constraints}")
        print(f"Size of lambda: {self.num_eq_constraints}")
        print(f"Number of variables (size of y): {self.num_variables}")
        print(f"Number of inputs (size of X): {self.num_inputs}")
        # print(f"Size of X: {self.X.shape[1]}")

        # Create masks for e_nodebal (node balancing rule).
        self._generate_nodebal_masks()

    @property
    def trainX(self):
        return self._trainX
    
    @property
    def validX(self):
        return self._validX
    
    @property
    def testX(self):
        return self._testX
    
    @property
    def train_time_ranges(self):
        return self._train_time_ranges
    
    @property
    def val_time_ranges(self):
        return self._val_time_ranges

    @property
    def xdim(self):
        return self._xdim
    
    @property
    def ydim(self):
        return self._ydim
    
    def _split_X_in_batches(self):
        """As input to the primal and dual nets, we use only the parameters that change over time (D_{n,t} and GA_{g,t})
        """
        # self.pDemand_tensor has shape [N, T]
        # self.pGenAva_tensor has shape [G, T]
        # Num Samples [B] = len(self.T) [8760] / self.sample_duration [12] = 730
        # Output should be of shape [B, N*sample_duration + G*sample_duration][730, N*12 + G*12]

        B = len(self.T) // self.sample_duration
        # Reshape tensors directly for batching
        pDemand_batched = self.pDemand_tensor.view(len(self.N), B, self.sample_duration).permute(1, 0, 2).reshape(B, -1)  # [B, N*sample_duration]
        pGenAva_batched = self.pGenAva_tensor.view(len(self.G), B, self.sample_duration).permute(1, 0, 2).reshape(B, -1)  # [B, G*sample_duration]

        # Concatenate along the last dimension
        batched_tensor = torch.cat((pDemand_batched, pGenAva_batched), dim=1)  # [B, N*sample_duration + G*sample_duration]
        return batched_tensor

    def _split_X_in_sets(self, X, train=0.8, valid=0.1, test=0.1):
        # Ensure the split ratios sum to 1
        assert train + valid + test == 1.0

        # Total number of samples
        B = X.size(0)

        # Shuffle the indices if self.shuffle = True
        if self.shuffle:
            indices = torch.randperm(B)
        else:
            indices = torch.arange(B)

        # Compute sizes for each set
        train_size = int(train * B)
        valid_size = int(valid * B)

        # Split the indices
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:train_size+valid_size]
        test_indices = indices[train_size+valid_size:]

        # Split the data
        trainX = X[train_indices]
        validX = X[valid_indices]
        testX = X[test_indices]

        # Convert time_ranges to a tensor or use list comprehension
        time_ranges_tensor = torch.tensor(self.time_ranges)

        # Split time ranges
        self._train_time_ranges = time_ranges_tensor[train_indices].tolist()
        self._val_time_ranges = time_ranges_tensor[valid_indices].tolist()
        self._test_time_ranges = time_ranges_tensor[test_indices].tolist()

        print(f"Size of train set: {train_size}")
        print(f"Size of val set: {valid_size}")
        print(f"Size of test set: {B - train_size - valid_size}")
        
        return trainX, validX, testX
    
    def _split_y(self, y):
        assert(y.shape[1] == self.vGenProd_size+self.vLineFlow_size+self.vLossLoad_size+self.vGenInv_size)
        vGenProd = y[:, :self.vGenProd_size]
        vLineFlow = y[:, self.vGenProd_size:self.vGenProd_size+self.vLineFlow_size]
        vLossLoad = y[:, self.vGenProd_size+self.vLineFlow_size:self.vGenProd_size+self.vLineFlow_size+self.vLossLoad_size]
        vGenInv = y[:, self.vGenProd_size+self.vLineFlow_size+self.vLossLoad_size:]

        vGenProd = vGenProd.reshape(y.shape[0], len(self.G), self.sample_duration) # Reshape to 2d array of size (batch_size, G, T)
        vLineFlow = vLineFlow.reshape(y.shape[0], len(self.L), self.sample_duration)
        vLossLoad = vLossLoad.reshape(y.shape[0], len(self.N), self.sample_duration)

        return vGenProd, vLineFlow, vLossLoad, vGenInv
    
    def _split_x(self, X):
        # Shape is equal to [B, N*T+G*T]
        assert(X.shape[1] == (self.pDemand_tensor.shape[0]*self.sample_duration+self.pGenAva_tensor.shape[0]*self.sample_duration))
        # Sizes
        N = self.pDemand_tensor.shape[0]  # Number of nodes
        G = self.pGenAva_tensor.shape[0]  # Number of generators
        T = self.sample_duration          # Sample duration

        # Slice and reshape
        pDemand = X[:, :N * T].reshape(-1, N, T)  # Shape [B, N, T]
        pGenAva = X[:, N * T:].reshape(-1, G, T)  # Shape [B, G, T]

        # Return shapes equal to [B, N, T], [B, G, T]
        return pDemand, pGenAva
    
    def ineq_dist(self, X, y):
        resids = self.g(X, y)
        return torch.clamp(resids, 0)

    # 3.1a
    # def obj_fn(self, y):
    #     # Split the decision variables
    #     vGenProd, _, vLossLoad, vGenInv = self._split_y(y)

    #     # Investment cost (summed over generators)
    #     inv_cost = torch.sum(self.pInvCost_tensor * self.pUnitCap_tensor * vGenInv, dim=1)

    #     # Operational cost (summed over generators and nodes, weighted by time)
    #     ope_cost = self.pWeight_tensor * (
    #         torch.sum(self.pVarCost_tensor.view(-1, 1) * vGenProd, dim=(1, 2)) +
    #         torch.sum(self.pVOLL_tensor.view(1, -1, 1) * vLossLoad, dim=(1, 2))
    #     )

    #     # Combine costs
    #     return torch.sum(inv_cost + ope_cost)

    #! Batched!
    def obj_fn(self, y):
        # Split the decision variables
        vGenProd, _, vLossLoad, vGenInv = self._split_y(y)

        # Investment cost (summed over generators)
        inv_cost = torch.sum(self.pInvCost_tensor * self.pUnitCap_tensor * vGenInv, dim=1)  # Shape: [batch_size]

        # Operational cost (summed over generators and nodes, weighted by time)
        ope_cost = torch.sum(
            self.pWeight_tensor.view(1, -1) * (  # Broadcast weights across batch dimension
                torch.sum(self.pVarCost_tensor.view(1, -1, 1) * vGenProd, dim=2) +  # Sum over generator dim
                torch.sum(self.pVOLL_tensor.view(1, -1, 1) * vLossLoad, dim=2)  # Sum over load dim
            ), dim=1  # Sum across time dim
        )  # Shape: [batch_size]

        # Combine costs
        total_cost = inv_cost + ope_cost  # Shape: [batch_size]
        return total_cost
    
    # 3.1b
    # Ensure production never exceeds capacity
    def _e_max_prod(self, vGenProd, vGenInv):
        ret = []
        for idxg, g in enumerate(self.G):
            for idxt, t in enumerate(self.T):
                availability = self.pGenAva.get((*g, t), 1.0)  # Default availability to 1.0
                # return vGenProd[g, t] <= availability * self.pUnitCap[g] * vGenInv[g]    
                ret.append(vGenProd[:, idxg, idxt] - availability * self.pUnitCap[g] * vGenInv[:, idxg])
        
        return torch.tensor(ret, device=self.DEVICE)
    
    def _e_max_prod_tensor(self, vGenProd, vGenInv, pGenAva):
        # pGenAva [B, G, T]
        # pUnitCap [G]
        max_cap = pGenAva * self.pUnitCap_tensor.view(1, len(self.G), 1)   # [B, G, T]
        vGenInv_expanded = vGenInv.unsqueeze(-1)                            # [B, G, 1]
        capacity_constraint = max_cap * vGenInv_expanded                    # [B, G, T]   
        # return (vGenProd - capacity_constraint).flatten()                   # [B*G*T]
        return (vGenProd - capacity_constraint).reshape(vGenProd.shape[0], -1) # [B, G*T]
    
    def _e_max_prod_tensor_scaled(self, vGenProd, vGenInv, pGenAva):
        # pGenAva [B, G, T]
        # pUnitCap [G]
        max_cap = pGenAva * self.pUnitCap_tensor.view(1, len(self.G), 1)   # [B, G, T]
        vGenInv_expanded = vGenInv.unsqueeze(-1)                            # [B, G, 1]
        capacity_constraint = max_cap * vGenInv_expanded                    # [B, G, T]   
        # return (vGenProd - capacity_constraint).flatten()                   # [B*G*T]
        return ((vGenProd - capacity_constraint) / max_cap).reshape(vGenProd.shape[0], -1) # [B, G*T]
    
    
    # 3.1d and 3.1e
    def _e_lineflow(self, vLineFlow):
        ret_lb = []
        ret_ub = []
        for idxl, l in enumerate(self.L):
            for idxt, t in enumerate(self.T):
                lb = -vLineFlow[:, idxl, idxt] - self.pImpCap[l]
                ub = vLineFlow[:, idxl, idxt] - self.pExpCap[l]
                ret_lb.append(lb)
                ret_ub.append(ub)
        
        return torch.tensor(ret_lb), torch.tensor(ret_ub)

    def _e_lineflow_tensor(self, vLineFlow):
        # Ensure shapes:
        # self.pImpCap_tensor: [L]
        # self.pExpCap_tensor: [L]
        # vLineFlow: [B, L, T]

        # Expand capacities for broadcasting
        pImpCap_tensor = self.pImpCap_tensor.unsqueeze(0).unsqueeze(-1)  # [1, L, 1]
        pExpCap_tensor = self.pExpCap_tensor.unsqueeze(0).unsqueeze(-1)  # [1, L, 1]

        # Compute lower and upper bounds
        ret_lb = -vLineFlow - pImpCap_tensor  # [B, L, T]
        ret_ub = vLineFlow - pExpCap_tensor  # [B, L, T]

        # Return tensors directly
        return ret_lb.reshape(vLineFlow.shape[0], -1), ret_ub.reshape(vLineFlow.shape[0], -1) # [B, L*T], [B, L*T]
    
    def _e_lineflow_tensor_scaled(self, vLineFlow):
        # Ensure shapes:
        # self.pImpCap_tensor: [L]
        # self.pExpCap_tensor: [L]
        # vLineFlow: [B, L, T]

        # Expand capacities for broadcasting
        pImpCap_tensor = self.pImpCap_tensor.unsqueeze(0).unsqueeze(-1)  # [1, L, 1]
        pExpCap_tensor = self.pExpCap_tensor.unsqueeze(0).unsqueeze(-1)  # [1, L, 1]

        # Compute lower and upper bounds
        #! Scale by the constraint parameter.
        ret_lb = (-vLineFlow - pImpCap_tensor) / pImpCap_tensor  # [B, L, T]
        ret_ub = (vLineFlow - pExpCap_tensor) / pExpCap_tensor  # [B, L, T]

        # Return tensors directly
        return ret_lb.reshape(vLineFlow.shape[0], -1), ret_ub.reshape(vLineFlow.shape[0], -1) # [B, L*T], [B, L*T]

    # 3.1g
    def _e_ramping_up(self, vGenProd, vGenInv):
        ret = []
        for idxg, g in enumerate(self.G):
            for idxt, t in enumerate(self.T[1:]):
                ret.append(-1*(vGenProd[:, idxg, idxt+1] - vGenProd[:, idxg, idxt]) - self.pRamping * self.pUnitCap[g] * vGenInv[:, idxg])
        return torch.tensor(ret)

    def _e_ramping_up_tensor(self, vGenProd, vGenInv):
        # vGenProd [B, G, T] [1, 2, 4]
        # vGenInv [B, G] [1, 2]
        # self.pRamping_tensor = [] (scalar)
        # self.pUnitCap_tensor = [G] [2]

        # p_gt - p_gt-1 <= R * UCAP_g * ui_g
        # p_gt - p_gt-1 - R * UCAP_g * ui_g <= 0

        production_diff = vGenProd[:, :, 1:] - vGenProd[:, :, :-1] # [B, G, T-1] [1, 2, 3]
        ramping_allowed = self.pRamping_tensor * self.pUnitCap_tensor.unsqueeze(0) * vGenInv # [B, G]
        ramping_allowed = ramping_allowed.unsqueeze(-1).expand(-1, -1, production_diff.shape[2]) # [B, G, T-1]
        ret = production_diff - ramping_allowed # [B, G, T-1]
        flattened = ret.reshape(vGenProd.shape[0], -1) # [B, G * (T-1)]

        return flattened
    
    def _e_ramping_up_tensor_scaled(self, vGenProd, vGenInv):
        # vGenProd [B, G, T] [1, 2, 4]
        # vGenInv [B, G] [1, 2]
        # self.pRamping_tensor = [] (scalar)
        # self.pUnitCap_tensor = [G] [2]

        # p_gt - p_gt-1 <= R * UCAP_g * ui_g
        # p_gt - p_gt-1 - R * UCAP_g * ui_g <= 0

        production_diff = vGenProd[:, :, 1:] - vGenProd[:, :, :-1] # [B, G, T-1] [1, 2, 3]
        ramping_allowed = self.pRamping_tensor * self.pUnitCap_tensor.unsqueeze(0) * vGenInv # [B, G]
        ramping_allowed = ramping_allowed.unsqueeze(-1).expand(-1, -1, production_diff.shape[2]) # [B, G, T-1]
        ret = production_diff - ramping_allowed # [B, G, T-1]
        ret = ret / (self.pRamping_tensor * self.pUnitCap_tensor.unsqueeze(0).unsqueeze(-1).expand(-1, -1, production_diff.shape[2]))
        flattened = ret.reshape(vGenProd.shape[0], -1) # [B, G * (T-1)]

        return flattened

    # 3.1f
    def _e_ramping_down(self, vGenProd, vGenInv):
        ret = []
        for idxg, g in enumerate(self.G):
            for idxt, t in enumerate(self.T[1:]):
                ret.append(vGenProd[:, idxg, idxt+1] - vGenProd[:, idxg, idxt] - self.pRamping * self.pUnitCap[g] * vGenInv[:, idxg])
        return torch.tensor(ret)
    
    def _e_ramping_down_tensor(self, vGenProd, vGenInv):
        # vGenProd [B, G, T]
        # vGenInv [B, G]
        # self.pRamping_tensor = [] (scalar)
        # self.pUnitCap_tensor = [G]

        # p_{gt} - p_{g,t-1} >= -R * UCAP_g * ui_g
        # -(p_{gt} - p_{g,t-1}) <= R * UCAP_g * ui_g
        # p_{g, t-1} - p_{gt} - R * UCAP_g * ui_g <= 0

        # Production =  [0, T-1] - [1, T]
        production_diff = vGenProd[:, :, :-1] - vGenProd[:, :, 1:] # [B, G, T-1]
        ramping_allowed = self.pRamping_tensor * self.pUnitCap_tensor.unsqueeze(0) * vGenInv # [B, G]
        ramping_allowed = ramping_allowed.unsqueeze(-1).expand(-1, -1, production_diff.shape[2]) # [B, G, T-1]
        ret = production_diff - ramping_allowed # [B, G, T-1]
        flattened = ret.reshape(vGenProd.shape[0], -1) # [B, G * (T-1)]

        return flattened
    
    def _e_ramping_down_tensor_scaled(self, vGenProd, vGenInv):
        # vGenProd [B, G, T]
        # vGenInv [B, G]
        # self.pRamping_tensor = [] (scalar)
        # self.pUnitCap_tensor = [G]

        # p_{gt} - p_{g,t-1} >= -R * UCAP_g * ui_g
        # -(p_{gt} - p_{g,t-1}) <= R * UCAP_g * ui_g
        # p_{g, t-1} - p_{gt} - R * UCAP_g * ui_g <= 0

        # Production =  [0, T-1] - [1, T]
        production_diff = vGenProd[:, :, :-1] - vGenProd[:, :, 1:] # [B, G, T-1]
        ramping_allowed = self.pRamping_tensor * self.pUnitCap_tensor.unsqueeze(0) * vGenInv # [B, G]
        ramping_allowed = ramping_allowed.unsqueeze(-1).expand(-1, -1, production_diff.shape[2]) # [B, G, T-1]
        ret = production_diff - ramping_allowed # [B, G, T-1]
        ret = ret / (self.pRamping_tensor * self.pUnitCap_tensor.unsqueeze(0).unsqueeze(-1).expand(-1, -1, production_diff.shape[2]))
        flattened = ret.reshape(vGenProd.shape[0], -1) # [B, G * (T-1)]

        return flattened
            
    # 3.1h
    def _e_gen_prod_positive(self, vGenProd):
        ret = []
        for idxg, g in enumerate(self.G):
            for idxt, t in enumerate(self.T):
                ret.append(-vGenProd[:, idxg, idxt])
        return torch.tensor(ret)

    def _e_gen_prod_positive_tensor(self, vGenProd):
        # vGenProd [B, G, T] [1, 2, 4]
        return -1*vGenProd.reshape(vGenProd.shape[0], -1)
    
    # 3.1i
    def _e_missed_demand_positive(self, vLossLoad):
        ret = []
        for idxn, n in enumerate(self.N):
            for idxt, t in enumerate(self.T):
                ret.append(-vLossLoad[:, idxn, idxt])
        return torch.tensor(ret)
    
    def _e_missed_demand_positive_tensor(self, vLossLoad):
        # vLossLoad [B, N, T] [1, 2, 4]
        return -1*vLossLoad.reshape(vLossLoad.shape[0], -1)
    

    # 3.1j
    def _e_missed_demand_leq_demand(self, vLossLoad):
        ret = []
        for idxn, n in enumerate(self.N):
            for idxt, t in enumerate(self.T):
                ret.append(vLossLoad[:, idxn, idxt] - self.pDemand[(n, t)])
        return torch.tensor(ret)

    def _e_missed_demand_leq_demand_tensor(self, vLossLoad, pDemand):
        # vLossLoad [B, N, T] [1, 2, 4]
        # self.pDemand_tensor [N, T]
        # pDemand [B, N, T]
        return (vLossLoad - pDemand).reshape(vLossLoad.shape[0], -1)

    def _e_missed_demand_leq_demand_tensor_scaled(self, vLossLoad, pDemand):
        # vLossLoad [B, N, T] [1, 2, 4]
        # self.pDemand_tensor [N, T]
        # pDemand [B, N, T]
        return ((vLossLoad - pDemand) / pDemand).reshape(vLossLoad.shape[0], -1)
    
    # 3.1k
    def _e_num_power_generation_units_positive(self, vGenInv):
        ret = []
        for idxg, g in enumerate(self.G):
            ret.append(-vGenInv[:, idxg])
        return torch.tensor(ret)
    
    def _e_num_power_generation_units_positive_tensor(self, vGenInv):
        return -1*vGenInv.reshape(vGenInv.shape[0], -1)


    def g(self, X, y):
        """Assume y contains [*vGenProd, *vLineFlow, *vLossLoad, *vGenInv]

        Args:
            y (_type_): 

        Returns [eMaxProd, eLineFlow, eRampingDown, eRampingUp, eGenProdPositive, eMissedDemandPositive, eMissedDemandLeqDemand eNumPowerGenerationUnitsPositive]
        """
        vGenProd, vLineFlow, vLossLoad, vGenInv = self._split_y(y)

        # pDemand [B, N, T]
        # pGenAva [B, G, T]
        pDemand, pGenAva = self._split_x(X)


        # 3.1b
        # b = self._e_max_prod(vGenProd, vGenInv)
        b = self._e_max_prod_tensor(vGenProd, vGenInv, pGenAva)
        

        # 3.1d, 3.1e
        # d, e = self._e_lineflow(vLineFlow)
        d, e = self._e_lineflow_tensor(vLineFlow)
        # d, e = self._e_lineflow_tensor_scaled(vLineFlow)

        # 3.1f
        # f = self._e_ramping_down(vGenProd, vGenInv)
        f = self._e_ramping_down_tensor(vGenProd, vGenInv)
        # f = self._e_ramping_down_tensor_scaled(vGenProd, vGenInv)

        # 3.1g
        # g = self._e_ramping_up(vGenProd, vGenInv)
        g = self._e_ramping_up_tensor(vGenProd, vGenInv)
        # g = self._e_ramping_up_tensor_scaled(vGenProd, vGenInv)

        # 3.1h
        # h = self._e_gen_prod_positive(vGenProd)
        h = self._e_gen_prod_positive_tensor(vGenProd)

        # 3.1i
        # i = self._e_missed_demand_positive(vLossLoad)
        i = self._e_missed_demand_positive_tensor(vLossLoad)

        # 3.1j
        # j = self._e_missed_demand_leq_demand(vLossLoad)
        j = self._e_missed_demand_leq_demand_tensor(vLossLoad, pDemand)
        # j = self._e_missed_demand_leq_demand_tensor_scaled(vLossLoad, pDemand)

        # 3.1k
        # k = self._e_num_power_generation_units_positive(vGenInv)
        k = self._e_num_power_generation_units_positive_tensor(vGenInv)

        g_x_y = torch.cat((b, d, e, f, g, h, i, j, k), dim=1)

        return g_x_y
    
    # 3.1c
    def _e_nodebal(self, vGenProd, vLineFlow, vLossLoad):
        ret = []
        for idxn, n in enumerate(self.N):
            for idxt, t in enumerate(self.T):
                # print(self.pDemand[n, t])
                gen_sum = sum(vGenProd[:, idxg, idxt] for idxg, g in enumerate(self.G) if g[0] == n)
                # print(gen_sum)
                inflow_sum = sum(vLineFlow[:, idxl, idxt] for idxl, l in enumerate(self.L) if l[1] == n)
                # print(inflow_sum)
                outflow_sum = sum(vLineFlow[:, idxl, idxt] for idxl, l in enumerate(self.L) if l[0] == n)
                # print(outflow_sum)
                # print(vLossLoad[:, idxn, idxt])
                ret.append(self.pDemand[n, t] -
                                (gen_sum +
                                inflow_sum - 
                                outflow_sum +
                                vLossLoad[:, idxn, idxt]
                                ))
        
        return torch.tensor(ret)

    def _e_nodebal_tensor(self, vGenProd, vLineFlow, vLossLoad, pDemand):
        # ! Check
        # vGenProd [B, G, T]
        # vLineFlow [B, L, T]
        # vLossLoad [B, N, T]
        # self.pDemand_tensor [N, T]
        # pDemand [B, N, T]

        # D_nt - sum(power supplied to n by generators) + sum(power entering node n from other nodes) + sum(power leaving node n to other nodes)
        # Compute generation contribution for each node
        # Convert pDemand to tensor and add batch dimension
        vGenProd_expanded = vGenProd.unsqueeze(1)  # [B, 1, G, T]
        vLineFlow_expanded = vLineFlow.unsqueeze(1) # [B, 1, L, T]
        gen_mask = self.gen_mask.unsqueeze(0) # [1, N, G, T]
        inflow_mask = self.inflow_mask.unsqueeze(0) # [1, N, L, T]
        outflow_mask = self.outflow_mask.unsqueeze(0) # [1, N, L, T]
        # pDemand_tensor = self.pDemand_tensor.unsqueeze(0) # [1, N, T]

        # self.gen_mask [N, G, T]
        # self.inflow_mask [N, L, T]
        # self.outflow_mask [N, L, T]
        # Compute generation contribution for each node
        gen_sum = (gen_mask * vGenProd_expanded).sum(dim=2)  # [B, N, T]

        # Compute inflows and outflows for each node
        inflow_sum = (inflow_mask * vLineFlow_expanded).sum(dim=2)  # [B, N, T]
        outflow_sum = (outflow_mask * vLineFlow_expanded).sum(dim=2)  # [B, N, T]

        # print(pDemand_tensor)
        # print(gen_sum)
        # print(inflow_sum)
        # print(vLossLoad)
        # Node balance equation
        node_balance = (
            pDemand                                 # [B, N, T]
            - (gen_sum                              # [B, N, T]
            + inflow_sum                            # [B, N, T]
            - outflow_sum                           # [B, N, T]
            + vLossLoad)                            # [B, N, T]
        )

        return node_balance.reshape(vGenProd.shape[0], -1)  # Shape: [B, N * T]
    
        
    def _generate_nodebal_masks(self):
        # Create generator-to-node mapping mask
        self.gen_mask = torch.tensor(
            [[1 if g[0] == n else 0 for g in self.G] for n in self.N], 
            device=self.DEVICE, dtype=self.DTYPE
        ).unsqueeze(-1).expand(-1, -1, self.sample_duration)  # Shape: [N, G, T]

        # Create line-to-node mapping masks for inflows and outflows
        self.inflow_mask = torch.tensor(
            [[1 if l[1] == n else 0 for l in self.L] for n in self.N], 
            device=self.DEVICE, dtype=self.DTYPE
        ).unsqueeze(-1).expand(-1, -1, self.sample_duration)  # Shape: [N, L, T]

        self.outflow_mask = torch.tensor(
            [[1 if l[0] == n else 0 for l in self.L] for n in self.N], 
            device=self.DEVICE, dtype=self.DTYPE
        ).unsqueeze(-1).expand(-1, -1, self.sample_duration)  # Shape: [N, L, T]

    def h(self, X, y):
        """Assume y contains [*vGenProd, *vLineFlow, *vLossLoad, *vGenInv]

        Args:
            y (_type_): _description_
        """
        vGenProd, vLineFlow, vLossLoad, _ = self._split_y(y)
        pDemand, _ = self._split_x(X)

        # 3.1c
        # h_x_y = self._e_nodebal(vGenProd, vLineFlow, vLossLoad)
        h_x_y = self._e_nodebal_tensor(vGenProd, vLineFlow, vLossLoad, pDemand)

        return h_x_y