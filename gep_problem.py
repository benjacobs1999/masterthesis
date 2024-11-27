import torch
import numpy as np

DTYPE = torch.float32

torch.set_default_dtype(DTYPE)

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)

class GEPProblem():

    def __init__(self, T, G, L, N, pDemand, pGenAva, pVOLL, pWeight, pRamping, pInvCost, pVarCost, pUnitCap, pExpCap, pImpCap):
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
        # Convert dictionaries to tensors
        self.pDemand_tensor = torch.tensor( # (N, T)
            [[pDemand[(n, t)] for t in self.T] for n in self.N],
            device=DEVICE,
            dtype=DTYPE
        )

        self.pGenAva_tensor = torch.tensor( # (G, T)
            [[pGenAva.get((*g, t), 1.0) for t in self.T] for g in self.G],
            device=DEVICE,
            dtype=DTYPE
        )

        self.pVOLL_tensor = torch.tensor(pVOLL, device=DEVICE, dtype=DTYPE)
        self.pWeight_tensor = torch.tensor(pWeight, device=DEVICE, dtype=DTYPE)
        self.pRamping_tensor = torch.tensor(pRamping, device=DEVICE, dtype=DTYPE)

        self.pInvCost_tensor = torch.tensor( # (G,)
            [pInvCost[g] for g in self.G],
            device=DEVICE,
            dtype=DTYPE
        )

        self.pVarCost_tensor = torch.tensor( # (G,)
            [pVarCost[g] for g in self.G],
            device=DEVICE,
            dtype=DTYPE
        )

        self.pUnitCap_tensor = torch.tensor( # (G,)
            [pUnitCap[g] for g in self.G],
            device=DEVICE,
            dtype=DTYPE
        )
        self.pExpCap_tensor = torch.tensor( # (L,)
            [pExpCap[l] for l in self.L],
            device=DEVICE,
            dtype=DTYPE
        )

        self.pImpCap_tensor = torch.tensor( # (L,)
            [pImpCap[l] for l in self.L],
            device=DEVICE,
            dtype=DTYPE
        )

        # TODO: Is this how we want the input to be??
        self.X = torch.tensor(np.concatenate((list(pDemand.values()), list(pGenAva.values()), [pVOLL], [pWeight], [pRamping], list(pInvCost.values()), list(pVarCost.values()), list(pUnitCap.values()), list(pExpCap.values()), list(pImpCap.values()))), dtype=DTYPE).unsqueeze(0)

        self.vGenProd_size = len(self.G) * len(self.T)  # Forall g in G, t in T
        self.vLineFlow_size = len(self.L) * len(self.T) # Forall l in L, t in T
        self.vLossLoad_size = len(self.N) * len(self.T) # Forall n in N, t in T
        self.vGenInv_size = len(self.G) # Forall g in G

        self.y_size = self.vGenProd_size + self.vLineFlow_size + self.vLossLoad_size + self.vGenInv_size

        self._xdim = self.X.shape[1]
        self._ydim = self.y_size

        self.num_ineq_constraints = (len(self.G) * len(self.T) + # 3.1b
                                     len(self.L) * len(self.T) + # 3.1d
                                     len(self.L) * len(self.T) + # 3.1e
                                     len(self.G) * (len(self.T)-1) + # 3.1f
                                     len(self.G) * (len(self.T)-1) + # 3.1g
                                     len(self.G) * len(self.T) + # 3.1h
                                     len(self.N) * len(self.T) + # 3.1i
                                     len(self.N) * len(self.T) + # 3.1j
                                     len(self.G))           # 3.1k
        
        self.num_eq_constraints = len(self.N) * len(self.T) # 3.1c

        print(f"Size of mu: {self.num_ineq_constraints}")
        print(f"Size of lambda: {self.num_eq_constraints}")

        # Create masks for e_nodebal (node balancing rule).
        self._generate_nodebal_masks()

    @property
    def trainX(self):
        return self.X
    
    @property
    def xdim(self):
        return self._xdim
    
    @property
    def ydim(self):
        return self._ydim
    
    def ineq_dist(self, y):
        resids = self.g(y)
        return torch.clamp(resids, 0)
    
    def _split_y(self, y):
        assert(y.shape[1] == self.vGenProd_size+self.vLineFlow_size+self.vLossLoad_size+self.vGenInv_size)
        vGenProd = y[:, :self.vGenProd_size]
        vLineFlow = y[:, self.vGenProd_size:self.vGenProd_size+self.vLineFlow_size]
        vLossLoad = y[:, self.vGenProd_size+self.vLineFlow_size:self.vGenProd_size+self.vLineFlow_size+self.vLossLoad_size]
        vGenInv = y[:, self.vGenProd_size+self.vLineFlow_size+self.vLossLoad_size:]

        vGenProd = vGenProd.reshape(y.shape[0], len(self.G), len(self.T)) # Reshape to 2d array of size (batch_size, G, T)
        vLineFlow = vLineFlow.reshape(y.shape[0], len(self.L), len(self.T))
        vLossLoad = vLossLoad.reshape(y.shape[0], len(self.N), len(self.T))

        return vGenProd, vLineFlow, vLossLoad, vGenInv

    # 3.1a
    def obj_fn(self, y):        
        vGenProd, _, vLossLoad, vGenInv = self._split_y(y)
        # Investment cost
        inv_cost = sum(
            self.pInvCost[g] * self.pUnitCap[g] * vGenInv[:, idx] for idx, g in enumerate(self.G)
        )

        # Operating cost
        ope_cost = self.pWeight * (
            sum(self.pVarCost[g] * vGenProd[:, idxg, idxt] for idxg, g in enumerate(self.G) for idxt, t in enumerate(self.T))
            + sum(self.pVOLL * vLossLoad[:, idxn, idxt] for idxn, n in enumerate(self.N) for idxt, t in enumerate(self.T))
        )

        return inv_cost + ope_cost
    
    # 3.1b
    # Ensure production never exceeds capacity
    def _e_max_prod(self, vGenProd, vGenInv):
        ret = []
        for idxg, g in enumerate(self.G):
            for idxt, t in enumerate(self.T):
                availability = self.pGenAva.get((*g, t), 1.0)  # Default availability to 1.0
                # return vGenProd[g, t] <= availability * self.pUnitCap[g] * vGenInv[g]    
                ret.append(vGenProd[:, idxg, idxt] - availability * self.pUnitCap[g] * vGenInv[:, idxg])
        
        return torch.tensor(ret, device=DEVICE)
    
    def _e_max_prod_tensor(self, vGenProd, vGenInv):
        max_cap = self.pGenAva_tensor * self.pUnitCap_tensor.unsqueeze(1)   # [G, T]
        vGenInv_expanded = vGenInv.unsqueeze(-1)                            # [B, G, 1]
        capacity_constraint = max_cap.unsqueeze(0) * vGenInv_expanded       # [B, G, T]   
        return (vGenProd - capacity_constraint).flatten()                   # [B*G*T]
    
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
        return ret_lb.flatten(), ret_ub.flatten()

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

        # p_gt - p_gt-1 >= -R * UCAP_g * ui_g
        # -(p_gt - p_gt-1) <= R * UCAP_g * ui_g

        production_diff = vGenProd[:, :, 1:] - vGenProd[:, :, :-1] # [B, G, T-1] [1, 2, 3]
        ramping_allowed = self.pRamping_tensor * self.pUnitCap_tensor.unsqueeze(0) * vGenInv # [B, G]
        ramping_allowed = ramping_allowed.unsqueeze(-1).expand(-1, -1, production_diff.shape[2]) # [B, G, T-1]
        ret = -1*production_diff - ramping_allowed # [B, G, T-1]
        flattened = ret.flatten() # [B * G * (T-1)]

        return flattened

    # 3.1f
    def _e_ramping_down(self, vGenProd, vGenInv):
        ret = []
        for idxg, g in enumerate(self.G):
            for idxt, t in enumerate(self.T[1:]):
                ret.append(vGenProd[:, idxg, idxt+1] - vGenProd[:, idxg, idxt] - self.pRamping * self.pUnitCap[g] * vGenInv[:, idxg])
        return torch.tensor(ret)
    
    def _e_ramping_down_tensor(self, vGenProd, vGenInv):
        # vGenProd [B, G, T] [1, 2, 4]
        # vGenInv [B, G] [1, 2]
        # self.pRamping_tensor = [] (scalar)
        # self.pUnitCap_tensor = [G] [2]

        # p_gt - p_gt-1 <= R * UCAP_g * ui_g

        production_diff = vGenProd[:, :, 1:] - vGenProd[:, :, :-1] # [B, G, T-1] [1, 2, 3]
        ramping_allowed = self.pRamping_tensor * self.pUnitCap_tensor.unsqueeze(0) * vGenInv # [B, G]
        ramping_allowed = ramping_allowed.unsqueeze(-1).expand(-1, -1, production_diff.shape[2]) # [B, G, T-1]
        ret = production_diff - ramping_allowed # [B, G, T-1]
        flattened = ret.flatten() # [B * G * (T-1)]

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
        return -1*vGenProd.flatten()
    
    # 3.1i
    def _e_missed_demand_positive(self, vLossLoad):
        ret = []
        for idxn, n in enumerate(self.N):
            for idxt, t in enumerate(self.T):
                ret.append(-vLossLoad[:, idxn, idxt])
        return torch.tensor(ret)
    
    def _e_missed_demand_positive_tensor(self, vLossLoad):
        # vLossLoad [B, N, T] [1, 2, 4]
        return -1*vLossLoad.flatten()
    

    # 3.1j
    def _e_missed_demand_leq_demand(self, vLossLoad):
        ret = []
        for idxn, n in enumerate(self.N):
            for idxt, t in enumerate(self.T):
                ret.append(vLossLoad[:, idxn, idxt] - self.pDemand[(n, t)])
        return torch.tensor(ret)

    def _e_missed_demand_leq_demand_tensor(self, vLossLoad):
        # vLossLoad [B, G, T] [1, 2, 4]
        # self.pDemand_tensor [G, T]
        return (vLossLoad - self.pDemand_tensor.unsqueeze(0)).flatten()
    
    # 3.1k
    def _e_num_power_generation_units_positive(self, vGenInv):
        ret = []
        for idxg, g in enumerate(self.G):
            ret.append(-vGenInv[:, idxg])
        return torch.tensor(ret)
    
    def _e_num_power_generation_units_positive_tensor(self, vGenInv):
        return -1*vGenInv.flatten()


    def g(self, y):
        """Assume y contains [*vGenProd, *vLineFlow, *vLossLoad, *vGenInv]

        Args:
            y (_type_): _description_
        """
        vGenProd, vLineFlow, vLossLoad, vGenInv = self._split_y(y)

        # 3.1b
        # b = self._e_max_prod(vGenProd, vGenInv)
        b = self._e_max_prod_tensor(vGenProd, vGenInv)
        

        # 3.1d, 3.1e
        # d, e = self._e_lineflow(vLineFlow)
        d, e = self._e_lineflow_tensor(vLineFlow)

        # 3.1f
        # f = self._e_ramping_down(vGenProd, vGenInv)
        f = self._e_ramping_down_tensor(vGenProd, vGenInv)

        # 3.1g
        # g = self._e_ramping_up(vGenProd, vGenInv)
        g = self._e_ramping_up_tensor(vGenProd, vGenInv)

        # 3.1h
        # h = self._e_gen_prod_positive(vGenProd)
        h = self._e_gen_prod_positive_tensor(vGenProd)

        # 3.1i
        # i = self._e_missed_demand_positive(vLossLoad)
        i = self._e_missed_demand_positive_tensor(vLossLoad)

        # 3.1j
        # j = self._e_missed_demand_leq_demand(vLossLoad)
        j = self._e_missed_demand_leq_demand_tensor(vLossLoad)

        # 3.1k
        # k = self._e_num_power_generation_units_positive(vGenInv)
        k = self._e_num_power_generation_units_positive_tensor(vGenInv)

        g_x_y = torch.concatenate((b, d, e, f, g, h, i, j, k))

        return g_x_y.unsqueeze(0)
    
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

    def _e_nodebal_tensor(self, vGenProd, vLineFlow, vLossLoad):
        # vGenProd [B, G, T]
        # vLineFlow [B, L, T]
        # vLossLoad [B, N, T]
        # self.pDemand_tensor [N, T]

        # D_nt - sum(power supplied to n by generators) + sum(power entering node n from other nodes) + sum(power leaving node n to other nodes)
        # Compute generation contribution for each node
        # Convert pDemand to tensor and add batch dimension
        vGenProd_expanded = vGenProd.unsqueeze(1)  # [B, 1, G, T]
        vLineFlow_expanded = vLineFlow.unsqueeze(1) # [B, 1, L, T]
        gen_mask = self.gen_mask.unsqueeze(0) # [1, N, G, T]
        inflow_mask = self.inflow_mask.unsqueeze(0) # [1, N, L, T]
        outflow_mask = self.outflow_mask.unsqueeze(0) # [1, N, L, T]
        pDemand_tensor = self.pDemand_tensor.unsqueeze(0) # [1, N, T]

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
            pDemand_tensor                         # [1, N, T] (broadcast to [B, N, T])
            - (gen_sum                              # [B, N, T]
            + inflow_sum                           # [B, N, T]
            - outflow_sum                          # [B, N, T]
            + vLossLoad)                            # [B, N, T]
        )

        return node_balance.flatten()  # Shape: [B * N * T]
    
        
    def _generate_nodebal_masks(self):
        # Create generator-to-node mapping mask
        self.gen_mask = torch.tensor(
            [[1 if g[0] == n else 0 for g in self.G] for n in self.N], 
            device=DEVICE, dtype=DTYPE
        ).unsqueeze(-1).expand(-1, -1, len(self.T))  # Shape: [N, G, T]

        # Create line-to-node mapping masks for inflows and outflows
        self.inflow_mask = torch.tensor(
            [[1 if l[1] == n else 0 for l in self.L] for n in self.N], 
            device=DEVICE, dtype=DTYPE
        ).unsqueeze(-1).expand(-1, -1, len(self.T))  # Shape: [N, L, T]

        self.outflow_mask = torch.tensor(
            [[1 if l[0] == n else 0 for l in self.L] for n in self.N], 
            device=DEVICE, dtype=DTYPE
        ).unsqueeze(-1).expand(-1, -1, len(self.T))  # Shape: [N, L, T]

    def h(self, y):
        """Assume y contains [*vGenProd, *vLineFlow, *vLossLoad, *vGenInv]

        Args:
            y (_type_): _description_
        """
        vGenProd, vLineFlow, vLossLoad, _ = self._split_y(y)

        # 3.1c
        # h_x_y = self._e_nodebal(vGenProd, vLineFlow, vLossLoad)
        h_x_y = self._e_nodebal_tensor(vGenProd, vLineFlow, vLossLoad)

        return h_x_y.unsqueeze(0)