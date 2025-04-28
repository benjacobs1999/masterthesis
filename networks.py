import torch
import torch.nn as nn


class PrimalNet(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.data = data
        self.hidden_sizes = [int(args["hidden_size_factor"]*data.xdim)] * args["n_layers"]
        
        # Create the list of layer sizes
        layer_sizes = [data.xdim] + self.hidden_sizes + [data.ydim]
        layers = []

        # layers.append(nn.LayerNorm(data.xdim))

        # Create layers dynamically based on the provided hidden_sizes
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            if out_size != data.ydim:  # Add ReLU activation for hidden layers only
                layers.append(nn.ReLU())

        # Initialize all layers
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                # nn.init.xavier_uniform_(layer.weight)

        self.net = nn.Sequential(*layers)
    
    def forward(self, x, total_demands=None):
        #! If we are training, we do not scale the output by the total demands. Only for logging metrics.
        if not self.training and (total_demands != None):
            return self.net(x) * total_demands
        else:
            return self.net(x)


class DualNetEndToEnd(nn.Module):
    def __init__(self, args, data, hidden_size_factor=5.0, n_layers=4):
        super().__init__()
        self.data = data
        self.hidden_sizes = [int(hidden_size_factor*data.xdim)] * n_layers
        self.args = args
        self.ED_args = args["ED_args"]

        if self.ED_args["benders_compact"]:
            self.out_dim = data.num_g + data.neq
        else:
            self.out_dim = data.neq

        if args["device"] == "mps":
            self.DTYPE = torch.float32
            self.DEVICE = torch.device("mps")
        else:
            self.DTYPE = torch.float64
            self.DEVICE = torch.device("cpu")

        #! Only predict lambda, we infer mu from it.
        self.feed_forward = FeedForwardNet(args, data.xdim, self.hidden_sizes, output_dim=self.out_dim).to(self.DTYPE).to(self.DEVICE)

        # Set dual variables to 0 at the first iteration
        # nn.init.zeros_(self.feed_forward.net[-1].weight)  # Initialize output layer weights to 0
        # nn.init.zeros_(self.feed_forward.net[-1].bias)    # Initialize output layer biases to 0
    
    def complete_duals(self, lamb):
        # first num_g outputs are the equality constraints added in benders compact form
        # if self.data.args["benders_compact"]:
            #TODO
            # pass
            # lamb_D_nt = lamb[:, self.data.num_g:]
            # eq_cm_D_nt = self.data.eq_cm[:, self.data.num_g:, self.data.num_g:]
            # obj_coeff = self.data.obj_coeff[self.data.num_g:]
        # else:
        eq_cm_D_nt = self.data.eq_cm
        lamb_D_nt = lamb
        obj_coeff = self.data.obj_coeff

        # mu = obj_coeff - torch.matmul(eq_cm_D_nt.transpose(1, 2), lamb_D_nt.unsqueeze(-1)).squeeze(-1)
        mu = obj_coeff - torch.matmul(lamb_D_nt, eq_cm_D_nt)

        # Compute lower and upper bound multipliers
        mu_lb = torch.relu(mu)   # Lower bound multipliers |mu|^+
        mu_ub = torch.relu(-mu)  # Upper bound multipliers |mu|^-

        # Split into groups, following the exact structure of mu
        p_g_lb = mu_lb[:, :self.data.num_g]  # Lower bounds for p_g
        p_g_ub = mu_ub[:, :self.data.num_g]  # Upper bounds for p_g

        f_l_lb = mu_lb[:, self.data.num_g:self.data.num_g + self.data.num_l]  # Lower bounds for f_l
        f_l_ub = mu_ub[:, self.data.num_g:self.data.num_g + self.data.num_l]  # Upper bounds for f_l

        md_n_lb = mu_lb[:, self.data.num_g + self.data.num_l:]  # Lower bounds for md_n
        md_n_ub = mu_ub[:, self.data.num_g + self.data.num_l:]  # Upper bounds for md_n

        # Concatenate while maintaining order
        out_mu = torch.cat([
            p_g_lb, p_g_ub,  # Lower and Upper bounds for p_g
            f_l_lb, f_l_ub,  # Lower and Upper bounds for f_l
            md_n_lb, md_n_ub  # Lower and Upper bounds for md_n
        ], dim=1)

        return out_mu
        
        
    def forward(self, x):
        out_lamb = self.feed_forward(x)
        out_mu = self.complete_duals(out_lamb)

        return out_mu, out_lamb

class DualNet(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.data = data
        self.hidden_sizes = [int(args["hidden_size_factor"]*data.xdim)] * args["n_layers"]
        self.mu_size = self.data.nineq
        self.lamb_size = self.data.neq

        # Create the list of layer sizes
        layer_sizes = [data.xdim] + self.hidden_sizes
        # layer_sizes = [2*data.xdim + 1000] + self.hidden_sizes
        layers = []
        # Create layers dynamically based on the provided hidden_sizes
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())

        # Initialize all layers
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

        # Add the output layer
        self.out_layer = nn.Linear(self.hidden_sizes[-1], self.mu_size + self.lamb_size)
        nn.init.zeros_(self.out_layer.weight)  # Initialize output layer weights to 0
        nn.init.zeros_(self.out_layer.bias)    # Initialize output layer biases to 0
        layers.append(self.out_layer)

        self.net = nn.Sequential(*layers)
    
    def forward(self, x, *args):
        out = self.net(x)
        #! ReLU to enforce nonnegativity in mu. Test with it.
        #! Does this work with zero initialization?
        # out_mu = torch.relu(out[:, :self.mu_size])
        out_mu = out[:, :self.mu_size]
        out_lamb = out[:, self.mu_size:]
        return out_mu, out_lamb


class DualNetTwoOutputLayers(nn.Module):
    def __init__(self, data, hidden_size):
        super().__init__()
        self.data = data
        self.hidden_size = hidden_size
        layer_sizes = [data.xdim, self.hidden_size, self.hidden_size]
        layers = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)
        self.out_layer_mu = nn.Linear(self.hidden_size, data.nineq)
        self.out_layer_lamb = nn.Linear(self.hidden_size, data.neq)
        # Init last layers as 0, like in the paper
        nn.init.zeros_(self.out_layer_mu.weight)
        nn.init.zeros_(self.out_layer_mu.bias)
        nn.init.zeros_(self.out_layer_lamb.weight)
        nn.init.zeros_(self.out_layer_lamb.bias)

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.net(x)
        out_mu = self.out_layer_mu(out)
        out_lamb = self.out_layer_lamb(out)
        return out_mu, out_lamb


class FeedForwardNet(nn.Module):
    def __init__(self, args, input_dim, hidden_sizes, output_dim):
        """_summary_

        Args:
            input_dim (int): Number of input features
            output_dim (int): Number of output features
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.output_dim = output_dim

        if args["device"] == "mps":
            self.DTYPE = torch.float32
            self.DEVICE = torch.device("mps")
        else:
            self.DTYPE = torch.float64
            self.DEVICE = torch.device("cpu")

        torch.set_default_dtype(self.DTYPE)
        torch.set_default_device(self.DEVICE)

        # Create the list of layer sizes
        layer_sizes = [self.input_dim] + self.hidden_sizes + [self.output_dim]
        layers = []
        
        if args["layernorm"]:
            layers.append(nn.LayerNorm(input_dim)) #! This is necessary to prevent gradient saturation in the sigmoids.

        # Create layers dynamically based on the provided hidden_sizes
        for idx, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            # layers.append(nn.LayerNorm(in_size)) #! LayerNorm
            layers.append(nn.Linear(in_size, out_size))
            # Add ReLU only if it is not the last layer
            if idx < len(layer_sizes) - 2:  # The last layer does not need ReLU
                layers.append(nn.ReLU())
                # layers.append(nn.LayerNorm(out_size))
                # layers.append(nn.LeakyReLU())
        
        # Initialize all layers
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                # nn.init.xavier_uniform_(layer.weight)
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        x_out = self.net(x)
        return x_out

    # def forward(self, x):
    #     x_out = x
    #     # Iterate through layers manually for debugging
    #     for layer in self.net:
    #         x_out = layer(x_out)
            
    #         # If it's not a ReLU layer, check if any activations are negative
    #         if not isinstance(layer, nn.ReLU):
    #             # Check if any activation is < 0
    #             if torch.any(x_out < 0):
    #                 # print(f"Negative activations found: {x_out[x_out < 0]}")
    #                 print(f"Mean of activations: {x_out.mean().item()}, Min of activations: {x_out.min().item()}, Max of activations: {x_out.max().item()}")
        
    #     return x_out


class BoundRepairLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, lb, ub, k=1.0):
        """_summary_

        Args:
            x (_type_): Decision variables, shape [B, N, T]
            lb (_type_): Lower bounds of decision variables, shape [B, N, T]
            ub (_type_): Upper bounds of decision variables, shape [B, N, T]

        Returns:
            _type_: _description_
        """
        scaled = torch.sigmoid(k * x)

        repaired = lb + (ub - lb) * scaled

        if False:
            # Attach hooks to inspect gradients
            def print_grad(name):
                return lambda grad: print(f"Gradient for {name}: {grad.norm():.4e}")
            x.register_hook(print_grad("x"))
            scaled.register_hook(print_grad("sigmoid(kx)"))
            repaired.register_hook(print_grad("repaired output"))

        return repaired
        # return torch.sigmoid(k*x)
        # return torch.clamp(x, lb, ub)
        # return (lb + (ub - lb)/2 * (torch.tanh(x) + 1))

class EstimateSlackLayer(nn.Module):
    def __init__(self, node_to_gen_mask, lineflow_mask):
        super().__init__()

        self.node_to_gen_mask = node_to_gen_mask    # [N, G]
        self.lineflow_mask = lineflow_mask          # [N, L]

    def forward(self, p_gt, f_lt, D_nt):
        """Compute md_n,t

        Args:
            p_gt (_type_): Generator production, shape [B, G]
            f_lt (_type_): Line flow, shape [B, L]
            D_nt (_type_): Demand, shape [B, N]
        """
        combined_flow = torch.matmul(p_gt, self.node_to_gen_mask.T) + \
                        torch.matmul(f_lt, self.lineflow_mask.T)
        
        #! If there are no lineflows, this is the same:
        # combined_flow = p_gt.sum(dim=1, keepdim=True)
        md_nt = D_nt - combined_flow

        # md_nt = D_nt - p_gt

        return md_nt

class PrimalNetEndToEnd(nn.Module):
        def __init__(self, args, data):
            super().__init__()
            self.data = data
            self.hidden_sizes = [int(args["hidden_size_factor"]*data.xdim)] * args["n_layers"]
            self.args = args

            if self.args["device"] == "mps":
                self.DTYPE = torch.float32
                self.DEVICE = torch.device("mps")
            else:
                self.DTYPE = torch.float64
                self.DEVICE = torch.device("cpu")

            torch.set_default_dtype(self.DTYPE)
            torch.set_default_device(self.DEVICE)

            # TODO: Implement compact benders form.
            # if self.data.args["benders_compact"]:
                # self.out_dim = data.num_g + data.n_prod_vars + data.n_line_vars
            # else:
            self.out_dim = data.n_prod_vars + data.n_line_vars

            self.feed_forward = FeedForwardNet(args, data.xdim, self.hidden_sizes, output_dim=self.out_dim).to(self.DTYPE).to(self.DEVICE)
            

            # ! Test with init zeros.
            # nn.init.zeros_(self.feed_forward.net[-1].weight)  # Initialize output layer weights to 0
            # nn.init.zeros_(self.feed_forward.net[-1].bias)    # Initialize output layer biases to 0

            self.bound_repair_layer = BoundRepairLayer()
            # self.ramping_repair_layer = RampingRepairLayer()

            self.estimate_slack_layer = EstimateSlackLayer(data.node_to_gen_mask.to(self.DEVICE), data.lineflow_mask.to(self.DEVICE))
        
        def forward(self, x, total_demands=None):
            eq_rhs, ineq_rhs = self.data.split_X(x)
            
            x_out = self.feed_forward(x)
            # [B, G, T], [B, L, T]
            ui_g, p_gt, f_lt = self.data.split_dec_vars_from_Y_raw(x_out)

            # [B, bounds, T]
            p_gt_lb, p_gt_ub, f_lt_lb, f_lt_ub, md_nt_lb, md_nt_ub = self.data.split_ineq_constraints(ineq_rhs)

            if self.args["repair_bounds"]:
                p_gt = self.bound_repair_layer(p_gt, p_gt_lb, p_gt_ub)
                #! Easy repair if we only have a single node and a single generator.
                # p_gt_bound_repaired = self.bound_repair_layer(p_gt, p_gt_lb, torch.min(p_gt_ub, eq_rhs))

                # Lineflow lower bound is negative.
                f_lt = self.bound_repair_layer(f_lt, -f_lt_lb, f_lt_ub)


            UI_g, D_nt = self.data.split_eq_constraints(eq_rhs)
            md_nt = self.estimate_slack_layer(p_gt, f_lt, D_nt)

            y = torch.cat([p_gt, f_lt, md_nt], dim=1)
            # y = torch.cat([md_nt, f_lt, p_gt], dim=1)

            if not self.training and (total_demands != None):
                return y * total_demands
            else:
                return y

class PrimalNetEndToEnd2(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.data = data
        self.hidden_sizes = [int(args["hidden_size_factor"]*data.xdim)] * args["n_layers"]
        self.args = args

        if not self.args["repair"]:
            assert not self.args["repair_bounds"], "If repair is disabled, bounds repair should not be enabled."
            assert not self.args["repair_completion"], "If repair is disabled, completion repair should not be enabled."
            assert not self.args["repair_power_balance"], "If repair is disabled, power balance repair should not be enabled."

        if self.args["repair_power_balance"]:
            # Power balance repair requires bounds repair.
            assert self.args["repair_bounds"], "Power balance repair requires bounds repair."
        
        if self.args["repair_bounds"]:
            # Bound repairs need layernorm to prevent gradient saturation in sigmoids.
            assert self.args["layernorm"], "Bounds repair requires layernorm."

        if self.args["device"] == "mps":
                self.DTYPE = torch.float32
                self.DEVICE = torch.device("mps")
        else:
            self.DTYPE = torch.float64
            self.DEVICE = torch.device("cpu")

        torch.set_default_dtype(self.DTYPE)
        torch.set_default_device(self.DEVICE)

        # TODO: Implement compact benders form.
        # if self.data.args["benders_compact"]:
            # self.out_dim = data.num_g + data.n_prod_vars + data.n_line_vars
        # else:
        if args["repair_completion"]:
            self.out_dim = data.n_prod_vars + data.n_line_vars
        else:
            self.out_dim = data.n_prod_vars + data.n_line_vars + data.n_md_vars

        self.feed_forward = FeedForwardNet(args, data.xdim, self.hidden_sizes, output_dim=self.out_dim).to(self.DTYPE).to(self.DEVICE)
        

        # ! Test with init zeros.
        # nn.init.zeros_(self.feed_forward.net[-1].weight)  # Initialize output layer weights to 0
        # nn.init.zeros_(self.feed_forward.net[-1].bias)    # Initialize output layer biases to 0
        
        if self.args["repair_bounds"]:
            self.bound_repair_layer = BoundRepairLayer()
            # self.ramping_repair_layer = RampingRepairLayer()

        if self.args["repair_completion"]:
            self.estimate_slack_layer = EstimateSlackLayer(data.node_to_gen_mask.to(self.DTYPE).to(self.DEVICE), data.lineflow_mask.to(self.DTYPE).to(self.DEVICE))
    
    def forward(self, x, total_demands=None):
        eq_rhs, ineq_rhs = self.data.split_X(x)
        x_out = self.feed_forward(x)
        if not self.args["repair"]:
            return x_out
        # [B, G, T], [B, L, T]
        if self.args["repair_completion"]:
            ui_g, p_gt, f_lt = self.data.split_dec_vars_from_Y_raw(x_out)
        else:
            p_gt, f_lt, md_nt = self.data.split_dec_vars_from_Y(x_out)

        # [B, bounds, T]
        p_gt_lb, p_gt_ub, f_lt_lb, f_lt_ub, md_nt_lb, md_nt_ub = self.data.split_ineq_constraints(ineq_rhs)
        
        if self.args["repair_bounds"]:
            p_gt = self.bound_repair_layer(p_gt, p_gt_lb, p_gt_ub)
                # Lineflow lower bound is negative.
                #! Note: Lineflows cannot be repaired more, since they depend on other lineflows. 
                #! For example, if we repair a lineflow such that it cannot export more than (imports + generation),
                #! then it will affect other lineflows, and we would need to repair them again.
            f_lt = self.bound_repair_layer(f_lt, -f_lt_lb, f_lt_ub) #! Bounds need to be repaired for this to work!

        if self.args["repair_power_balance"]:
            net_flow = torch.matmul(f_lt, self.data.lineflow_mask.T)  # [B, N]
            updated_demand = eq_rhs - net_flow                                    # [B, N]
            # Clamp demand between lower and upper bound of total capacity, since generators must produce between that.
            total_capacity_lb = torch.matmul(p_gt_lb, self.data.node_to_gen_mask.T)
            total_capacity_ub = torch.matmul(p_gt_ub, self.data.node_to_gen_mask.T)
            demand_clamped = torch.clamp(updated_demand, min=total_capacity_lb, max=total_capacity_ub)

            # Repair generation.
            total_generation = torch.matmul(p_gt, self.data.node_to_gen_mask.T)
            # Calculate zeta_up and zeta_down using vectorized operations
            mask_up = (total_generation < demand_clamped).to(self.DTYPE)
            # Calculate zeta_up and zeta_down based on conditions
            zeta_up = (demand_clamped - total_generation) / ((total_capacity_ub - total_generation) + 1e-12)
            zeta_down = (total_generation - demand_clamped) / ((total_generation - total_capacity_lb) + 1e-12)

            # Expand masks to generator dimension.
            mask_up = torch.matmul(mask_up, self.data.node_to_gen_mask)
            zeta_up = torch.matmul(zeta_up, self.data.node_to_gen_mask)
            zeta_down = torch.matmul(zeta_down, self.data.node_to_gen_mask)
            # Apply the updates to p_gt based on the condition
            p_gt_repaired = torch.where(mask_up.bool(), (1 - zeta_up) * p_gt + zeta_up * p_gt_ub, (1 - zeta_down) * p_gt + zeta_down * p_gt_lb)
            # p_gt = torch.where(mask_down, (1 - zeta_down) * p_gt + zeta_down * p_gt_lb, p_gt)

            p_gt = p_gt_repaired

        if self.args["repair_completion"]:
            UI_g, D_nt = self.data.split_eq_constraints(eq_rhs)
            md_nt = self.estimate_slack_layer(p_gt, f_lt, D_nt)

        y = torch.cat([p_gt, f_lt, md_nt], dim=1)
        # y = torch.cat([md_nt, f_lt, p_gt], dim=1)

        # Only scale if we are not training.
        if not self.training and (total_demands != None):
            return y * total_demands
        else:
            return y
        
        
class ImplicitLayer(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.data = data
        self.lineflow_mask = data.lineflow_mask
        self.node_to_gen_mask = data.node_to_gen_mask
        self.args = args

        self.gen_cost_vec = data.obj_coeff[:data.num_g]
        self.penalty_unmet_demand = data.pVOLL

        if self.args["device"] == "mps":
            self.DTYPE = torch.float32
            self.DEVICE = torch.device("mps")
        else:
            self.DTYPE = torch.float64
            self.DEVICE = torch.device("cpu")
    
    def greedy_differentiable_allocation(self, max_gen, demand):
        #! Assumes that the generators are sorted increasing by cost within node.
        
        B, G = max_gen.shape
        N = demand.shape[1]
        device = demand.device

        # [B, N, G]: repeat demand per generator
        demand_per_gen = torch.einsum("bn,ng->bng", demand, self.node_to_gen_mask)

        max_gen_per_node = self.node_to_gen_mask.unsqueeze(0) * max_gen.unsqueeze(1)  # [B, N, G]

        cum_cap = torch.cumsum(max_gen_per_node, dim=2)                     # [B, N, G]
        prev_cum = torch.cat([torch.zeros(B, N, 1, device=device), cum_cap[:, :, :-1]], dim=2)

        alloc = torch.clamp(demand_per_gen - prev_cum, min=0.0)
        alloc = torch.minimum(alloc, max_gen_per_node)

        # Collapse node dimension → [B, G] (summing contributions from all N to each generator)
        p_gt = alloc.sum(dim=1)

        return p_gt

    def forward(self, f_lt_bound_repaired, D_nt, p_gt_ub):
        net_flow = torch.matmul(f_lt_bound_repaired, self.lineflow_mask.T)  # [B, N]
        updated_demand = D_nt - net_flow                                    # [B, N]

        p_gt = self.greedy_differentiable_allocation(p_gt_ub, updated_demand)  # [B, G]

        # Project back generator output to nodes: [B, N] = [B, G] @ [G, N]
        md_nt = updated_demand - torch.matmul(p_gt, self.node_to_gen_mask.T)

        return p_gt, md_nt

class PrimalNetImplicit(nn.Module):
    """Implementation of a primal net with an implicit layer. The primal net only predicts the lineflows. The optimal generator production is then derived from the lineflows using an implicit layer.

    Args:
        nn (_type_): _description_
    """
    def __init__(self, args, data):
        super().__init__()
        self.data = data
        self.hidden_sizes = [int(args["hidden_size_factor"]*data.xdim)] * args["n_layers"]
        self.args = args

        if self.args["device"] == "mps":
            self.DTYPE = torch.float32
            self.DEVICE = torch.device("mps")
        else:
            self.DTYPE = torch.float64
            self.DEVICE = torch.device("cpu")

        #! Out dim is only the lineflows
        self.out_dim = data.n_line_vars

        self.feed_forward = FeedForwardNet(args, data.xdim, self.hidden_sizes, output_dim=self.out_dim).to(self.DTYPE).to(self.DEVICE)

        self.bound_repair_layer = BoundRepairLayer()
        # self.ramping_repair_layer = RampingRepairLayer()
        self.implicit_layer = ImplicitLayer(self.args, self.data)

        self.estimate_slack_layer = EstimateSlackLayer(data.node_to_gen_mask.to(self.DEVICE), data.lineflow_mask.to(self.DEVICE))

    def forward(self, x):
        eq_rhs, ineq_rhs = self.data.split_X(x)
        
        # Output from the feed forward net is the lineflows.
        f_lt = self.feed_forward(x)

        p_gt_lb, p_gt_ub, f_lt_lb, f_lt_ub, md_nt_lb, md_nt_ub = self.data.split_ineq_constraints(ineq_rhs)

        # p_gt_bound_repaired = self.bound_repair_layer(p_gt, p_gt_lb, p_gt_ub)
        #! Easy repair if we only have a single node and a single generator.
        # p_gt_bound_repaired = self.bound_repair_layer(p_gt, p_gt_lb, torch.min(p_gt_ub, eq_rhs))

        # Repair the lineflows. Note that we need to negate the lower bound, since it is positive in the RHS, but should be negative.
        f_lt_bound_repaired = self.bound_repair_layer(f_lt, -f_lt_lb, f_lt_ub)


        p_gt, md_nt = self.implicit_layer(f_lt_bound_repaired, eq_rhs, p_gt_ub)

        # eq_rhs only contains the demand.
        # md_nt = self.estimate_slack_layer(p_gt, f_lt_bound_repaired, eq_rhs)

        y = torch.cat([p_gt, f_lt_bound_repaired, md_nt], dim=1)

        return y
         
         
def load(args, data, save_dir):
    primal_net = PrimalNetEndToEnd(args, data=data)
    primal_net.load_state_dict(torch.load(save_dir + '/primal_weights.pth', weights_only=True))
    dual_net = DualNetEndToEnd(args, data=data)
    dual_net.load_state_dict(torch.load(save_dir + '/dual_weights.pth', weights_only=True))

    return primal_net, dual_net