import copy
import os
import pickle
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
from mtadam import MTAdam
from logger import TensorBoardLogger
torch.autograd.set_detect_anomaly(True)
# torch.manual_seed(42)

# torch.set_num_threads(4)
# torch.set_num_interop_threads(4)

class IndexedDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __getitem__(self, index):
        # Return the data sample and its original index
        return self.X[index], index

    def __len__(self):
        return len(self.X)

        
class PrimalDualTrainer():

    def __init__(self, data, args, save_dir, problem_type="GEP", log=True):
        """_summary_

        Args:
            data (_type_): _description_
            args (_type_): _description_
            save_dir (_type_): _description_
            problem_type (str, optional): Either "GEP" or "Benchmark". Defaults to "GEP".
            optimal_objective_train (_type_, optional): _description_. Defaults to None.
            optimal_objective_val (_type_, optional): _description_. Defaults to None.
            log (bool, optional): _description_. Defaults to True.
        """
        assert problem_type in ["GEP", "Benchmark"]

        print(f"X dim: {data.xdim}")
        print(f"Y dim: {data.ydim}")

        print(f"Size of mu: {data.nineq}")
        print(f"Size of lambda: {data.neq}")

        self.data = data
        self.args = args
        self.save_dir = save_dir
        self.problem_type = problem_type
        self.logger = TensorBoardLogger(args, data, save_dir, args["opt_targets"])
        self.log_frequency = args["log_frequency"]
        
        if self.args["device"] == "mps":
            self.DTYPE = torch.float32
            self.DEVICE = torch.device("mps")
        else:
            self.DTYPE = torch.float64
            self.DEVICE = torch.device("cpu")

        torch.set_default_dtype(self.DTYPE)
        torch.set_default_device(self.DEVICE)

        self.outer_iterations = args["outer_iterations"]
        self.inner_iterations = args["inner_iterations"]
        self.tau = args["tau"]
        self.rho = args["rho"]
        self.rho_max = args["rho_max"]
        self.alpha = args["alpha"]
        self.batch_size = args["batch_size"]
        self.hidden_sizes = args["hidden_sizes"]

        self.primal_lr = args["primal_lr"]
        self.dual_lr = args["dual_lr"]
        self.decay = args["decay"]
        self.patience = args["patience"]
        
        self.clip_gradients_norm = args["clip_gradients_norm"]

        # for logging
        self.step = 0

        train = data.train_indices

        self.X_train = data.X[train].to(self.DTYPE).to(self.DEVICE)
        self.eq_cm_train = data.eq_cm[train].to(self.DTYPE).to(self.DEVICE)
        self.ineq_cm_train = data.ineq_cm[train].to(self.DTYPE).to(self.DEVICE)
        self.eq_rhs_train = data.eq_rhs[train].to(self.DTYPE).to(self.DEVICE)
        self.ineq_rhs_train = data.ineq_rhs[train].to(self.DTYPE).to(self.DEVICE)


        self.train_dataset = IndexedDataset(self.X_train)
        # self.valid_dataset = CustomDataset(X[valid].to(self.DEVICE), eq_cm[valid], ineq_cm[valid], eq_rhs[valid], ineq_rhs[valid])
        # self.test_dataset = TensorDataset(self.data.testX.to(self.DEVICE), self.data.testX_scaled.to(self.DEVICE))

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        # self.valid_loader = DataLoader(self.valid_dataset, batch_size=len(self.valid_dataset))
        # self.test_loader = DataLoader(self.test_dataset, batch_size=len(self.test_dataset))

        # self.primal_net = PrimalNet(self.data, self.hidden_sizes).to(dtype=self.DTYPE, device=self.DEVICE)
        self.primal_net = PrimalNetEndToEnd(self.args, self.data).to(dtype=self.DTYPE, device=self.DEVICE)
        # self.dual_net = DualNet(self.data, self.hidden_sizes, self.data.nineq, self.data.neq).to(dtype=self.DTYPE, device=self.DEVICE)
        self.dual_net = DualNetEndToEnd(self.args, self.data).to(dtype=self.DTYPE, device=self.DEVICE)

        # Compile networks if PyTorch 2.0+ is available
        # if hasattr(torch, 'compile'):
        #     try:
        #         print("Compiling networks with torch.compile()")
        #         self.primal_net = torch.compile(self.primal_net)
        #         self.dual_net = torch.compile(self.dual_net)
        #     except Exception as e:
        #         print(f"Network compilation failed: {e}")

        self.primal_optim = torch.optim.Adam(self.primal_net.parameters(), lr=self.primal_lr)
        self.dual_optim = torch.optim.Adam(self.dual_net.parameters(), lr=self.dual_lr)

        # self.primal_optim = MTAdam(self.primal_net.parameters(), lr=self.primal_lr)
        # self.dual_optim = MTAdam(self.dual_net.parameters(), lr=self.dual_lr)

        # Add schedulers
        self.primal_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.primal_optim, mode='min', factor=self.decay, patience=self.patience
        )
        self.dual_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.dual_optim, mode='min', factor=self.decay, patience=self.patience
        )

        known_missed_demand = self.data.opt_targets["y_operational"][:, self.data.md_indices]
        assert known_missed_demand.sum() == 0, "Missed demand > 0 --> primal is infeasible, dual is unbounded"

    def freeze(self, network):
        """
        Create a frozen copy of a network
        """
        if isinstance(network, PrimalNetEndToEnd):
            frozen_net = PrimalNetEndToEnd(self.args, self.data).to(device=self.DEVICE, dtype=self.DTYPE)
        elif isinstance(network, DualNetEndToEnd):
            frozen_net = DualNetEndToEnd(self.args, self.data).to(device=self.DEVICE, dtype=self.DTYPE)
        elif isinstance(network, PrimalNet):
            frozen_net = PrimalNet(self.args, self.data, self.hidden_sizes).to(device=self.DEVICE, dtype=self.DTYPE)
        elif isinstance(network, DualNet):
            frozen_net = DualNet(self.args, self.data, self.hidden_sizes, self.data.nineq, self.data.neq).to(device=self.DEVICE, dtype=self.DTYPE)
        else:
            raise TypeError(f"Unsupported network type: {type(network)}")
        
        # Load a deep copy of the state dictionary
        frozen_net.load_state_dict(copy.deepcopy(network.state_dict()))
    
        # Set to evaluation mode
        frozen_net.eval()
        
        return frozen_net

    def train_PDL(self,):
        try:
            # with torch.profiler.profile(
            #     activities=[
            #         torch.profiler.ProfilerActivity.CPU,
            #         # torch.profiler.ProfilerActivity.MPS,
            #     ],
            #     schedule=torch.profiler.schedule(
            #         wait=1,      # Skip first iteration
            #         warmup=1,    # Warmup for one iteration
            #         active=3,    # Profile for three iterations
            #     ),
            #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./profile_logs'),
            #     record_shapes=True,
            #     profile_memory=True,
            #     with_stack=True,
            # ) as prof:
                prev_v_k = 0
                for k in range(self.outer_iterations):
                    begin_time = time.time()
                    epoch_stats = {}
                    frozen_dual_net = self.freeze(self.dual_net)
                    with torch.no_grad():
                        self.logger.log_rho_vk(self.rho, prev_v_k, self.step)

                    for l1 in range(self.inner_iterations):
                        self.step += 1
                        # Update primal net using primal loss
                        self.primal_net.train()

                        # Accumulate training loss over all batches
                        total_train_loss = 0.0
                        num_batches = 0
                        for Xtrain, sample_indices in self.train_loader:
                            self.primal_optim.zero_grad()

                            train_eq_cm, train_ineq_cm, train_eq_rhs, train_ineq_rhs = self.eq_cm_train[sample_indices], self.ineq_cm_train[sample_indices], self.eq_rhs_train[sample_indices], self.ineq_rhs_train[sample_indices]
                            
                            y = self.primal_net(Xtrain, train_eq_rhs, train_ineq_rhs)

                            with torch.no_grad():
                                if k == 0:
                                    mu, lamb = torch.zeros_like(train_ineq_rhs), torch.zeros_like(train_eq_rhs)
                                else:
                                    mu, lamb = frozen_dual_net(Xtrain, train_eq_cm)
                                    # mu, lamb = self.data.opt_targets["mu_operational"][sample_indices], self.data.opt_targets["lamb_operational"][sample_indices]
                            batch_loss, obj, lagrange_eq, penalty = self.primal_loss(y, train_eq_cm, train_ineq_cm, train_eq_rhs, train_ineq_rhs, mu.detach(), lamb.detach())
                            batch_loss, obj, lagrange_eq, penalty = batch_loss.mean(), obj.mean(), lagrange_eq.mean(), penalty.mean()
                            total_train_loss += batch_loss.item()
                            if isinstance(self.primal_optim, MTAdam):
                                self.primal_optim.step(loss_array=[obj, lagrange_eq, penalty], ranks=[1, 1, 2], feature_map=None)
                            else:
                                batch_loss.backward()
                                #! Test gradient clipping
                                if k > 2 and self.clip_gradients_norm > 0:
                                    torch.nn.utils.clip_grad_norm_(self.primal_net.parameters(), self.clip_gradients_norm)
                                self.primal_optim.step()

                            num_batches += 1
                        
                        # Compute average loss for the epoch
                        #! Scheduler on training set (should be validation set)
                        avg_train_loss = total_train_loss / num_batches
                        self.primal_scheduler.step(avg_train_loss)

                        # Log training loss:
                        if self.log_frequency > 0 and self.step % self.log_frequency == 0:
                            with torch.no_grad():
                                self.logger.log_loss(avg_train_loss, "primal", self.step)
                                self.logger.log_train(self.data, primal_net=self.primal_net, dual_net=frozen_dual_net, rho=self.rho, step=self.step)

                        # Evaluate validation loss every epoch, and update learning rate
                        # with torch.no_grad():
                        #     self.primal_net.eval()
                        #     frozen_dual_net.eval()

                        #     curr_val_loss = 0
                        #     for Xvalid, valid_eq_cm, valid_ineq_cm, valid_eq_rhs, valid_ineq_rhs in self.valid_loader:
                        #         y = self.primal_net(Xvalid)
                        #         mu, lamb = frozen_dual_net(Xvalid)
                        #         loss = self.primal_loss(y, valid_eq_cm, valid_ineq_cm, valid_eq_rhs, valid_ineq_rhs, mu, lamb).mean()
                        #         curr_val_loss += loss
                        #     curr_val_loss /= len(self.valid_loader)
                        #     # Normalize by rho, so that the schedular still works correctly if rho is increased
                            # self.primal_scheduler.step(torch.sign(curr_val_loss) * (torch.abs(curr_val_loss) / self.rho))

                        # prof.step()

                    with torch.no_grad():
                        # Copy primal net into frozen primal net
                        frozen_primal_net = self.freeze(self.primal_net)

                        # Calculate v_k
                        y = frozen_primal_net(self.X_train, self.eq_rhs_train, self.ineq_rhs_train)
                        mu_k, lamb_k = frozen_dual_net(self.X_train, self.eq_cm_train)
                        v_k = self.violation(y, self.eq_cm_train, self.ineq_cm_train, self.eq_rhs_train, self.ineq_rhs_train, mu_k)

                    for l in range(self.inner_iterations):
                        self.step += 1
                        # Update dual net using dual loss
                        self.dual_net.train()
                        frozen_primal_net.train()
                        total_train_loss = 0.0
                        num_batches = 0
                        for Xtrain, sample_indices in self.train_loader:
                            train_eq_cm, train_ineq_cm, train_eq_rhs, train_ineq_rhs = self.eq_cm_train[sample_indices], self.ineq_cm_train[sample_indices], self.eq_rhs_train[sample_indices], self.ineq_rhs_train[sample_indices]
                            self.dual_optim.zero_grad()
                            mu, lamb = self.dual_net(Xtrain, train_eq_cm)
                            with torch.no_grad():
                                # mu_k, lamb_k = frozen_dual_net(Xtrain, train_eq_cm)
                                y = frozen_primal_net(Xtrain, train_eq_rhs, train_ineq_rhs)
                            # ! Test other loss!
                            # batch_loss = self.dual_loss(y, train_eq_cm, train_ineq_cm, train_eq_rhs, train_ineq_rhs, mu, lamb, mu_k, lamb_k).mean()
                            batch_loss = self.dual_loss_changed(train_eq_rhs, train_ineq_rhs, mu, lamb).mean()
                            batch_loss.backward()

                            #! Test gradient clipping
                            if k > 2 and self.clip_gradients_norm > 0:
                                torch.nn.utils.clip_grad_norm_(self.dual_net.parameters(), self.clip_gradients_norm)

                            self.dual_optim.step()
                            total_train_loss += batch_loss.item()
                            num_batches += 1
                        
                        #! Scheduler on training set (should be validation set)
                        avg_train_loss = total_train_loss / num_batches
                        self.dual_scheduler.step(avg_train_loss)
                        if self.log_frequency > 0 and self.step % self.log_frequency == 0:
                            with torch.no_grad():
                                # Logg training loss:
                                self.logger.log_loss(avg_train_loss, "dual", self.step)
                                self.logger.log_train(self.data, primal_net=frozen_primal_net, dual_net=self.dual_net, rho=self.rho, step=self.step)

                        # Evaluate validation loss every epoch, and update learning rate
                        # TODO! Does scheduler correctly decrease LR when rho is increased, if the training set is small?
                        # with torch.no_grad():
                        #     frozen_primal_net.eval()
                        #     self.dual_net.eval()
                        #     curr_val_loss = 0
                        #     for Xvalid, valid_eq_cm, valid_ineq_cm, valid_eq_rhs, valid_ineq_rhs in self.valid_loader:
                        #         y = frozen_primal_net(Xvalid)
                        #         mu_valid, lamb_valid = self.dual_net(Xvalid)
                        #         mu_k_valid, lamb_k_valid = frozen_dual_net(Xvalid)
                        #         curr_val_loss += self.dual_loss(y, valid_eq_cm, valid_ineq_cm, valid_eq_rhs, valid_ineq_rhs, mu_valid, lamb_valid, mu_k_valid, lamb_k_valid).mean()
                            
                        #     curr_val_loss /= len(self.valid_loader)
                        #     # Normalize by rho, so that the schedular still works correctly if rho is increased
                        #     self.dual_scheduler.step(torch.sign(curr_val_loss) * (torch.abs(curr_val_loss) / self.rho))
                    
                    with torch.no_grad():
                        self.logger.log_train(self.data, primal_net=frozen_primal_net, dual_net=self.dual_net, rho=self.rho, step=self.step)
                        self.logger.log_val(self.data, self.primal_net, self.dual_net, self.step)

                    end_time = time.time()
                    stats = epoch_stats
                    print(f"Epoch {k} done. Time taken: {end_time - begin_time}. Rho: {self.rho}. Primal LR: {self.primal_optim.param_groups[0]['lr']}, Dual LR: {self.dual_optim.param_groups[0]['lr']}")

                    # Update rho from the second iteration onward.
                    if k > 0 and v_k > self.tau * prev_v_k:
                        self.rho = np.min([self.alpha * self.rho, self.rho_max])

                    prev_v_k = v_k
                    # Step the profiler
                    
                # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        
        except Exception as e:
            print(e, flush=True)
            # Ensure writer is closed even if an exception occurs
            if self.logger:
                self.logger.close()
            raise

        with open(os.path.join(self.save_dir, 'stats.dict'), 'wb') as f:
            pickle.dump(stats, f)
        
        self.save(self.save_dir)

        return self.primal_net, self.dual_net, stats

    def primal_loss(self, y, eq_cm, ineq_cm, eq_rhs, ineq_rhs, mu, lamb):
        obj = self.data.obj_fn_train(y)
        
        # g(y)
        # ineq = self.data.ineq_resid(y, ineq_cm, ineq_rhs)
        # h(y)
        eq = self.data.eq_resid(y, eq_cm, eq_rhs)

        # ! Clamp mu?
        # Element-wise clamping of mu_i when g_i (ineq) is negative
        # mu = torch.where(ineq < 0, torch.zeros_like(mu), mu)
        # ! Clamp ineq_resid?
        # ineq = ineq.clamp(min=0)

        # lagrange_ineq = torch.sum(mu * ineq, dim=1)  # Shape (batch_size,)
        # ! Clamp the lagrangian inequality term --> we do not adhere to KKT (why not?)
        # lagrange_ineq = torch.sum(mu * ineq, dim=1).clamp(min=0)  # Shape (batch_size,)

        lagrange_eq = torch.sum(lamb * eq, dim=1)   # Shape (batch_size,)

        # violation_ineq = torch.sum(torch.maximum(ineq, torch.zeros_like(ineq)) ** 2, dim=1)
        violation_eq = torch.sum(eq ** 2, dim=1)
        # penalty = self.rho/2 * (violation_ineq + violation_eq)

        # ! Alternative penalty: missed demand ** 2
        # lagrange_eq = torch.sum(lamb * y[:, self.data.md_indices])
        # violation_eq = torch.sum(y[:, self.data.md_indices] ** 2, dim=1)
        # violation_eq = torch.sum(y[:, self.data.md_indices].abs(), dim=1)

        penalty = self.rho/2 * violation_eq

        # ! Primal loss might need to be scaled to work.
        # loss = (obj*1e3 + (lagrange_ineq + lagrange_eq + penalty))
        # loss = (obj + (lagrange_ineq + lagrange_eq + penalty))

        # loss = (obj + (lagrange_eq.clamp(min=0) + penalty))
        # loss = penalty
        loss = (obj + penalty)

        # loss = (obj*1e3 + penalty)


        # relaxation_threshold = 0.01  # Allow 1% imbalance
        # relaxed_penalty = torch.clamp(penalty - eq_rhs.abs().sum(dim=1) * relaxation_threshold, min=0)
        # loss = obj + relaxed_penalty


        #! Test with scaling the loss function
        # penalty = violation_eq
        # lagrange_eq = lagrange_eq.clamp(min=0)
        # loss = (obj/obj.detach() + lagrange_eq/lagrange_eq.detach() + self.rho/2*(penalty/penalty.detach()))
        # loss = (obj/obj.detach() + self.rho/2*(penalty/penalty.detach()))

        # ! Test with term regularizing the distance between previous solution.
        # reg = torch.norm(y - self.prev_solution)

        #! Test only optimizing objective.
        # loss = obj

        return loss, obj, lagrange_eq.clamp(min=0), penalty

    def dual_loss(self, y, eq_cm, ineq_cm, eq_rhs, ineq_rhs, mu, lamb, mu_k, lamb_k):
        # mu = [batch, g]
        # lamb = [batch, h]

        # g(y)
        ineq = self.data.ineq_resid(y, ineq_cm, ineq_rhs) # [batch, g]
        # h(y)
        eq = self.data.eq_resid(y, eq_cm, eq_rhs)   # [batch, h]

        #! From 2nd PDL paper, fix to 1e-1, not rho
        # target_mu = torch.maximum(mu_k + self.rho * ineq, torch.zeros_like(ineq))
        target_mu = torch.maximum(mu_k + 1e-1 * ineq, torch.zeros_like(ineq))

        dual_resid_ineq = mu - target_mu # [batch, g]

        dual_resid_ineq = torch.norm(dual_resid_ineq, dim=1)  # [batch]

        # Compute the dual residuals for equality constraints
        #! From 2nd PDL paper, fix to 1e-1, not rho
        # dual_resid_eq = lamb - (lamb_k + self.rho * eq)
        dual_resid_eq = lamb - (lamb_k + 1e-1 * eq)
        dual_resid_eq = torch.norm(dual_resid_eq, dim=1)  # Norm along constraint dimension

        loss = (dual_resid_ineq + dual_resid_eq)

        return loss
    
    def dual_loss_changed(self, eq_rhs, ineq_rhs, mu, lamb):
        #! We maximize the dual obj func, so to use it in the loss, take the negation.
        dual_obj = -self.data.dual_obj_fn(eq_rhs, ineq_rhs, mu, lamb)

        #! Dual solutions are always feasible because of the completion step, so we do not include penalty and lagrangian terms.
        return dual_obj

    def violation(self, y, eq_cm, ineq_cm, eq_rhs, ineq_rhs, mu_k):
        # Calculate the equality constraint function h_x(y)
        eq = self.data.eq_resid(y, eq_cm, eq_rhs)  # Assume shape (num_samples, n_eq)
        
        # Calculate the infinity norm of h_x(y)
        eq_inf_norm = torch.abs(eq).max(dim=1).values  # Shape: (num_samples,)

        # Calculate the inequality constraint function g_x(y)
        ineq = self.data.ineq_resid(y, ineq_cm, ineq_rhs)  # Assume shape (num_samples, n_ineq)
        
        # Calculate sigma_x(y) for each inequality constraint
        sigma_y = torch.maximum(ineq, -mu_k / self.rho)  # Element-wise max
        
        # Calculate the infinity norm of sigma_x(y)
        sigma_y_inf_norm = torch.abs(sigma_y).max(dim=1).values  # Shape: (num_samples,)

        # Compute v_k as the maximum of the two norms
        v_k = torch.maximum(eq_inf_norm, sigma_y_inf_norm)  # Shape: (num_samples,)
        
        return v_k.max().item()

    def save(self, save_dir):
        print("saving")
        torch.save(self.primal_net.state_dict(), save_dir + '/primal_weights.pth')
        torch.save(self.dual_net.state_dict(), save_dir + '/dual_weights.pth')


class PrimalNet(nn.Module):
    def __init__(self, data, hidden_sizes):
        super().__init__()
        self._data = data
        self._hidden_sizes = hidden_sizes
        
        # Create the list of layer sizes
        layer_sizes = [data.xdim] + self._hidden_sizes + [data.ydim]
        layers = []

        # Create layers dynamically based on the provided hidden_sizes
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            if out_size != data.ydim:  # Add ReLU activation for hidden layers only
                layers.append(nn.ReLU())

        # Initialize all layers
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

        self.net = nn.Sequential(*layers)
    
    def forward(self, x, eq_rhs, ineq_rhs):
        return self.net(x)


class DualNetEndToEnd(nn.Module):
    def __init__(self, args, data, hidden_size_factor=5.0, n_layers=4):
        super().__init__()
        self._data = data
        self._hidden_sizes = [int(hidden_size_factor*data.xdim)] * n_layers
        self.args = args

        if self.args["benders_compact"]:
            self._out_dim = data.num_g + data.neq
        else:
            self._out_dim = data.neq

        if args["device"] == "mps":
            self.DTYPE = torch.float32
            self.DEVICE = torch.device("mps")
        else:
            self.DTYPE = torch.float64
            self.DEVICE = torch.device("cpu")

        #! Only predict lambda, we infer mu from it.
        self.feed_forward = FeedForwardNet(data.xdim, self._hidden_sizes, output_dim=self._out_dim).to(self.DEVICE)

        # Set dual variables to 0 at the first iteration
        nn.init.zeros_(self.feed_forward.net[-1].weight)  # Initialize output layer weights to 0
        nn.init.zeros_(self.feed_forward.net[-1].bias)    # Initialize output layer biases to 0
    
    def complete_duals(self, lamb, eq_cm):
         # first num_g outputs are the equality constraints added in benders compact form
        if self._data.args["benders_compact"]:
            # lamb_ui_g = out_lamb[:, :self._data.num_g]
            lamb_D_nt = lamb[:, self._data.num_g:]
            eq_cm_D_nt = eq_cm[:, self._data.num_g:, self._data.num_g:]
            obj_coeff = self._data.obj_coeff[self._data.num_g:]
        else:
            eq_cm_D_nt = eq_cm
            lamb_D_nt = lamb
            obj_coeff = self._data.obj_coeff

        eq_cm = eq_cm[0]

        # mu = obj_coeff - torch.bmm(eq_cm_D_nt.transpose(1, 2), lamb_D_nt.unsqueeze(-1)).squeeze(-1)
        mu = obj_coeff - lamb @ eq_cm

        # mu = mu.view(mu.shape[0], self._data.sample_duration, -1)  # Shape: (batch_size, t, constraints)

        # Compute lower and upper bound multipliers
        mu_lb = torch.relu(mu)   # Lower bound multipliers |mu|^+
        mu_ub = torch.relu(-mu)  # Upper bound multipliers |mu|^-
        
        # Split into groups, following the exact structure of mu
        p_g_lb = mu_lb[:, :self._data.num_g]  # Lower bounds for p_g
        p_g_ub = mu_ub[:, :self._data.num_g]  # Upper bounds for p_g

        # f_l_lb = mu_lb[:, :, self._data.num_g:self._data.num_g + self._data.num_l]  # Lower bounds for f_l
        # f_l_ub = mu_ub[:, :, self._data.num_g:self._data.num_g + self._data.num_l]  # Upper bounds for f_l
        f_l_lb = mu_lb[:, self._data.num_g:]  # Lower bounds for f_l
        f_l_ub = mu_ub[:, self._data.num_g:]  # Upper bounds for f_l

        # md_n_lb = mu_lb[:, :, self._data.num_g + self._data.num_l:]  # Lower bounds for md_n
        # md_n_ub = mu_ub[:, :, self._data.num_g + self._data.num_l:]  # Upper bounds for md_n

        # Reshape back into (batch_size, constraints * t) while maintaining order
        out_mu = torch.cat([
            p_g_lb, p_g_ub,  # Lower and Upper bounds for p_g
            f_l_lb, f_l_ub,  # Lower and Upper bounds for f_l
            # md_n_lb, md_n_ub  # Lower and Upper bounds for md_n
        ], dim=-1).reshape(lamb.shape[0], -1)  # Flatten back to (batch_size, constraints * t)
        return out_mu
        
    def forward(self, x, eq_cm):
        out_lamb = self.feed_forward(x)
        out_mu = self.complete_duals(out_lamb, eq_cm)
        # print(out_mu)
        # print(out_lamb)
        return out_mu, out_lamb

class DualNet(nn.Module):
    def __init__(self, data, hidden_sizes, mu_size, lamb_size):
        super().__init__()
        self._data = data
        self._hidden_sizes = hidden_sizes
        self._mu_size = mu_size
        self._lamb_size = lamb_size

        # Create the list of layer sizes
        layer_sizes = [data.xdim] + self._hidden_sizes
        # layer_sizes = [2*data.xdim + 1000] + self._hidden_sizes
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
        self.out_layer = nn.Linear(self._hidden_sizes[-1], self._mu_size + self._lamb_size)
        nn.init.zeros_(self.out_layer.weight)  # Initialize output layer weights to 0
        nn.init.zeros_(self.out_layer.bias)    # Initialize output layer biases to 0
        layers.append(self.out_layer)

        self.net = nn.Sequential(*layers)
    
    def forward(self, x, *args):
        out = self.net(x)
        #! ReLU to enforce nonnegativity in mu. Test with it.
        #! Does this work with zero initialization?
        # out_mu = torch.relu(out[:, :self._mu_size])
        out_mu = out[:, :self._mu_size]
        out_lamb = out[:, self._mu_size:]
        return out_mu, out_lamb


class DualNetTwoOutputLayers(nn.Module):
    def __init__(self, data, hidden_size):
        super().__init__()
        self._data = data
        self._hidden_size = hidden_size
        layer_sizes = [data.xdim, self._hidden_size, self._hidden_size]
        layers = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)
        self.out_layer_mu = nn.Linear(self._hidden_size, data.nineq)
        self.out_layer_lamb = nn.Linear(self._hidden_size, data.neq)
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
    def __init__(self, input_dim, hidden_sizes, output_dim):
        """_summary_

        Args:
            input_dim (int): Number of input features
            output_dim (int): Number of output features
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.output_dim = output_dim

        # Create the list of layer sizes
        layer_sizes = [self.input_dim] + self.hidden_sizes + [self.output_dim]
        layers = []
        layers.append(nn.LayerNorm(input_dim))
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
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        x_out = self.net(x)
        return x_out


class BoundRepairLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, lb, ub, k=2.0):
        """_summary_

        Args:
            x (_type_): Decision variables, shape [B, N, T]
            lb (_type_): Lower bounds of decision variables, shape [B, N, T]
            ub (_type_): Upper bounds of decision variables, shape [B, N, T]

        Returns:
            _type_: _description_
        """

        return lb + (ub - lb) * torch.sigmoid(k*x)
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
            p_gt (_type_): Generator production, shape [B, G, T]
            f_lt (_type_): Line flow, shape [B, L, T]
            D_nt (_type_): Demand, shape [B, N, T]
        """

        # Map generator production to nodes
        # p_nt = torch.einsum('ng,bgt->bnt', self.node_to_gen_mask, p_gt)

        # In lineflow mask (adjacency matrix) starting nodes are -1, receiving nodes are 1.
        # Thus, this mask is f_in_nt - f_out_nt (net_flow)
        # net_flow_nt = torch.einsum('nl,blt->bnt', self.lineflow_mask, f_lt)
        # Compute md_n,t
        # md_nt = D_nt - (p_nt + net_flow_nt)  # [B, N, T]

        combined_flow = torch.einsum('ng,bgt->bnt', self.node_to_gen_mask, p_gt) + \
                        torch.einsum('nl,blt->bnt', self.lineflow_mask, f_lt)
        
        # combined_flow = p_gt.sum(dim=1, keepdim=True)
        md_nt = D_nt - combined_flow

        return md_nt

class PrimalNetEndToEnd(nn.Module):
        def __init__(self, args, data, hidden_size_factor=5.0, n_layers=4):
            super().__init__()
            self._data = data
            self._hidden_sizes = [int(hidden_size_factor*data.xdim)] * n_layers
            self.args = args

            if self.args["device"] == "mps":
                self.DTYPE = torch.float32
                self.DEVICE = torch.device("mps")
            else:
                self.DTYPE = torch.float64
                self.DEVICE = torch.device("cpu")

            if self._data.args["benders_compact"]:
                self._out_dim = data.num_g + data.n_prod_vars + data.n_line_vars
            else:
                self._out_dim = data.n_prod_vars + data.n_line_vars

            self.feed_forward = FeedForwardNet(data.xdim, self._hidden_sizes, output_dim=self._out_dim).to(self.DEVICE)

            # ! Test with init zeros.
            # nn.init.zeros_(self.feed_forward.net[-1].weight)  # Initialize output layer weights to 0
            # nn.init.zeros_(self.feed_forward.net[-1].bias)    # Initialize output layer biases to 0

            self.bound_repair_layer = BoundRepairLayer()
            # self.ramping_repair_layer = RampingRepairLayer()

            self.estimate_slack_layer = EstimateSlackLayer(data.node_to_gen_mask.to(self.DEVICE), data.lineflow_mask.to(self.DEVICE))
        
        def forward(self, x, eq_rhs, ineq_rhs):
            x_out = self.feed_forward(x)

            # [B, G, T], [B, L, T]
            ui_g, p_gt, f_lt = self._data.split_dec_vars_from_Y_raw(x_out)
            
            # [B, bounds, T]
            # p_gt_lb, p_gt_ub, f_lt_lb, f_lt_ub, md_nt_lb, md_nt_ub = self._data.split_ineq_constraints(ineq_rhs)
            p_gt_lb, p_gt_ub, f_lt_lb, f_lt_ub = self._data.split_ineq_constraints(ineq_rhs)

            p_gt_bound_repaired = self.bound_repair_layer(p_gt, p_gt_lb, p_gt_ub)
            # print(p_gt)
            # print(p_gt_bound_repaired)

            # Lineflow lower bound is negative.
            f_lt_bound_repaired = self.bound_repair_layer(f_lt, -f_lt_lb, f_lt_ub)


            #! Test without slack.
            # UI_g, D_nt = self._data.split_eq_constraints(eq_rhs)
            # md_nt = self.estimate_slack_layer(p_gt_bound_repaired, f_lt_bound_repaired, D_nt)

            # y = torch.cat([p_gt_bound_repaired, f_lt_bound_repaired, md_nt], dim=1).permute(0, 2, 1).reshape(x_out.shape[0], -1)

            y = torch.cat([p_gt_bound_repaired, f_lt_bound_repaired], dim=1).permute(0, 2, 1).reshape(x_out.shape[0], -1)

            if self._data.args["benders_compact"]:
                y = torch.cat([ui_g, y], dim=1)

            if self._data.args["scale_input"]:
                y = self._data.scaler.inverse_transform(y)
            return y
         
         
def load(args, data, save_dir):
    primal_net = PrimalNetEndToEnd(args, data=data)
    primal_net.load_state_dict(torch.load(save_dir + '/primal_weights.pth', weights_only=True))
    dual_net = DualNetEndToEnd(args, data=data)
    dual_net.load_state_dict(torch.load(save_dir + '/dual_weights.pth', weights_only=True))

    return primal_net, dual_net