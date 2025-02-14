import copy
import os
import pickle
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
# import torch_geometric

DTYPE = torch.float64
DEVICE = torch.device="cpu"
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(42)
print(f"Running on {DEVICE}")

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x, eq_cm, ineq_cm, eq_rhs, ineq_rhs):
        self.x = x
        self.eq_cm = eq_cm 
        self.ineq_cm = ineq_cm
        self.eq_rhs = eq_rhs
        self.ineq_rhs = ineq_rhs
        self._index = 0  # Internal index for tracking iteration

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Return a tuple of input and target for the given index
        return self.x[idx], self.eq_cm[idx], self.ineq_cm[idx], self.eq_rhs[idx], self.ineq_rhs[idx]

    def __iter__(self):
        # Reset the internal index and return the dataset itself as an iterator
        self._index = 0
        return self

    def __next__(self):
        # Check if the index is within bounds
        if self._index < len(self):
            result = self[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration  # Raise StopIteration when iteration is done

class TensorBoardLogger():
    def __init__(self, save_dir):
         self.writer = SummaryWriter(log_dir=save_dir)
    
    def close(self):
        self.writer.close()

    def log_loss(self, loss, network, step):

        self.writer.add_scalar(f"Train_loss/{network}_loss", loss, step)

    def log_train(self, data, primal_net, dual_net, step):
        with torch.no_grad():
            Y = primal_net(data.X[data.train_indices], data.eq_rhs[data.train_indices], data.ineq_rhs[data.train_indices])
            mu, lamb = dual_net(data.X[data.train_indices], data.eq_cm[data.train_indices])
            obj = data.obj_fn(Y)
            dual_obj = data.dual_obj_fn(data.eq_rhs[data.train_indices], data.ineq_rhs[data.train_indices], mu, lamb)

            Y_target = data.opt_targets["y_operational"][data.train_indices]
            mu_target = data.opt_targets["mu_operational"][data.train_indices]
            lamb_target = data.opt_targets["lamb_operational"][data.train_indices]

            # print(Y[0, 5*14:5*15].tolist())

            ineq_resid = data.ineq_resid(Y, data.ineq_cm[data.train_indices], data.ineq_rhs[data.train_indices])
            ineq_dist = data.ineq_dist(Y, data.ineq_cm[data.train_indices], data.ineq_rhs[data.train_indices])

            eq_resid = data.eq_resid(Y, data.eq_cm[data.train_indices], data.eq_rhs[data.train_indices])

            obj_target = data.obj_fn(Y_target)
            dual_obj_target = data.dual_obj_fn(data.eq_rhs[data.train_indices], data.ineq_rhs[data.train_indices], mu_target, lamb_target)

            # Obj funcs
            self.writer.add_scalar(f"Train_obj/obj", obj.mean(), step)
            self.writer.add_scalar(f"Train_obj/dual_obj", dual_obj.mean(), step)
            self.writer.add_scalar(f"Train_obj/obj_optimality_gap", ((obj - obj_target)/obj_target).mean(), step)
            self.writer.add_scalar(f"Train_obj/dual_obj_optimality_gap", ((dual_obj - dual_obj_target)/dual_obj_target).mean(), step)

            # Neural network outputs and targets
            Y_diff = (Y - Y_target).abs()
            mu_diff = (mu - mu_target).abs()
            lamb_diff = (lamb - lamb_target).abs()
            self.writer.add_scalar(f"Train_outputs/Y", Y.mean(), step)
            self.writer.add_scalar(f"Train_outputs/mu", mu.mean(), step)
            self.writer.add_scalar(f"Train_outputs/lamb", lamb.mean(), step)
            self.writer.add_scalar(f"Train_outputs/Y_diff", Y_diff.mean(), step)
            self.writer.add_scalar(f"Train_outputs/mu_diff", mu_diff.mean(), step)
            self.writer.add_scalar(f"Train_outputs/lamb_diff", lamb_diff.mean(), step)

            # Constraint violations
            self.writer.add_scalar(f"Train_constraints/eq_resid", eq_resid.mean(), step)
            self.writer.add_scalar(f"Train_constraints/ineq_resid", ineq_resid.mean(), step)
            self.writer.add_scalar(f"Train_constraints/ineq_mean", ineq_dist.mean(), step)
            self.writer.add_scalar(f"Train_constraints/ineq_max", ineq_dist.max(), step)
            self.writer.add_scalar(f"Train_constraints/eq_mean", eq_resid.abs().mean(), step)
            self.writer.add_scalar(f"Train_constraints/eq_max", eq_resid.abs().max(), step)

            # Primal variable specific differences
            p_gt, f_lt, md_nt = data.split_dec_vars_from_Y(Y)
            p_gt_target, f_lt_target, md_nt_target = data.split_dec_vars_from_Y(Y_target)
            diff_p_gt = p_gt - p_gt_target
            diff_f_lt = f_lt - f_lt_target
            diff_md_nt = md_nt - md_nt_target

            # diff_ui_g = (Y[:, data.ui_g_indices] - Y_target[:, data.ui_g_indices])
            self.writer.add_scalar(f"Train_var_diffs/diff_p_gt", diff_p_gt.mean(), step)
            self.writer.add_scalar(f"Train_var_diffs/diff_f_lt", diff_f_lt.mean(), step)
            self.writer.add_scalar(f"Train_var_diffs/diff_md_nt", diff_md_nt.mean(), step)
            # self.writer.add_scalar(f"Train_var_diffs/diff_ui_g", diff_ui_g.mean(), step)

            h, b, d, e, i, j = data.split_ineq_constraints(ineq_dist)
            c = data.split_eq_constraints(eq_resid)

            self.writer.add_scalar(f"Train_constraint_specific/p_gt_ub", b.mean(), step)
            self.writer.add_scalar(f"Train_constraint_specific/node_balance", c.mean(), step)
            self.writer.add_scalar(f"Train_constraint_specific/f_lt_lb", d.mean(), step)
            self.writer.add_scalar(f"Train_constraint_specific/f_lt_ub", e.mean(), step)
            # self.writer.add_scalar(f"Train_constraint_specific/f", f.mean(), step)
            # self.writer.add_scalar(f"Train_constraint_specific/g", g.mean(), step)
            self.writer.add_scalar(f"Train_constraint_specific/p_gt_lb", h.mean(), step)
            self.writer.add_scalar(f"Train_constraint_specific/md_nt_lb", i.mean(), step)
            self.writer.add_scalar(f"Train_constraint_specific/md_nt_ub", j.mean(), step)
            # self.writer.add_scalar(f"Train_constraint_specific/k", k.mean(), step)

            # Dual variable specific differences
            # inequality
            mu_h, mu_b, mu_d, mu_e, mu_i, mu_j = data.split_ineq_constraints(mu)
            mu_target_h, mu_target_b, mu_target_d, mu_target_e, mu_target_i, mu_target_j = data.split_ineq_constraints(mu_target)
            mu_h_diff = mu_target_h - mu_h
            mu_b_diff = mu_target_b - mu_b
            mu_d_diff = mu_target_d - mu_d
            mu_e_diff = mu_target_e - mu_e
            mu_i_diff = mu_target_i - mu_i
            mu_j_diff = mu_target_j - mu_j

            # # equality
            lamb_c = data.split_eq_constraints(lamb)
            lamb_target_c = data.split_eq_constraints(lamb_target)
            lamb_c_diff = lamb_target_c - lamb_c

            self.writer.add_scalar(f"Train_dual_var_diffs/gen_ub", mu_b_diff.mean(), step)
            self.writer.add_scalar(f"Train_dual_var_diffs/node_balance", lamb_c_diff.mean(), step)
            self.writer.add_scalar(f"Train_dual_var_diffs/lineflow_lb", mu_d_diff.mean(), step)
            self.writer.add_scalar(f"Train_dual_var_diffs/lineflow_ub", mu_e_diff.mean(), step)
            # self.writer.add_scalar(f"Train_dual_var_diffs/mu_f", mu_f_diff.mean(), step)
            # self.writer.add_scalar(f"Train_dual_var_diffs/mu_g", mu_g_diff.mean(), step)
            self.writer.add_scalar(f"Train_dual_var_diffs/gen_lb", mu_h_diff.mean(), step)
            self.writer.add_scalar(f"Train_dual_var_diffs/md_lb", mu_i_diff.mean(), step)
            self.writer.add_scalar(f"Train_dual_var_diffs/md_ub", mu_j_diff.mean(), step)
            # self.writer.add_scalar(f"Train_dual_var_diffs/mu_k", mu_k_diff.mean(), step)

            # Dual constraints
            dual_eq_resid = data.dual_eq_resid(mu, lamb, data.eq_cm[data.train_indices], data.ineq_cm[data.train_indices])
            dual_ineq_resid = data.dual_ineq_resid(mu, lamb)
            dual_ineq_dist = torch.clamp(dual_ineq_resid, 0)
            self.writer.add_scalar("Dual_constraints/eq_resid", dual_eq_resid.abs().mean(), step)
            self.writer.add_scalar("Dual_constraints/ineq_mean", dual_ineq_dist.mean(), step)


            # Log gradients
            # Iterate over all layers and log their gradients
            for name, param in primal_net.named_parameters():
                if param.grad is not None:  # Skip parameters without gradients
                    self.writer.add_scalar(f"Gradients_primal/{name}", param.grad.norm().item(), step)
            # self.writer.add_scalar(f"Gradient/dual", dual_net[-1].weight.grad.mean())

    def log_rho_vk(self, rho, v_k, step):
        self.writer.add_scalar(f"Rho_and_violation/rho", rho, step)
        self.writer.add_scalar(f"Rho_and_violation/v_k", v_k, step)

    def log_val(self, data, primal_net, dual_net):
        pass
    
        
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
        if log:
            self.logger = TensorBoardLogger(save_dir)
        else:
            self.logger = None

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
        
        # for logging
        self.step = 0

        X = data.X
        eq_cm = data.eq_cm
        ineq_cm = data.ineq_cm
        eq_rhs = data.eq_rhs
        ineq_rhs = data.ineq_rhs

        train = data.train_indices
        valid = data.valid_indices
        test = data.test_indices

        self.train_dataset = CustomDataset(X[train].to(DEVICE), eq_cm[train], ineq_cm[train], eq_rhs[train], ineq_rhs[train])
        # self.valid_dataset = CustomDataset(X[valid].to(DEVICE), eq_cm[valid], ineq_cm[valid], eq_rhs[valid], ineq_rhs[valid])
        # self.test_dataset = TensorDataset(self.data.testX.to(DEVICE), self.data.testX_scaled.to(DEVICE))

        # self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)
        # self.valid_loader = DataLoader(self.valid_dataset, batch_size=len(self.valid_dataset))
        # self.test_loader = DataLoader(self.test_dataset, batch_size=len(self.test_dataset))

        # self.primal_net = PrimalNet(self.data, self.hidden_sizes).to(dtype=DTYPE, device=DEVICE)
        self.primal_net = PrimalNetEndToEnd(self.data).to(dtype=DTYPE, device=DEVICE)
        self.dual_net = DualNet(self.data, self.hidden_sizes, self.data.nineq, self.data.neq).to(dtype=DTYPE, device=DEVICE)
        # self.dual_net = DualNetEndToEnd(self.data).to(dtype=DTYPE, device=DEVICE)

        self.primal_optim = torch.optim.Adam(self.primal_net.parameters(), lr=self.primal_lr)
        self.dual_optim = torch.optim.Adam(self.dual_net.parameters(), lr=self.dual_lr)

        # Add schedulers
        self.primal_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.primal_optim, mode='min', factor=self.decay, patience=self.patience
        )
        self.dual_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.dual_optim, mode='min', factor=self.decay, patience=self.patience
        )

    def train_PDL(self,):
        try:
            prev_v_k = 0
            for k in range(self.outer_iterations):
                begin_time = time.time()
                epoch_stats = {}
                frozen_dual_net = copy.deepcopy(self.dual_net)
                self.logger.log_rho_vk(self.rho, prev_v_k, self.step)
                for l1 in range(self.inner_iterations):
                    self.step += 1
                    # Update primal net using primal loss
                    self.primal_net.train()

                    # Accumulate training loss over all batches
                    total_train_loss = 0.0
                    num_batches = 0
                    for Xtrain, train_eq_cm, train_ineq_cm, train_eq_rhs, train_ineq_rhs in self.train_loader:
                        self.primal_optim.zero_grad()
                        y = self.primal_net(Xtrain, train_eq_rhs, train_ineq_rhs)
                        with torch.no_grad():
                            mu, lamb = frozen_dual_net(Xtrain, train_eq_cm)
                        batch_loss = self.primal_loss(y, train_eq_cm, train_ineq_cm, train_eq_rhs, train_ineq_rhs, mu, lamb).mean()
                        total_train_loss += batch_loss.item()
                        batch_loss.backward()
                        self.primal_optim.step()
                        num_batches += 1
                    
                    # Compute average loss for the epoch
                    avg_train_loss = total_train_loss / num_batches
                    self.primal_scheduler.step(avg_train_loss)

                    # Step optimizer with gradients accumulated over the epoch

                    # Logg training loss:
                    with torch.no_grad():
                        self.logger.log_loss(batch_loss, "primal", self.step)
                        self.logger.log_train(self.data, primal_net=self.primal_net, dual_net=frozen_dual_net, step=self.step)

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
                
                with torch.no_grad():
                    # Copy primal net into frozen primal net
                    frozen_primal_net = copy.deepcopy(self.primal_net)

                    # Calculate v_k
                    y = frozen_primal_net(self.train_dataset.x, self.train_dataset.eq_rhs, self.train_dataset.ineq_rhs)
                    mu_k, lamb_k = frozen_dual_net(self.train_dataset.x, self.train_dataset.eq_cm)
                    v_k = self.violation(y, self.train_dataset.eq_cm, self.train_dataset.ineq_cm, self.train_dataset.eq_rhs, self.train_dataset.ineq_rhs, mu_k)

                self.logger.log_rho_vk(self.rho, prev_v_k, self.step)

                for l in range(self.inner_iterations):
                    self.step += 1
                    # Update dual net using dual loss
                    self.dual_net.train()
                    frozen_primal_net.train()
                    total_train_loss = 0.0
                    num_batches = 0
                    for Xtrain, train_eq_cm, train_ineq_cm, train_eq_rhs, train_ineq_rhs in self.train_loader:
                        self.dual_optim.zero_grad()
                        mu, lamb = self.dual_net(Xtrain, train_eq_cm)
                        with torch.no_grad():
                            mu_k, lamb_k = frozen_dual_net(Xtrain, train_eq_cm)
                            y = frozen_primal_net(Xtrain, train_eq_rhs, train_ineq_rhs)
                        # ! Test other loss!
                        batch_loss = self.dual_loss(y, train_eq_cm, train_ineq_cm, train_eq_rhs, train_ineq_rhs, mu, lamb, mu_k, lamb_k).mean()
                        # batch_loss = self.dual_loss_changed(y, train_eq_cm, train_ineq_cm, train_eq_rhs, train_ineq_rhs, mu, lamb, mu_k, lamb_k).mean()
                        batch_loss.backward()
                        self.dual_optim.step()
                        total_train_loss += batch_loss.item()
                        num_batches += 1
                    
                    avg_train_loss = total_train_loss / num_batches
                    self.dual_scheduler.step(avg_train_loss)
                    
                    with torch.no_grad():
                        # Logg training loss:
                        self.logger.log_loss(batch_loss, "dual", self.step)
                        self.logger.log_train(self.data, primal_net=frozen_primal_net, dual_net=self.dual_net, step=self.step)

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

                end_time = time.time()
                stats = epoch_stats
                print(f"Epoch {k} done. Time taken: {end_time - begin_time}. Rho: {self.rho}. Primal LR: {self.primal_optim.param_groups[0]['lr']}, Dual LR: {self.dual_optim.param_groups[0]['lr']}")

                # Update rho from the second iteration onward.
                if k > 0 and v_k > self.tau * prev_v_k:
                    self.rho = np.min([self.alpha * self.rho, self.rho_max])

                prev_v_k = v_k

            # self.logger.plot_variable_differences(self.data, self.primal_net, self.dual_net)
        
        except Exception as e:
            print(e, flush=True)
            # Ensure writer is closed even if an exception occurs
            if self.logger:
                self.logger.close()
            raise

        with open(os.path.join(self.save_dir, 'stats.dict'), 'wb') as f:
            pickle.dump(stats, f)
        with open(os.path.join(self.save_dir, 'primal_net.dict'), 'wb') as f:
            torch.save(self.primal_net.state_dict(), f)
        with open(os.path.join(self.save_dir, 'dual_net.dict'), 'wb') as f:
            torch.save(self.dual_net.state_dict(), f)

        return self.primal_net, self.dual_net, stats

    def primal_loss(self, y, eq_cm, ineq_cm, eq_rhs, ineq_rhs, mu, lamb):
        obj = self.data.obj_fn(y)
        
        # g(y)
        ineq = self.data.ineq_resid(y, ineq_cm, ineq_rhs)
        # h(y)
        eq = self.data.eq_resid(y, eq_cm, eq_rhs)

        # ! Clamp mu?
        # Element-wise clamping of mu_i when g_i (ineq) is negative
        # mu = torch.where(ineq < 0, torch.zeros_like(mu), mu)
        # ! Clamp ineq_resid?
        # ineq = ineq.clamp(min=0)

        lagrange_ineq = torch.sum(mu * ineq, dim=1)  # Shape (batch_size,)

        lagrange_eq = torch.sum(lamb * eq, dim=1)   # Shape (batch_size,)

        violation_ineq = torch.sum(torch.maximum(ineq, torch.zeros_like(ineq)) ** 2, dim=1)
        violation_eq = torch.sum(eq ** 2, dim=1)
        penalty = self.rho/2 * (violation_ineq + violation_eq)

        loss = (obj + (lagrange_ineq + lagrange_eq + penalty))

        return loss

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
    
    def dual_loss_changed(self, y, eq_cm, ineq_cm, eq_rhs, ineq_rhs, mu, lamb, mu_k, lamb_k):
        #! We maximize the dual obj func, so to use it in the loss, take the negation.
        dual_obj = -self.data.dual_obj_fn(eq_rhs, ineq_rhs, mu, lamb)

        #! Enforced with ReLU.
        # ineq = self.data.dual_ineq_resid(mu, lamb)

        eq = self.data.dual_eq_resid(mu, lamb, eq_cm, ineq_cm)
        # Lagrange multiplier becomes y
        lagrange_eq = torch.sum(y * eq, dim=1)

        violation_eq = torch.sum(eq ** 2, dim=1)

        penalty = self.rho/2 * violation_eq

        loss = dual_obj + lagrange_eq + penalty
        # loss = dual_obj + penalty

        return loss

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

    # Modifies stats in place
    def dict_agg(self, stats, key, value, op='concat'):
        if key in stats.keys():
            if op == 'sum':
                stats[key] += value
            elif op == 'concat':
                stats[key] = np.concatenate((stats[key], value), axis=0)
            else:
                raise NotImplementedError
        else:
            stats[key] = value

    # Modifies stats in place
    def eval_pdl(self, X, eq_cm, ineq_cm, eq_rhs, ineq_rhs, primal_net, dual_net, prefix, stats, targets=None):

        eps_converge = self.args['corrEps']
        make_prefix = lambda x: "{}_{}".format(prefix, x)
        start_time = time.time()
        # Y = primal_net(X_scaled)
        # mu, lamb = dual_net(X_scaled)
        Y = primal_net(X)
        mu, lamb = dual_net(X)
        raw_end_time = time.time()
        Ycorr = Y

        # Ycorr, steps = grad_steps_all(data, X, Y, args)

        self.dict_agg(stats, make_prefix('time'), time.time() - start_time, op='sum')
        # self.dict_agg(stats, make_prefix('steps'), np.array([steps]))
        self.dict_agg(stats, make_prefix('primal_loss'), self.primal_loss(Y, eq_cm, ineq_cm, eq_rhs, ineq_rhs, mu, lamb).detach().cpu().numpy())
        # self.dict_agg(stats, make_prefix('dual_loss'), self.dual_loss(X, Y, mu, lamb, mu_k, lamb_k, log_type=None).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('eval'), self.data.obj_fn(Ycorr).detach().cpu().numpy())

        self.dict_agg(stats, make_prefix('ineq_max'), torch.max(self.data.ineq_dist(Y, ineq_cm, ineq_rhs), dim=1)[0].detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('ineq_mean'), torch.mean(self.data.ineq_dist(Y, ineq_cm, ineq_rhs,), dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('ineq_num_viol_0'),
                torch.sum(self.data.ineq_dist(Y, ineq_cm, ineq_rhs) > eps_converge, dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('ineq_num_viol_1'),
                torch.sum(self.data.ineq_dist(Y, ineq_cm, ineq_rhs) > 10 * eps_converge, dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('ineq_num_viol_2'),
                torch.sum(self.data.ineq_dist(Y, ineq_cm, ineq_rhs) > 100 * eps_converge, dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('eq_max'),
                torch.max(torch.abs(self.data.eq_resid(Y, eq_cm, eq_rhs)), dim=1)[0].detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('eq_mean'), torch.mean(torch.abs(self.data.eq_resid(Y, eq_cm, eq_rhs)), dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('eq_num_viol_0'),
                torch.sum(torch.abs(self.data.eq_resid(Y, eq_cm, eq_rhs)) > eps_converge, dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('eq_num_viol_1'),
                torch.sum(torch.abs(self.data.eq_resid(Y, eq_cm, eq_rhs)) > 10 * eps_converge, dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('eq_num_viol_2'),
                torch.sum(torch.abs(self.data.eq_resid(Y, eq_cm, eq_rhs)) > 100 * eps_converge, dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('raw_time'), raw_end_time - start_time, op='sum')
        self.dict_agg(stats, make_prefix('raw_eval'), self.data.obj_fn(Y).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('raw_ineq_max'), torch.max(self.data.ineq_dist(Y, ineq_cm, ineq_rhs), dim=1)[0].detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('raw_ineq_mean'), torch.mean(self.data.ineq_dist(Y, ineq_cm, ineq_rhs), dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('raw_ineq_num_viol_0'),
                torch.sum(self.data.ineq_dist(Y, ineq_cm, ineq_rhs) > eps_converge, dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('raw_ineq_num_viol_1'),
                torch.sum(self.data.ineq_dist(Y, ineq_cm, ineq_rhs) > 10 * eps_converge, dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('raw_ineq_num_viol_2'),
                torch.sum(self.data.ineq_dist(Y, ineq_cm, ineq_rhs) > 100 * eps_converge, dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('raw_eq_max'),
                torch.max(torch.abs(self.data.eq_resid(Y, eq_cm, eq_rhs)), dim=1)[0].detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('raw_eq_mean'),
                torch.mean(torch.abs(self.data.eq_resid(Y, eq_cm, eq_rhs)), dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('raw_eq_num_viol_0'),
                torch.sum(torch.abs(self.data.eq_resid(Y, eq_cm, eq_rhs)) > eps_converge, dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('raw_eq_num_viol_1'),
                torch.sum(torch.abs(self.data.eq_resid(Y, eq_cm, eq_rhs)) > 10 * eps_converge, dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('raw_eq_num_viol_2'),
                torch.sum(torch.abs(self.data.eq_resid(Y, eq_cm, eq_rhs)) > 100 * eps_converge, dim=1).detach().cpu().numpy())

        return stats

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

class PrimalGCNNet(nn.Module):
    def __init__(self, data, hidden_sizes):
        super().__init__()
        self._data = data
        self._hidden_sizes = hidden_sizes
        
        self.gcn = torch_geometric.nn.conv.GCNConv(1, 1)
        self.gcn.reset_parameters()
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
    
    def forward(self, x):
        # ! Only works for batches of 1!
        edge_index = self._data.edge_index_demand
        x_pDemand = x[:, :self._data.neq]
        x_pGenAva = x[:, self._data.neq:]
        # Reshape x_pDemand for GCN: [batch_size * num_nodes, 1]
        batch_size, num_nodes = x_pDemand.shape
        x_pDemand = x_pDemand.T.reshape(-1, 1)  # Shape: [batch_size * num_nodes, 1]

         # Apply GCN layer
        x_pDemand = self.gcn(x_pDemand, edge_index.repeat(1, batch_size))  # Use repeated edge_index
        x_pDemand = torch.relu(x_pDemand)

        # Reshape back to batch dimension: [batch_size, num_nodes]
        x_pDemand = x_pDemand.view(num_nodes, batch_size).T

        x = torch.concat([x_pDemand, x_pGenAva], dim=1)

        return self.net(x)
    
class PrimalCNNNet(nn.Module):
    def __init__(self, data, hidden_channels, kernel_size=3):
        super().__init__()
        self._data = data
        self._hidden_channels = hidden_channels
        
        # Define input and output sizes
        input_features = data.xdim
        output_features = data.ydim

        # Create convolutional layers
        layers = []
        channels = [1] + hidden_channels  # Start with single input channel

        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())

        # Final layer to map to output dimensions
        layers.append(nn.Conv1d(hidden_channels[-1], 1, kernel_size=kernel_size, padding=kernel_size // 2))

        self.conv_net = nn.Sequential(*layers)
        self.fc = nn.Linear(input_features, output_features)

    def forward(self, x):
        # Reshape input to [batch_size, 1, num_features] for Conv1d
        x = x.unsqueeze(1)

        # Pass through convolutional layers
        x = self.conv_net(x)

        # Flatten back to [batch_size, num_features]
        x = x.squeeze(1)

        # Map to final output using a fully connected layer
        return self.fc(x)

class DualNetEndToEnd(nn.Module):
    def __init__(self, data, n_size_factor=1.5, n_layers=4):
        super().__init__()
        self._data = data
        self._hidden_sizes = [int(n_size_factor*data.xdim)] * n_layers

        self._out_dim = data.n_prod_vars + data.n_line_vars

        #! Only predict lambda, we infer mu from it.
        self.feed_forward = FeedForwardNet(data.xdim, self._hidden_sizes, output_dim=data.neq)
        
        
    def forward(self, x, eq_cm):
        out_lamb = self.feed_forward(x)

        # out_mu = torch.relu(x_out[:, :self._data.nineq])
        # out_lamb = x_out[:, self._data.nineq:]
        mu = self._data.obj_coeff + torch.bmm(eq_cm.transpose(1, 2), out_lamb.unsqueeze(-1)).squeeze(-1)

        mu = mu.view(mu.shape[0], self._data.sample_duration, -1)  # Shape: (batch_size, t, constraints)

        # Compute lower and upper bound multipliers
        mu_lb = torch.relu(mu)   # Lower bound multipliers |mu|^+
        mu_ub = torch.relu(-mu)  # Upper bound multipliers |mu|^-

        # Split into groups, following the exact structure of mu
        p_g_lb = mu_lb[:, :, :self._data.num_g]  # Lower bounds for p_g
        p_g_ub = mu_ub[:, :, :self._data.num_g]  # Upper bounds for p_g

        f_l_lb = mu_lb[:, :, self._data.num_g:self._data.num_g + self._data.num_l]  # Lower bounds for f_l
        f_l_ub = mu_ub[:, :, self._data.num_g:self._data.num_g + self._data.num_l]  # Upper bounds for f_l

        md_n_lb = mu_lb[:, :, self._data.num_g + self._data.num_l:]  # Lower bounds for md_n
        md_n_ub = mu_ub[:, :, self._data.num_g + self._data.num_l:]  # Upper bounds for md_n

        # Reshape back into (batch_size, constraints * t) while maintaining order
        out_mu = torch.cat([
            p_g_lb, p_g_ub,  # Lower and Upper bounds for p_g
            f_l_lb, f_l_ub,  # Lower and Upper bounds for f_l
            md_n_lb, md_n_ub  # Lower and Upper bounds for md_n
        ], dim=-1).reshape(mu.shape[0], -1)  # Flatten back to (batch_size, constraints * t)

        return out_mu, out_lamb

class DualCompleteLayer(nn.Module):
    def __init__():
        super().__init__()

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


class DualGCNNet(nn.Module):
    def __init__(self, data, hidden_sizes, mu_size, lamb_size):
        super().__init__()
        self._data = data
        self._hidden_sizes = hidden_sizes
        self._mu_size = mu_size
        self._lamb_size = lamb_size

        self.gcn = torch_geometric.nn.conv.GCNConv(1, 1)
        self.gcn.reset_parameters()
        # Create the list of layer sizes
        layer_sizes = [data.xdim] + self._hidden_sizes
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
        
         # Add the output layer
        self.out_layer = nn.Linear(self._hidden_sizes[-1], self._mu_size + self._lamb_size)
        nn.init.zeros_(self.out_layer.weight)  # Initialize output layer weights to 0
        nn.init.zeros_(self.out_layer.bias)    # Initialize output layer biases to 0
        layers.append(self.out_layer)

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        # ! Only works for batches of 1!
        edge_index = self._data.edge_index_demand
        x_pDemand = x[:, :self._data.neq]
        x_pGenAva = x[:, self._data.neq:]
        # Reshape x_pDemand for GCN: [batch_size * num_nodes, 1]
        batch_size, num_nodes = x_pDemand.shape
        x_pDemand = x_pDemand.T.reshape(-1, 1)  # Shape: [batch_size * num_nodes, 1]

         # Apply GCN layer
        x_pDemand = self.gcn(x_pDemand, edge_index.repeat(1, batch_size))  # Use repeated edge_index
        x_pDemand = torch.relu(x_pDemand)

        # Reshape back to batch dimension: [batch_size, num_nodes]
        x_pDemand = x_pDemand.view(num_nodes, batch_size).T

        x = torch.concat([x_pDemand, x_pGenAva], dim=1)
        x = torch.concat([x_pDemand, x_pGenAva], dim=1)

        out = self.net(x)
        out_mu = out[:, :self._mu_size]
        out_lamb = out[:, self._mu_size:]

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

        # Create layers dynamically based on the provided hidden_sizes
        for idx, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layers.append(nn.LayerNorm(in_size))
            layers.append(nn.Linear(in_size, out_size))
            # Add ReLU only if it is not the last layer
            if idx < len(layer_sizes) - 2:  # The last layer does not need ReLU
                layers.append(nn.ReLU())
        
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
    
    def forward(self, x, lb, ub):
        """_summary_

        Args:
            x (_type_): Decision variables, shape [B, N, T]
            lb (_type_): Lower bounds of decision variables, shape [B, N, T]
            ub (_type_): Upper bounds of decision variables, shape [B, N, T]

        Returns:
            _type_: _description_
        """

        return lb + (ub - lb) * torch.sigmoid(x)
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
        p_nt = torch.einsum('ng,bgt->bnt', self.node_to_gen_mask.to(dtype=torch.float64), p_gt)

        # In lineflow mask (adjacency matrix) starting nodes are -1, receiving nodes are 1.
        # Thus, this mask is f_in_nt - f_out_nt (net_flow)
        net_flow_nt = torch.einsum('nl,blt->bnt', self.lineflow_mask.to(dtype=torch.float64), f_lt)
        # Compute md_n,t
        md_nt = D_nt - (p_nt + net_flow_nt)  # [B, N, T]

        return md_nt

class PrimalNetEndToEnd(nn.Module):
        # TODO! Validate the repair layers in a jupyter notebook!
        def __init__(self, data, n_size_factor=1.5, n_layers=4):
            super().__init__()
            self._data = data
            self._hidden_sizes = [int(n_size_factor*data.xdim)] * n_layers

            self._out_dim = data.n_prod_vars + data.n_line_vars

            self.feed_forward = FeedForwardNet(data.xdim, self._hidden_sizes, output_dim=data.n_prod_vars+data.n_line_vars)
            self.bound_repair_layer = BoundRepairLayer()
            # self.ramping_repair_layer = RampingRepairLayer()

            self.estimate_slack_layer = EstimateSlackLayer(data.node_to_gen_mask, data.lineflow_mask)
        
        def forward(self, x, eq_rhs, ineq_rhs):
            x_out = self.feed_forward(x)

            # [B, G, T], [B, L, T]
            p_gt, f_lt = self._data.split_dec_vars_from_Y_raw(x_out)
            
            # [B, bounds, T]
            p_gt_lb, p_gt_ub, f_lt_lb, f_lt_ub, md_nt_lb, md_nt_ub = self._data.split_ineq_constraints(ineq_rhs)

            p_gt_bound_repaired = self.bound_repair_layer(p_gt, p_gt_lb, p_gt_ub)

            # Lineflow lower bound is negative.
            f_lt_bound_repaired = self.bound_repair_layer(f_lt, -f_lt_lb, f_lt_ub)

            D_nt = self._data.split_eq_constraints(eq_rhs)
            md_nt = self.estimate_slack_layer(p_gt_bound_repaired, f_lt_bound_repaired, D_nt)

            y = torch.cat([p_gt_bound_repaired, f_lt_bound_repaired, md_nt], dim=1).permute(0, 2, 1).reshape(x_out.shape[0], -1)

            return y
         
         
