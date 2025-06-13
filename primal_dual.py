import copy
import os
import time
import torch
from torch.utils.data import TensorDataset, DataLoader
from logger import TensorBoardLogger
import numpy as np
from mtadam import MTAdam
from networks import DualClassificationNetEndToEnd, DualNet, DualNetEndToEnd, PrimalNet, PrimalNetEndToEnd
import optuna
torch.autograd.set_detect_anomaly(True)

class EarlyStopping():
    def __init__(self, patience=1000):
        self.patience = patience        # epochs to wait after last improvement
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        
class PrimalDualTrainer():

    def __init__(self, data, args, save_dir):
        """_summary_

        Args:
            data (_type_): _description_
            args (_type_): _description_
            save_dir (_type_): _description_
            problem_type (str, optional): Either "GEP" "ED" or "QP". Defaults to "ED".
            optimal_objective_train (_type_, optional): _description_. Defaults to None.
            optimal_objective_val (_type_, optional): _description_. Defaults to None.
            log (bool, optional): _description_. Defaults to True.
            optuna (bool, optional): Whether to use Optuna. Defaults to False.
        """

        print(f"X dim: {data.xdim}")
        print(f"Y dim: {data.ydim}")

        print(f"Size of mu: {data.nineq}")
        print(f"Size of lambda: {data.neq}")

        self.data = data
        self.args = args
        self.save_dir = save_dir
        self.problem_type = args["problem_type"]
        self.log = args["log"]
        self.log_frequency = args["log_frequency"]
        
        if self.args["device"] == "mps":
            self.DTYPE = torch.float32
            self.DEVICE = torch.device("mps")
            self.data.to_mps()
        else:
            self.DTYPE = torch.float64
            self.DEVICE = torch.device("cpu")

        torch.set_default_dtype(self.DTYPE)
        torch.set_default_device(self.DEVICE)

        print(f"DTYPE: {self.DTYPE}, DEVICE: {self.DEVICE}")

        self.train = args["train"]
        self.valid = args["valid"]
        self.test = args["test"]
        self.outer_iterations = args["outer_iterations"]
        self.inner_iterations = args["inner_iterations"]
        self.tau = args["tau"]
        self.rho = args["rho"]
        self.rho_max = args["rho_max"]
        self.alpha = args["alpha"]
        self.batch_size = args["batch_size"]
        self.primal_lr = args["primal_lr"]
        self.dual_lr = args["dual_lr"]
        self.decay = args["decay"]
        self.patience = args["patience"]
        self.clip_gradients_norm = args["clip_gradients_norm"]
        self.max_violation_save_thresholds = args["max_violation_save_thresholds"]  
        self.early_stopping_patience = args["early_stopping_patience"]
        self.X = data.X.to(self.DTYPE).to(self.DEVICE)
        if self.problem_type == "ED":
            self.total_demands = data.total_demands.to(self.DTYPE).to(self.DEVICE)
        else:
            self.total_demands = torch.ones((self.X.shape[0], 1))
        
        # for logging
        self.step = 0
        indices = torch.arange(self.X.shape[0])
        # Compute sizes for each set
        train_size = int(self.train * self.X.shape[0])
        valid_size = int(self.valid * self.X.shape[0])
        print(f"Train size: {train_size}, Valid size: {valid_size}, Test size: {self.X.shape[0] - train_size - valid_size}")

        # Split the indices
        self.train_indices = indices[:train_size]
        self.valid_indices = indices[train_size:train_size+valid_size]
        self.test_indices = indices[train_size+valid_size:]

        self.X_train = self.X[self.train_indices]
        self.X_valid = self.X[self.valid_indices]
        self.total_demands_train = self.total_demands[self.train_indices]
        self.total_demands_valid = self.total_demands[self.valid_indices]

        if self.log == True:
            self.logger = TensorBoardLogger(args, data, self.X, self.total_demands, self.train_indices, self.valid_indices, save_dir, args["opt_targets"])
        else:
            self.logger = None
        
        self.train_dataset = TensorDataset(self.X_train, self.total_demands_train)
        self.valid_dataset = TensorDataset(self.X_valid, self.total_demands_valid)
        # self.test_dataset = TensorDataset(self.data.testX.to(self.DEVICE), self.data.testX_scaled.to(self.DEVICE))

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, generator=torch.Generator(device=self.DEVICE))
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=len(self.valid_dataset), generator=torch.Generator(device=self.DEVICE))
        # self.test_loader = DataLoader(self.test_dataset, batch_size=len(self.test_dataset))

        if self.problem_type == "QP":
            self.primal_loss_fn = self.primal_loss_QP
            self.dual_loss_fn = self.dual_loss
            self.primal_net = PrimalNet(self.args, self.data).to(dtype=self.DTYPE, device=self.DEVICE)
            self.dual_net = DualNet(self.args, self.data).to(dtype=self.DTYPE, device=self.DEVICE)
        elif self.problem_type == "ED":
            self.primal_loss_fn = self.primal_loss
            #! PrimalNetEndToEnd takes into account whether repairs are used or not.
            self.primal_net = PrimalNetEndToEnd(self.args, self.data).to(dtype=self.DTYPE, device=self.DEVICE)
            if self.args["dual_alternate_loss"]:
                if self.args["dual_classification"]:
                    self.dual_net = DualClassificationNetEndToEnd(self.args, self.data).to(dtype=self.DTYPE, device=self.DEVICE)
                elif self.args["dual_completion"]:
                    self.dual_net = DualNetEndToEnd(self.args, self.data).to(dtype=self.DTYPE, device=self.DEVICE)
                else:
                    self.dual_net = DualNet(self.args, self.data).to(dtype=self.DTYPE, device=self.DEVICE)
                self.dual_loss_fn = self.alternate_dual_loss
            else:
                self.dual_net = DualNet(self.args, self.data).to(dtype=self.DTYPE, device=self.DEVICE)
                self.dual_loss_fn = self.dual_loss

        elif self.problem_type == "GEP":
            # TODO: Implement GEP networks
            pass
        self.primal_net.to(self.DTYPE).to(self.DEVICE)
        self.primal_optim = torch.optim.Adam(self.primal_net.parameters(), lr=self.primal_lr)
        self.dual_optim = torch.optim.Adam(self.dual_net.parameters(), lr=self.dual_lr)

        #! For MTAdam
        # self.primal_optim = MTAdam(self.primal_net.parameters(), lr=self.primal_lr)
        # self.dual_optim = MTAdam(self.dual_net.parameters(), lr=self.dual_lr)

        # Add schedulers
        self.primal_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.primal_optim, mode='min', factor=self.decay, patience=self.patience
        )
        self.dual_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.dual_optim, mode='min', factor=self.decay, patience=self.patience
        )

        # For saving best models:
        self.best_primal_objs = [float('inf') for _ in range(len(self.max_violation_save_thresholds))]
        self.best_dual_obj = -1*float('inf')

        self.opt_targets_train = self.data.opt_targets["y_operational"].to(self.DTYPE).to(self.DEVICE)[self.train_indices]
        self.target_obj_train = self.data.obj_fn(self.X_train, self.opt_targets_train)

        pred_y = self.primal_net(self.X_train, self.total_demands_train)
        pred_obj_train = self.data.obj_fn(self.X_train, pred_y)
        # print((pred_obj_train - self.target_obj_train) / self.target_obj_train)
        print(f"Initial known; pred; gap: {self.target_obj_train.mean()}, {pred_obj_train.mean()}, {((pred_obj_train - self.target_obj_train) / self.target_obj_train).mean()}")

        if not self.args["learn_primal"]:
            assert self.args["dual_alternate_loss"] == True, "Cannot disable primal learning without alternate dual loss."
        
        self.primal_early_stopping = EarlyStopping(patience=self.early_stopping_patience)
        self.dual_early_stopping = EarlyStopping(patience=self.early_stopping_patience)

        self.primal_net_best = None
        self.dual_net_best = None

        self.train_time = 0

        self.best_time = 0

    def freeze(self, network):
        """
        Create a frozen copy of a network
        """
        if isinstance(network, PrimalNetEndToEnd):
            frozen_net = PrimalNetEndToEnd(self.args, self.data).to(device=self.DEVICE, dtype=self.DTYPE)
        elif isinstance(network, DualNetEndToEnd):
            frozen_net = DualNetEndToEnd(self.args, self.data).to(device=self.DEVICE, dtype=self.DTYPE)
        elif isinstance(network, DualClassificationNetEndToEnd):
            frozen_net = DualClassificationNetEndToEnd(self.args, self.data).to(device=self.DEVICE, dtype=self.DTYPE)
        elif isinstance(network, PrimalNet):
            frozen_net = PrimalNet(self.args, self.data).to(device=self.DEVICE, dtype=self.DTYPE)
        elif isinstance(network, DualNet):
            frozen_net = DualNet(self.args, self.data).to(device=self.DEVICE, dtype=self.DTYPE)
        else:
            raise TypeError(f"Unsupported network type: {type(network)}")
        
        # Load a deep copy of the state dictionary
        frozen_net.load_state_dict(copy.deepcopy(network.state_dict()))
    
        # Set to evaluation mode
        frozen_net.eval()
        
        return frozen_net

    def train_PDL(self, optuna_trial=None):
        prev_v_k = 0
        for k in range(self.outer_iterations):
            begin_time = time.time()
            frozen_dual_net = self.freeze(self.dual_net)
            if self.logger:
                with torch.no_grad():
                    self.logger.log_rho_vk(self.rho, prev_v_k, self.step)
            if self.args["learn_primal"]:
                for l1 in range(self.inner_iterations):
                    self.step += 1
                    # Update primal net using primal loss
                    self.primal_net.train()
                    frozen_dual_net.train()

                    # Accumulate training loss over all batches
                    # For logging
                    total_train_loss = 0.0
                    total_obj = 0.0
                    total_lagrange_eq = 0.0
                    total_lagrange_ineq = 0.0
                    total_penalty = 0.0

                    num_batches = 0

                    for (Xtrain, total_demands) in self.train_loader:
                        compute_begin_time = time.time()

                        self.primal_optim.zero_grad()
                        
                        y = self.primal_net(Xtrain, total_demands)
                        # y.requires_grad_(True) # If logging gradients

                        with torch.no_grad():
                            if k == 0 and self.problem_type != "QP":
                                mu, lamb = torch.zeros((Xtrain.shape[0], self.data.nineq)), torch.zeros((Xtrain.shape[0], self.data.neq))
                            else:
                                mu, lamb = frozen_dual_net(Xtrain)
                        batch_loss, obj, lagrange_eq, lagrange_ineq, penalty = self.primal_loss_fn(Xtrain, y, mu.detach(), lamb.detach())
                        batch_loss, obj, lagrange_eq, lagrange_ineq, penalty = batch_loss.mean(), obj.mean(), lagrange_eq.mean(), lagrange_ineq.mean(), penalty.mean()
                        total_train_loss += batch_loss.item()
                        total_obj += obj.item()
                        total_lagrange_eq += lagrange_eq.item()
                        total_lagrange_ineq += lagrange_ineq.item()
                        total_penalty += penalty.item()
                        if isinstance(self.primal_optim, MTAdam):
                            self.primal_optim.step(loss_array=[obj, lagrange_eq, lagrange_ineq, penalty], ranks=[1, 1, 1, 1], feature_map=None)
                        else:
                            # y.retain_grad()
                            batch_loss.backward()
                            # Log the gradients of each decision variable
                            # p_gt, f_lt, md_nt = self.data.split_dec_vars_from_Y(y.grad)
                            # print("Gradients of p_gt:", p_gt.mean())
                            # print("Gradients of f_lt:", f_lt)
                            # print("Gradients of md_nt:", md_nt.mean())
                            self.primal_optim.step()
                        
                        compute_end_time = time.time()
                        self.train_time += compute_end_time - compute_begin_time
                        num_batches += 1
                    
                    # Compute average loss for the epoch
                    avg_train_loss = total_train_loss / num_batches
                    avg_obj = total_obj / num_batches
                    avg_lagrange_eq = total_lagrange_eq / num_batches
                    avg_lagrange_ineq = total_lagrange_ineq / num_batches
                    avg_penalty = total_penalty / num_batches

                    # Log training loss:
                    if self.logger and self.log_frequency > 0 and self.step % self.log_frequency == 0:
                        with torch.no_grad():
                            self.logger.log_primal_loss(avg_train_loss, avg_obj, avg_lagrange_eq, avg_lagrange_ineq, avg_penalty, self.step)
                            self.logger.log_train(self.data, primal_net=self.primal_net, dual_net=frozen_dual_net, rho=self.rho, step=self.step)
                    
                    with torch.no_grad():
                        self.primal_net.eval()
                        frozen_dual_net.eval()
                        obj_val_mean, val_loss_mean, ineq_max, ineq_mean, eq_max, eq_mean, dual_obj_val_mean, dual_loss_mean = self.evaluate(self.valid_dataset.tensors[0], self.valid_dataset.tensors[1], self.primal_net, self.dual_net)    
                        if k > 0:
                            self.save_if_best(obj_val_mean, ineq_max, ineq_mean, eq_max, eq_mean, dual_obj_val_mean)
                        # Normalize by rho, so that the scheduler still works correctly if rho is increased

                        if self.primal_early_stopping.step(val_loss_mean):
                            self.primal_net_best = self.freeze(self.primal_net)
                            self.best_time = self.train_time
                        if self.early_stopping_patience > 0 and self.primal_early_stopping.early_stop:
                            print(f"Early stopping at step {self.step}")
                            # Return the best primal net, and the best loss.
                            self.save(self.save_dir)
                            with open(os.path.join(self.save_dir, "train_time.txt"), "w") as f:
                                f.write(f"Train time: {self.train_time}")
                            return self.primal_net_best, self.dual_net, self.primal_early_stopping.best_loss, dual_loss_mean, self.train_time

                        if optuna_trial:
                            optuna_trial.report(val_loss_mean.item(), self.step)
                            if optuna_trial.should_prune():
                                raise optuna.TrialPruned()
                        
                        if self.rho > 0:
                            self.primal_scheduler.step(torch.sign(val_loss_mean) * (torch.abs(val_loss_mean) / self.rho))
                        else:
                            self.primal_scheduler.step(val_loss_mean)


                with torch.no_grad():
                    # Copy primal net into frozen primal net
                    self.primal_net.train() # Otherwise, we are still on eval, and inverse normalize.
                    frozen_primal_net = self.freeze(self.primal_net)

                    # Calculate v_k
                    y = frozen_primal_net(self.X_train, self.total_demands_train)
                    mu_k, lamb_k = frozen_dual_net(self.X_train)
                    v_k = self.violation(self.X_train, y, mu_k)

            if self.args["learn_dual"]:
                for l in range(self.inner_iterations):
                    self.step += 1
                    # Update dual net using dual loss
                    self.dual_net.train()
                    if self.args["learn_primal"]:
                        frozen_primal_net.eval()
                    # For logging
                    total_train_loss = 0.0
                    total_obj = 0.0
                    total_lagrange_eq = 0.0
                    total_lagrange_ineq = 0.0
                    total_penalty = 0.0

                    num_batches = 0
                    for (Xtrain, total_demands) in self.train_loader:
                        compute_begin_time = time.time()
                        self.dual_optim.zero_grad()
                        mu, lamb = self.dual_net(Xtrain)
                        # print(lamb.mean(), lamb.max(), lamb.min())
                        if self.args["learn_primal"]:
                            with torch.no_grad():
                                mu_k, lamb_k = frozen_dual_net(Xtrain)
                                y = frozen_primal_net(Xtrain, total_demands).detach()
                        else:
                            mu_k, lamb_k = None, None
                            y = None

                        batch_loss, obj, lagrange_eq, lagrange_ineq, penalty = self.dual_loss_fn(Xtrain, y, mu, lamb, mu_k, lamb_k)
                        batch_loss, obj, lagrange_eq, lagrange_ineq, penalty = batch_loss.mean(), obj.mean(), lagrange_eq.mean(), lagrange_ineq.mean(), penalty.mean()
                        total_train_loss += batch_loss.item()
                        total_obj += obj.item()
                        total_lagrange_eq += lagrange_eq.item()
                        total_lagrange_ineq += lagrange_ineq.item()
                        total_penalty += penalty.item()

                        batch_loss.backward()

                        self.dual_optim.step()
                        compute_end_time = time.time()
                        self.train_time += compute_end_time - compute_begin_time

                        total_train_loss += batch_loss.item()
                        num_batches += 1
                    
                    if self.logger and self.log_frequency > 0 and self.step % self.log_frequency == 0:
                        with torch.no_grad():
                            # Logg training loss:
                            # Compute average loss for the epoch
                            avg_train_loss = total_train_loss / num_batches
                            avg_obj = total_obj / num_batches
                            avg_lagrange_eq = total_lagrange_eq / num_batches
                            avg_lagrange_ineq = total_lagrange_ineq / num_batches
                            avg_penalty = total_penalty / num_batches


                            self.logger.log_dual_loss(avg_train_loss, self.step, avg_obj, avg_lagrange_eq, avg_lagrange_ineq, avg_penalty)
                            self.logger.log_train(self.data, primal_net=self.primal_net, dual_net=self.dual_net, rho=self.rho, step=self.step)
                    
                    # Evaluate validation loss every epoch, and update learning rate
                    with torch.no_grad():
                        self.primal_net.eval()
                        self.dual_net.eval()
                        obj_val_mean, val_loss_mean, ineq_max, ineq_mean, eq_max, eq_mean, dual_obj_val_mean, dual_loss_mean = self.evaluate(self.valid_dataset.tensors[0], self.valid_dataset.tensors[1], self.primal_net, self.dual_net)    

                        # Early stopper also checks whether the dual loss is better than the best seen so far.
                        if self.dual_early_stopping.step(dual_loss_mean):
                            self.dual_net_best = self.freeze(self.dual_net)
                            self.best_time = self.train_time
                        if self.early_stopping_patience > 0 and self.dual_early_stopping.early_stop:
                            print(f"Early stopping at step {self.step}")
                            self.save(self.save_dir)
                            with open(os.path.join(self.save_dir, "train_time.txt"), "w") as f:
                                f.write(f"Train time: {self.train_time}")
                            return self.primal_net, self.dual_net_best, val_loss_mean, self.dual_early_stopping.best_loss, self.train_time
                        
                        if optuna_trial:
                            optuna_trial.report(dual_loss_mean.item(), self.step)
                            if optuna_trial.should_prune():
                                raise optuna.TrialPruned()
                        
                        # Normalize by rho, so that the schedular still works correctly if rho is increased
                        if self.rho > 0:
                            self.dual_scheduler.step(torch.sign(dual_loss_mean) * (torch.abs(dual_loss_mean) / self.rho))
                        else:
                            self.dual_scheduler.step(dual_loss_mean)
    
            if self.logger:
                with torch.no_grad():
                    self.logger.log_train(self.data, primal_net=self.primal_net, dual_net=self.dual_net, rho=self.rho, step=self.step)
                    self.logger.log_val(self.data, self.primal_net, self.dual_net, self.step)

            end_time = time.time()
            print(f"Epoch {k} done. Time taken: {end_time - begin_time}. Rho: {self.rho}. Primal LR: {self.primal_optim.param_groups[0]['lr']}, Dual LR: {self.dual_optim.param_groups[0]['lr']}")

            if self.args["learn_primal"]:
                # Update rho from the second iteration onward.
                if k > 0 and v_k > self.tau * prev_v_k:
                    self.rho = np.min([self.alpha * self.rho, self.rho_max])

                prev_v_k = v_k
        
        self.save(self.save_dir)
        with open(os.path.join(self.save_dir, "train_time.txt"), "w") as f:
            f.write(f"Train time: {self.best_time}")

        return self.primal_net_best, self.dual_net_best, val_loss_mean, dual_loss_mean, self.best_time

    def save_if_best(self, obj_val_mean, ineq_max, ineq_mean, eq_max, eq_mean, dual_obj_val_mean):
        """
        Saves the primal and/or dual model if they meet the criteria for improvement.
        - Primal model is saved if objective is the best for a given violation threshold.
        - Dual model is saved if its objective is the best seen so far.
        """
        # Primal Model Saving Logic
        for i in range(len(self.max_violation_save_thresholds)):
            threshold = self.max_violation_save_thresholds[i]
            # Check if validation metrics meet the threshold and objective is improved
            if ineq_max < threshold and eq_max < threshold and obj_val_mean < self.best_primal_objs[i]:
                print(f"Saving new best primal model for threshold {threshold}: "
                    f"Obj: {obj_val_mean:.4f}, Eq Max: {eq_max:.4f}, Ineq Max: {ineq_max:.4f}")
                
                primal_save_path = os.path.join(self.save_dir, f'{threshold}_best_primal_net.pth')
                torch.save(self.primal_net.state_dict(), primal_save_path)
                
                # Update the best objective for this threshold
                self.best_primal_objs[i] = obj_val_mean

        # Dual Model Saving Logic
        # if dual_obj_val_mean > self.best_dual_obj:
        #     print(f"Saving new best dual model: Obj: {dual_obj_val_mean:.4f}")
            
        #     dual_save_path = os.path.join(self.save_dir, 'best_dual_net.pth')
        #     torch.save(self.dual_net.state_dict(), dual_save_path)
            
        #     # Update the best overall dual objective
        #     self.best_dual_obj = dual_obj_val_mean

    def evaluate(self, X, total_demands, primal_net, dual_net):        
        # Forward pass through networks
        Y = primal_net(X, total_demands)
        mu, lamb = dual_net(X)

        ineq_dist = self.data.ineq_dist(X, Y)
        eq_resid = self.data.eq_resid(X, Y)

        # Convert lists to arrays for easier handling
        obj_values = self.data.obj_fn(X, Y).detach()
        primal_losses, obj, lagrange_eq, lagrange_ineq, penalty = self.primal_loss_fn(X, Y, mu, lamb)
        dual_losses, dual_obj, dual_lagrange_eq, dual_lagrange_ineq, dual_penalty = self.alternate_dual_loss(X, Y, mu, lamb)
        primal_losses = primal_losses.detach()
        dual_losses = dual_losses.detach()
        ineq_max_vals = torch.max(ineq_dist, dim=1)[0].detach() # First element is the max, second is the index
        ineq_mean_vals = torch.mean(ineq_dist, dim=1).detach()
        eq_max_vals = torch.max(torch.abs(eq_resid), dim=1)[0].detach() # First element is the max, second is the index
        eq_mean_vals = torch.mean(torch.abs(eq_resid), dim=1).detach()

        return torch.mean(obj_values), torch.mean(primal_losses), torch.mean(ineq_max_vals), torch.mean(ineq_mean_vals), torch.mean(eq_max_vals), torch.mean(eq_mean_vals), torch.mean(dual_obj), torch.mean(dual_losses)

    def primal_loss_QP(self, X, Y, mu, lamb):
        obj = self.data.obj_fn(X, Y)
        
        # g(Y)
        ineq = self.data.ineq_resid(X, Y)
        # h(Y)
        eq = self.data.eq_resid(X, Y)

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

        return loss, obj, lagrange_eq, lagrange_ineq, penalty
    
    def primal_loss(self, X, Y, mu, lamb):
        # if self.args["penalize_md_obj"]:
        obj = self.data.obj_fn(X, Y)
        if self.rho > 0:
            ineq = self.data.ineq_resid(X, Y)
            # ineq = self.data.ineq_resid(X, Y)
            eq = self.data.eq_resid(X, Y)
            lagrange_eq = torch.sum(lamb * eq, dim=1)
            lagrange_ineq = torch.sum(mu * ineq, dim=1).clamp(min=0)  # Shape (batch_size,)
            violation_ineq = torch.sum(torch.maximum(ineq, torch.zeros_like(ineq)) ** 2, dim=1)
            violation_eq = torch.sum(eq ** 2, dim=1)
            penalty = self.rho/2 * (violation_ineq + violation_eq)
            loss = obj + lagrange_ineq + lagrange_eq + penalty
            return loss, obj, lagrange_eq, lagrange_ineq, penalty
        else:   
            loss = obj
            return loss, obj, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
        

    def dual_loss(self, X, y, mu, lamb, mu_k, lamb_k):
        # mu = [batch, g]
        # lamb = [batch, h]

        # g(y)
        ineq = self.data.ineq_resid(X, y) # [batch, g]
        # h(y)
        eq = self.data.eq_resid(X, y)   # [batch, h]

        #! From 2nd PDL paper, fix to 1e-1, not rho
        target_mu = torch.maximum(mu_k + self.rho * ineq, torch.zeros_like(ineq))
        # target_mu = torch.maximum(mu_k + 1e-1 * ineq, torch.zeros_like(ineq))

        dual_resid_ineq = mu - target_mu # [batch, g]

        dual_resid_ineq = torch.norm(dual_resid_ineq, dim=1)  # [batch]

        # Compute the dual residuals for equality constraints
        #! From 2nd PDL paper, fix to 1e-1, not rho
        dual_resid_eq = lamb - (lamb_k + self.rho * eq)
        # dual_resid_eq = lamb - (lamb_k + 1e-1 * eq)
        dual_resid_eq = torch.norm(dual_resid_eq, dim=1)  # Norm along constraint dimension

        loss = (dual_resid_ineq + dual_resid_eq)

        return loss, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
    
    def alternate_dual_loss(self, X, Y, mu, lamb, mu_k=None, lamb_k=None):
        #! We maximize the dual obj func, so to use it in the loss, take the negation.
        dual_obj = self.data.dual_obj_fn(X, mu, lamb)

        loss = -dual_obj

        #! Dual constraints are never violated, so we do not include penalty and lagrangian terms.
        return loss, dual_obj, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

    def violation(self, X, Y, mu_k):
        # Calculate the equality constraint function h_x(y)
        eq = self.data.eq_resid(X, Y)  # Assume shape (num_samples, n_eq)
        
        # Calculate the infinity norm of h_x(y)
        eq_inf_norm = torch.abs(eq).max(dim=1).values  # Shape: (num_samples,)

        # Calculate the inequality constraint function g_x(y)
        ineq = self.data.ineq_resid(X, Y)  # Assume shape (num_samples, n_ineq)
        
        # Calculate sigma_x(y) for each inequality constraint
        sigma_y = torch.maximum(ineq, -mu_k / self.rho)  # Element-wise max
        
        # Calculate the infinity norm of sigma_x(y)
        sigma_y_inf_norm = torch.abs(sigma_y).max(dim=1).values  # Shape: (num_samples,)

        # Compute v_k as the maximum of the two norms
        v_k = torch.maximum(eq_inf_norm, sigma_y_inf_norm)  # Shape: (num_samples,)
        
        return v_k.max().item()

    def save(self, save_dir):
        print("saving")
        if self.primal_net_best is not None:
            torch.save(self.primal_net_best.state_dict(), save_dir + '/primal_weights.pth')
        if self.dual_net_best is not None:
            torch.save(self.dual_net_best.state_dict(), save_dir + '/dual_weights.pth')