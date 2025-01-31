import copy
import os
import pickle
import time
from setproctitle import setproctitle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np


# DTYPE = torch.float32
# torch.set_default_dtype(DTYPE)

# DEVICE = (
#     torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
# )

DTYPE = torch.float64
DEVICE = torch.device="cpu"

# Train with same seed!
# Set a fixed seed for reproducibility
# SEED = 42
# NumPy random seed
# np.random.seed(SEED)
# PyTorch random seed
# torch.manual_seed(SEED)

# DEVICE = "cpu"
print(f"Running on {DEVICE}")

# OPTIMAL_OBJ = 874269730.1062025 # Entire set (Train, Val, Test)
# OPTIMAL_OBJ = 1132056901.8343623 # Validation set
# OPTIMAL_OBJ = 8.18648032e+08 # First sample

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, rhs_eq, rhs_ineq, targets):
        self.inputs = inputs  # Inputs should be a PyTorch tensor
        self.rhs_eq = rhs_eq
        self.rhs_ineq = rhs_ineq
        self.targets = targets  # Targets can be any data type
        self._index = 0  # Internal index for tracking iteration

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Return a tuple of input and target for the given index
        return self.inputs[idx], self.rhs_eq[idx], self.rhs_ineq[idx], self.targets[idx]

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
        
class PrimalDualTrainer():

    def __init__(self, data, args, save_dir, problem_type="GEP", optimal_objective_train=None, optimal_objective_val=None, log=True):
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

        print(f"X dim: {data._xdim}")
        print(f"Y dim: {data._ydim}")

        print(f"Size of mu: {data.nineq}")
        print(f"Size of lambda: {data.neq}")

        self.data = data
        self.args = args
        self.save_dir = save_dir
        self.problem_type = problem_type
        if log:
            self.logger = TensorBoardLogger(data, save_dir, optimal_objective_train, optimal_objective_val)
        else:
            self.logger = None

        self.K = args["K"]
        self.L = args["L"]
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

        if problem_type == "GEP":
            self.train_dataset = CustomDataset(self.data.trainX.to(DEVICE), self.data.train_rhs_eq.to(DEVICE), self.data.train_rhs_ineq.to(DEVICE), self.data.targets_train)
            self.valid_dataset = CustomDataset(self.data.validX.to(DEVICE), self.data.valid_rhs_eq.to(DEVICE), self.data.valid_rhs_ineq.to(DEVICE), self.data.targets_valid)
        elif problem_type == "Benchmark":
            self.train_dataset = TensorDataset(self.data.trainX.to(DEVICE))
            self.valid_dataset = TensorDataset(self.data.validX.to(DEVICE))
        # self.test_dataset = TensorDataset(self.data.testX.to(DEVICE), self.data.testX_scaled.to(DEVICE))

        # self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=len(self.valid_dataset))
        # self.test_loader = DataLoader(self.test_dataset, batch_size=len(self.test_dataset))

        self.primal_net = PrimalNet(self.data, self.hidden_sizes).to(dtype=DTYPE, device=DEVICE)
        # dual_net = DualNetTwoOutputLayers(data, hidden_size).to(dtype=DTYPE, device=DEVICE)
        if problem_type == "GEP":
            self.dual_net = DualNet(self.data, self.hidden_sizes, self.data.nineq, self.data.neq).to(dtype=DTYPE, device=DEVICE)
        elif problem_type == "Benchmark":
            self.dual_net = DualNet(self.data, self.hidden_sizes, self.data.nineq, self.data.neq).to(dtype=DTYPE, device=DEVICE)


        self.primal_optim = torch.optim.Adam(self.primal_net.parameters(), lr=self.primal_lr)
        self.dual_optim = torch.optim.Adam(self.dual_net.parameters(), lr=self.dual_lr)

        # Add schedulers
        self.primal_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.primal_optim, mode='min', factor=self.decay, patience=self.patience
        )
        self.dual_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.dual_optim, mode='min', factor=self.decay, patience=self.patience
        )

    def train_PDL(self, ):

        try:

            if self.problem_type == "GEP":
                for k in range(self.K):
                    begin_time = time.time()
                    epoch_stats = {}
                    frozen_dual_net = copy.deepcopy(self.dual_net)
                    frozen_dual_net.eval()

                    for l1 in range(self.L):
                        self.step += 1
                        # Update primal net using primal loss
                        self.primal_net.train()
                        for Xtrain, train_rhs_eq, train_rhs_ineq, train_targets in self.train_loader:
                            self.primal_optim.zero_grad()
                            y = self.primal_net(Xtrain)
                            mu, lamb = frozen_dual_net(Xtrain)
                            train_loss = self.primal_loss(Xtrain, y, mu, lamb, train_targets, log_type="train")
                            train_loss.mean().backward()
                            self.primal_optim.step()
                        if self.logger is not None and self.step % 10 == 0:
                            self.logger.log_primal(self.step)

                        # Evaluate validation loss every epoch, and update learning rate
                        curr_val_loss = 0
                        self.primal_net.eval()
                        for Xvalid, valid_rhs_eq, valid_rhs_ineq, valid_targets in self.valid_loader:
                            y = self.primal_net(Xvalid)
                            mu, lamb = frozen_dual_net(Xvalid)
                            loss = self.primal_loss(Xvalid, y, mu, lamb, valid_targets, log_type="valid").mean()
                            curr_val_loss += loss
                        curr_val_loss /= len(self.valid_loader)
                        # Normalize by rho, so that the schedular still works correctly if rho is increased
                        self.primal_scheduler.step(torch.sign(curr_val_loss) * (torch.abs(curr_val_loss) / self.rho))
                    

                    # Copy primal net into frozen primal net
                    frozen_primal_net = copy.deepcopy(self.primal_net)
                    frozen_primal_net.eval()

                    # Calculate v_k
                    y = frozen_primal_net(self.train_dataset.inputs)
                    mu_k, lamb_k = frozen_dual_net(self.train_dataset.inputs)
                    v_k = self.violation(self.train_dataset.inputs, y, mu_k)

                    for l2 in range(self.L):
                        self.step += 1
                        # Update dual net using dual loss
                        self.dual_net.train()
                        for Xtrain, train_rhs_eq, train_rhs_ineq, train_targets in self.train_loader:
                            self.dual_optim.zero_grad()
                            mu, lamb = self.dual_net(Xtrain)
                            mu_k, lamb_k = frozen_dual_net(Xtrain)
                            y = frozen_primal_net(Xtrain)
                            train_loss = self.dual_loss(Xtrain, y, mu, lamb, mu_k, lamb_k, train_rhs_eq, train_rhs_ineq, train_targets, log_type="train")
                            train_loss.mean().backward()
                            self.dual_optim.step()

                        if self.logger is not None and self.step % 10 == 0:
                            self.logger.log_dual(self.step)

                        # Evaluate validation loss every epoch, and update learning rate
                        self.dual_net.eval()
                        curr_val_loss = 0
                        for Xvalid, valid_rhs_eq, valid_rhs_ineq, valid_targets in self.valid_loader:
                            y = frozen_primal_net(Xvalid)
                            mu_valid, lamb_valid = self.dual_net(Xvalid)
                            mu_k_valid, lamb_k_valid = frozen_dual_net(Xvalid)
                            curr_val_loss += self.dual_loss(Xvalid, y, mu_valid, lamb_valid, mu_k_valid, lamb_k_valid, valid_rhs_eq, valid_rhs_ineq, valid_targets, log_type="valid").mean()
                        
                        curr_val_loss /= len(self.valid_loader)
                        # Normalize by rho, so that the schedular still works correctly if rho is increased
                        self.dual_scheduler.step(torch.sign(curr_val_loss) * (torch.abs(curr_val_loss) / self.rho))

                    ##### Evaluate #####
                    self.primal_net.train()
                    self.dual_net.train()
                    self.logger.log_weights_gradients(model_name="Primal", model=self.primal_net, step=self.step)
                    self.logger.log_weights_gradients(model_name="Dual", model=self.dual_net, step=self.step)

                    self.primal_net.eval()
                    self.dual_net.eval()
                    
                    for Xtrain, train_rhs_eq, train_rhs_ineq, train_targets in self.train_loader:
                        self.eval_pdl(Xtrain, self.primal_net, self.dual_net, 'train', epoch_stats, train_targets)

                    # Get valid loss
                    for Xvalid, valid_rhs_eq, valid_rhs_ineq, valid_targets in self.valid_loader:
                        self.eval_pdl(Xvalid, self.primal_net, self.dual_net, 'valid', epoch_stats, train_targets)

                    end_time = time.time()
                    stats = epoch_stats
                    print(f"Epoch {k} done. Time taken: {end_time - begin_time}. Rho: {self.rho}. Primal LR: {self.primal_optim.param_groups[0]['lr']}, Dual LR: {self.dual_optim.param_groups[0]['lr']}")
                    print(
                        '{}: p-loss: {:.4E}, obj. val {:.4E}, Max eq.: {:.4E}, Max ineq.: {:.4E}, Mean eq.: {:.4E}, Mean ineq.: {:.4E}\n'.format(
                            k, np.mean(epoch_stats['valid_primal_loss']),
                            np.mean(epoch_stats['valid_eval']),
                            np.mean(epoch_stats['valid_eq_max']),
                            np.mean(epoch_stats['valid_ineq_max']),
                            np.mean(epoch_stats['valid_eq_mean']),
                            np.mean(epoch_stats['valid_ineq_mean']))
                    )
                    # Write to tensorboard
                    if self.logger is not None:
                        # Eval logg
                        self.logger.add_eval_log("valid", epoch_stats['valid_eval'], epoch_stats['valid_eq_max'], epoch_stats['valid_ineq_max'], epoch_stats['valid_ineq_mean'], epoch_stats["valid_eq_mean"])
                        self.logger.add_eval_log("train", epoch_stats['train_eval'], epoch_stats['train_eq_max'], epoch_stats['train_ineq_max'], epoch_stats['train_ineq_mean'], epoch_stats["train_eq_mean"])
                        # self.logger.add_param_log(v_k, self.rho, self.primal_optim.param_groups[0]['lr'], self.dual_optim.param_groups[0]['lr'])
                        self.logger.log_eval(self.step)
                        self.logger.log_param(self.step)

                    # Update rho from the second iteration onward.
                    if k > 0 and v_k > self.tau * prev_v_k:
                        self.rho = np.min([self.alpha * self.rho, self.rho_max])
                    prev_v_k = v_k
                
            elif self.problem_type == "Benchmark":

                for k in range(self.K):
                    begin_time = time.time()
                    epoch_stats = {}
                    frozen_dual_net = copy.deepcopy(self.dual_net)
                    frozen_dual_net.eval()

                    for l1 in range(self.L):
                        self.step += 1
                        # Update primal net using primal loss
                        self.primal_net.train()
                        for Xtrain in self.train_loader:
                            start_time = time.time()
                            self.primal_optim.zero_grad()
                            y = self.primal_net(Xtrain[0])
                            mu, lamb = frozen_dual_net(Xtrain[0])
                            train_loss = self.primal_loss(Xtrain[0], y, mu, lamb, targets=None, log_type="train")
                            train_loss.mean().backward()
                            self.primal_optim.step()
                            train_time = time.time() - start_time
                        if self.logger is not None and self.step % 10 == 0:
                            self.logger.log_primal(self.step)

                        # Evaluate validation loss every epoch, and update learning rate
                        curr_val_loss = 0
                        self.primal_net.eval()
                        for Xvalid in self.valid_loader:
                            y = self.primal_net(Xvalid[0])
                            mu, lamb = frozen_dual_net(Xvalid[0])
                            loss = self.primal_loss(Xvalid[0], y, mu, lamb, targets=None, log_type="valid").mean()
                            curr_val_loss += loss
                        curr_val_loss /= len(self.valid_loader)
                        # Normalize by rho, so that the schedular still works correctly if rho is increased
                        self.primal_scheduler.step(torch.sign(curr_val_loss) * (torch.abs(curr_val_loss) / self.rho))
                    
                    # self.dict_agg(epoch_stats, 'train_loss_primal', train_loss.detach().cpu().numpy())

                    # Copy primal net into frozen primal net
                    frozen_primal_net = copy.deepcopy(self.primal_net)
                    frozen_primal_net.eval()

                    # Calculate v_k
                    y = frozen_primal_net(self.train_dataset.tensors[0])
                    mu_k, lamb_k = frozen_dual_net(self.train_dataset.tensors[0])
                    v_k = self.violation(self.train_dataset.tensors[0], y, mu_k)

                    for l2 in range(self.L):
                        self.step += 1
                        # Update dual net using dual loss
                        self.dual_net.train()
                        for Xtrain in self.train_loader:
                            start_time = time.time()
                            self.dual_optim.zero_grad()
                            mu, lamb = self.dual_net(Xtrain[0])
                            mu_k, lamb_k = frozen_dual_net(Xtrain[0])
                            y = frozen_primal_net(Xtrain[0])
                            train_loss = self.dual_loss(Xtrain[0], y, mu, lamb, mu_k, lamb_k, rhs_eq=None, rhs_ineq=None, targets=None, log_type="train")
                            train_loss.mean().backward()
                            self.dual_optim.step()

                        if self.logger is not None and self.step % 10 == 0:
                            self.logger.log_dual(self.step)

                        # Evaluate validation loss every epoch, and update learning rate
                        self.dual_net.eval()
                        curr_val_loss = 0
                        for Xvalid in self.valid_loader:
                            y = frozen_primal_net(Xvalid[0])
                            mu_valid, lamb_valid = self.dual_net(Xvalid[0])
                            mu_k_valid, lamb_k_valid = frozen_dual_net(Xvalid[0])
                            curr_val_loss += self.dual_loss(Xvalid[0], y, mu_valid, lamb_valid, mu_k_valid, lamb_k_valid, rhs_eq=None, rhs_ineq=None, targets=None, log_type="valid").mean()
                        
                        curr_val_loss /= len(self.valid_loader)
                        # Normalize by rho, so that the schedular still works correctly if rho is increased
                        self.dual_scheduler.step(torch.sign(curr_val_loss) * (torch.abs(curr_val_loss) / self.rho))

                    ##### Evaluate #####

                    self.primal_net.eval()
                    self.dual_net.eval()
                    
                    for Xtrain in self.train_loader:
                        # Xtrain = Xtrain[0].to(DEVICE)
                        self.eval_pdl(Xtrain[0], self.primal_net, self.dual_net, 'train', epoch_stats, targets=None)

                    # Get valid loss
                    for Xvalid in self.valid_loader:
                        # Xvalid = Xvalid[0].to(DEVICE)
                        self.eval_pdl(Xvalid[0], self.primal_net, self.dual_net, 'valid', epoch_stats, targets=None)

                    end_time = time.time()
                    stats = epoch_stats
                    print(f"Epoch {k} done. Time taken: {end_time - begin_time}. Rho: {self.rho}. Primal LR: {self.primal_optim.param_groups[0]['lr']}, Dual LR: {self.dual_optim.param_groups[0]['lr']}")
                    print(
                        # '{}: p-loss: {:.4E}, d-loss: {:.4E}, obj. val {:.4E}, Max eq.: {:.4E}, Max ineq.: {:.4E}, Mean eq.: {:.4E}, Mean ineq.: {:.4E}\n'.format(
                        '{}: p-loss: {:.4E}, obj. val {:.4E}, Max eq.: {:.4E}, Max ineq.: {:.4E}, Mean eq.: {:.4E}, Mean ineq.: {:.4E}\n'.format(
                            k, np.mean(epoch_stats['valid_primal_loss']),
                            # np.mean(epoch_stats['valid_dual_loss']),
                            np.mean(epoch_stats['valid_eval']),
                            np.mean(epoch_stats['valid_eq_max']),
                            np.mean(epoch_stats['valid_ineq_max']),
                            np.mean(epoch_stats['valid_eq_mean']),
                            np.mean(epoch_stats['valid_ineq_mean']))
                    )
                    # Write to tensorboard
                    if self.logger is not None:
                        # Eval logg
                        self.logger.add_eval_log("valid", epoch_stats['valid_eval'], epoch_stats['valid_eq_max'], epoch_stats['valid_ineq_max'], epoch_stats['valid_ineq_mean'], epoch_stats["valid_eq_mean"])
                        self.logger.add_eval_log("train", epoch_stats['train_eval'], epoch_stats['train_eq_max'], epoch_stats['train_ineq_max'], epoch_stats['train_ineq_mean'], epoch_stats["train_eq_mean"])
                        self.logger.add_param_log(v_k, self.rho, self.primal_optim.param_groups[0]['lr'], self.dual_optim.param_groups[0]['lr'])
                        self.logger.log_eval(self.step)
                        self.logger.log_param(self.step)

                    # Update rho from the second iteration onward.
                    if k > 0 and v_k > self.tau * prev_v_k:
                        self.rho = np.min([self.alpha * self.rho, self.rho_max])
                    prev_v_k = v_k
        
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

    def primal_loss(self, X, y, mu, lamb, targets=None, log_type=None):
        obj = self.data.obj_fn(y)
        
        # g(y)
        ineq = self.data.ineq_resid(X, y)
        # h(y)
        eq = self.data.eq_resid(X, y)

        # ! Clamp mu!
        # Element-wise clamping of mu_i when g_i (ineq) is negative
        mu = torch.where(ineq < 0, torch.zeros_like(mu), mu)

        lagrange_ineq = torch.sum(mu * ineq, dim=1)  # Shape (batch_size,)

        lagrange_eq = torch.sum(lamb * eq, dim=1)   # Shape (batch_size,)

        violation_ineq = torch.sum(torch.maximum(ineq, torch.zeros_like(ineq)) ** 2, dim=1)
        violation_eq = torch.sum(eq ** 2, dim=1)
        penalty = self.rho/2 * (violation_ineq + violation_eq)

        # loss = (obj / 1e5 + (lagrange_ineq + lagrange_eq + penalty) / 1e6)
        loss = (obj + (lagrange_ineq + lagrange_eq + penalty))
        # if log_type == 'train' and (self.step in [1, 1001, 2001, 3001, 4001, 5001, 6001, 7001, 8001, 9001, 9999]  ):
        #     print(f"ratio obj, ineq, eq, penalty: {(obj/loss).item()}, {(lagrange_ineq/loss).item()}, {(lagrange_eq/loss).item()}, {(penalty/loss).item()}")

        if self.logger is not None and log_type is not None:
            self.logger.add_primal_log(X=X, log_type=log_type, loss=loss.detach(), penalty=penalty.detach(), violation_ineq=violation_ineq.detach(), violation_eq=violation_eq.detach(), mu=mu.detach(), lamb=lamb.detach(), ineq=ineq.detach(), eq=eq.detach(), lagrange_ineq=lagrange_ineq.detach(), lagrange_eq=lagrange_eq.detach(), y=y.detach(), target=targets)

        return loss

    def dual_loss(self, X, y, mu, lamb, mu_k, lamb_k, rhs_eq=None, rhs_ineq=None, targets=None, log_type=None):
        # mu = [batch, g]
        # lamb = [batch, h]

        # g(y)
        ineq = self.data.ineq_resid(X, y) # [batch, g]
        # h(y)
        eq = self.data.eq_resid(X, y)   # [batch, h]

        target_mu = torch.maximum(mu_k + self.rho * ineq, torch.zeros_like(ineq))

        dual_resid_ineq = mu - target_mu # [batch, g]

        dual_resid_ineq = torch.norm(dual_resid_ineq, dim=1)  # [batch]

        # Compute the dual residuals for equality constraints
        dual_resid_eq = lamb - (lamb_k + self.rho * eq)
        dual_resid_eq = torch.norm(dual_resid_eq, dim=1)  # Norm along constraint dimension

        loss = (dual_resid_ineq + dual_resid_eq)

        if self.problem_type == "GEP":
            dual_obj = self.data.dual_obj_fn(mu.detach(), lamb.detach(), rhs_eq.detach(), rhs_ineq.detach())

        if self.logger is not None and log_type is not None:
            if self.problem_type == "GEP":
                self.logger.add_dual_log(log_type, loss.detach(), mu.detach(), mu_k.detach(), lamb.detach(), lamb_k.detach(), ineq.detach(), eq.detach(), dual_resid_ineq.detach(), dual_resid_eq.detach(), dual_obj.detach(), targets,)
            else:    
                self.logger.add_dual_log(log_type, loss.detach(), mu.detach(), mu_k.detach(), lamb.detach(), lamb_k.detach(), ineq.detach(), eq.detach(), dual_resid_ineq.detach(), dual_resid_eq.detach(), None, targets)

        return loss

    def violation(self, X, y, mu_k):
        # Calculate the equality constraint function h_x(y)
        eq = self.data.eq_resid(X, y)  # Assume shape (num_samples, n_eq)
        
        # Calculate the infinity norm of h_x(y)
        eq_inf_norm = torch.abs(eq).max(dim=1).values  # Shape: (num_samples,)

        # Calculate the inequality constraint function g_x(y)
        ineq = self.data.ineq_resid(X, y)  # Assume shape (num_samples, n_ineq)
        
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
    def eval_pdl(self, X, primal_net, dual_net, prefix, stats, targets=None):

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
        self.dict_agg(stats, make_prefix('primal_loss'), self.primal_loss(X, Y, mu, lamb, targets, log_type=None).detach().cpu().numpy())
        # self.dict_agg(stats, make_prefix('dual_loss'), self.dual_loss(X, Y, mu, lamb, mu_k, lamb_k, log_type=None).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('eval'), self.data.obj_fn(Ycorr).detach().cpu().numpy())

        self.dict_agg(stats, make_prefix('ineq_max'), torch.max(self.data.ineq_dist(X, Ycorr), dim=1)[0].detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('ineq_mean'), torch.mean(self.data.ineq_dist(X, Ycorr), dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('ineq_num_viol_0'),
                torch.sum(self.data.ineq_dist(X, Ycorr) > eps_converge, dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('ineq_num_viol_1'),
                torch.sum(self.data.ineq_dist(X, Ycorr) > 10 * eps_converge, dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('ineq_num_viol_2'),
                torch.sum(self.data.ineq_dist(X, Ycorr) > 100 * eps_converge, dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('eq_max'),
                torch.max(torch.abs(self.data.eq_resid(X, Ycorr)), dim=1)[0].detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('eq_mean'), torch.mean(torch.abs(self.data.eq_resid(X, Ycorr)), dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('eq_num_viol_0'),
                torch.sum(torch.abs(self.data.eq_resid(X, Ycorr)) > eps_converge, dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('eq_num_viol_1'),
                torch.sum(torch.abs(self.data.eq_resid(X, Ycorr)) > 10 * eps_converge, dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('eq_num_viol_2'),
                torch.sum(torch.abs(self.data.eq_resid(X, Ycorr)) > 100 * eps_converge, dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('raw_time'), raw_end_time - start_time, op='sum')
        self.dict_agg(stats, make_prefix('raw_eval'), self.data.obj_fn(Y).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('raw_ineq_max'), torch.max(self.data.ineq_dist(X, Y), dim=1)[0].detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('raw_ineq_mean'), torch.mean(self.data.ineq_dist(X, Y), dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('raw_ineq_num_viol_0'),
                torch.sum(self.data.ineq_dist(X, Y) > eps_converge, dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('raw_ineq_num_viol_1'),
                torch.sum(self.data.ineq_dist(X, Y) > 10 * eps_converge, dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('raw_ineq_num_viol_2'),
                torch.sum(self.data.ineq_dist(X, Y) > 100 * eps_converge, dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('raw_eq_max'),
                torch.max(torch.abs(self.data.eq_resid(X, Y)), dim=1)[0].detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('raw_eq_mean'),
                torch.mean(torch.abs(self.data.eq_resid(X, Y)), dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('raw_eq_num_viol_0'),
                torch.sum(torch.abs(self.data.eq_resid(X, Y)) > eps_converge, dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('raw_eq_num_viol_1'),
                torch.sum(torch.abs(self.data.eq_resid(X, Y)) > 10 * eps_converge, dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('raw_eq_num_viol_2'),
                torch.sum(torch.abs(self.data.eq_resid(X, Y)) > 100 * eps_converge, dim=1).detach().cpu().numpy())

        return stats
    
class TensorBoardLogger:
    def __init__(self, data, save_dir, optimal_objective_train, optimal_objective_val):
        self.writer = SummaryWriter(log_dir=save_dir)
        self.data = data
        self.optimal_objective = {"train": optimal_objective_train, "valid": optimal_objective_val}
        self.init_primal_log_dict()
        self.init_dual_log_dict()
        self.init_eval_log_dict()
        self.init_param_log_dict()

    def close(self):
        self.writer.close()

    def init_primal_log_dict(self):
        self.primal_log_dict = {
            "train": {
                "predicted": {key: [] for key in [
                    "loss", "penalty", "violation_ineq", "violation_eq",
                    "mu", "lamb", "ineq", "eq", "lagrange_ineq", "lagrange_eq", "y"
                ]},
                "target": {key: [] for key in [
                    "mu", "lamb", "ineq", "eq", "y", "obj"
                ]}
            },
            "valid": {
                "predicted": {key: [] for key in [
                    "loss", "penalty", "violation_ineq", "violation_eq",
                    "mu", "lamb", "ineq", "eq", "lagrange_ineq", "lagrange_eq", "y"
                ]},
                "target": {key: [] for key in [
                    "mu", "lamb", "ineq", "eq", "y", "obj"
                ]}
            },
        }

    def init_dual_log_dict(self):
        self.dual_log_dict = {
            "train": {
                "predicted": {key: [] for key in [
                    "loss", "mu", "mu_k", "lamb", "lamb_k",
                    "ineq", "eq", "dual_resid_ineq", "dual_resid_eq", "dual_obj"
                ]}, 
                "target": {key: [] for key in [
                    "mu", "lamb", "ineq", "eq"
                ]}
            },
            "valid": {
                "predicted": {key: [] for key in [
                    "loss", "mu", "mu_k", "lamb", "lamb_k",
                    "ineq", "eq", "dual_resid_ineq", "dual_resid_eq", "dual_obj"
                ]}, 
                "target": {key: [] for key in [
                    "mu", "lamb", "ineq", "eq"
                ]}
            }
        }

    def init_eval_log_dict(self):
        self.eval_log_dict = {
            "train": {
                "predicted": {key: [] for key in [
                    "obj", "eq_max", "ineq_max", "ineq_mean", "eq_mean"
                ]},
                "target": {key: [] for key in [
                    "obj"
                ]}
            },
            "valid": {
                "predicted": {key: [] for key in [
                    "obj", "eq_max", "ineq_max", "ineq_mean", "eq_mean"
                ]},
                "target": {key: [] for key in [
                    "obj"
                ]}
            }
        }

    def init_param_log_dict(self):
        self.param_log_dict = {
            key: [] for key in ["v_k", "rho", "primal_lr", "dual_lr"]
        }

    def add_primal_log(self, X, log_type, loss, penalty, violation_ineq, violation_eq, mu, lamb, ineq, eq, lagrange_ineq, lagrange_eq, y, target=None):
        with torch.no_grad():
            for key, value in zip(
                ["loss", "penalty", "violation_ineq", "violation_eq", "mu", "lamb", 
                 "ineq", "eq", "lagrange_ineq", "lagrange_eq", "y"],
                [loss, penalty, violation_ineq, violation_eq, mu, lamb, 
                 ineq, eq, lagrange_ineq, lagrange_eq, y]
            ):
                self.primal_log_dict[log_type]["predicted"][key].append(value.to(DEVICE))
            # "mu", "lamb", "ineq", "eq", "y"
            if target:
                self.primal_log_dict[log_type]["target"]["mu"].append(target["mu"])
                self.primal_log_dict[log_type]["target"]["lamb"].append(target["lamb"])
                self.primal_log_dict[log_type]["target"]["ineq"].append(self.data.ineq_resid(X, target["y"]))
                self.primal_log_dict[log_type]["target"]["eq"].append(self.data.eq_resid(X, target["y"]))
                self.primal_log_dict[log_type]["target"]["y"].append(target["y"])
                self.primal_log_dict[log_type]["target"]["obj"].append(target["obj"])

    def add_dual_log(self, log_type, loss, mu, mu_k, lamb, lamb_k, ineq, eq, dual_resid_ineq, dual_resid_eq, dual_obj=None, target=None):
        with torch.no_grad():
            for key, value in zip(
                ["loss", "mu", "mu_k", "lamb", "lamb_k", "ineq", "eq", 
                 "dual_resid_ineq", "dual_resid_eq", "dual_obj"],
                [loss, mu, mu_k, lamb, lamb_k, ineq, eq, dual_resid_ineq, dual_resid_eq, dual_obj]
            ):  
                if value is not None:
                    self.dual_log_dict[log_type]["predicted"][key].append(value.to(DEVICE))
            
            if target:
                # "mu", "lamb", "ineq", "eq"
                self.dual_log_dict[log_type]["target"]["mu"].append(target["mu"])
                self.dual_log_dict[log_type]["target"]["lamb"].append(target["lamb"])
                self.dual_log_dict[log_type]["target"]["ineq"].append(target["ineq"])
                self.dual_log_dict[log_type]["target"]["eq"].append(target["eq"])

    def add_eval_log(self, log_type, obj, eq_max, ineq_max, ineq_mean, eq_mean):
        with torch.no_grad():
            for key, value in zip(
                ["obj", "eq_max", "ineq_max", "ineq_mean", "eq_mean"],
                [obj, eq_max, ineq_max, ineq_mean, eq_mean]
            ):
                self.eval_log_dict[log_type]["predicted"][key].append(torch.tensor(value, device=DEVICE))
            # self.eval_log_dict[log_type]["target"]["obj"].append(target["obj"])
    
    def add_param_log(self, v_k, rho, primal_lr, dual_lr):
        #Append the parameters to their respective lists
        with torch.no_grad():
            self.param_log_dict["v_k"].append(v_k)
            self.param_log_dict["rho"].append(rho)
            self.param_log_dict["primal_lr"].append(primal_lr)
            self.param_log_dict["dual_lr"].append(dual_lr)

    def log_primal(self, step):
        for log_type in ["train", "valid"]:
            for key in self.primal_log_dict[log_type]["predicted"]:
                if self.primal_log_dict[log_type]["predicted"][key]:
                    if key == "y":
                        concatenated_values = torch.cat(self.primal_log_dict[log_type]["predicted"][key], dim=0)  # Concatenate across batches
                        mean_value = self.data.obj_fn(concatenated_values).mean().item()
                        self.writer.add_scalar(f"{log_type.capitalize()}_Primal/obj", mean_value, step)
                        if self.optimal_objective[log_type]:
                            self.writer.add_scalar(f"{log_type.capitalize()}_Primal/opt_gap", 
                                                (mean_value - self.optimal_objective[log_type]) / self.optimal_objective[log_type], step)
                    else:
                        concatenated_values = torch.cat(self.primal_log_dict[log_type]["predicted"][key], dim=0)  # Concatenate across batches
                        mean_value = concatenated_values.mean().item()
                        self.writer.add_scalar(f"{log_type.capitalize()}_Primal/{key}", mean_value, step)

        self.init_primal_log_dict()

    def log_dual(self, step):
        for log_type in ["train", "valid"]:
            for key in self.dual_log_dict[log_type]["predicted"]:
                if self.dual_log_dict[log_type]["predicted"][key]:
                    concatenated_values = torch.cat(self.dual_log_dict[log_type]["predicted"][key], dim=0)  # Concatenate across batches
                    mean_value = concatenated_values.mean().item()
                    self.writer.add_scalar(f"{log_type.capitalize()}_Dual/{key}", mean_value, step)

            for key in self.dual_log_dict[log_type]["target"]:
                if self.dual_log_dict[log_type]["target"][key]:
                    if key == "obj":
                        pass
                    else:
                        concatenated_values = torch.cat(self.dual_log_dict[log_type]["predicted"][key], dim=0)  # Concatenate across batches
                        concatenated_targets = torch.cat(self.dual_log_dict[log_type]["target"][key], dim=0)
                        diff = concatenated_values - concatenated_targets
                        mean_differences = diff.abs().mean().item()
                        mean_values = concatenated_targets.mean().item()
                        self.writer.add_scalar(f"Targets_Dual_{log_type}/{key}_diff", mean_differences, step)
                        self.writer.add_scalar(f"Targets_Dual_{log_type}/{key}_value", mean_values, step)

        # g = torch.cat(self.dual_log_dict["train"]["predicted"]["ineq"])
        # mu = torch.cat(self.dual_log_dict["train"]["predicted"]["mu"])
        # satisfied = g < 0
        # unsatisfied = ~satisfied
        # self.writer.add_scalar(f"Train_Dual/ratio_g_satisfied", torch.sum(satisfied) / (torch.sum(unsatisfied) + torch.sum(satisfied)), step)
        # self.writer.add_scalar(f"Train_Dual/mean_mag_mu_unsatisfied", torch.mean(torch.abs(mu[unsatisfied])), step)
        # self.writer.add_scalar(f"Train_Dual/mean_mag_mu_satisfied", torch.mean(torch.abs(mu[satisfied])), step)

        self.init_dual_log_dict()

    def log_eval(self, step):
        for log_type in ["train", "valid"]:
            for key in self.eval_log_dict[log_type]["predicted"]:
                if self.eval_log_dict[log_type]["predicted"][key]:
                    concatenated_values = torch.cat(self.eval_log_dict[log_type]["predicted"][key], dim=0)  # Concatenate across batches
                    mean_value = concatenated_values.mean().item()
                    self.writer.add_scalar(f"{log_type.capitalize()}_Eval/{key}", mean_value, step)

            # Compute the mean of the individual optimality gaps
            if self.optimal_objective[log_type]:
                obj_values = torch.cat(self.eval_log_dict[log_type]["predicted"]["obj"], dim=0)  # Concatenate obj values
                opt_gaps = (obj_values - self.optimal_objective[log_type]) / self.optimal_objective[log_type]
                mean_opt_gap = opt_gaps.mean().item()  # Mean of the optimality gaps

                self.writer.add_scalar(
                    f"{log_type.capitalize()}_Eval/opt_gap",
                    mean_opt_gap,
                    step,
                )
        self.init_eval_log_dict()

    def log_param(self, step):
        for key, values in self.param_log_dict.items():
            if values:
                mean_value = sum(values) / len(values)
                self.writer.add_scalar(f"Params/{key}", mean_value, step)
        self.init_param_log_dict()

    def log_weights_gradients(self, model_name, model, step):
        for name, param in model.named_parameters():
            # self.writer.add_scalar(f"{model_name}/weights_mean/{name}", param.data.mean(), step)
            self.writer.add_scalar(f"{model_name}/weights_norm/{name}", param.data.norm(), step)
            if param.grad is not None:
                # self.writer.add_scalar(f"{model_name}/gradients_mean/{name}", param.grad.mean(), step)
                self.writer.add_scalar(f"{model_name}/gradients_norm/{name}", param.grad.norm(), step)

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
    
    def forward(self, x):
        return self.net(x)
    
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
    
    def forward(self, x):
        out = self.net(x)
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
    

         
