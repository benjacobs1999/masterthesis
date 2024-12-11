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


class PrimalDualTrainer():

    def __init__(self, data, args, save_dir, optimal_objective):
        self.data = data
        self.args = args
        self.save_dir = save_dir
        self.logger = TensorBoardLogger(data, save_dir, optimal_objective)

        self.K = args["K"]
        self.L = args["L"]
        self.tau = args["tau"]
        self.rho = args["rho"]
        self.rho_max = args["rho_max"]
        self.alpha = args["alpha"]
        self.batch_size = args["batch_size"]
        self.hidden_size = args["hidden_size"]

        self.primal_lr = args["primal_lr"]
        self.dual_lr = args["dual_lr"]
        self.decay = args["decay"]
        self.patience = args["patience"]
        
        # for logging
        self.step = 0
        v_k = 0


        self.train_dataset = TensorDataset(self.data.trainX.to(DEVICE))
        self.valid_dataset = TensorDataset(self.data.validX.to(DEVICE))
        self.test_dataset = TensorDataset(self.data.testX.to(DEVICE))

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=len(self.valid_dataset))
        self.test_loader = DataLoader(self.test_dataset, batch_size=len(self.test_dataset))

        self.primal_net = PrimalNet(self.data, self.hidden_size).to(dtype=DTYPE, device=DEVICE)
        # dual_net = DualNetTwoOutputLayers(data, hidden_size).to(dtype=DTYPE, device=DEVICE)
        self.dual_net = DualNet(self.data, self.hidden_size).to(dtype=DTYPE, device=DEVICE)

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
        # For logging:

        try:

            for k in range(self.K):
                begin_time = time.time()
                epoch_stats = {}
                frozen_dual_net = copy.deepcopy(self.dual_net)
                frozen_dual_net.eval()

                self.primal_net.train()
                for l1 in range(self.L):
                    self.step += 1
                    # Update primal net using primal loss
                    for Xtrain in self.train_loader:
                        Xtrain = Xtrain[0].to(DEVICE)
                        start_time = time.time()
                        self.primal_optim.zero_grad()
                        y = self.primal_net(Xtrain)
                        mu, lamb = frozen_dual_net(Xtrain)
                        train_loss = self.primal_loss(Xtrain, y, mu, lamb, "train")
                        train_loss.mean().backward()
                        self.primal_optim.step()
                        train_time = time.time() - start_time
                        if self.logger is not None and self.step % 10 == 0:
                            self.logger.log_primal(self.step)

                    # Log weights and gradients
                    # for i, param in enumerate(primal_net.parameters()):
                        # Log weights
                        # weight_norm = param.data.norm(2).item()
                        # writer.add_scalar(f"Weights/primal/layer_{i}", weight_norm, k*self.L + l1)
                        
                        # Log gradients
                        # if param.grad is not None:
                            # grad_norm = param.grad.norm(2).item()
                            # writer.add_scalar(f"Gradients/primal/layer_{i}", grad_norm, k*self.L + l1)

                    # Evaluate validation loss every epoch, and update learning rate
                    curr_val_loss = 0
                    self.primal_net.eval()
                    for Xvalid in self.valid_loader:
                        Xvalid = Xvalid[0].to(DEVICE)
                        y = self.primal_net(Xvalid)
                        mu, lamb = frozen_dual_net(Xvalid)
                        loss = self.primal_loss(Xvalid, y, mu, lamb, log_type="valid").sum()
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

                self.dual_net.train()
                for l2 in range(self.L):
                    self.step += 1
                    # Update dual net using dual loss
                    for Xtrain in self.train_loader:
                        Xtrain = Xtrain[0].to(DEVICE)
                        start_time = time.time()
                        self.dual_optim.zero_grad()
                        mu, lamb = self.dual_net(Xtrain)
                        mu_k, lamb_k = frozen_dual_net(Xtrain)
                        y = frozen_primal_net(Xtrain)
                        train_loss = self.dual_loss(Xtrain, y, mu, lamb, mu_k, lamb_k, "train")
                        # train_loss.sum().backward()
                        train_loss.mean().backward()
                        self.dual_optim.step()
                        if self.logger is not None and self.step % 10 == 0:
                            self.logger.log_dual(self.step)

                    # # Log weights and gradients
                    # for i, param in enumerate(dual_net.parameters()):
                    #     # Log weights
                    #     weight_norm = param.data.norm(2).item()
                    #     writer.add_scalar(f"Weights/dual/layer_{i}", weight_norm, k*self.L + l2)
                        
                    #     # Log gradients
                    #     if param.grad is not None:
                    #         grad_norm = param.grad.norm(2).item()
                    #         writer.add_scalar(f"Gradients/dual/layer_{i}", grad_norm, k*self.L + l2)

                    # Evaluate validation loss every epoch, and update learning rate
                    self.dual_net.eval()
                    curr_val_loss = 0
                    for Xvalid in self.valid_loader:
                        Xvalid = Xvalid[0].to(DEVICE)
                        y = frozen_primal_net(Xvalid)
                        mu_valid, lamb_valid = self.dual_net(Xvalid)
                        mu_k_valid, lamb_k_valid = frozen_dual_net(Xvalid)
                        curr_val_loss += self.dual_loss(Xvalid, y, mu_valid, lamb_valid, mu_k_valid, lamb_k_valid, log_type="valid").sum()
                    
                    curr_val_loss /= len(self.valid_loader)
                    # Normalize by rho, so that the schedular still works correctly if rho is increased
                    self.dual_scheduler.step(torch.sign(curr_val_loss) * (torch.abs(curr_val_loss) / self.rho))
                

                # self.dict_agg(epoch_stats, 'train_loss_dual', train_loss.detach().cpu().numpy())

                ##### Evaluate #####

                self.primal_net.eval()
                self.dual_net.eval()
                
                for Xtrain in self.train_loader:
                    Xtrain = Xtrain[0].to(DEVICE)
                    self.eval_pdl(Xtrain, self.primal_net, self.dual_net, 'train', epoch_stats)

                # Get valid loss
                for Xvalid in self.valid_loader:
                    Xvalid = Xvalid[0].to(DEVICE)
                    self.eval_pdl(Xvalid, self.primal_net, self.dual_net, 'valid', epoch_stats)

                # # Get test loss
                # primal_net.eval()
                # for Xtest in test_loader:
                #     Xtest = Xtest[0].to(DEVICE)
                #     self.eval_pdl(Xtest, primal_net, 'test', epoch_stats)

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
                # writer.add_scalar("Eval/Objective_Value", np.mean(epoch_stats['valid_eval']), 2*k*self.L + l1 + l2+2)
                # writer.add_scalar("Eval/Max_Equality", np.mean(epoch_stats['valid_eq_max']), 2*k*self.L + l1 + l2+2)
                # writer.add_scalar("Eval/Max_Inequality", np.mean(epoch_stats['valid_ineq_max']), 2*k*self.L + l1 + l2+2)
                # writer.add_scalar("Eval/Mean_Equality", np.mean(epoch_stats['valid_eq_mean']), 2*k*self.L + l1 + l2+2)
                # writer.add_scalar("Eval/Mean_Inequality", np.mean(epoch_stats['valid_ineq_mean']), 2*k*self.L + l1 + l2+2)


                # Log additional metrics
                # writer.add_scalar("Parameters/Violation", v_k, step)
                # writer.add_scalar("Parameters/Rho", rho, step)
                # writer.add_scalar("Parameters/Primal_LR", primal_optim.param_groups[0]['lr'], step)
                # writer.add_scalar("Parameters/Dual_LR", dual_optim.param_groups[0]['lr'], step)


                # Update rho from the second iteration onward.
                if k > 0 and v_k > self.tau * prev_v_k:
                    self.rho = np.min([self.alpha * self.rho, self.rho_max])
                prev_v_k = v_k
        
        finally:
            # Ensure writer is closed even if an exception occurs
            self.logger.close()

        with open(os.path.join(self.save_dir, 'stats.dict'), 'wb') as f:
            pickle.dump(stats, f)
        with open(os.path.join(self.save_dir, 'primal_net.dict'), 'wb') as f:
            torch.save(self.primal_net.state_dict(), f)
        with open(os.path.join(self.save_dir, 'dual_net.dict'), 'wb') as f:
            torch.save(self.dual_net.state_dict(), f)

        return self.primal_net, self.dual_net, stats

    def primal_loss(self, X, y, mu, lamb, log_type=None):
        obj = self.data.obj_fn(y)
        
        # g(y)
        ineq = self.data.g(X, y)
        # h(y)
        eq = self.data.h(X, y)

        # ! Clamp mu!
        # Element-wise clamping of mu_i when g_i (ineq) is negative
        mu = torch.where(ineq < 0, torch.zeros_like(mu), mu)

        lagrange_ineq = torch.sum(mu * ineq, dim=1)  # Shape (batch_size,)

        lagrange_eq = torch.sum(lamb * eq, dim=1)   # Shape (batch_size,)

        violation_ineq = torch.sum(torch.maximum(ineq, torch.zeros_like(ineq)) ** 2, dim=1)
        violation_eq = torch.sum(eq ** 2, dim=1)
        penalty = self.rho/2 * (violation_ineq + violation_eq)

        # ! Scale obj:
        # loss = ((obj / OPTIMAL_OBJ) + lagrange_ineq + lagrange_eq + penalty)
        loss = (obj + lagrange_ineq + lagrange_eq + penalty)

        if self.logger is not None and log_type is not None:
            self.logger.add_primal_log(log_type, loss.detach(), penalty.detach(), violation_ineq.detach(), violation_eq.detach(), mu.detach(), lamb.detach(), ineq.detach(), eq.detach(), lagrange_ineq.detach(), lagrange_eq.detach(), y.detach())

            # writer.add_scalar(f"Primal_loss/{writer_title}/loss", loss.detach().mean().item(), writer_steps)
            # writer.add_scalar(f"Primal_loss/{writer_title}/penalty", penalty.detach().mean().item(), writer_steps)
            # writer.add_scalar(f"Primal_loss/{writer_title}/penalty_g", violation_ineq.detach().mean().item(), writer_steps)
            # writer.add_scalar(f"Primal_loss/{writer_title}/penalty_h", violation_eq.detach().mean().item(), writer_steps)
            # writer.add_scalar(f"Primal_loss/{writer_title}/mu", mu.detach().mean().item(), writer_steps)
            # writer.add_scalar(f"Primal_loss/{writer_title}/lamb", lamb.detach().mean().item(), writer_steps)
            # writer.add_scalar(f"Primal_loss/{writer_title}/g", ineq.detach().mean().item(), writer_steps)
            # writer.add_scalar(f"Primal_loss/{writer_title}/h", eq.detach().mean().item(), writer_steps)
            # writer.add_scalar(f"Primal_loss/{writer_title}/mu*g", lagrange_ineq.detach().mean().item(), writer_steps)
            # writer.add_scalar(f"Primal_loss/{writer_title}/lamb*h", lagrange_eq.detach().mean().item(), writer_steps)
            # writer.add_scalar(f"Obj_Function/{writer_title}/obj", data.obj_fn(y.detach()).mean().item(), writer_steps)
            # writer.add_scalar(f"Obj_Function/{writer_title}/optimality_gap", (data.obj_fn(y.detach()).mean().item() - OPTIMAL_OBJ) / OPTIMAL_OBJ, writer_steps)

        return loss

    def dual_loss(self, X, y, mu, lamb, mu_k, lamb_k, log_type=None):
        # mu = [batch, g]
        # lamb = [batch, h]

        # g(y)
        ineq = self.data.g(X, y) # [batch, g]
        # h(y)
        eq = self.data.h(X, y)   # [batch, h]

        target_mu = torch.maximum(mu_k + self.rho * ineq, torch.zeros_like(ineq))

        dual_resid_ineq = mu - target_mu # [batch, g]

        dual_resid_ineq = torch.norm(dual_resid_ineq, dim=1)  # [batch]

        # Compute the dual residuals for equality constraints
        dual_resid_eq = lamb - (lamb_k + self.rho * eq)
        dual_resid_eq = torch.norm(dual_resid_eq, dim=1)  # Norm along constraint dimension

        loss = (dual_resid_ineq + dual_resid_eq)

        if self.logger is not None and log_type is not None:
            self.logger.add_dual_log(log_type, loss.detach(), mu.detach(), mu_k.detach(), lamb.detach(), lamb_k.detach(), ineq.detach(), eq.detach(), dual_resid_ineq.detach(), dual_resid_eq.detach())
            # writer.add_scalar(f"Dual_loss/{writer_title}/loss", loss.detach().mean().item(), writer_steps)
            # writer.add_scalar(f"Dual_loss/{writer_title}/mu", mu.detach().mean().item(), writer_steps)
            # writer.add_scalar(f"Dual_loss/{writer_title}/mu_k", mu_k.detach().mean().item(), writer_steps)
            # writer.add_scalar(f"Dual_loss/{writer_title}/lamb", lamb.detach().mean().item(), writer_steps)
            # writer.add_scalar(f"Dual_loss/{writer_title}/lamb_k", lamb_k.detach().mean().item(), writer_steps)
            # writer.add_scalar(f"Dual_loss/{writer_title}/g", ineq.detach().mean().item(), writer_steps)
            # writer.add_scalar(f"Dual_loss/{writer_title}/h", eq.detach().mean().item(), writer_steps)
            # writer.add_scalar(f"Dual_loss/{writer_title}/ineq_resid", dual_resid_ineq.detach().mean().item(), writer_steps)
            # writer.add_scalar(f"Dual_loss/{writer_title}/eq_resid", dual_resid_eq.detach().mean().item(), writer_steps)
            # writer.add_scalar(f"Obj_Function/{writer_title}/obj", data.obj_fn(y.detach()).mean().item(), writer_steps)
            # writer.add_scalar(f"Obj_Function/{writer_title}/optimality_gap", (data.obj_fn(y.detach()).mean().item() - OPTIMAL_OBJ) / OPTIMAL_OBJ, writer_steps)

        return loss

    def violation(self, X, y, mu_k):
        # Calculate the equality constraint function h_x(y)
        eq = self.data.h(X, y)  # Assume shape (num_samples, n_eq)
        
        # Calculate the infinity norm of h_x(y)
        eq_inf_norm = torch.abs(eq).max(dim=1).values  # Shape: (num_samples,)

        # Calculate the inequality constraint function g_x(y)
        ineq = self.data.g(X, y)  # Assume shape (num_samples, n_ineq)
        
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
    def eval_pdl(self, X, primal_net, dual_net, prefix, stats):

        eps_converge = self.args['corrEps']
        make_prefix = lambda x: "{}_{}".format(prefix, x)
        start_time = time.time()
        Y = primal_net(X)
        mu, lamb = dual_net(X)
        raw_end_time = time.time()
        Ycorr = Y

        # Ycorr, steps = grad_steps_all(data, X, Y, args)

        self.dict_agg(stats, make_prefix('time'), time.time() - start_time, op='sum')
        # self.dict_agg(stats, make_prefix('steps'), np.array([steps]))
        self.dict_agg(stats, make_prefix('primal_loss'), self.primal_loss(X, Y, mu, lamb, log_type=None).detach().cpu().numpy())
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
                torch.max(torch.abs(self.data.h(X, Ycorr)), dim=1)[0].detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('eq_mean'), torch.mean(torch.abs(self.data.h(X, Ycorr)), dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('eq_num_viol_0'),
                torch.sum(torch.abs(self.data.h(X, Ycorr)) > eps_converge, dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('eq_num_viol_1'),
                torch.sum(torch.abs(self.data.h(X, Ycorr)) > 10 * eps_converge, dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('eq_num_viol_2'),
                torch.sum(torch.abs(self.data.h(X, Ycorr)) > 100 * eps_converge, dim=1).detach().cpu().numpy())
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
                torch.max(torch.abs(self.data.h(X, Y)), dim=1)[0].detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('raw_eq_mean'),
                torch.mean(torch.abs(self.data.h(X, Y)), dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('raw_eq_num_viol_0'),
                torch.sum(torch.abs(self.data.h(X, Y)) > eps_converge, dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('raw_eq_num_viol_1'),
                torch.sum(torch.abs(self.data.h(X, Y)) > 10 * eps_converge, dim=1).detach().cpu().numpy())
        self.dict_agg(stats, make_prefix('raw_eq_num_viol_2'),
                torch.sum(torch.abs(self.data.h(X, Y)) > 100 * eps_converge, dim=1).detach().cpu().numpy())

        return stats
    
class TensorBoardLogger:
    def __init__(self, data, save_dir, optimal_objective):
        self.writer = SummaryWriter(log_dir=save_dir)
        self.data = data
        self.optimal_objective = optimal_objective
        self.init_primal_log_dict()
        self.init_dual_log_dict()
        self.init_eval_log_dict()
        self.init_param_log_dict()

    def close(self):
        self.writer.close()

    def init_primal_log_dict(self):
        self.primal_log_dict = {
            "train": {key: [] for key in [
                "loss", "penalty", "violation_ineq", "violation_eq",
                "mu", "lamb", "ineq", "eq", "lagrange_ineq", "lagrange_eq", "y"
            ]},
            "valid": {key: [] for key in [
                "loss", "penalty", "violation_ineq", "violation_eq",
                "mu", "lamb", "ineq", "eq", "lagrange_ineq", "lagrange_eq", "y"
            ]}
        }

    def init_dual_log_dict(self):
        self.dual_log_dict = {
            "train": {key: [] for key in [
                "loss", "mu", "mu_k", "lamb", "lamb_k",
                "ineq", "eq", "dual_resid_ineq", "dual_resid_eq"
            ]},
            "valid": {key: [] for key in [
                "loss", "mu", "mu_k", "lamb", "lamb_k",
                "ineq", "eq", "dual_resid_ineq", "dual_resid_eq"
            ]}
        }

    def init_eval_log_dict(self):
        self.eval_log_dict = {
            "train": {key: [] for key in [
                "obj", "eq_max", "ineq_max", "ineq_mean", "eq_mean"
            ]},
            "valid": {key: [] for key in [
                "obj", "eq_max", "ineq_max", "ineq_mean", "eq_mean"
            ]}
        }

    def init_param_log_dict(self):
        self.param_log_dict = {
            key: [] for key in ["v_k", "rho", "primal_lr", "dual_lr"]
        }

    def add_primal_log(self, log_type, loss, penalty, violation_ineq, violation_eq, mu, lamb, ineq, eq, lagrange_ineq, lagrange_eq, y):
        with torch.no_grad():
            for key, value in zip(
                ["loss", "penalty", "violation_ineq", "violation_eq", "mu", "lamb", 
                 "ineq", "eq", "lagrange_ineq", "lagrange_eq", "y"],
                [loss, penalty, violation_ineq, violation_eq, mu, lamb, 
                 ineq, eq, lagrange_ineq, lagrange_eq, y]
            ):
                self.primal_log_dict[log_type][key].append(value.to(DEVICE))

    def add_dual_log(self, log_type, loss, mu, mu_k, lamb, lamb_k, ineq, eq, dual_resid_ineq, dual_resid_eq):
        with torch.no_grad():
            for key, value in zip(
                ["loss", "mu", "mu_k", "lamb", "lamb_k", "ineq", "eq", 
                 "dual_resid_ineq", "dual_resid_eq"],
                [loss, mu, mu_k, lamb, lamb_k, ineq, eq, dual_resid_ineq, dual_resid_eq]
            ):
                self.dual_log_dict[log_type][key].append(value.to(DEVICE))

    def add_eval_log(self, log_type, obj, eq_max, ineq_max, ineq_mean, eq_mean):
        with torch.no_grad():
            for key, value in zip(
                ["obj", "eq_max", "ineq_max", "ineq_mean", "eq_mean"],
                [obj, eq_max, ineq_max, ineq_mean, eq_mean]
            ):
                self.eval_log_dict[log_type][key].append(torch.tensor(value, device=DEVICE))
    
    def add_param_log(self, v_k, rho, primal_lr, dual_lr):
        #Append the parameters to their respective lists
        with torch.no_grad():
            self.param_log_dict["v_k"].append(v_k)
            self.param_log_dict["rho"].append(rho)
            self.param_log_dict["primal_lr"].append(primal_lr)
            self.param_log_dict["dual_lr"].append(dual_lr)

    def log_primal(self, step):
        for log_type in ["train", "valid"]:
            for key in self.primal_log_dict[log_type]:
                if self.primal_log_dict[log_type][key]:
                    if key == "y":
                        mean_value = self.data.obj_fn(torch.concat(self.primal_log_dict[log_type][key])).mean().item()
                        self.writer.add_scalar(f"{log_type.capitalize()}_Primal/obj", mean_value, step)
                        self.writer.add_scalar(f"{log_type.capitalize()}_Primal/opt_gap", (mean_value - self.optimal_objective) / self.optimal_objective, step)
                    else:
                        stacked_tensor = torch.stack(self.primal_log_dict[log_type][key], dim=0)
                        mean_value = stacked_tensor.mean().item()
                        self.writer.add_scalar(f"{log_type.capitalize()}_Primal/{key}", mean_value, step)
        self.init_primal_log_dict()

    def log_dual(self, step):
        for log_type in ["train", "valid"]:
            for key in self.dual_log_dict[log_type]:
                if self.dual_log_dict[log_type][key]:
                    stacked_tensor = torch.stack(self.dual_log_dict[log_type][key], dim=0)
                    mean_value = stacked_tensor.mean().item()
                    self.writer.add_scalar(f"{log_type.capitalize()}_Dual/{key}", mean_value, step)
        self.init_dual_log_dict()

    def log_eval(self, step):
        for log_type in ["train", "valid"]:
            for key in self.eval_log_dict["train"]:
                if self.eval_log_dict[log_type][key]:
                    stacked_tensor = torch.stack(self.eval_log_dict[log_type][key], dim=0)
                    mean_value = stacked_tensor.mean().item()
                    self.writer.add_scalar(f"{log_type.capitalize()}_Eval/{key}", mean_value, step)
            # self.writer.add_scalar(f"{log_type.capitalize()}_Eval/opt_gap", (torch.stack(self.eval_log_dict[log_type]["obj"]).mean().item() - OPTIMAL_OBJ) / OPTIMAL_OBJ, step)
        self.init_eval_log_dict()

    def log_param(self, step):
        for key, values in self.param_log_dict.items():
            if values:
                mean_value = sum(values) / len(values)
                self.writer.add_scalar(f"Params/{key}", mean_value, step)
        self.init_param_log_dict()

class PrimalNet(nn.Module):
    def __init__(self, data, hidden_size):
        super().__init__()
        self._data = data
        self._hidden_size = hidden_size
        layer_sizes = [data.xdim, self._hidden_size, self._hidden_size]
        layers = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
        layers += [nn.Linear(layer_sizes[-1], data.ydim)]

        # All layers initialized.
        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

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
        self.out_layer_mu = nn.Linear(self._hidden_size, data.num_ineq_constraints)
        self.out_layer_lamb = nn.Linear(self._hidden_size, data.num_eq_constraints)
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
    
class DualNet(nn.Module):
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
        # output layer = [mu, lamb]
        self.out_layer = nn.Linear(self._hidden_size, self._data.num_ineq_constraints + self._data.num_eq_constraints)
        # Init last layers as 0, like in the paper
        nn.init.zeros_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)
        layers += [self.out_layer]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.net(x)
        out_mu = out[:, :self._data.num_ineq_constraints]
        out_lamb = out[:, self._data.num_ineq_constraints:]
        return out_mu, out_lamb
         
