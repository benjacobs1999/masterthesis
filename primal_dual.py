import copy
import os
import pickle
import time
from setproctitle import setproctitle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


DTYPE = torch.float32

torch.set_default_dtype(DTYPE)

import numpy as np

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)
# DEVICE = "cpu"
print(f"Running on {DEVICE}")

def train_PDL(data, args, save_dir):
    K = args["K"]
    L = args["L"]
    tau = args["tau"]
    rho = args["rho"]
    rho_max = args["rho_max"]
    alpha = args["alpha"]
    batch_size = args["batch_size"]
    hidden_size = args["hidden_size"]

    primal_lr = args["primal_lr"]
    dual_lr = args["dual_lr"]
    decay = args["decay"]
    patience = args["patience"]

    train_dataset = TensorDataset(data.trainX.to(DEVICE))
    #! There is only one problem, so we only have train set.
    # valid_dataset = TensorDataset(data.validX.to(DEVICE))
    # test_dataset = TensorDataset(data.testX.to(DEVICE))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    #! There is only one problem, so we only have train loader
    # valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))
    # test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    primal_net = PrimalNet(data, hidden_size).to(dtype=DTYPE, device=DEVICE)
    dual_net = DualNetTwoOutputLayers(data, hidden_size).to(dtype=DTYPE, device=DEVICE)

    primal_optim = torch.optim.Adam(primal_net.parameters(), lr=primal_lr)
    dual_optim = torch.optim.Adam(dual_net.parameters(), lr=dual_lr)

    # Add schedulers
    primal_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        primal_optim, mode='min', factor=decay, patience=patience
    )
    dual_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        dual_optim, mode='min', factor=decay, patience=patience
    )

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=save_dir)

    try:

        for k in range(K):
            begin_time = time.time()
            epoch_stats = {}
            frozen_dual_net = copy.deepcopy(dual_net)
            frozen_dual_net.eval()

            for l1 in range(L):
                # print(f"Outer: {k}, Primal inner: {l1}")
                # Update primal net using primal loss
                for Xtrain in train_loader:
                    Xtrain = Xtrain[0].to(DEVICE)
                    start_time = time.time()
                    primal_optim.zero_grad()
                    y = primal_net(Xtrain)
                    mu, lamb = frozen_dual_net(Xtrain)
                    train_loss = primal_loss(data, Xtrain, y, mu, lamb, rho, writer=writer, writer_title="Inner_Iterations", writer_steps=2*k*L + l1)
                    train_loss.mean().backward()
                    primal_optim.step()
                    train_time = time.time() - start_time

                # ! Cant do that right now, only train set.
                # Evaluate validation loss every epoch, and update learning rate
                # curr_val_loss = 0
                # primal_net.eval()
                # for Xvalid in valid_loader:
                #     Xvalid = Xvalid[0].to(DEVICE)
                #     y = primal_net(Xvalid)
                #     mu, lamb = frozen_dual_net(Xvalid)
                #     loss = primal_loss(data, Xvalid, y, mu, lamb, rho).sum()
                #     curr_val_loss += loss
                # curr_val_loss /= len(valid_loader)
                # # Normalize by rho, so that the schedular still works correctly if rho is increased
                # primal_scheduler.step(torch.sign(curr_val_loss) * (torch.abs(curr_val_loss) / rho))
                # Log primal loss
            
            dict_agg(epoch_stats, 'train_loss_primal', train_loss.detach().cpu().numpy())

            # Copy primal net into frozen primal net
            frozen_primal_net = copy.deepcopy(primal_net)
            frozen_primal_net.eval()

            # Calculate v_k
            y = frozen_primal_net(data.trainX.to(DEVICE))
            mu_k, lamb_k = frozen_dual_net(data.trainX.to(DEVICE))
            v_k = violation(data, data.trainX.to(DEVICE), y, mu_k, rho)

            for l2 in range(L):
                # Update dual net using dual loss
                # print(f"Outer: {k}, Dual inner: {l1}")
                for Xtrain in train_loader:
                    Xtrain = Xtrain[0].to(DEVICE)
                    start_time = time.time()
                    dual_optim.zero_grad()
                    mu, lamb = dual_net(Xtrain)
                    mu_k, lamb_k = frozen_dual_net(Xtrain)
                    y = frozen_primal_net(Xtrain)
                    train_loss = dual_loss(data, Xtrain, y, mu, lamb, mu_k, lamb_k, rho, writer=writer, writer_title="Inner_Iterations", writer_steps=2*k*L + l1 + l2)
                    # train_loss.sum().backward()
                    train_loss.mean().backward()
                    dual_optim.step()

                # ! Cant do that right now, only train set.
                # Evaluate validation loss every epoch, and update learning rate
                # dual_net.eval()
                # curr_val_loss = 0
                # for Xvalid in valid_loader:
                #     Xvalid = Xvalid[0].to(DEVICE)
                #     y = frozen_primal_net(Xvalid)
                #     mu, lamb = dual_net(Xvalid)
                #     mu_k, lamb_k = frozen_dual_net(Xvalid)
                #     curr_val_loss += dual_loss(data, Xvalid, y, mu, lamb, mu_k, lamb_k, rho).sum()
                
                # curr_val_loss /= len(valid_loader)
                # # Normalize by rho, so that the schedular still works correctly if rho is increased
                # dual_scheduler.step(torch.sign(curr_val_loss) * (torch.abs(curr_val_loss) / rho))

            dict_agg(epoch_stats, 'train_loss_dual', train_loss.detach().cpu().numpy())

            # Log additional metrics
            writer.add_scalar("Metrics/Violation", v_k, k)
            writer.add_scalar("Metrics/Rho", rho, k)
            writer.add_scalar("Metrics/Primal_LR", primal_optim.param_groups[0]['lr'], k)
            writer.add_scalar("Metrics/Dual_LR", dual_optim.param_groups[0]['lr'], k)

            ##### Evaluate #####
            # ! Evaluate on train set, because we don't have valid set.
            primal_net.eval()
            for Xvalid in train_loader:
                Xvalid = Xvalid[0].to(DEVICE)
                eval_pdl(data, Xvalid, primal_net, dual_net, mu_k, lamb_k, rho, args, 'valid', epoch_stats)


            # # Get valid loss
            # primal_net.eval()
            # for Xvalid in valid_loader:
            #     Xvalid = Xvalid[0].to(DEVICE)
            #     eval_pdl(data, Xvalid, primal_net, dual_net, mu_k, lamb_k, rho, args, 'valid', epoch_stats)

            # Get test loss
            # primal_net.eval()
            # for Xtest in test_loader:
            #     Xtest = Xtest[0].to(DEVICE)
            #     eval_pdl(data, Xtest, primal_net, args, 'test', epoch_stats)

            end_time = time.time()
            stats = epoch_stats
            print(f"Epoch {k} done. Time taken: {end_time - begin_time}. Rho: {rho}. Primal LR: {primal_optim.param_groups[0]['lr']}, Dual LR: {dual_optim.param_groups[0]['lr']}")
            print(
                '{}: p-loss: {:.4E}, d-loss: {:.4E}, obj. val {:.4E}, Max eq.: {:.4E}, Max ineq.: {:.4E}, Mean eq.: {:.4E}, Mean ineq.: {:.4E}\n'.format(
                    k, np.mean(epoch_stats['train_loss_primal']),
                    np.mean(epoch_stats['train_loss_dual']),
                    np.mean(epoch_stats['valid_eval']),
                    np.mean(epoch_stats['valid_eq_max']),
                    np.mean(epoch_stats['valid_ineq_max']),
                    np.mean(epoch_stats['valid_eq_mean']),
                    np.mean(epoch_stats['valid_ineq_mean']))
            )
            # Write to tensorboard
            writer.add_scalar("Eval/Objective_Value", np.mean(epoch_stats['valid_eval']), k)
            writer.add_scalar("Eval/Max_Equality", np.mean(epoch_stats['valid_eq_max']), k)
            writer.add_scalar("Eval/Max_Inequality", np.mean(epoch_stats['valid_ineq_max']), k)
            writer.add_scalar("Eval/Mean_Equality", np.mean(epoch_stats['valid_eq_mean']), k)
            writer.add_scalar("Eval/Mean_Inequality", np.mean(epoch_stats['valid_ineq_mean']), k)

            # Update rho from the second iteration onward.
            if k > 0 and v_k > tau * prev_v_k:
                rho = np.min([alpha * rho, rho_max])
            prev_v_k = v_k
    
    finally:
        # Ensure writer is closed even if an exception occurs
        writer.close()

    with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
        pickle.dump(stats, f)
    with open(os.path.join(save_dir, 'primal_net.dict'), 'wb') as f:
        torch.save(primal_net.state_dict(), f)
    with open(os.path.join(save_dir, 'dual_net.dict'), 'wb') as f:
        torch.save(dual_net.state_dict(), f)

    return primal_net, dual_net, stats

def primal_loss(data, X, y, mu, lamb, rho, writer=None, writer_title=None, writer_steps=None):
    obj = data.obj_fn(y)
    # g(y)
    ineq = data.g(y)
    # h(y)
    eq = data.h(y)

    lagrange_ineq = torch.sum(mu * ineq, dim=1)  # Shape (batch_size,)
    lagrange_eq = torch.sum(lamb * eq, dim=1)   # Shape (batch_size,)

    violation_ineq = torch.sum(torch.maximum(ineq, torch.zeros_like(ineq)) ** 2, dim=1)
    violation_eq = torch.sum(eq ** 2, dim=1)
    penalty = rho/2 * (violation_ineq + violation_eq)

    loss = obj + lagrange_ineq + lagrange_eq + penalty

    if writer is not None and writer_title is not None and writer_steps is not None:
        writer.add_scalar(f"Primal_Loss/{writer_title}", loss.detach().mean().item(), writer_steps)
        writer.add_scalar(f"Constraints/{writer_title}/mu", mu.detach().mean().item(), writer_steps)
        writer.add_scalar(f"Constraints/{writer_title}/lamb", lamb.detach().mean().item(), writer_steps)
        writer.add_scalar(f"Constraints/{writer_title}/g", ineq.detach().mean().item(), writer_steps)
        writer.add_scalar(f"Constraints/{writer_title}/h", eq.detach().mean().item(), writer_steps)
        writer.add_scalar(f"Constraints/{writer_title}/mu*g", lagrange_ineq.detach().mean().item(), writer_steps)
        writer.add_scalar(f"Constraints/{writer_title}/lamb*h", lagrange_eq.detach().mean().item(), writer_steps)
        writer.add_scalar(f"Obj_Function/{writer_title}", data.obj_fn(y.detach()).item(), writer_steps)

    return loss

def dual_loss(data, X, y, mu, lamb, mu_k, lamb_k, rho, writer=None, writer_title=None, writer_steps=None):
    # g(y)
    ineq = data.g(y)
    # h(y)
    eq = data.h(y)

    dual_resid_ineq = mu - torch.maximum(mu_k + rho * ineq, torch.zeros_like(ineq))
    dual_resid_ineq = torch.norm(dual_resid_ineq, dim=1)  # Norm along constraint dimension
    
    # Compute the dual residuals for equality constraints
    dual_resid_eq = lamb - (lamb_k + rho * eq)
    dual_resid_eq = torch.norm(dual_resid_eq, dim=1)  # Norm along constraint dimension

    loss = dual_resid_ineq + dual_resid_eq

    if writer is not None and writer_title is not None and writer_steps is not None:
        writer.add_scalar(f"Dual_Loss/{writer_title}", loss.detach().mean().item(), writer_steps)
        writer.add_scalar(f"Constraints/{writer_title}/mu", mu.detach().mean().item(), writer_steps)
        writer.add_scalar(f"Constraints/{writer_title}/lamb", lamb.detach().mean().item(), writer_steps)
        writer.add_scalar(f"Constraints/{writer_title}/g", ineq.detach().mean().item(), writer_steps)
        writer.add_scalar(f"Constraints/{writer_title}/h", eq.detach().mean().item(), writer_steps)
        writer.add_scalar(f"Constraints/{writer_title}/ineq_resid", dual_resid_ineq.detach().mean().item(), writer_steps)
        writer.add_scalar(f"Constraints/{writer_title}/eq_resid", dual_resid_eq.detach().mean().item(), writer_steps)
        writer.add_scalar(f"Obj_Function/{writer_title}", data.obj_fn(y.detach()).item(), writer_steps)

    return loss

def violation(data, X, y, mu_k, rho):
    # Calculate the equality constraint function h_x(y)
    eq = data.h(y)  # Assume shape (num_samples, n_eq)
    
    # Calculate the infinity norm of h_x(y)
    eq_inf_norm = torch.abs(eq).max(dim=1).values  # Shape: (num_samples,)

    # Calculate the inequality constraint function g_x(y)
    ineq = data.g(y)  # Assume shape (num_samples, n_ineq)
    
    # Calculate sigma_x(y) for each inequality constraint
    # ! Is this a typo in the PDL paper!? Lamb should be mu
    # sigma_y = torch.maximum(ineq, -lamb_k / rho)  # Element-wise max
    sigma_y = torch.maximum(ineq, -mu_k / rho)  # Element-wise max
    
    # Calculate the infinity norm of sigma_x(y)
    sigma_y_inf_norm = torch.abs(sigma_y).max(dim=1).values  # Shape: (num_samples,)

    # Compute v_k as the maximum of the two norms
    v_k = torch.maximum(eq_inf_norm, sigma_y_inf_norm)  # Shape: (num_samples,)
    
    return v_k.max().item()

# Modifies stats in place
def dict_agg(stats, key, value, op='concat'):
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
def eval_pdl(data, X, primal_net, dual_net, mu_k, lamb_k, rho, args, prefix, stats):

    eps_converge = args['corrEps']
    make_prefix = lambda x: "{}_{}".format(prefix, x)
    start_time = time.time()
    Y = primal_net(X)
    mu, lamb = dual_net(X)
    raw_end_time = time.time()
    Ycorr = Y

    # Ycorr, steps = grad_steps_all(data, X, Y, args)

    dict_agg(stats, make_prefix('time'), time.time() - start_time, op='sum')
    # dict_agg(stats, make_prefix('steps'), np.array([steps]))
    # TODO: Change to correct loss function.
    dict_agg(stats, make_prefix('primal_loss'), primal_loss(data, X, Y, mu, lamb, rho).detach().cpu().numpy())
    dict_agg(stats, make_prefix('dual_loss'), dual_loss(data, X, Y, mu, lamb, mu_k, lamb_k, rho).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eval'), data.obj_fn(Ycorr).detach().cpu().numpy())
    dict_agg(stats, make_prefix('dist'), torch.norm(Ycorr - Y, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_max'), torch.max(data.ineq_dist(Ycorr), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_mean'), torch.mean(data.ineq_dist(Ycorr), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_num_viol_0'),
             torch.sum(data.ineq_dist(Ycorr) > eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_num_viol_1'),
             torch.sum(data.ineq_dist(Ycorr) > 10 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_num_viol_2'),
             torch.sum(data.ineq_dist(Ycorr) > 100 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_max'),
             torch.max(torch.abs(data.h(Ycorr)), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_mean'), torch.mean(torch.abs(data.h(Ycorr)), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_num_viol_0'),
             torch.sum(torch.abs(data.h(Ycorr)) > eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_num_viol_1'),
             torch.sum(torch.abs(data.h(Ycorr)) > 10 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_num_viol_2'),
             torch.sum(torch.abs(data.h(Ycorr)) > 100 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_time'), raw_end_time - start_time, op='sum')
    dict_agg(stats, make_prefix('raw_eval'), data.obj_fn(Y).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_ineq_max'), torch.max(data.ineq_dist(Y), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_ineq_mean'), torch.mean(data.ineq_dist(Y), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_ineq_num_viol_0'),
             torch.sum(data.ineq_dist(Y) > eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_ineq_num_viol_1'),
             torch.sum(data.ineq_dist(Y) > 10 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_ineq_num_viol_2'),
             torch.sum(data.ineq_dist(Y) > 100 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_eq_max'),
             torch.max(torch.abs(data.h(Y)), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_eq_mean'),
             torch.mean(torch.abs(data.h(Y)), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_eq_num_viol_0'),
             torch.sum(torch.abs(data.h(Y)) > eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_eq_num_viol_1'),
             torch.sum(torch.abs(data.h(Y)) > 10 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_eq_num_viol_2'),
             torch.sum(torch.abs(data.h(Y)) > 100 * eps_converge, dim=1).detach().cpu().numpy())

    return stats

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
         
