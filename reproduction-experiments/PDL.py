import copy
import os
import pickle
import time
from setproctitle import setproctitle
import torch
import torch.nn as nn
import torch.optim as optim
torch.set_default_dtype(torch.float64)

import operator
from functools import reduce
from torch.utils.data import TensorDataset, DataLoader

import default_args
from utils import my_hash

torch.set_default_dtype(torch.float32)

import numpy as np

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)
print(f"Running on {DEVICE}")

# For both the primal and dual nets in PDL and for other baselines,
# MLP with 2 hidden layers of size 500 is used. 
# Also, a ReLU activation is attached to each layer. 
# Batch normalization and Dropout layers are excluded because it is found that
# adding those degrades the performance. But for DC3, those are not excluded as 
# used in the original work. For the baselines, the number of maximum epochs is 
# set to 10000, which is equivalent to using the maximum outer iteration of K = 10
#  with the 500 inner epochs (for each primal or dual learning) for PDL.

def train_PDL(data, args, save_dir):
    K = 10
    L = 500
    # L = 1 # for testing
    tau = 0.8
    rho = 0.5
    rho_max = 5000
    alpha = 10
    # batch_size = 200
    batch_size = 200 # For testing
    hidden_size = 500

    train_dataset = TensorDataset(data.trainX.to(DEVICE))
    valid_dataset = TensorDataset(data.validX.to(DEVICE))
    test_dataset = TensorDataset(data.testX.to(DEVICE))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    primal_net = PrimalNet(data, hidden_size).to(dtype=torch.float32, device=DEVICE)
    dual_net = DualNetTwoOutputLayers(data, hidden_size).to(dtype=torch.float32, device=DEVICE)

    primal_lr = 1e-4
    dual_lr = 1e-4
    decay = 0.99
    patience = 10

    primal_optim = torch.optim.Adam(primal_net.parameters(), lr=primal_lr)
    dual_optim = torch.optim.Adam(dual_net.parameters(), lr=dual_lr)

    # Add schedulers
    primal_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        primal_optim, mode='min', factor=decay, patience=patience
    )
    dual_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        dual_optim, mode='min', factor=decay, patience=patience
    )
    for k in range(K):
        begin_time = time.time()
        epoch_stats = {}
        frozen_dual_net = copy.deepcopy(dual_net)
        frozen_dual_net.eval()

        for l1 in range(L):
            # Update primal net using primal loss
            for Xtrain in train_loader:
                Xtrain = Xtrain[0].to(DEVICE)
                start_time = time.time()
                primal_optim.zero_grad()
                y = primal_net(Xtrain)
                mu, lamb = frozen_dual_net(Xtrain)
                train_loss = primal_loss(data, Xtrain, y, mu, lamb, rho)
                train_loss.mean().backward()
                primal_optim.step()
                train_time = time.time() - start_time

            # Evaluate validation loss every epoch, and update learning rate
            curr_val_loss = 0
            primal_net.eval()
            for Xvalid in valid_loader:
                Xvalid = Xvalid[0].to(DEVICE)
                y = primal_net(Xvalid)
                mu, lamb = frozen_dual_net(Xvalid)
                loss = primal_loss(data, Xvalid, y, mu, lamb, rho).sum()
                curr_val_loss += loss
            curr_val_loss /= len(valid_loader)
            # Normalize by rho, so that the schedular still works correctly if rho is increased
            primal_scheduler.step(torch.sign(curr_val_loss) * (torch.abs(curr_val_loss) / rho))
        
        dict_agg(epoch_stats, 'train_loss_primal', train_loss.detach().cpu().numpy())

        # Copy primal net into frozen primal net
        frozen_primal_net = copy.deepcopy(primal_net)
        frozen_primal_net.eval()

        # Calculate v_k
        y = frozen_primal_net(data.trainX.to(DEVICE))
        mu_k, lamb_k = frozen_dual_net(data.trainX.to(DEVICE))
        v_k = violation(data, data.trainX.to(DEVICE), y, lamb_k, rho)

        for l2 in range(L):
            # Update dual net using dual loss
            for Xtrain in train_loader:
                Xtrain = Xtrain[0].to(DEVICE)
                start_time = time.time()
                dual_optim.zero_grad()
                mu, lamb = dual_net(Xtrain)
                mu_k, lamb_k = frozen_dual_net(Xtrain)
                y = frozen_primal_net(Xtrain)
                train_loss = dual_loss(data, Xtrain, y, mu, lamb, mu_k, lamb_k, rho)
                # train_loss.sum().backward()
                train_loss.mean().backward()
                dual_optim.step()

            # Evaluate validation loss every epoch, and update learning rate
            dual_net.eval()
            curr_val_loss = 0
            for Xvalid in valid_loader:
                Xvalid = Xvalid[0].to(DEVICE)
                y = frozen_primal_net(Xvalid)
                mu, lamb = dual_net(Xvalid)
                mu_k, lamb_k = frozen_dual_net(Xvalid)
                curr_val_loss += dual_loss(data, Xvalid, y, mu, lamb, mu_k, lamb_k, rho).sum()
            
            curr_val_loss /= len(valid_loader)
            # Normalize by rho, so that the schedular still works correctly if rho is increased
            dual_scheduler.step(torch.sign(curr_val_loss) * (torch.abs(curr_val_loss) / rho))

        dict_agg(epoch_stats, 'train_loss_dual', train_loss.detach().cpu().numpy())

        ##### Evaluate #####
        # Get valid loss
        primal_net.eval()
        for Xvalid in valid_loader:
            Xvalid = Xvalid[0].to(DEVICE)
            eval_pdl(data, Xvalid, primal_net, args, 'valid', epoch_stats)

        # Get test loss
        # primal_net.eval()
        # for Xtest in test_loader:
        #     Xtest = Xtest[0].to(DEVICE)
        #     eval_pdl(data, Xtest, primal_net, args, 'test', epoch_stats)

        end_time = time.time()
        stats = epoch_stats
        print(f"Epoch {k} done. Time taken: {end_time - begin_time}. Rho: {rho}. Primal LR: {primal_optim.param_groups[0]['lr']}, Dual LR: {dual_optim.param_groups[0]['lr']}")
        print(
            '{}: p-loss: {:.4f}, d-loss: {:.4f}, obj. val {:.4f}, Max eq.: {:.4f}, Max ineq.: {:.4f}, Mean eq.: {:.4f}, Mean ineq.: {:.4f}\n'.format(
                k, np.mean(epoch_stats['train_loss_primal']),
                np.mean(epoch_stats['train_loss_dual']),
                np.mean(epoch_stats['valid_eval']),
                np.mean(epoch_stats['valid_eq_max']),
                np.mean(epoch_stats['valid_ineq_max']),
                np.mean(epoch_stats['valid_eq_mean']),
                np.mean(epoch_stats['valid_ineq_mean']))
        )

        # Update rho from the second iteration onward.
        if k > 0 and v_k > tau * prev_v_k:
            rho = np.min([alpha * rho, rho_max])
        prev_v_k = v_k

    with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
        pickle.dump(stats, f)
    with open(os.path.join(save_dir, 'primal_net.dict'), 'wb') as f:
        torch.save(primal_net.state_dict(), f)
    with open(os.path.join(save_dir, 'dual_net.dict'), 'wb') as f:
        torch.save(dual_net.state_dict(), f)

    return primal_net, dual_net, stats

def primal_loss(data, X, y, mu, lamb, rho):
    obj = data.obj_fn(y)
    # g(y)
    ineq = data.ineq_resid(X, y)
    # h(y)
    eq = data.eq_resid(X, y)

    lagrange_ineq = torch.sum(mu * ineq, dim=1)  # Shape (batch_size,)
    lagrange_eq = torch.sum(lamb * eq, dim=1)   # Shape (batch_size,)

    violation_ineq = torch.sum(torch.maximum(ineq, torch.zeros_like(ineq)) ** 2, dim=1)
    violation_eq = torch.sum(eq ** 2, dim=1)
    penalty = rho/2 * (violation_ineq + violation_eq)

    return obj + lagrange_ineq + lagrange_eq + penalty

def dual_loss(data, X, y, mu, lamb, mu_k, lamb_k, rho):
    # g(y)
    ineq = data.ineq_resid(X, y)
    # h(y)
    eq = data.eq_resid(X, y)

    dual_resid_ineq = mu - torch.maximum(mu_k + rho * ineq, torch.zeros_like(ineq))
    dual_resid_ineq = torch.norm(dual_resid_ineq, dim=1)  # Norm along constraint dimension
    
    # Compute the dual residuals for equality constraints
    dual_resid_eq = lamb - (lamb_k + rho * eq)
    dual_resid_eq = torch.norm(dual_resid_eq, dim=1)  # Norm along constraint dimension
    
    # Total dual loss as the sum of both residuals
    return dual_resid_ineq + dual_resid_eq

def violation(data, X, y, lamb_k, rho):
    # Calculate the equality constraint function h_x(y)
    eq = data.eq_resid(X, y)  # Assume shape (num_samples, n_eq)
    
    # Calculate the infinity norm of h_x(y)
    eq_inf_norm = torch.abs(eq).max(dim=1).values  # Shape: (num_samples,)

    # Calculate the inequality constraint function g_x(y)
    ineq = data.ineq_resid(X, y)  # Assume shape (num_samples, n_ineq)
    
    # Calculate sigma_x(y) for each inequality constraint
    sigma_y = torch.maximum(ineq, -lamb_k / rho)  # Element-wise max
    
    # Calculate the infinity norm of sigma_x(y)
    sigma_y_inf_norm = torch.abs(sigma_y).max(dim=1).values  # Shape: (num_samples,)

    # Compute v_k as the maximum of the two norms
    v_k = torch.maximum(eq_inf_norm, sigma_y_inf_norm)  # Shape: (num_samples,)
    
    return v_k.max().item()

def softloss(data, X, Y, args):
    obj_cost = data.obj_fn(Y)
    ineq_cost = torch.norm(data.ineq_dist(X, Y), dim=1)
    eq_cost = torch.norm(data.eq_resid(X, Y), dim=1)
    return obj_cost + args['softWeight'] * (1 - args['softWeightEqFrac']) * ineq_cost + \
           args['softWeight'] * args['softWeightEqFrac'] * eq_cost

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
def eval_pdl(data, X, primal_net, args, prefix, stats):

    eps_converge = args['corrEps']
    make_prefix = lambda x: "{}_{}".format(prefix, x)
    start_time = time.time()
    Y = primal_net(X)
    raw_end_time = time.time()
    Ycorr = Y

    # Ycorr, steps = grad_steps_all(data, X, Y, args)

    dict_agg(stats, make_prefix('time'), time.time() - start_time, op='sum')
    # dict_agg(stats, make_prefix('steps'), np.array([steps]))
    # TODO: Change to correct loss function.
    dict_agg(stats, make_prefix('loss'), softloss(data, X, Y, args).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eval'), data.obj_fn(Ycorr).detach().cpu().numpy())
    dict_agg(stats, make_prefix('dist'), torch.norm(Ycorr - Y, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_max'), torch.max(data.ineq_dist(X, Ycorr), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_mean'), torch.mean(data.ineq_dist(X, Ycorr), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_num_viol_0'),
             torch.sum(data.ineq_dist(X, Ycorr) > eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_num_viol_1'),
             torch.sum(data.ineq_dist(X, Ycorr) > 10 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_num_viol_2'),
             torch.sum(data.ineq_dist(X, Ycorr) > 100 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_max'),
             torch.max(torch.abs(data.eq_resid(X, Ycorr)), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_mean'), torch.mean(torch.abs(data.eq_resid(X, Ycorr)), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_num_viol_0'),
             torch.sum(torch.abs(data.eq_resid(X, Ycorr)) > eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_num_viol_1'),
             torch.sum(torch.abs(data.eq_resid(X, Ycorr)) > 10 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_num_viol_2'),
             torch.sum(torch.abs(data.eq_resid(X, Ycorr)) > 100 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_time'), raw_end_time - start_time, op='sum')
    dict_agg(stats, make_prefix('raw_eval'), data.obj_fn(Y).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_ineq_max'), torch.max(data.ineq_dist(X, Y), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_ineq_mean'), torch.mean(data.ineq_dist(X, Y), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_ineq_num_viol_0'),
             torch.sum(data.ineq_dist(X, Y) > eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_ineq_num_viol_1'),
             torch.sum(data.ineq_dist(X, Y) > 10 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_ineq_num_viol_2'),
             torch.sum(data.ineq_dist(X, Y) > 100 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_eq_max'),
             torch.max(torch.abs(data.eq_resid(X, Y)), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_eq_mean'),
             torch.mean(torch.abs(data.eq_resid(X, Y)), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_eq_num_viol_0'),
             torch.sum(torch.abs(data.eq_resid(X, Y)) > eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_eq_num_viol_1'),
             torch.sum(torch.abs(data.eq_resid(X, Y)) > 10 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_eq_num_viol_2'),
             torch.sum(torch.abs(data.eq_resid(X, Y)) > 100 * eps_converge, dim=1).detach().cpu().numpy())

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
        # mu_k = np.zeros(self.G_np.shape[0])  # (50,)
        # lamb_k = np.zeros(self.A_np.shape[0])  # (50,)
        self.out_layer_mu = nn.Linear(self._hidden_size, data.G.shape[0])
        self.out_layer_lamb = nn.Linear(self._hidden_size, data.A.shape[0])
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
        self.out_layer = nn.Linear(self._hidden_size, self._data.G.shape[0] + self._data.A.shape[0])
        # Init last layers as 0, like in the paper
        nn.init.zeros_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)
        layers += [self.out_layer]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.net(x)
        out_mu = out[:, :self._data.G.shape[0]]
        out_lamb = out[:, self._data.A.shape[0]:]
        return out_mu, out_lamb

def main():
    # args = {'probType': 'simple'}
    args = {'probType': 'nonconvex'}
    defaults = default_args.baseline_nn_default_args(args['probType'])
    # for key in defaults.keys():
    #     if args[key] is None:
    #         args[key] = defaults[key]
    args.update(defaults)
    print(args)

    setproctitle('PDL-{}'.format(args['probType']))

    # Load data, and put on GPU if needed
    prob_type = args['probType']
    if prob_type == 'simple':
        torch.set_default_dtype(torch.float64)
        filepath = os.path.join('datasets', 'simple', "random_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(
            args['simpleVar'], args['simpleIneq'], args['simpleEq'], args['simpleEx']))
    elif prob_type == 'nonconvex':
        filepath = os.path.join('datasets', 'nonconvex', "random_nonconvex_dataset_var{}_ineq{}_eq{}_ex{}".format(
            args['nonconvexVar'], args['nonconvexIneq'], args['nonconvexEq'], args['nonconvexEx']))
    else:
        raise NotImplementedError

    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    for attr in dir(data):
        var = getattr(data, attr)
        if not callable(var) and not attr.startswith("__") and torch.is_tensor(var):
            try:
                setattr(data, attr, var.to(dtype=torch.float32, device=DEVICE))
            except AttributeError:
                pass
    data._device = DEVICE

    save_dir = os.path.join('results', str(data), 'PDL',
        my_hash(str(sorted(list(args.items())))), str(time.time()).replace('.', '-'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'args.dict'), 'wb') as f:
        pickle.dump(args, f)

    # Run PDL
    primal_net, dual_net, stats = train_PDL(data, args, save_dir)

if __name__ == "__main__":
    main()