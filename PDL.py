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
    # L = 500
    L = 500 # for testing
    tau = 0.8
    rho = 0.5
    rho_max = 5000
    alpha = 10
    batch_size = args['batchSize']

    train_dataset = TensorDataset(data.trainX.to(DEVICE))
    valid_dataset = TensorDataset(data.validX.to(DEVICE))
    test_dataset = TensorDataset(data.testX.to(DEVICE))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    primal_net = PrimalNet(data, args).to(dtype=torch.float32, device=DEVICE)
    dual_net = DualNet(data, args).to(dtype=torch.float32, device=DEVICE)

    primal_optim = torch.optim.Adam(primal_net.parameters(), lr=1e-4)
    dual_optim = torch.optim.Adam(dual_net.parameters(), lr=1e-4)

    for k in range(K):
        epoch_stats = {}
        frozen_dual_net = copy.deepcopy(dual_net)

        for l1 in range(L):
            # print(f"outer: {k+1}/{K}, primal:{l1+1}/{L}, primal:{0}/{L}")
            # Update primal net using primal loss
            for Xtrain in train_loader:
                Xtrain = Xtrain[0].to(DEVICE)
                start_time = time.time()
                primal_optim.zero_grad()
                y = primal_net(Xtrain)
                mu, lamb = frozen_dual_net(Xtrain)
                train_loss = primal_loss(data, Xtrain, y, mu, lamb, rho, args)
                train_loss.sum().backward()
                primal_optim.step()
                train_time = time.time() - start_time
                # dict_agg(epoch_stats, 'train_loss', train_loss.detach().cpu().numpy())
                # dict_agg(epoch_stats, 'train_time', train_time, op='sum')
        
        dict_agg(epoch_stats, 'train_loss_primal', train_loss.detach().cpu().numpy())
        
        # Calculate v_k

        # Copy dual net into dual_net_k
        frozen_primal_net = copy.deepcopy(primal_net)
        for l2 in range(L):
            # print(f"outer: {k+1}/{K}, primal:{l1+1}/{L}, primal:{l2+1}/{L}")
            # Update dual net using dual loss
            for Xtrain in train_loader:
                Xtrain = Xtrain[0].to(DEVICE)
                start_time = time.time()
                dual_optim.zero_grad()
                mu, lamb = dual_net(Xtrain)
                mu_k, lamb_k = frozen_dual_net(Xtrain)
                y = frozen_primal_net(Xtrain)
                train_loss = dual_loss(data, Xtrain, y, mu, lamb, mu_k, lamb_k, rho, args)
                train_loss.sum().backward()
                dual_optim.step()
                train_time = time.time() - start_time
                # dict_agg(epoch_stats, 'train_loss', train_loss.detach().cpu().numpy())
                # dict_agg(epoch_stats, 'train_time', train_time, op='sum')

        dict_agg(epoch_stats, 'train_loss_dual', train_loss.detach().cpu().numpy())

        # Calculate v_k for the entire training dataset
        y = primal_net(data.trainX.to(DEVICE))
        mu_k, lamb_k = dual_net(data.trainX.to(DEVICE))
        v_k = violation(data, data.trainX.to(DEVICE), y, lamb_k, rho)
        
        # Update rho from the second iteration onward.
        if k > 0 and v_k > tau * prev_v_k:
            rho = min(alpha * rho, rho_max)
        prev_v_k = v_k

        ##### Evaluate #####
        # Get valid loss
        primal_net.eval()
        for Xvalid in valid_loader:
            Xvalid = Xvalid[0].to(DEVICE)
            eval_pdl(data, Xvalid, primal_net, args, 'valid', epoch_stats)

        # Get test loss
        primal_net.eval()
        for Xtest in test_loader:
            Xtest = Xtest[0].to(DEVICE)
            eval_pdl(data, Xtest, primal_net, args, 'test', epoch_stats)

        stats = epoch_stats
        print(
            'Epoch {}: train loss primal {:.4f}, train loss dual {:.4f}, obj. value {:.4f},'.format(
                k, np.mean(epoch_stats['train_loss_primal']), np.mean(epoch_stats['train_loss_dual']), np.mean(epoch_stats['valid_eval']))
        )
        # print(
        #     'Epoch {}: train loss primal {:.4f}, train loss dual {:.4f}, eval {:.4f}, dist {:.4f}, ineq max {:.4f}, ineq mean {:.4f}, ineq num viol {:.4f}, eq max {:.4f}, time {:.4f}'.format(
        #         k, np.mean(epoch_stats['train_loss_primal']), np.mean(epoch_stats['train_loss_dual']), np.mean(epoch_stats['valid_eval']),
        #         np.mean(epoch_stats['valid_dist']), np.mean(epoch_stats['valid_ineq_max']),
        #         np.mean(epoch_stats['valid_ineq_mean']), np.mean(epoch_stats['valid_ineq_num_viol_0']),
        #         np.mean(epoch_stats['valid_eq_max']), np.mean(epoch_stats['valid_time'])))

    with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
        pickle.dump(stats, f)
    with open(os.path.join(save_dir, 'primal_net.dict'), 'wb') as f:
        torch.save(primal_net.state_dict(), f)
    with open(os.path.join(save_dir, 'dual_net.dict'), 'wb') as f:
        torch.save(dual_net.state_dict(), f)

    return primal_net, dual_net, stats

def primal_loss(data, X, y, mu, lamb, rho, args):
    obj = data.pdl_obj_fn(y)
    # g(y)
    ineq = data.pdl_g(X, y)
    # h(y)
    eq = data.pdl_h(X, y)
    # Lagrange multiplier terms (element-wise multiplication followed by sum along the constraint dimension)
    lagrange_ineq = torch.sum(mu * ineq, dim=1)  # Shape (batch_size,)
    lagrange_eq = torch.sum(lamb * eq, dim=1)  # Shape (batch_size,)
    violation_ineq = torch.sum(torch.maximum(ineq, torch.zeros_like(ineq)) ** 2, dim=1)
    violation_eq = torch.sum(eq ** 2)
    penalty = rho/2 * (violation_ineq + violation_eq)

    return obj + lagrange_ineq + lagrange_eq + penalty

def dual_loss(data, X, y, mu, lamb, mu_k, lamb_k, rho, args):
    # g(y)
    ineq = data.pdl_g(X, y)
    # h(y)
    eq = data.pdl_h(X, y)

    dual_resid_ineq = mu - torch.maximum(mu_k + rho * ineq, torch.zeros_like(ineq))
    dual_resid_ineq = torch.norm(dual_resid_ineq, dim=1)  # Norm along constraint dimension
    
    # Compute the dual residuals for equality constraints
    dual_resid_eq = lamb - (lamb_k + rho * eq)
    dual_resid_eq = torch.norm(dual_resid_eq, dim=1)  # Norm along constraint dimension
    
    # Total dual loss as the sum of both residuals
    return dual_resid_ineq + dual_resid_eq

def violation(data, X, y, lamb_k, rho):
    # Calculate the equality constraint function h_x(y)
    eq = data.pdl_h(X, y)  # Assume shape (batch_size, n_eq)
    
    # Calculate the infinity norm of h_x(y)
    eq_inf_norm = torch.norm(eq, p=float('inf'), dim=1)  # Shape: (batch_size,)

    # Calculate the inequality constraint function g_x(y)
    eq = data.pdl_g(X, y)  # Assume shape (batch_size, n_ineq)
    
    # Calculate sigma_x(y) for each inequality constraint
    sigma_y = torch.maximum(eq, -lamb_k / rho)  # Element-wise max
    
    # Calculate the infinity norm of sigma_x(y)
    sigma_y_inf_norm = torch.norm(sigma_y, p=float('inf'), dim=1)  # Shape: (batch_size,)

    # Compute v_k as the maximum of the two norms
    v_k = torch.maximum(eq_inf_norm, sigma_y_inf_norm)  # Shape: (batch_size,)
    
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
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        layer_sizes = [data.xdim, self._args['hiddenSize'], self._args['hiddenSize']]
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

class DualNet(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self._data = data
        self._args = args
        layer_sizes = [data.xdim, self._args['hiddenSize'], self._args['hiddenSize']]
        layers = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)
        # mu_k = np.zeros(self.G_np.shape[0])  # (50,)
        # lamb_k = np.zeros(self.A_np.shape[0])  # (50,)
        self.out_layer_mu = nn.Linear(self._args['hiddenSize'], data.G.shape[0])
        self.out_layer_lamb = nn.Linear(self._args['hiddenSize'], data.A.shape[0])
        # Init last layers as 0, like in the paper
        nn.init.zeros_(self.out_layer_mu.weight)
        nn.init.zeros_(self.out_layer_lamb.weight)
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.net(x)
        out_mu = self.out_layer_mu(out)
        out_lamb = self.out_layer_lamb(out)
        return out_mu, out_lamb

def main():
    args = {'probType': 'simple'}
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