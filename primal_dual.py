import copy
import os
import pickle
import time
from setproctitle import setproctitle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.set_default_dtype(torch.float32)

import numpy as np

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)
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

    primal_net = PrimalNet(data, hidden_size).to(dtype=torch.float32, device=DEVICE)
    dual_net = DualNetTwoOutputLayers(data, hidden_size).to(dtype=torch.float32, device=DEVICE)

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
            print(f"Outer: {k}, Primal inner: {l1}")
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
            print(f"Outer: {k}, Dual inner: {l1}")
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
    ineq = data.g(y)
    # h(y)
    eq = data.h(y)

    lagrange_ineq = torch.sum(mu * ineq, dim=1)  # Shape (batch_size,)
    lagrange_eq = torch.sum(lamb * eq, dim=1)   # Shape (batch_size,)

    violation_ineq = torch.sum(torch.maximum(ineq, torch.zeros_like(ineq)) ** 2, dim=1)
    violation_eq = torch.sum(eq ** 2, dim=1)
    penalty = rho/2 * (violation_ineq + violation_eq)

    return obj + lagrange_ineq + lagrange_eq + penalty

def dual_loss(data, X, y, mu, lamb, mu_k, lamb_k, rho):
    # g(y)
    ineq = data.g(y)
    # h(y)
    eq = data.h(y)

    dual_resid_ineq = mu - torch.maximum(mu_k + rho * ineq, torch.zeros_like(ineq))
    dual_resid_ineq = torch.norm(dual_resid_ineq, dim=1)  # Norm along constraint dimension
    
    # Compute the dual residuals for equality constraints
    dual_resid_eq = lamb - (lamb_k + rho * eq)
    dual_resid_eq = torch.norm(dual_resid_eq, dim=1)  # Norm along constraint dimension
    
    # Total dual loss as the sum of both residuals
    return dual_resid_ineq + dual_resid_eq

def violation(data, X, y, lamb_k, rho):
    # Calculate the equality constraint function h_x(y)
    eq = data.h(y)  # Assume shape (num_samples, n_eq)
    
    # Calculate the infinity norm of h_x(y)
    eq_inf_norm = torch.abs(eq).max(dim=1).values  # Shape: (num_samples,)

    # Calculate the inequality constraint function g_x(y)
    ineq = data.g(y)  # Assume shape (num_samples, n_ineq)
    
    # Calculate sigma_x(y) for each inequality constraint
    sigma_y = torch.maximum(ineq, -lamb_k / rho)  # Element-wise max
    
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
    dict_agg(stats, make_prefix('ineq_max'), torch.max(data.ineq_dist(X, Ycorr), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_mean'), torch.mean(data.ineq_dist(X, Ycorr), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_num_viol_0'),
             torch.sum(data.ineq_dist(X, Ycorr) > eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_num_viol_1'),
             torch.sum(data.ineq_dist(X, Ycorr) > 10 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('ineq_num_viol_2'),
             torch.sum(data.ineq_dist(X, Ycorr) > 100 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_max'),
             torch.max(torch.abs(data.h(X, Ycorr)), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_mean'), torch.mean(torch.abs(data.h(X, Ycorr)), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_num_viol_0'),
             torch.sum(torch.abs(data.h(X, Ycorr)) > eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_num_viol_1'),
             torch.sum(torch.abs(data.h(X, Ycorr)) > 10 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('eq_num_viol_2'),
             torch.sum(torch.abs(data.h(X, Ycorr)) > 100 * eps_converge, dim=1).detach().cpu().numpy())
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
             torch.max(torch.abs(data.h(X, Y)), dim=1)[0].detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_eq_mean'),
             torch.mean(torch.abs(data.h(X, Y)), dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_eq_num_viol_0'),
             torch.sum(torch.abs(data.h(X, Y)) > eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_eq_num_viol_1'),
             torch.sum(torch.abs(data.h(X, Y)) > 10 * eps_converge, dim=1).detach().cpu().numpy())
    dict_agg(stats, make_prefix('raw_eq_num_viol_2'),
             torch.sum(torch.abs(data.h(X, Y)) > 100 * eps_converge, dim=1).detach().cpu().numpy())

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
        self.pDemand_tensor = torch.tensor(
            [[pDemand[(n, t)] for t in self.T] for n in self.N],
            device=DEVICE,
            dtype=torch.float32
        )

        self.pGenAva_tensor = torch.tensor(
            [[pGenAva.get((g, t), 1.0) for t in self.T] for g in self.G],
            device=DEVICE,
            dtype=torch.float32
        )

        self.pVOLL_tensor = torch.tensor(pVOLL, device=DEVICE, dtype=torch.float32)
        self.pWeight_tensor = torch.tensor(pWeight, device=DEVICE, dtype=torch.float32)
        self.pRamping_tensor = torch.tensor(pRamping, device=DEVICE, dtype=torch.float32)

        self.pInvCost_tensor = torch.tensor(
            [pInvCost[g] for g in self.G],
            device=DEVICE,
            dtype=torch.float32
        )

        self.pVarCost_tensor = torch.tensor(
            [pVarCost[g] for g in self.G],
            device=DEVICE,
            dtype=torch.float32
        )

        self.pUnitCap_tensor = torch.tensor(
            [pUnitCap[g] for g in self.G],
            device=DEVICE,
            dtype=torch.float32
        )

        self.pExpCap_tensor = torch.tensor(
            [pExpCap[l] for l in self.L],
            device=DEVICE,
            dtype=torch.float32
        )

        self.pImpCap_tensor = torch.tensor(
            [pImpCap[l] for l in self.L],
            device=DEVICE,
            dtype=torch.float32
        )

        # TODO: Is this how we want the input to be??
        sizes = [len(T), len(G), len(L), len(N)] # Do we need these??
        self.X = torch.tensor(np.concatenate((sizes, list(pDemand.values()), list(pGenAva.values()), [pVOLL], [pWeight], [pRamping], list(pInvCost.values()), list(pVarCost.values()), list(pUnitCap.values()), list(pExpCap.values()), list(pImpCap.values()))), dtype=torch.float32).unsqueeze(0)

        self.vGenProd_size = len(self.G) * len(self.T)  # Forall g in G, t in T
        self.vLineFlow_size = len(self.L) * len(self.T) # Forall l in L, t in T
        self.vLossLoad_size = len(self.N) * len(self.T) # Forall n in N, t in T
        self.vGenInv_size = len(self.G) # Forall g in G

        self.y_size = self.vGenProd_size + self.vLineFlow_size + self.vLossLoad_size + self.vGenInv_size

        self._xdim = self.X.shape[1]
        self._ydim = self.y_size

        self.num_ineq_constraints = (len(G) * len(T) + # 3.1b
                                     len(L) * len(T) + # 3.1d
                                     len(L) * len(T) + # 3.1e
                                     len(G) * (len(T)-1) + # 3.1f
                                     len(G) * (len(T)-1) + # 3.1g
                                     len(G) * len(T) + # 3.1h
                                     len(N) * len(T) + # 3.1i
                                     len(N) * len(T) + # 3.1j
                                     len(G))           # 3.1k
        
        self.num_eq_constraints = len(N) * len(T) # 3.1c

    @property
    def trainX(self):
        return self.X
    
    @property
    def xdim(self):
        return self._xdim
    
    @property
    def ydim(self):
        return self._ydim
    
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
        
        return torch.tensor(ret)
    
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

    # 3.1g
    def _e_ramping_up(self, vGenProd, vGenInv):
        ret = []
        for idxg, g in enumerate(self.G):
            for idxt, t in enumerate(self.T[1:]):
                ret.append(-1*(vGenProd[:, idxg, idxt] - vGenProd[:, idxg, idxt-1]) - self.pRamping * self.pUnitCap[g] * vGenInv[:, idxg])
        return torch.tensor(ret)

    # 3.1f
    def _e_ramping_down(self, vGenProd, vGenInv):
        ret = []
        for idxg, g in enumerate(self.G):
            for idxt, t in enumerate(self.T[1:]):
                ret.append(vGenProd[:, idxg, idxt] - vGenProd[:, idxg, idxt-1] - self.pRamping * self.pUnitCap[g] * vGenInv[:, idxg])
        return torch.tensor(ret)
    
    # 3.1h
    def _e_gen_prod_positive(self, vGenProd):
        ret = []
        for idxg, g in enumerate(self.G):
            for idxt, t in enumerate(self.T):
                ret.append(-vGenProd[:, idxg, idxt])
        return torch.tensor(ret)
    
    # 3.1i
    def _e_missed_demand_positive(self, vLossLoad):
        ret = []
        for idxn, n in enumerate(self.N):
            for idxt, t in enumerate(self.T):
                ret.append(-vLossLoad[:, idxn, idxt])
        return torch.tensor(ret)
    
    # 3.1j
    def _e_missed_demand_leq_demand(self, vLossLoad):
        ret = []
        for idxn, n in enumerate(self.N):
            for idxt, t in enumerate(self.T):
                ret.append(vLossLoad[:, idxn, idxt] - self.pDemand[(n, t)])
        return torch.tensor(ret)
    
    # 3.1k
    def _e_num_power_generation_units_positive(self, vGenInv):
        ret = []
        for idxg, g in enumerate(self.G):
            ret.append(-vGenInv[:, idxg])
        return torch.tensor(ret)


    def g(self, y):
        """Assume y contains [*vGenProd, *vLineFlow, *vLossLoad, *vGenInv]

        Args:
            y (_type_): _description_
        """
        vGenProd, vLineFlow, vLossLoad, vGenInv = self._split_y(y)

        # 3.1b
        b = self._e_max_prod(vGenProd, vGenInv)

        # 3.1d, 3.1e
        d, e = self._e_lineflow(vLineFlow)

        # 3.1f
        f = self._e_ramping_down(vGenProd, vGenInv)

        # 3.1g
        g = self._e_ramping_up(vGenProd, vGenInv)

        # 3.1h
        h = self._e_gen_prod_positive(vGenProd)

        # 3.1i
        i = self._e_missed_demand_positive(vLossLoad)

        # 3.1j
        j = self._e_missed_demand_leq_demand(vLossLoad)

        # 3.1k
        k = self._e_num_power_generation_units_positive(vGenInv)

        g_x_y = torch.concatenate((b, d, e, f, g, h, i, j, k)).to(DEVICE)

        return g_x_y.unsqueeze(0)
    
    # 3.1c
    def _e_nodebal(self, vGenProd, vLineFlow, vLossLoad):
        ret = []
        for idxn, n in enumerate(self.N):
            for idxt, t in enumerate(self.T):
                ret.append(self.pDemand[n, t] - (sum(vGenProd[:, idxg, idxt] for idxg, g in enumerate(self.G) if g[0] == n) +
                                sum(vLineFlow[:, idxl, idxt] for idxl, l in enumerate(self.L) if l[1] == n) - 
                                sum(vLineFlow[:, idxl, idxt] for idxl, l in enumerate(self.L) if l[0] == n) +
                                vLossLoad[:, idxn, idxt]
                                ))
        
        return torch.tensor(ret)

    def h(self, y):
        """Assume y contains [*vGenProd, *vLineFlow, *vLossLoad, *vGenInv]

        Args:
            y (_type_): _description_
        """
        vGenProd, vLineFlow, vLossLoad, _ = self._split_y(y)

        # 3.1c
        h_x_y = self._e_nodebal(vGenProd, vLineFlow, vLossLoad).to(DEVICE)

        return h_x_y.unsqueeze(0)
         
