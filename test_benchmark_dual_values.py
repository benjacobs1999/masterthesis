import numpy as np
import pickle
import torch
from QP_problem import OriginalQPProblem
import torch.nn as nn

np.random.seed(17)

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

def create_QP_dataset(num_var, num_ineq, num_eq, num_examples):
    Q = np.diag(np.random.random(num_var))
    p = np.random.random(num_var)
    A = np.random.normal(loc=0, scale=1., size=(num_eq, num_var))
    X = np.random.uniform(-1, 1, size=(num_examples, num_eq))
    G = np.random.normal(loc=0, scale=1., size=(num_ineq, num_var))
    h = np.sum(np.abs(G@np.linalg.pinv(A)), axis=1)

    problem = OriginalQPProblem(Q, p, A, G, X, h)
    problem.calc_Y()
    problem.dual_calc_Y()
    print(len(problem.Y))

    with open("./QP_data/Dual_QP_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(num_var, num_ineq, num_eq, num_examples), 'wb') as f:
        pickle.dump(problem, f)
    
    return problem

# num_var = 100
# num_ineq = 50
# num_eq = 50
# num_examples = 10000
# original_data = create_QP_dataset(num_var, num_ineq, num_eq, num_examples)

with open("QP_data/Dual_QP_simple_dataset_var100_ineq50_eq50_ex10000", 'rb') as f:
    original_data = pickle.load(f)

X = original_data.X[original_data.test_indices]

primal_y = original_data.Y[original_data.test_indices]
mu, lamb = original_data.mu[original_data.test_indices], original_data.lamb[original_data.test_indices]

true_primal_obj = original_data.obj_fn(primal_y)
true_dual_obj = original_data.dual_obj_fn(X, mu, lamb)

primal_net = PrimalNet(original_data, [500, 500])
dual_net = DualNet(original_data, [500, 500], original_data.nineq, original_data.neq)

# Load state dictionaries.
primal_state_dict = torch.load("benchmark_experiment_output/original/0.006_primal_net.dict", weights_only=True)
dual_state_dict = torch.load("benchmark_experiment_output/original/0.006_dual_net.dict", weights_only=True)

# Load the state dictionaries into the networks.
primal_net.load_state_dict(primal_state_dict)
dual_net.load_state_dict(dual_state_dict)

y_pred = primal_net(X)
mu_pred, lamb_pred = dual_net(X)

pred_primal_obj = original_data.obj_fn(y_pred)
pred_dual_obj = original_data.dual_obj_fn(X, mu_pred, lamb_pred)

print(f"Primal dec var mean difference: {(y_pred - primal_y).detach().abs().mean()}")
print(f"Primal optimality gap: {((pred_primal_obj - true_primal_obj)/true_primal_obj).detach().abs().mean()}")

print(f"Dual ineq dec var mean difference: {(mu_pred - mu).detach().abs().mean()}")
print(f"Dual eq dec var mean difference: {(lamb_pred - lamb).detach().abs().mean()}")

print(f"Dual optimality gap: {((pred_dual_obj - true_dual_obj)/true_dual_obj).detach().abs().mean()}")

print(mu.clamp(max=0).detach().abs().mean())
print(mu.clamp(max=0).detach().abs().mean())