import numpy as np
import pickle

from QP_problem import *

def create_QP_dataset(num_var, num_ineq, num_eq, num_examples):
    np.random.seed(17)
    Q = np.diag(np.random.random(num_var))
    p = np.random.random(num_var)
    A = np.random.normal(loc=0, scale=1., size=(num_eq, num_var))
    X = np.random.uniform(-1, 1, size=(num_examples, num_eq))
    G = np.random.normal(loc=0, scale=1., size=(num_ineq, num_var))
    h = np.sum(np.abs(G@np.linalg.pinv(A)), axis=1)

    problem = OriginalQPProblem(Q, p, A, G, X, h)
    problem.calc_Y()
    print(len(problem.Y))

    with open("./QP_data/QP_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(num_var, num_ineq, num_eq, num_examples), 'wb') as f:
        pickle.dump(problem, f)
    
    return problem

def create_varying_G_dataset(num_var, num_ineq, num_eq, num_examples, vary):
    """Creates a modified QP data set that differs in the inequality constraint matrix, instead of the RHS variables.
    """
    np.random.seed(17)
    Q = np.diag(np.random.random(num_var))
    p = np.random.random(num_var)
    A = np.random.normal(loc=0, scale=1., size=(num_eq, num_var))
    # X is the same for all samples:
    b = np.random.uniform(-1, 1, size=(num_eq))
    G = np.random.normal(loc=0, scale=1., size=(num_ineq, num_var))
    d = np.sum(np.abs(G@np.linalg.pinv(A)), axis=1)

    X = np.random.normal(loc=0, scale=1., size=(num_examples, num_ineq))

    # Try first with changing a single row!
    if vary == 'row':
        row_indices = [0] * num_ineq
        col_indices = list(range(num_ineq))
    if vary == 'column':
        col_indices = [0] * num_ineq
        row_indices = list(range(num_ineq))
    if vary == 'random':
        col_indices = np.random.choice(num_var, num_ineq, replace=False)
        row_indices = np.random.choice(num_ineq, num_ineq, replace=True)

    problem = QPProblemVaryingG(X=torch.tensor(X), Q=Q, p=p, A=A, G=G, b=b, d=d, row_indices=row_indices, col_indices=col_indices)
    problem.calc_Y()
    print(len(problem.Y))

    with open("./QP_data/Varying_G_type={}_dataset_var{}_ineq{}_eq{}_ex{}".format(vary, num_var, num_ineq, num_eq, num_examples), 'wb') as f:
        pickle.dump(problem, f)
    
    return problem

def create_varying_G_b_d_dataset(num_var, num_ineq, num_eq, num_examples, num_varying_rows):
    """Creates a modified QP data set that differs in the inequality constraint matrix, instead of the RHS variables.
    """
    np.random.seed(17)
    Q = np.diag(np.random.random(num_var))
    p = np.random.random(num_var)
    A = np.random.normal(loc=0, scale=1., size=(num_eq, num_var))
    # X is the same for all samples:
    B = np.random.uniform(-1, 1, size=(num_examples, num_eq))
    G_base = np.random.normal(loc=0, scale=1., size=(num_ineq, num_var))

    G_list = []
    # For each sample, create a different inequality constraint matrix
    for _ in range(num_examples):
        G_sample = G_base.copy()
        # Vary the first n rows, (specified by num_varying_rows).
        G_sample[:num_varying_rows, :] = np.random.normal(loc=0, scale=1., size=(num_varying_rows, num_var))
        G_list.append(G_sample)
    
    # Create H matrix for each example
    D_list = []
    for Gi in G_list:
        d = np.sum(np.abs(Gi @ np.linalg.pinv(A)), axis=1)  # Compute bounds for all inequalities
        D_list.append(d)  # Resulting shape will be (num_ineq,)

    G = np.array(G_list)
    D = np.stack(D_list, axis=0)  # Shape (num_examples, num_ineq)
    problem = QPProblemVaryingGbd(Q=Q, p=p, A=A, G_base=G_base, G_varying=G, b=B, d=D, n_varying_rows=num_varying_rows)
    problem.calc_Y()
    print(len(problem.Y))

    with open("./QP_data/modified/MODIFIED_random_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(num_var, num_ineq, num_eq, num_examples), 'wb') as f:
        pickle.dump(problem, f)
    
    return problem

if __name__ == "__main__":
    num_var = 100
    num_ineq = 50
    num_eq = 50
    num_examples = 10000
    
    original_data = create_QP_dataset(num_var, num_ineq, num_eq, num_examples)
    varying_cm_row_data = create_varying_G_dataset(num_var, num_ineq, num_eq, num_examples, vary='row')
    varying_cm_column_data = create_varying_G_dataset(num_var, num_ineq, num_eq, num_examples, vary='column')
    varying_cm_random_data = create_varying_G_dataset(num_var, num_ineq, num_eq, num_examples, vary='random')