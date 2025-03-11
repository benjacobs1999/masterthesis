import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch
import json
import pickle
import pandas as pd
import time

from gep_problem import GEPProblemSet
from gep_problem_operational import GEPOperationalProblemSet
from gep_primal_dual_main import prep_data
from primal_dual import load
from gep_config_parser import *

CONFIG_FILE_NAME        = "config.toml"

def solve_matrix_problem(data,i):

    # inspect data
    print('data.obj_coeff.size()',data.obj_coeff.size())
    print('data.ineq_cm.size()',data.ineq_cm.size())
    print('data.ineq_rhs.size()',data.ineq_rhs.size())
    print('data.eq_cm.size()',data.eq_cm.size())
    print('data.eq_rhs.size()',data.eq_rhs.size())
    print('data.ydim', data.ydim)

    # Create a new model
    m = gp.Model("Matrix problem")

    # Create variables
    x = m.addMVar(shape=data.ydim, vtype=GRB.CONTINUOUS, name="x")

    # Set objective
    obj = np.array(data.obj_coeff)
    m.setObjective(obj @ x, GRB.MINIMIZE)

    # Add ineq constraints
    A = np.array(data.ineq_cm[i])
    b = np.array(data.ineq_rhs[i])
    m.addConstr(A @ x <= b, name="ineq")

    # Add eq constraints
    A = np.array(data.eq_cm[i])
    b = np.array(data.eq_rhs[i])
    m.addConstr(A @ x == b, name="eq")

    # Optimize model
    m.optimize()

    print(x.X)
    print(f"Obj: {m.ObjVal:g}")

    return 

def solve_matrix_problem_simple(obj, A_ineq, b_ineq, A_eq, b_eq, master):

    # Create environment where Gurobi output is muted
    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    
    # Create a new model
    m = gp.Model("Matrix problem", env=env)

    # Create variables
    ydim = obj.size()[0]
    if master:
        vtypes = np.array([GRB.INTEGER for _ in range(ydim-1)])
        vtypes = np.append(vtypes,GRB.CONTINUOUS)
    else:
        vtypes = np.array([GRB.CONTINUOUS for _ in range(ydim)])
    x = m.addMVar(shape=ydim, vtype=vtypes, name="x")

    # Set objective
    obj = np.array(obj)
    m.setObjective(obj @ x, GRB.MINIMIZE)

    # Add ineq constraints
    A = np.array(A_ineq)
    b = np.array(b_ineq)
    m.addConstr(A @ x <= b, name="ineq")

    # Add eq constraints
    if not master:
        A = np.array(A_eq)
        b = np.array(b_eq)
        m.addConstr(A @ x == b, name="eq")

    # Optimize model
    m.optimize()

    # print(x.X)
    # print(f"Obj: {m.ObjVal:g}")
    if master:
        dual_val = []
    else:
        dual_val = m.Pi
        # print(m.Pi)

    return m.ObjVal, x.X, dual_val

def solve_matrix_problem_PDL(data, primal_net, dual_net, obj, A_ineq, b_ineq, A_eq, b_eq):
    # Set objective
    # obj = torch.tensor(obj)

    # Add ineq constraints
    ineq_cm = A_ineq.unsqueeze(0)
    ineq_rhs  = b_ineq.unsqueeze(0)

    # Add eq constraints
    eq_cm = A_eq.unsqueeze(0)
    eq_rhs = b_eq.unsqueeze(0)

    x = torch.concat([eq_rhs, ineq_rhs], dim=1)

    primal_val = primal_net(x, eq_rhs, ineq_rhs)
    dual_val = dual_net(x, eq_cm)

    obj_val = obj @ primal_val.T
    print(obj_val)
    obj_val_from_data = data.obj_fn(primal_val)

    return obj_val.item(), primal_val, dual_val

def solve_master_problem(data,sample,investments,obj_val,dual_val):
    # Solves the master problem in Benders decomposition
    # Returns the optimal objective function value in two parts: investment costs and value of alpha
    # And returns the optimal investment solution

    obj, A_ineq, b_ineq, A_eq, b_eq = find_master_problem_cm_rhs_obj(data,sample,investments,obj_val,dual_val)

    obj_val, primal_val, dual_val = solve_matrix_problem_simple(obj, A_ineq, b_ineq, A_eq, b_eq, True)

    obj_val_master = [obj_val-primal_val[-1], primal_val[-1]] # primal_val[-1] is the value of alpha

    new_investments = primal_val[:-1]

    return obj_val_master, new_investments

def find_master_problem_cm_rhs_obj(data,sample,investments,obj_val,dual_val):

    # Find objective of master problem
    obj = data.obj_coeff[:data.num_g] # first g columns are ui_g variables
    obj = torch.cat((obj,torch.tensor([1.])),0) # add alpha column with coeff 1

    # Find constraint ineq submatrices
    A_ineq = data.ineq_cm[sample,:data.num_g,0:0 + data.num_g] # first g rows are 3.1k constraints
    A_ineq = torch.cat((A_ineq,torch.zeros((data.num_g,1))),1) # add alpha column with coeff 0
    b_ineq = data.ineq_rhs[sample,:data.num_g]

    # Find constraint eq submatrices
    # A_eq = data.eq_cm[sample,0:0,:data.num_g+1] #take empty submatrix of correct size
    # b_eq = data.eq_rhs[sample,0:0]
    A_eq = torch.zeros((1,data.num_g+1)) # make one zeros equality, otherwise it gives an error for multiplying empty arrays
    b_eq = torch.zeros((1))

    # print('obj',obj)
    # print('A_ineq',A_ineq)
    # print('b_ineq',b_ineq)
    # print('A_eq',A_eq)
    # print('b_eq',b_eq)

    # Add lower bound for alpha
    lb_constraint = torch.zeros((1,data.num_g+1))
    lb_constraint[0,-1] = -1 # coeff for alpha: -1
    A_ineq = torch.cat((A_ineq,lb_constraint),0)
    rhs = 1e06
    b_ineq = torch.cat((b_ineq, torch.tensor([rhs])),0)

    # Add Benders cuts for alpha
    for iteration in range(len(obj_val)):
        for g in range(data.num_g):
            coeff_ui = sum(dual_val[iteration,t,g] for t in range(data.sample_duration))
            lb_constraint[0,g] = coeff_ui # coeff for ui_g
        A_ineq = torch.cat((A_ineq,lb_constraint),0)
        rhs = -obj_val[iteration] - sum(sum(dual_val[iteration,t,g]*-investments[iteration,g] for g in range(data.num_g)) for t in range(data.sample_duration))
        b_ineq = torch.cat((b_ineq, torch.tensor([rhs])),0)

    # Add upper bounds on the investments (this is not necessary)
    rhs = 100000. #TODO change this value for different instances
    for g in range(data.num_g):
        ub_constraint = torch.zeros((1,data.num_g+1))
        ub_constraint[0,g] = 1 # coeff for ui_g: 1
        A_ineq = torch.cat((A_ineq,ub_constraint),0)
        b_ineq = torch.cat((b_ineq, torch.tensor([rhs])),0)

    # print('A_ineq',A_ineq)
    # print('b_ineq',b_ineq)

    return obj, A_ineq, b_ineq, A_eq, b_eq

def solve_subproblems(data,sample,investments, primal_net, dual_net):
    # Solves the subproblems in Benders decomposition
    # Returns the optimal objective function value of the subproblems for all time periods added together
    # And returns the dual values of the ui_g = investment constraints

    # Calculate information about subproblem sizes
    time_range = data.time_ranges[sample]
    num_timesteps = len(time_range)
    
    obj_val_total = 0
    dual_val_all = []

    for time_step in range(num_timesteps):

        # print("Solving subproblem", time_step)

        # Find constraint matrices, right hand side vectors and objective vector of subproblem
        obj, A_ineq, b_ineq, A_eq, b_eq = find_subproblem_cm_rhs_obj(data,sample,investments,time_step)

        # Solve subproblem
        # obj_val, primal_val, dual_val = solve_matrix_problem_simple(obj, A_ineq, b_ineq, A_eq, b_eq, False)

        obj_val, primal_val, dual_val = solve_matrix_problem_PDL(data, primal_net, dual_net, obj, A_ineq, b_ineq, A_eq, b_eq)
        obj_val_original, primal_val_original, dual_val_original = solve_matrix_problem_simple(obj, A_ineq, b_ineq, A_eq, b_eq, master=False)



        print(f"Obj -- PDL: {obj_val}, original: {obj_val_original}")
        # print(f"Obj PDL: {obj_val}")

        # Find dual variables of ui_g = investments constraint
        # dual_val = dual_val[-data.num_n:]
        dual_val = dual_val[data.num_g:] #! Changed!

        dual_val_all.append(dual_val)

        # Find total objective value for cuts
        obj_val_total += obj_val

    return obj_val_total, dual_val_all 

def find_subproblem_cm_rhs_obj(data,sample,investments,time_step):

    # Calculate information about subproblem sizes
    num_rows_per_t_ineq = 2 * (data.num_g + data.num_l + data.num_n) # lower and upper bounds for p_g, f_l and md_n 
    num_rows_per_t_eq = data.num_n # energy balance equality for each node
    num_columns_per_t = data.n_var_per_t
    columns_ui = range(data.num_g)

    # Find objective of subproblem
    column_index = data.num_g + time_step*num_columns_per_t # first g columns are ui_g variables
    obj = data.obj_coeff[column_index:column_index + num_columns_per_t] # sample index is not needed, obj is same for all samples
    obj = torch.cat((torch.zeros(data.num_g),obj),0) # add 0 for ui_g variables

    # Find constraint ineq submatrices
    row_index = data.num_g + time_step*num_rows_per_t_ineq # first g rows are 3.1k constraints
    A_ineq = data.ineq_cm[sample, row_index:row_index + num_rows_per_t_ineq, column_index:column_index + num_columns_per_t] #take submatrix of time_step
    A_ineq = torch.cat((data.ineq_cm[sample,row_index:row_index + num_rows_per_t_ineq,columns_ui],A_ineq),1) # add ui_g columns
    b_ineq = data.ineq_rhs[sample,row_index:row_index + num_rows_per_t_ineq]

    # Find constraint eq submatrices
    row_index = time_step*num_rows_per_t_eq
    A_eq = data.eq_cm[sample,row_index:row_index + num_rows_per_t_eq,column_index:column_index + num_columns_per_t] #take submatrix of time_step
    A_eq = torch.cat((data.eq_cm[sample,row_index:row_index + num_rows_per_t_eq,columns_ui],A_eq),1) # add ui_g columns
    b_eq = data.eq_rhs[sample,row_index:row_index + num_rows_per_t_eq]

    # print('obj',obj)
    # print('A_ineq',A_ineq)
    # print('b_ineq',b_ineq)
    # print('A_eq',A_eq)
    # print('b_eq',b_eq)

    # Replace investment variables with constants in right hand side # NOT needed in compact form
    # row_index = data.num_g # first set of g constraints is lower bound on p_g variables, second set is upper bound
    # column_index = 0 # first g variables are ui_g
    # for g in range(data.num_g):
    #     upper_bound_p = A_ineq[row_index + g,column_index + g]
    #     b_ineq[row_index + g] = upper_bound_p * investments[g]

    # Fix investments: add constraint ui_g = investments
    ui_g = torch.eye(data.num_g)
    # Prepend ui_g
    ui_g = torch.cat((ui_g,torch.zeros(data.num_g,num_columns_per_t)),1)
    A_eq = torch.cat((A_eq,ui_g),0)
    b_eq = torch.cat((b_eq,investments),0) 

    # print('A_eq',A_eq)
    # print('b_eq',b_eq)

    return obj, A_ineq, b_ineq, A_eq, b_eq

def solve_with_benders(data, sample, primal_net, dual_net):

    # Create lists for algorithm
    investments_all = [] # list of tensors of size (num_g), one for every iteration
    obj_val_subproblems_all = [] # list of floats, one for every iteration
    dual_val_subproblems_all = [] # list of tensors of size (num_t,num_g), one for every iteration

    # Parameters for Benders algorithm
    epsilon = 1e-6

    # Start Benders algorithm
    optimal = False
    iter = 0
    while not optimal:

        print("Iteration", iter)

        # Find the investment decisions
        if iter == 0:
            # Generate initial investment solution
            investments_iter_k = [0. for _ in range(data.num_g)] #TODO find better initial solution?
            # Calculate objective of master problem of this solution
            obj_val_master = 0
            for g_idx, g in enumerate(data.G):
                obj_val_master += data.pInvCost[g] * data.pUnitCap[g] * investments_iter_k[g_idx]
            obj_val_master = [obj_val_master, 0] # alpha is zero in the first iteration
        else:
            # Solve master problem to find investments
            print("Solving the master problem in iteration", iter)
            obj_val_master, investments_iter_k = solve_master_problem(data,sample,torch.stack(investments_all),torch.tensor(obj_val_subproblems_all),torch.stack(dual_val_subproblems_all))

        # Add investment values of current iteration to list
        print("The investment decisions are", investments_iter_k)
        investments_iter_k = torch.tensor(investments_iter_k)
        investments_all.append(investments_iter_k)

        # Solve subproblems to find new cuts
        obj_val_total, dual_val_all  = solve_subproblems(data,sample,investments_iter_k, primal_net, dual_net)

        # Add total objective value of all subproblems of current iteration together to list
        obj_val_subproblems_all.append(obj_val_total)

        # Add dual variable values of current iteration to list
        dual_val_subproblems_all.append(torch.tensor(dual_val_all))

        # print('dual_val_all',dual_val_all)

        # Check for optimality
        lower_bound = obj_val_master[0] + obj_val_master[1] 
        upper_bound = obj_val_master[0] + obj_val_total
        print("Upper bound:",upper_bound)
        print("Lower bound",lower_bound)
        #TODO: Should we keep track of a best upper bound? The upper bound is not strictly better each iteration, it seems.
        if upper_bound - lower_bound < epsilon:
            optimal = True
            print('Done! Optimal solution found')
            print('Total number of iterations needed:', iter)
            print('Optimal objective value:', upper_bound)

        iter += 1
        
    return

if __name__ == "__main__":
        ## Step 1: parse the input data
    print("Parsing the config file")

    data = parse_config(CONFIG_FILE_NAME)
    experiment = data["experiment"]
    outputs_config = data["outputs_config"]

    with open("config.json", "r") as file:
        args = json.load(file)
    
    print(args)

    # Train the model:
    for i, experiment_instance in enumerate(experiment["experiments"]):
        # Setup output dataframe
        df_res = pd.DataFrame(columns=["setup_time", "presolve_time", "barrier_time", "crossover_time", "restore_time", "objective_value"])

        for j in range(experiment["repeats"]):
            # Run one experiment for j repeats
            run_name = f"refactored_train:{args['train']}_rho:{args['rho']}_rhomax:{args['rho_max']}_alpha:{args['alpha']}_L:{args['alpha']}"
            # save_dir = os.path.join('outputs', 'PDL',
            #     run_name + "-" + str(time.time()).replace('.', '-'))
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)
            # with open(os.path.join(save_dir, 'args.dict'), 'wb') as f:
            #     pickle.dump(args, f)

            target_path = f"outputs/Gurobi/Operational={True}_T={args['sample_duration']}_{args['G']}"

            # Prep problem data:
            gep_data = prep_data(args=args, inputs=experiment_instance, target_path=target_path, operational=False)
            operational_data = prep_data(args=args, inputs=experiment_instance, target_path=target_path, operational=True)

            # Load primal and dual net
            model_dir = "outputs/PDL/refactored_train:0.004_rho:0.5_rhomax:5000_alpha:10_L:10-1741007291-764719"
            primal_net, dual_net = load(operational_data, model_dir)

            primal_net.eval(), dual_net.eval()

            # data.plot_balance(primal_net, dual_net)
            # data.plot_decision_variable_diffs(primal_net, dual_net)

            # Solve single sample with matrix formulation
            # sample = 1 # only solve first sample for now 
            # solution = solve_matrix_problem(data, sample) # solution = Obj: 2374.99
            # solution sample 1 = 2790.09

            # Solve single sample with Benders decomposition
            sample = 0 # solution = Obj: 2374.99
            solve_with_benders(gep_data, sample, primal_net, dual_net)