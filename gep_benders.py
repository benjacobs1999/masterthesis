import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch
from gep_problem import GEPProblemSet
from gep_problem_operational import GEPOperationalProblemSet

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

def solve_master_problem(data,compact,sample,investments,obj_val,benders_cuts):
    # Solves the master problem in Benders decomposition
    # Returns the optimal objective function value in two parts: investment costs and value of alpha
    # And returns the optimal investment solution

    obj, A_ineq, b_ineq, A_eq, b_eq = find_master_problem_cm_rhs_obj(data,compact,sample,investments,obj_val,benders_cuts)

    obj_val, primal_val, dual_val = solve_matrix_problem_simple(obj, A_ineq, b_ineq, A_eq, b_eq, True)

    obj_val_master = [obj_val-primal_val[-1], primal_val[-1]] # primal_val[-1] is the value of alpha

    new_investments = primal_val[:-1]

    return obj_val_master, new_investments

def find_master_problem_cm_rhs_obj(data,compact,sample,investments,obj_val,benders_cuts):

    # Find objective of master problem
    obj = data.obj_coeff[:data.num_g] # first g columns are ui_g variables
    obj = torch.cat((obj,torch.tensor([1.])),0) # add alpha column with coeff 1

    # Find constraint ineq submatrices
    A_ineq = data.ineq_cm[sample,:data.num_g,0:0 + data.num_g] # first g rows are 3.1k constraints
    A_ineq = torch.cat((A_ineq,torch.zeros((data.num_g,1))),1) # add alpha column with coeff 0
    b_ineq = data.ineq_rhs[sample,:data.num_g]

    # Find constraint eq submatrices
    A_eq = torch.zeros((1,data.num_g+1)) # make one zeros equality, otherwise it gives an error for multiplying empty arrays
    b_eq = torch.zeros((1))

    # Add lower bound for alpha
    lb_constraint = torch.zeros((1,data.num_g+1))
    lb_constraint[0,-1] = -1 # coeff for alpha: -1
    A_ineq = torch.cat((A_ineq,lb_constraint),0)
    rhs = 1e06
    b_ineq = torch.cat((b_ineq, torch.tensor([rhs])),0)

    # Add Benders cuts for alpha
    for iteration in range(len(obj_val)):
        A_ineq = torch.cat((A_ineq,benders_cuts[iteration][0]),0)
        b_ineq = torch.cat((b_ineq, torch.tensor([benders_cuts[iteration][1]])),0)

    # Add upper bounds on the investments (this is not necessary) #TODO do we also want upper bound on alpha?
    rhs = 100000. #TODO change this value for different instances
    for g in range(data.num_g):
        ub_constraint = torch.zeros((1,data.num_g+1))
        ub_constraint[0,g] = 1 # coeff for ui_g: 1
        A_ineq = torch.cat((A_ineq,ub_constraint),0)
        b_ineq = torch.cat((b_ineq, torch.tensor([rhs])),0)

    return obj, A_ineq, b_ineq, A_eq, b_eq

def solve_subproblems(data,compact,sample,investments):
    # Solves the subproblems in Benders decomposition
    # Returns the optimal objective function value of the subproblems for all time periods added together
    # And returns the dual values of the ui_g = investment constraints

    # Calculate information about subproblem sizes
    time_range = data.time_ranges[sample]
    num_timesteps = len(time_range)
    
    obj_val_total = 0
    benders_cut_lhs = torch.zeros((1,data.num_g+1)) #coefficients for ui_g and for alpha
    benders_cut_lhs[0,-1] = -1 # coeff for alpha: -1
    benders_cut_rhs = 0

    for time_step in range(num_timesteps):

        # Find constraint matrices, right hand side vectors and objective vector of subproblem
        obj, A_ineq, b_ineq, A_eq, b_eq = find_subproblem_cm_rhs_obj(data,compact,sample,investments,time_step)

        # Solve subproblem
        obj_val, primal_val, dual_val = solve_matrix_problem_simple(obj, A_ineq, b_ineq, A_eq, b_eq, False)

        # Add objective value to the total
        obj_val_total += obj_val

        # Find coefficients of Benders cut and add to current cut
        benders_cut_lhs, benders_cut_rhs = find_benders_cut(data, compact, sample, investments, 
                                                            (benders_cut_lhs, benders_cut_rhs), 
                                                            time_step, b_ineq, b_eq, obj_val, dual_val)
        
    # Obtain the final Benders cut of all subproblems together
    benders_cut = benders_cut_lhs, benders_cut_rhs

    return obj_val_total, benders_cut

def find_subproblem_cm_rhs_obj(data,compact,sample,investments,time_step):

    # Calculate information about subproblem sizes
    num_rows_per_t_ineq = 2 * (data.num_g + data.num_l + data.num_n) # lower and upper bounds for p_g, f_l and md_n 
    num_rows_per_t_eq = data.num_n # energy balance equality for each node
    num_columns_per_t = data.n_var_per_t
    columns_ui = range(data.num_g)

    # Find objective of subproblem
    column_index = data.num_g + time_step*num_columns_per_t # first g columns are ui_g variables
    obj = data.obj_coeff[column_index:column_index + num_columns_per_t] # sample index is not needed, obj is same for all samples
    if compact:
        obj = torch.cat((torch.zeros(data.num_g),obj),0) # add 0 for ui_g variables

    # Find constraint ineq submatrices
    row_index = data.num_g + time_step*num_rows_per_t_ineq # first g rows are 3.1k constraints
    A_ineq = data.ineq_cm[sample,row_index:row_index + num_rows_per_t_ineq,column_index:column_index + num_columns_per_t] #take submatrix of time_step
    if compact:
        A_ineq = torch.cat((data.ineq_cm[sample,row_index:row_index + num_rows_per_t_ineq,columns_ui],A_ineq),1) # add ui_g columns
    # Find constraint ineq rhs
    b_ineq = data.ineq_rhs[sample,row_index:row_index + num_rows_per_t_ineq]
    if not compact:
        # Replace investment variables with constants in right hand side, NOT needed in compact form
        for g in range(data.num_g):
            upper_bound_p = investments[g]*-data.ineq_cm[sample,row_index+ data.num_g + g,g] #first g constraints are 3.1c, we want to take 3.1b coeff of ui_g
            b_ineq[data.num_g+g] = upper_bound_p  # second set of g constraints are 3.1b, we want to replace rhs 0 of 3.1b with upper_bound_p
            
    # Find constraint eq submatrices
    row_index = time_step*num_rows_per_t_eq
    A_eq = data.eq_cm[sample,row_index:row_index + num_rows_per_t_eq,column_index:column_index + num_columns_per_t] #take submatrix of time_step
    if compact:
        A_eq = torch.cat((data.eq_cm[sample,row_index:row_index + num_rows_per_t_eq,columns_ui],A_eq),1) # add ui_g columns
    b_eq = data.eq_rhs[sample,row_index:row_index + num_rows_per_t_eq]

    # Fix investments: add constraint ui_g = investments, ONLY in compact form
    if compact:
        ui_g = torch.eye(data.num_g)
        ui_g = torch.cat((ui_g,torch.zeros(data.num_g,num_columns_per_t)),1)
        A_eq = torch.cat((A_eq,ui_g),0)
        b_eq = torch.cat((b_eq,investments),0) 

    return obj, A_ineq, b_ineq, A_eq, b_eq

def find_benders_cut(data, compact, sample, investments, old_benders_cut, time_step, b_ineq, b_eq, obj_val, dual_val):

    benders_cut_lhs = old_benders_cut[0]
    benders_cut_rhs = old_benders_cut[1]

    # Find the coefficients of ui_g (lhs of Benders cut)
    for g in range(data.num_g):
        if compact:
            # Add dual variables of ui_g = investments constraint (last g equalities)
            coeff_ui = dual_val[-(data.num_g-g)]
        else:
            # Add dual term for upperbound on p constraint
            num_rows_per_t_ineq = 2 * (data.num_g + data.num_l + data.num_n) # lower and upper bounds for p_g, f_l and md_n 
            # coefficient of ui_g is - sum(pi_g,t * GA_g,t for t) * UCAP_g
            # we take this from 3.1b constraint in the original problem
            row_index = data.num_g + time_step*num_rows_per_t_ineq + data.num_g # we want the 3.1b constraints
            coeff_ui = dual_val[data.num_g + g] * -data.ineq_cm[sample,row_index+g,g] 

        benders_cut_lhs[0,g] = benders_cut_lhs[0,g] + coeff_ui

    # Compute right hand side of Benders cut
    if compact:
        # Add objective of subproblem
        benders_cut_rhs += -obj_val
        # Add dual term for ui_g = investments constraint
        for g in range(data.num_g):
            # rhs is - dual * -investment
            benders_cut_rhs += - dual_val[-(data.num_g-g)]*-investments[g]
    else:
        # Create array of constraint nr's of inequalties of which we want to include the dual term (3.1d,3.1e,3.1j)
        # because we only need to consider the constraints of which the rhs is not 0
        constraint_nrs = []
        constraint_nrs.extend([2*data.num_g+l for l in range(data.num_l)]) # 3.1d: Lineflow lower bound
        constraint_nrs.extend([2*data.num_g+data.num_l+l for l in range(data.num_l)]) # 3.1e: Lineflow upper bound
        constraint_nrs.extend([2*data.num_g+2*data.num_l+data.num_n+n for n in range(data.num_n)]) # 3.1j: Missed demand upper bound

        # Add dual term for inequalities 
        for constraint_nr in constraint_nrs:
            benders_cut_rhs += -dual_val[constraint_nr] * b_ineq[constraint_nr]
            
        # Add dual term for equalities
        num_rows_per_t_ineq = 2 * (data.num_g + data.num_l + data.num_n) # lower and upper bounds for p_g, f_l and md_n
        for constraint_nr in range(data.num_n):
            benders_cut_rhs += -dual_val[num_rows_per_t_ineq+constraint_nr] * b_eq[constraint_nr]
    
    new_benders_cut = benders_cut_lhs, benders_cut_rhs

    return new_benders_cut

def solve_with_benders(data, compact, sample):

    # Create lists for algorithm
    investments_all = [] # list of tensors of size (num_g), one for every iteration
    obj_val_subproblems_all = [] # list of floats, one for every iteration
    benders_cut_all = [] # list of benders cuts ([lhs],rhs), one for every iteration

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
            obj_val_master, investments_iter_k = solve_master_problem(data,compact,sample,torch.stack(investments_all),torch.tensor(obj_val_subproblems_all),benders_cut_all)

        # Add investment values of current iteration to list
        print("The investment decisions are", investments_iter_k)
        investments_iter_k = torch.tensor(investments_iter_k)
        investments_all.append(investments_iter_k)

        # Solve subproblems to find new cuts
        obj_val_total, benders_cut  = solve_subproblems(data,compact,sample,investments_iter_k)

        # Add total objective value of all subproblems of current iteration together to list
        obj_val_subproblems_all.append(obj_val_total)

        # Add Benders cut of current iteration to list
        benders_cut_all.append(benders_cut)
        print('Subproblems solved. Benders_cut:',benders_cut)

        # print('dual_val_all',dual_val_all)

        # Check for optimality
        lower_bound = obj_val_master[0] + obj_val_master[1]
        upper_bound = obj_val_master[0] + obj_val_total
        print("Upper bound:",upper_bound)
        print("Lower bound",lower_bound)
        if upper_bound - lower_bound < epsilon:
            optimal = True
            print('Done! Optimal solution found')
            print('Total number of iterations needed:', iter)
            print('Optimal objective value:', upper_bound)

        iter += 1
        
    return
