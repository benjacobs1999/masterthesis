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
from create_gep_dataset import prep_data
from primal_dual import load
from gep_config_parser import *

CONFIG_FILE_NAME        = "config.toml"

class BendersSolver():
    def __init__(self, gep_data, operational_data, primal_net, dual_net, sample, exact=True):
        self.gep_data = gep_data
        self.operational_data = operational_data
        self.primal_net = primal_net
        self.dual_net = dual_net
        self.exact = exact
        self.sample = sample

    def solve_matrix_problem(self, data,i):

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
        # x = m.addMVar(shape=data.ydim, vtype=GRB.CONTINUOUS, name="x")
        #! Important! We need the lb=-GRB.INFINITY, because otherwise the lower bound is automatically set to 0 by Gurobi.
        x = m.addMVar(shape=data.ydim, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")


        # Set objective
        obj = np.array(data.obj_coeff)
        # print(obj)
        m.setObjective(obj @ x, GRB.MINIMIZE)

        # Add ineq constraints
        A = np.array(data.ineq_cm[i])
        b = np.array(data.ineq_rhs[i])
        m.addConstr(A @ x <= b, name="ineq")

        # print(A)
        # print(b)

        # Add eq constraints
        A = np.array(data.eq_cm[i])
        b = np.array(data.eq_rhs[i])
        m.addConstr(A @ x == b, name="eq")

        # print(A)
        # print(b)

        # Optimize model
        m.optimize()

        print(x.X)
        print(f"Obj: {m.ObjVal:g}")

        return x.X, m.ObjVal

    def solve_matrix_problem_simple(self, obj, A_ineq, b_ineq, A_eq, b_eq, master):

        # Create environment where Gurobi output is muted
        env = gp.Env(empty=True)
        env.setParam("OutputFlag",0)
        env.start()
        
        # Create a new model
        m = gp.Model("Matrix problem", env=env)

        # Create variables
        ydim = obj.size()[0]
        if master:
            # vtypes = np.array([GRB.INTEGER for _ in range(ydim-1)])
            vtypes = np.array([GRB.CONTINUOUS for _ in range(ydim-1)]) #! For now, test with continuous variables. We can do integer later.
            vtypes = np.append(vtypes,GRB.CONTINUOUS)
        else:
            vtypes = np.array([GRB.CONTINUOUS for _ in range(ydim)])
        
        #! Important! We need the lb=-GRB.INFINITY, because otherwise the lower bound is automatically set to 0 by Gurobi.
        x = m.addMVar(shape=ydim, lb=-GRB.INFINITY, vtype=vtypes, name="x")

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

    def solve_matrix_problem_PDL(self, obj, A_ineq, b_ineq, A_eq, b_eq, time_step):
        # For ineq RHS, only 3.1b varies across instances --> first |G| constraints are 3.1h, second |G| constraints are 3.1b.
        # ineq_rhs_varying_indices = [t * self.operational_data.nineq + self.operational_data.num_g + i for t in range(self.operational_data.sample_duration) for i in range(self.operational_data.num_g)]

        # The entire RHS changes for equality constraints
        X = torch.concat([b_eq, b_ineq]).unsqueeze(0)

        # print(f"X: {X}")
        # print(f"trained_X: {self.operational_data.X[sample*24+time_step]}")

        # print(X - self.operational_data.X[sample*24+time_step])


        primal_sol = self.primal_net(X)
        mu, lamb = dual_net(X)
        # mu, lamb = self.dual_net(X, data.eq_cm[:1])

        # mu = -mu

        # print(f"Dual obj val: {365*data.dual_obj_fn(b_eq, b_ineq, mu, lamb)}")
        # print(f"Dual obj val: {data.dual_obj_fn(data.eq_rhs[0], data.ineq_rhs[0], mu, lamb)}")

        exact_obj_val, exact_primal_val, exact_dual_val = self.solve_matrix_problem_simple(obj, A_ineq, b_ineq, A_eq, b_eq, False)

        dual_sol = torch.concat([mu, lamb], dim=1).squeeze()

        # exact_mu = exact_dual_val[:mu.shape[1]]
        # exact_lamb = exact_dual_val[mu.shape[1]:]

        #! Negative mu evaluates to the same as dual obj value as from the prediction. This is also the same as the primal obj value.
        dual_obj_exact = self.operational_data.dual_obj_fn(X, -torch.tensor(exact_dual_val[:mu.shape[1]]).unsqueeze(0), torch.tensor(exact_dual_val[mu.shape[1]:]).unsqueeze(0))

        # self.operational_data.dual_obj_fn(b_eq.unsqueeze(0), b_ineq.unsqueeze(0), torch.tensor(exact_dual_val[:mu.shape[1]]).unsqueeze(0), torch.tensor(exact_dual_val[mu.shape[1]:]).unsqueeze(0))

        dual_obj = self.operational_data.dual_obj_fn(X, mu*self.operational_data.pWeight, lamb*self.operational_data.pWeight)

        print(f"Dual optimality gap: {((dual_obj - dual_obj_exact)/dual_obj_exact).item()}")
        # self.operational_data.dual_obj_fn(b_eq.unsqueeze(0), b_ineq.unsqueeze(0), -mu*self.operational_data.pWeight, lamb*self.operational_data.pWeight)

        #! This is only correct in the first sample. why?
        obj_val = primal_sol @ obj

        primal_sol *= self.operational_data.pWeight
        obj_val = self.operational_data.obj_fn(primal_sol)

        print(f"Primal optimality gap: {((obj_val - exact_obj_val)/exact_obj_val).item()}")

        #! b_ineq might differ from what the model has been trained on. --> because the investment variables are different

        dual_sol *= self.operational_data.pWeight

        # dual_sol = exact_dual_val
        # dual_sol[:mu.shape[1]] *= -1

        return obj_val.detach().numpy().item(), primal_sol.squeeze().detach().numpy(), dual_sol.detach().numpy()

    def solve_master_problem(self, data,compact,sample,investments,obj_val,benders_cuts):
        # Solves the master problem in Benders decomposition
        # Returns the optimal objective function value in two parts: investment costs and value of alpha
        # And returns the optimal investment solution

        obj, A_ineq, b_ineq, A_eq, b_eq = self.find_master_problem_cm_rhs_obj(data,compact,sample,investments,obj_val,benders_cuts)

        obj_val, primal_val, dual_val = self.solve_matrix_problem_simple(obj, A_ineq, b_ineq, A_eq, b_eq, True)

        obj_val_master = [obj_val-primal_val[-1], primal_val[-1]] # primal_val[-1] is the value of alpha

        new_investments = primal_val[:-1]

        return obj_val_master, new_investments

    def find_master_problem_cm_rhs_obj(self, data,compact,sample,investments,obj_val,benders_cuts):

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

    def solve_subproblems(self, data,compact,sample,investments):
        # Solves the subproblems in Benders decomposition
        # Returns the optimal objective function value of the subproblems for all time periods added together
        # And returns the dual values of the ui_g = investment constraints

        # Calculate information about subproblem sizes
        time_range = data.time_ranges[sample]
        num_timesteps = len(time_range)
        
        obj_val_total = 0
        obj_val_exact = 0
        benders_cut_lhs = torch.zeros((1,data.num_g+1)) #coefficients for ui_g and for alpha
        benders_cut_lhs[0,-1] = -1 # coeff for alpha: -1
        benders_cut_rhs = 0

        benders_cut_lhs_exact = torch.zeros((1,data.num_g+1)) #coefficients for ui_g and for alpha
        benders_cut_lhs_exact[0,-1] = -1 # coeff for alpha: -1
        benders_cut_rhs_exact = 0

        for time_step in range(num_timesteps):

            # Find constraint matrices, right hand side vectors and objective vector of subproblem
            obj, A_ineq, b_ineq, A_eq, b_eq = self.find_subproblem_cm_rhs_obj(data,compact,sample,investments,time_step)

            # Solve subproblem
            obj_val_original, primal_val_original, dual_val_original = self.solve_matrix_problem_simple(obj, A_ineq, b_ineq, A_eq, b_eq, False)

            #! Solve with PDL:
            if self.exact:
                obj_val, primal_val, dual_val = self.solve_matrix_problem_simple(obj, A_ineq, b_ineq, A_eq, b_eq, False)
            else:
                obj_val, primal_val, dual_val = self.solve_matrix_problem_PDL(obj, A_ineq, b_ineq, A_eq, b_eq, time_step)

            # print(obj_val_original)
            # print(self.operational_data.pWeight*obj_val)

            # print(primal_val_original)
            # print(primal_val)

            # print(dual_val_original)
            # print(dual_val)

            # obj_val = obj_val_original
            # print('-'*10)
            # print(f"PDL obj: {obj_val}, Exact obj: {obj_val_original}")
            # print(obj_val - obj_val_original)
            # print('-'*10)
            # obj_val = obj_val_original
            # primal_val = primal_val_original
            # dual_val = dual_val_original

            # Add objective value to the total
            obj_val_total += obj_val
            obj_val_exact += obj_val_original

            # Find coefficients of Benders cut and add to current cut
            benders_cut_lhs, benders_cut_rhs = self.find_benders_cut(data, compact, sample, investments, 
                                                                (benders_cut_lhs, benders_cut_rhs), 
                                                                time_step, b_ineq, b_eq, obj_val, dual_val)
            benders_cut_lhs_exact, benders_cut_rhs_exact = self.find_benders_cut(data, compact, sample, investments, 
                                                                (benders_cut_lhs_exact, benders_cut_rhs_exact), 
                                                                time_step, b_ineq, b_eq, obj_val_original, dual_val_original)
            
            
        # Obtain the final Benders cut of all subproblems together
        if self.exact:
            benders_cut = benders_cut_lhs, benders_cut_rhs
        else:
            benders_cut = -benders_cut_lhs, benders_cut_rhs #! Negate lhs to get correct cut if we are predicting (if self.exact = False)
        benders_cut_exact = benders_cut_lhs_exact, benders_cut_rhs_exact
        print(f"Benders cut: {benders_cut}")
        print(f"Benders cut exact: {benders_cut_exact}")
        return obj_val_total, benders_cut

    def find_subproblem_cm_rhs_obj(self, data,compact,sample,investments,time_step):

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

    def find_benders_cut(self, data, compact, sample, investments, old_benders_cut, time_step, b_ineq, b_eq, obj_val, dual_val):

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

    def solve_with_benders(self, data, compact, sample):

        # Create lists for algorithm
        investments_all = [] # list of tensors of size (num_g), one for every iteration
        obj_val_subproblems_all = [] # list of floats, one for every iteration
        benders_cut_all = [] # list of benders cuts ([lhs],rhs), one for every iteration

        # Parameters for Benders algorithm
        epsilon = 1e-6

        # Start Benders algorithm
        optimal = False
        i = 0
        while not optimal and i < 100:
            print("-"*50)
            print("Iteration", i)

            # Find the investment decisions
            if i == 0:
                # Generate initial investment solution
                investments_iter_k = [0. for _ in range(data.num_g)] #TODO find better initial solution?
                # investments_iter_k = self.operational_data.opt_targets['y_investment'][0] #! Test with optimal solution
                # Calculate objective of master problem of this solution
                obj_val_master = 0
                for g_idx, g in enumerate(data.G):
                    obj_val_master += data.pInvCost[g] * data.pUnitCap[g] * investments_iter_k[g_idx]
                obj_val_master = [obj_val_master, 0] # alpha is zero in the first iteration
            else:
                # Solve master problem to find investments
                print("Solving the master problem in iteration", i)
                obj_val_master, investments_iter_k = self.solve_master_problem(data,compact,sample,torch.stack(investments_all),torch.tensor(obj_val_subproblems_all),benders_cut_all)

            # Add investment values of current iteration to list
            print("The investment decisions are", investments_iter_k)
            investments_iter_k = torch.tensor(investments_iter_k)
            investments_all.append(investments_iter_k)

            # Solve subproblems to find new cuts
            obj_val_total, benders_cut  = self.solve_subproblems(data,compact,sample,investments_iter_k)

            # Add total objective value of all subproblems of current iteration together to list
            obj_val_subproblems_all.append(obj_val_total)

            # Add Benders cut of current iteration to list
            benders_cut_all.append(benders_cut)
            print('Subproblems solved. Benders_cut:',benders_cut)

            # print('dual_val_all',dual_val_all)

            # Check for optimality
            #TODO: Upper bound should be 'best found' instead of 'current' upper bound (and for lower bound as well?)
            lower_bound = obj_val_master[0] + obj_val_master[1]
            upper_bound = obj_val_master[0] + obj_val_total
            print("Upper bound:",upper_bound)
            print("Lower bound:",lower_bound)
            if upper_bound - lower_bound < epsilon:
                optimal = True
                print('Done! Optimal solution found')
                print('Total number of iterations needed:', i)
                print('Optimal objective value:', upper_bound)

            i += 1
            
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
            run_name = f"refactored_train:{args['train']}_rho:{args['rho']}_rhomax:{args['rho_max']}_alpha:{args['alpha']}_L:{args['L']}"
            # save_dir = os.path.join('outputs', 'PDL',
            #     run_name + "-" + str(time.time()).replace('.', '-'))
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)
            # with open(os.path.join(save_dir, 'args.dict'), 'wb') as f:
            #     pickle.dump(args, f)

            target_path_gep = f"outputs/Gurobi/Operational={args['operational']}_T={args['sample_duration']}_Scale={args['scale_problem']}_{args['G']}_{args['L']}"
            target_path_operational = f"outputs/Gurobi/Operational={args['operational']}_T={1}_Scale={args['scale_problem']}_{args['G']}_{args['L']}"

            # Prep problem data:
            gep_data = prep_data(args=args, inputs=experiment_instance, target_path=target_path_gep, operational=False)
            operational_data = prep_data(args=args, inputs=experiment_instance, target_path=target_path_operational, operational=True)

            # Load primal and dual net
            model_dir = "benders_models/1-node-1-generator"
            primal_net, dual_net = load(args, operational_data, model_dir)
            primal_net.eval(), dual_net.eval()
            # primal_net = None
            # dual_net = None

            # Solve single sample with matrix formulation
            sample = 1 # only solve first sample for now
            exact = False
            solver = BendersSolver(gep_data=gep_data, operational_data=operational_data, primal_net=primal_net, dual_net=dual_net, sample=sample, exact=exact)
            y, obj = solver.solve_matrix_problem(gep_data, sample) # solution = Obj: 2374.99
            # solution sample 1 = 2790.09

            # Solve single sample with Benders decomposition
            # sample = 1 # solution = Obj: 2374.99
            compact = False
            # solve_with_benders(gep_data, operational_data, compact, sample, primal_net, dual_net)
            solver.solve_with_benders(gep_data, compact, sample)

            print(f"Known optimum: {obj}")