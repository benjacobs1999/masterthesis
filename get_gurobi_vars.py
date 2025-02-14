import pyomo.environ as pyo
import numpy as np
import torch

class OptValueExtractor:
    def __init__(self, constant_gen_inv=False):
        self.constant_gen_inv = constant_gen_inv
        self.targets = {"y_gep": [], "y_operational": [], "y_investment": [], "mu_gep": [], "mu_operational": [], "lamb_gep": [], "lamb_operational": [], "obj": []}
    
    @property
    def opt_targets(self):
        return {"y_gep": torch.stack(self.targets["y_gep"]),
                "y_operational": torch.stack(self.targets["y_operational"]),
                "y_investment": torch.stack(self.targets["y_investment"]),
                "mu_gep": torch.stack(self.targets["mu_gep"]), 
                "mu_operational": torch.stack(self.targets["mu_operational"]), 
                "lamb_gep": torch.stack(self.targets["lamb_gep"]),
                "lamb_operational": torch.stack(self.targets["lamb_operational"]),
                "obj": self.targets["obj"]}

    def extract_values(self, model):
        decision_vars_gep, decision_vars_operational, decision_vars_investment = self.extract_decision_variables(model)
        dual_vars_ineq_gep, dual_vars_ineq_operational, dual_vars_eq = self.extract_dual_vars(model)
        self.targets["obj"].append(model.obj())
        self.targets["y_gep"].append(torch.tensor(decision_vars_gep, dtype=torch.float64))
        self.targets["y_operational"].append(torch.tensor(decision_vars_operational, dtype=torch.float64))
        self.targets["y_investment"].append(torch.tensor(decision_vars_investment, dtype=torch.float64))
        self.targets["mu_gep"].append(torch.tensor(dual_vars_ineq_gep, dtype=torch.float64))
        self.targets["mu_operational"].append(torch.tensor(dual_vars_ineq_operational, dtype=torch.float64))
        # Lamb is the same for gep and operational.
        self.targets["lamb_gep"].append(torch.tensor(dual_vars_eq, dtype=torch.float64))
        self.targets["lamb_operational"].append(torch.tensor(dual_vars_eq, dtype=torch.float64))


    def extract_decision_variables(self, model):
        """Extracts decision variables in time-first order: [c0t0, c1t0, c0t1, c1t1, ...]."""
        
        # Select variables based on whether generator investment is constant
        # if self.constant_gen_inv:
        variables = [model.vGenProd, model.vLineFlow, model.vLossLoad]
        # else:
            # raise NotImplementedError("This method needs to be implemented for non-fixed investment variables")

        decision_vars_operational = []
        decision_vars_investment = []

        # Extract time indices from one of the indexed variables (assuming they share time structure)
        sample_var = variables[0]
        if sample_var.is_indexed():
            time_indices = sorted(set(idx[-1] for idx in sample_var))  # Extract unique time indices
        else:
            time_indices = [None]  # If variables are not indexed, assume a single time step

        # Iterate over time first, then over decision variables
        for t in time_indices:
            for var in variables:
                if var.is_indexed():
                    for idx in var:
                        if idx[-1] == t:  # Extract values in time order
                            decision_vars_operational.append(var[idx].value)
                else:
                    # Handle scalar variables (not indexed)
                    decision_vars_operational.append(var.value)
        
        # Add investment variables
        if model.vGenInv.is_indexed():
            for idx in model.vGenInv:
                decision_vars_investment.append(model.vGenInv[idx].value)
        else:
            decision_vars_investment.append(model.vGenInv.value)

        decision_vars_gep = decision_vars_investment + decision_vars_operational

        return decision_vars_gep, decision_vars_operational, decision_vars_investment


    def extract_dual_vars(self, model):
        """Extracts dual variables with time-first ordering: [c0t0, c1t0, c0t1, c1t1, ...]"""
        # Define constraints based on whether generator investment is constant
        ineq_constraints = [
            model.eGenProdPositive, model.eMaxProd, model.eLineFlowLB, model.eLineFlowUB, model.eMissedDemandPositive, model.eMissedDemandLeqDemand
        ]  # gen_lb, gen_ub, lineflow_lb, lineflow_ub, md_lb, md_ub

        # Equality constraints
        eq_constraints = [model.eNodeBal]

        dual_vars_ineq_operational = []
        dual_vars_ineq_investment = []
        dual_vars_eq = []

        # Extract time indices from one of the indexed constraints (assuming all have the same time structure)
        sample_constraint = ineq_constraints[0]
        if sample_constraint.is_indexed():
            time_indices = sorted(set(idx[-1] for idx in sample_constraint))  # Extract all unique time indices
        else:
            time_indices = [None]  # If constraints aren't indexed, treat it as a single time step

        #! For some reason, Gurobi flips the sign of dual variables of the inequality constraints
        # Iterate over time steps first
        for t in time_indices:
            for item in ineq_constraints:
                if item.is_indexed():
                    for idx in item:
                        if idx[-1] == t:  # Ensure we only take constraints at the current time step
                            if item[idx] in model.dual:
                                dual_vars_ineq_operational.append(-model.dual[item[idx]])
                                # dual_vars_ineq_operational.append(model.dual[item[idx]])
                            else:
                                dual_vars_ineq_operational.append(0)  # Append 0 if no dual exists
                else:  # Handle scalar constraints
                    if item in model.dual:
                        dual_vars_ineq_operational.append(-model.dual[item])
                        # dual_vars_ineq_operational.append(model.dual[item])
                    else:
                        dual_vars_ineq_operational.append(0)
        
        # Add investment constraints
        if model.eGenInvPositive.is_indexed():
            for idx in model.eGenInvPositive:
                if model.eGenInvPositive[idx] in model.dual:
                    dual_vars_ineq_investment.append(-model.dual[model.eGenInvPositive[idx]])
                    # dual_vars_ineq_investment.append(model.dual[model.eGenInvPositive[idx]])
                else:
                    dual_vars_ineq_investment.append(0)
        else:
            if model.eGenInvPositive in model.dual:
                dual_vars_ineq_investment.append(-model.dual[model.eGenInvPositive])
                # dual_vars_ineq_investment.append(model.dual[model.eGenInvPositive])
            else:
                dual_vars_ineq_investment.append(0)

        # Iterate over time for equality constraints
        for t in time_indices:
            for constr in eq_constraints:
                if constr.is_indexed():
                    for idx in constr:
                        if idx[-1] == t:
                            if constr[idx] in model.dual:
                                # dual_vars_eq.append(-model.dual[constr[idx]])
                                dual_vars_eq.append(model.dual[constr[idx]])
                            else:
                                dual_vars_eq.append(0)
                else:
                    if constr in model.dual:
                        # dual_vars_eq.append(-model.dual[constr])
                        dual_vars_eq.append(model.dual[constr])
                    else:
                        dual_vars_eq.append(0)

        # Investment vars are prepended.
        dual_vars_ineq_gep = dual_vars_ineq_investment + dual_vars_ineq_operational

        return dual_vars_ineq_gep, dual_vars_ineq_operational, dual_vars_eq

if __name__ == "__main__":
    import pickle

    import numpy as np
    from gep_config_parser import *
    from gep_main import run_model_no_bounds as run_Gurobi_no_bounds
    from get_gurobi_vars import OptValueExtractor
    from gep_primal_dual_main import prep_data
    import torch

    import pyomo as pyo
    import json

    CONFIG_FILE_NAME        = "config.toml"
    VISUALIZATION_FILE_NAME = "visualization.toml"

    ## Step 1: parse the input data
    print("Parsing the config file")

    data = parse_config(CONFIG_FILE_NAME)
    experiment = data["experiment"]
    outputs_config = data["outputs_config"]

    with open("config.json", "r") as file:
        args = json.load(file)


    for i, experiment_instance in enumerate(experiment["experiments"]):
        # Setup output dataframe
        df_res = pd.DataFrame(columns=["setup_time", "presolve_time", "barrier_time", "crossover_time", "restore_time", "objective_value"])
            
        # Prep problem data:
        data = prep_data(experiment_instance, N=args["N"], G=args["G"], L=args["L"], sample_duration=args["sample_duration"])

        # Run Gurobi
        # experiment_instance, t, N, G, L, pDemand, pGenAva, pVOLL, pWeight, pRamping, pInvCost, pVarCost, pUnitCap, pExpCap, pImpCap
        extractor = OptValueExtractor(args["operational"])
        for t in data.time_ranges:
            model, solver, time_taken = run_Gurobi_no_bounds(experiment_instance,
                        t,
                        data.N,
                        data.G,
                        data.L,
                        data.pDemand,
                        data.pGenAva,
                        data.pVOLL,
                        data.pWeight,
                        data.pRamping,
                        data.pInvCost,
                        data.pVarCost,
                        data.pUnitCap,
                        data.pExpCap,
                        data.pImpCap,
                        )
            extractor.extract_values(model)
        
        with open(os.path.join("outputs/Gurobi", f"{args['G']}-OPT_TARGETS_T={args['sample_duration']}"), 'wb') as f:
                pickle.dump(extractor.opt_targets, f)