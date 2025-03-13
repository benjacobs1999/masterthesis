
import torch
import pickle
import json

from gep_config_parser import *
from gep_main import run_model_no_bounds as run_Gurobi_no_bounds
from gep_main import run_operational_model_no_bounds as run_operational_Gurobi_no_bounds
import torch

class OptValueExtractor:
    def __init__(self):
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

    def extract_gep_values(self, model):
        decision_vars_gep, decision_vars_operational, decision_vars_investment = self.extract_decision_variables(model)
        dual_vars_ineq, dual_vars_eq = self.extract_dual_vars(model, operational=False)
        self.targets["obj"].append(model.obj())
        self.targets["y_gep"].append(torch.tensor(decision_vars_gep, dtype=torch.float64))
        self.targets["y_operational"].append(torch.tensor(decision_vars_operational, dtype=torch.float64))
        self.targets["y_investment"].append(torch.tensor(decision_vars_investment, dtype=torch.float64))
        self.targets["mu_gep"].append(torch.tensor(dual_vars_ineq, dtype=torch.float64))
        # self.targets["mu_operational"].append(torch.tensor(dual_vars_ineq_operational, dtype=torch.float64))
        # Lamb is the same for gep and operational.
        self.targets["lamb_gep"].append(torch.tensor(dual_vars_eq, dtype=torch.float64))
        # self.targets["lamb_operational"].append(torch.tensor(dual_vars_eq, dtype=torch.float64))
    
    def extract_operational_duals(self, model):
        dual_vars_ineq, dual_vars_eq = self.extract_dual_vars(model, operational=True)
        self.targets["mu_operational"].append(torch.tensor(dual_vars_ineq, dtype=torch.float64))
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


    def extract_dual_vars(self, model, operational):
        """Extracts dual variables with time-first ordering: [c0t0, c1t0, c0t1, c1t1, ...]
        Dual variables depend on the objective function, therefore they differ between operational and GEP problem.
        """
        # Define constraints based on whether generator investment is constant
        if operational:
            ineq_constraints = [
                model.eGenProdPositive, model.eMaxProd, model.eLineFlowLB, model.eLineFlowUB, model.eMissedDemandPositive, model.eMissedDemandLeqDemand
            ]  # gen_lb, gen_ub, lineflow_lb, lineflow_ub, md_lb, md_ub
        else:
            ineq_constraints = [
                model.eGenInvPositive, model.eGenProdPositive, model.eMaxProd, model.eLineFlowLB, model.eLineFlowUB, model.eMissedDemandPositive, model.eMissedDemandLeqDemand
            ]  # gen_lb, gen_ub, lineflow_lb, lineflow_ub, md_lb, md_ub

        # Equality constraints
        eq_constraints = [model.eNodeBal]

        dual_vars_ineq = []
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
                                dual_vars_ineq.append(-model.dual[item[idx]])
                                # dual_vars_ineq.append(model.dual[item[idx]])
                            else:
                                dual_vars_ineq.append(0)  # Append 0 if no dual exists
                else:  # Handle scalar constraints
                    if item in model.dual:
                        dual_vars_ineq.append(-model.dual[item])
                        # dual_vars_ineq.append(model.dual[item])
                    else:
                        dual_vars_ineq.append(0)

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

        return dual_vars_ineq, dual_vars_eq


def save_opt_targets(args, experiment_instance, target_path, T, N, G, L, pDemand, pGenAva, pVOLL, pWeight, pRamping, pInvCost, pVarCost, pUnitCap, pExpCap, pImpCap, time_ranges):
    extractor = OptValueExtractor()
    print('-'*40)
    print('Get GEP variables')
    print('-'*40)
    for t in time_ranges:
        model, solver, time_taken = run_Gurobi_no_bounds(experiment_instance,
                    t,
                    N,
                    G,
                    L,
                    pDemand,
                    pGenAva,
                    pVOLL,
                    pWeight,
                    pRamping,
                    pInvCost,
                    pVarCost,
                    pUnitCap,
                    pExpCap,
                    pImpCap,
                    )
        extractor.extract_gep_values(model)
    
    pGenInv = torch.stack(extractor.targets["y_investment"]).tolist()
    print('-'*40)
    print('Get operational duals')
    print('-'*40)
    for idx, t in enumerate(time_ranges):
        model, solver, time_taken = run_operational_Gurobi_no_bounds(experiment_instance,
                    t,
                    N,
                    G,
                    L,
                    pDemand,
                    pGenAva,
                    pVOLL,
                    pWeight,
                    pRamping,
                    pInvCost,
                    pVarCost,
                    pUnitCap,
                    pExpCap,
                    pImpCap,
                    pGenInv[idx]
                    )
        extractor.extract_operational_duals(model)

    with open(target_path, 'wb') as f:
        pickle.dump(extractor.opt_targets, f)