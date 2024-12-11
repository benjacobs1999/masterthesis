import pickle
import time

import numpy as np
from gep_config_parser import *
from data_wrangling import dataframe_to_dict

from primal_dual import PrimalDualTrainer
from gep_problem import GEPProblem
from gep_main import run_model as run_Gurobi
import sys
import torch

CONFIG_FILE_NAME        = "config.toml"
VISUALIZATION_FILE_NAME = "visualization.toml"
SAMPLE_DURATION = 12
# SAMPLE_DURATION = 120

## Step 1: parse the input data
print("Parsing the config file")

data = parse_config(CONFIG_FILE_NAME)
experiment = data["experiment"]
outputs_config = data["outputs_config"]

def prep_data(inputs, shuffle=False):
    print("Wrangling the input data")

    # Extract sets
    T = inputs["times"] # [1, 2, 3, ... 8760] ---> 8760
    G = inputs["generators"] # [('Country1', 'EnergySource1'), ...] ---> 107
    L = inputs["transmission_lines"] # [('Country1', 'Country2'), ...] ---> 44
    N = inputs["nodes"] # ['Country1', 'Country2', ...] ---> 20

    ### SET UP CUSTOM CONFIG ###
    # N = ['BEL', 'FRA', 'GER', 'NED'] # 4 nodes
    N = ['BEL', 'GER', 'NED'] # 3 nodes
    # G = [('BEL', 'SunPV'), ('FRA', 'SunPV'), ('GER', 'SunPV'), ('NED', 'SunPV')] # 4 generators
    G = [('BEL', 'SunPV'), ('GER', 'SunPV'), ('NED', 'SunPV')] # 3 generators
    # L = [('BEL', 'FRA'), ('BEL', 'GER'), ('BEL', 'NED'), ('GER', 'FRA'), ('GER', 'NED')] # 5 lines
    L = [('BEL', 'GER'), ('BEL', 'NED'), ('GER', 'NED')] # 3 lines

    # Extract time series data
    pDemand = dataframe_to_dict(
        inputs["demand_data"],
        keys=["Country", "Time"],
        value="Demand_MW"
    )
    pGenAva = dataframe_to_dict(
        inputs["generation_availability_data"],
        keys=["Country", "Technology", "Time"],
        value="Availability_pu"
    )

    # Extract scalar parameters
    pVOLL = inputs["value_of_lost_load"]

    # WOP
    # Scale inversely proportional to times (T)
    pWeight = inputs["representative_period_weight"] / (SAMPLE_DURATION / 8760)

    pRamping = inputs["ramping_value"]

    # Extract generator parameters
    pInvCost = dataframe_to_dict(
        inputs["generation_data"],
        keys=["Country", "Technology"],
        value="InvCost_kEUR_MW_year"
    )
    pVarCost = dataframe_to_dict(
        inputs["generation_data"],
        keys=["Country", "Technology"],
        value="VarCost_kEUR_per_MWh"
    )
    pUnitCap = dataframe_to_dict(
        inputs["generation_data"],
        keys=["Country", "Technology"],
        value="UnitCap_MW"
    )

    # Extract line parameters
    pExpCap = dataframe_to_dict(
        inputs["transmission_lines_data"],
        keys=["CountryA", "CountryB"],
        value="ExpCap_MW"
    )
    pImpCap = dataframe_to_dict(
        inputs["transmission_lines_data"],
        keys=["CountryA", "CountryB"],
        value="ImpCap_MW"
    )

    # We need to sort the dictionaries for changing to tensors!
    pDemand = dict(sorted(pDemand.items()))
    pGenAva = dict(sorted(pGenAva.items()))
    pInvCost = dict(sorted(pInvCost.items()))
    pVarCost = dict(sorted(pVarCost.items()))
    pUnitCap = dict(sorted(pUnitCap.items()))
    pExpCap = dict(sorted(pExpCap.items()))
    pImpCap = dict(sorted(pImpCap.items()))


    print("Creating problem instance")
    data = GEPProblem(T, G, L, N, pDemand, pGenAva, pVOLL, pWeight, pRamping, pInvCost, pVarCost, pUnitCap, pExpCap, pImpCap, sample_duration=SAMPLE_DURATION, shuffle=shuffle)

    return data

def run_PDL(data, args, save_dir, optimal_objective):
    # Run PDL
    print("Training the PDL")
    trainer = PrimalDualTrainer(data, args, save_dir, optimal_objective=optimal_objective)
    primal_net, dual_net, stats = trainer.train_PDL()
    


if __name__ == "__main__":
    args = {
            # "K": 2,
            "K": 25,
            # "L": 10,
            "L": 500,
            "tau": 0.8,
            "rho": 0.01,
            # "rho": 1e-5,
            "rho_max": 100000,
            # "rho_max": sys.maxsize * 2 + 1,
            "alpha": 5,
            # "alpha": 2,
            # "batch_size": 584, # Full training set!
            "batch_size": 10000,
            "hidden_size": 500,
            # "hidden_size": 1000,
            "primal_lr": 1e-4,
            "dual_lr": 1e-4,
            # "primal_lr": 1e-5,
            # "dual_lr": 1e-5,
            # "decay": 0.99,
            "decay": 0.99,
            "patience": 10,
            "corrEps": 1e-4,
            "shuffle": False,
    }

    # Train the model:
    for i, experiment_instance in enumerate(experiment["experiments"]):
        # Setup output dataframe
        df_res = pd.DataFrame(columns=["setup_time", "presolve_time", "barrier_time", "crossover_time", "restore_time", "objective_value"])

        for j in range(experiment["repeats"]):
            # Run one experiment for j repeats
            run_name = "2024-12-09"
            save_dir = os.path.join('outputs', 'PDL',
                run_name + "-" + str(time.time()).replace('.', '-'))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(os.path.join(save_dir, 'args.dict'), 'wb') as f:
                pickle.dump(args, f)
            
            # Prep proble data:
            data = prep_data(experiment_instance, shuffle=args["shuffle"])

            # Run Gurobi
            # experiment_instance, t, N, G, L, pDemand, pGenAva, pVOLL, pWeight, pRamping, pInvCost, pVarCost, pUnitCap, pExpCap, pImpCap
            opt_objs_val = []
            for t in data.val_time_ranges:
                model, solver, time_taken = run_Gurobi(experiment_instance,
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
                           data.pImpCap)
                opt_objs_val.append(model.obj())
            
            avg_opt_obj = np.mean(opt_objs_val)

            print(avg_opt_obj)

            # Run PDL
            run_PDL(data, args, save_dir, optimal_objective=avg_opt_obj)


