import pickle
import time

import numpy as np
from gep_config_parser import *
from data_wrangling import dataframe_to_dict

from old_gep_code.primal_dual import PrimalDualTrainer
from old_gep_code.gep_problem import GEPProblem
from gep_main import run_model as run_Gurobi
from gep_main import run_model_no_bounds as run_Gurobi_no_bounds
from get_gurobi_vars import OptValueExtractor
import sys
import torch

import pyomo as pyo

CONFIG_FILE_NAME        = "config.toml"
VISUALIZATION_FILE_NAME = "visualization.toml"
SAMPLE_DURATION = 24
# SAMPLE_DURATION = 120


SCALE_FACTORS = {
    "pDemand": 1/1000,  # MW -> GW
    "pGenAva": 1,       # Don't scale
    "pVOLL": 1,         # kEUR/MWh -> mEUR/GWh
    "pWeight": 1,       # Don't scale
    "pRamping": 1,      # Don't scale
    "pInvCost": 1,      # kEUR/MW -> mEUR/GW
    "pVarCost": 1,      # kEUR/MWh -> mEUR/GWh
    "pUnitCap": 1/1000, # MW -> GW
    "pExpCap": 1/1000,  # MW -> GW
    "pImpCap": 1/1000,  # MW -> GW
}


## Step 1: parse the input data
print("Parsing the config file")

data = parse_config(CONFIG_FILE_NAME)
experiment = data["experiment"]
outputs_config = data["outputs_config"]

def scale_dict(data_dict, scale_factor):
    return {key: value * scale_factor for key, value in data_dict.items()}


def prep_data(inputs, shuffle=False, scale_input=False, train=0.8, valid=0.1, test=0.1, scale=False):
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
    # G = [('BEL', 'WindOn'), ('GER', 'WindOn'), ('NED', 'WindOn')] # 3 generators
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

    if scale:
        pDemand = scale_dict(pDemand, SCALE_FACTORS["pDemand"])
        pGenAva = scale_dict(pGenAva, SCALE_FACTORS["pGenAva"])
        pVOLL *= SCALE_FACTORS["pVOLL"]
        pWeight *= SCALE_FACTORS["pWeight"]
        pRamping *= SCALE_FACTORS["pRamping"]
        pInvCost = scale_dict(pInvCost, SCALE_FACTORS["pInvCost"])
        pVarCost = scale_dict(pVarCost, SCALE_FACTORS["pVarCost"])
        pUnitCap = scale_dict(pUnitCap, SCALE_FACTORS["pUnitCap"])
        pExpCap = scale_dict(pExpCap, SCALE_FACTORS["pExpCap"])
        pImpCap = scale_dict(pImpCap, SCALE_FACTORS["pImpCap"])


    # We need to sort the dictionaries for changing to tensors!
    pDemand = dict(sorted(pDemand.items()))
    pGenAva = dict(sorted(pGenAva.items()))
    pInvCost = dict(sorted(pInvCost.items()))
    pVarCost = dict(sorted(pVarCost.items()))
    pUnitCap = dict(sorted(pUnitCap.items()))
    pExpCap = dict(sorted(pExpCap.items()))
    pImpCap = dict(sorted(pImpCap.items()))


    print("Creating problem instance")
    data = GEPProblem(T, N, G, L, pDemand, pGenAva, pVOLL, pWeight, pRamping, pInvCost, pVarCost, pUnitCap, pExpCap, pImpCap, sample_duration=SAMPLE_DURATION, shuffle=shuffle, train=train, valid=valid, test=test, scale_input=scale_input)

    return data

def run_PDL(data, args, save_dir, optimal_objective_train, optimal_objective_val):
    # Run PDL
    print("Training the PDL")
    trainer = PrimalDualTrainer(data, args, save_dir, optimal_objective_train=optimal_objective_train, optimal_objective_val=optimal_objective_val)
    primal_net, dual_net, stats = trainer.train_PDL()

if __name__ == "__main__":
    args = {
            # "K": 2,
            "K": 10,
            # "L": 1,
            "L": 500,
            # "L": 2000,
            "tau": 0.8,
            # "rho": 0.1,
            "rho": 0.5,
            # "rho": 0.1,
            # "rho_max": 10,
            "rho_max": 5000,
            # "rho_max": 100,
            # "rho_max": sys.maxsize * 2 + 1,
            "alpha": 10,
            # "alpha": 2,
            "batch_size": 100,
            "hidden_sizes": [500, 500],
            # "hidden_sizes": [500, 500, 500, 500],
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
            "scale_input": False,
            # "train": 0.002, # 1 sample
            # "valid": 0.002,
            # "test": 0.996,
            # "train": 0.004, # 2 samples
            # "valid": 0.004,
            # "test": 0.992,
            # "train": 0.8,
            # "valid": 0.1,
            # "test": 0.1
            "train": 0.02,
            "valid": 0.02,
            "test": 0.96,
            # "train": 0.01,
            # "valid": 0.01,
            # "test": 0.98,
    }

    # Train the model:
    for i, experiment_instance in enumerate(experiment["experiments"]):
        # Setup output dataframe
        df_res = pd.DataFrame(columns=["setup_time", "presolve_time", "barrier_time", "crossover_time", "restore_time", "objective_value"])

        for j in range(experiment["repeats"]):
            # Run one experiment for j repeats
            run_name = f"train:{args['train']}_rho:{args['rho']}_rhomax:{args['rho_max']}_alpha:{args['alpha']}_L:{args['alpha']}_scaled:{args['scale_input']}"
            save_dir = os.path.join('outputs', 'PDL',
                run_name + "-" + str(time.time()).replace('.', '-'))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(os.path.join(save_dir, 'args.dict'), 'wb') as f:
                pickle.dump(args, f)
            
            # Prep proble data:
            data = prep_data(experiment_instance, shuffle=args["shuffle"], scale_input=args["scale_input"], train=args["train"], valid=args["valid"], test=args["test"])

            # Run Gurobi
            # experiment_instance, t, N, G, L, pDemand, pGenAva, pVOLL, pWeight, pRamping, pInvCost, pVarCost, pUnitCap, pExpCap, pImpCap
            # opt_objs_train = []
            # opt_vars = []
            # opt_dual_vars = []
            train_extractor = OptValueExtractor()
            for t in data.train_time_ranges:
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
                           data.pImpCap)
                train_extractor.extract_values(model)
            
            data.set_train_extractor(train_extractor)

            valid_extractor = OptValueExtractor()
            for t in data.val_time_ranges:
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
                           data.pImpCap)
                valid_extractor.extract_values(model)

            data.set_valid_extractor(valid_extractor)
            
            avg_opt_obj_train = train_extractor.get_avg_obj() 
            avg_opt_obj_val = valid_extractor.get_avg_obj()

            print(f"Avg obj train: {avg_opt_obj_train}")
            print(f"Avg obj valid: {avg_opt_obj_val}")

            # Run PDL
            run_PDL(data, args, save_dir, optimal_objective_train=avg_opt_obj_train, optimal_objective_val=avg_opt_obj_val)


