import pickle
import time
from gep_config_parser import *
from data_wrangling import dataframe_to_dict
import pyomo.environ as pyo

from primal_dual import GEPProblem, train_PDL

CONFIG_FILE_NAME        = "config.toml"
VISUALIZATION_FILE_NAME = "visualization.toml"

## Step 1: parse the input data
print("Parsing the config file")

data = parse_config(CONFIG_FILE_NAME)
experiment = data["experiment"]
outputs_config = data["outputs_config"]

def run_model(inputs, args):
    
    if outputs_config["terminal"]["input_plots"]:
        pass

    print("Wrangling the input data")

    # Extract sets
    T = inputs["times"] # [1, 2, 3, ... 8761] ---> 8761
    G = inputs["generators"] # [('Country1', 'EnergySource1'), ...] ---> 107
    L = inputs["transmission_lines"] # [('Country1', 'Country2'), ...] ---> 44
    N = inputs["nodes"] # ['Country1', 'Country2', ...] ---> 20

    SCALE_TIMES = 0.01
    T = range(1, int(SCALE_TIMES*len(T)))

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
    pWeight = inputs["representative_period_weight"] / SCALE_TIMES

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

    data = GEPProblem(T, G, L, N, pDemand, pGenAva, pVOLL, pWeight, pRamping, pInvCost, pVarCost, pUnitCap, pExpCap, pImpCap)

    save_dir = os.path.join('outputs', 'PDL',
        str(time.time()).replace('.', '-'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'args.dict'), 'wb') as f:
        pickle.dump(args, f)

    # Run PDL
    primal_net, dual_net, stats = train_PDL(data, args, save_dir)

if __name__ == "__main__":
    args = {"K": 10,
            "L": 500,
            "tau": 0.8,
            "rho": 0.5,
            "rho_max": 5000,
            "alpha": 10,
            "batch_size": 200,
            "hidden_size": 500,
            "primal_lr": 1e-4,
            "dual_lr": 1e-4,
            "decay": 0.99,
            "patience": 10}

    # Train the model:
    for i, experiment_instance in enumerate(experiment["experiments"]):
        # Setup output dataframe
        df_res = pd.DataFrame(columns=["setup_time", "presolve_time", "barrier_time", "crossover_time", "restore_time", "objective_value"])

        for j in range(experiment["repeats"]):
            # Run one experiment for j repeats
            run_model(experiment_instance, args)

