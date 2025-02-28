import pickle
import time

from gep_config_parser import *
from data_wrangling import dataframe_to_dict

from primal_dual import PrimalDualTrainer, load
from gep_problem import GEPProblemSet
from gep_problem_operational import GEPOperationalProblemSet
from get_gurobi_vars import save_opt_targets

CONFIG_FILE_NAME        = "config.toml"
VISUALIZATION_FILE_NAME = "visualization.toml"


SCALE_FACTORS = {
    "pDemand": 1/1000,  # MW -> GW
    "pGenAva": 1,       # Don't scale
    "pVOLL": 1/1000,         # kEUR/MWh -> mEUR/GWh
    "pWeight": 1,       # Don't scale
    "pRamping": 1,      # Don't scale
    "pInvCost": 1/1000,      # kEUR/MW -> mEUR/GW
    "pVarCost": 1/1000,      # kEUR/MWh -> mEUR/GWh
    "pUnitCap": 1/1000, # MW -> GW
    "pExpCap": 1/1000,  # MW -> GW
    "pImpCap": 1/1000,  # MW -> GW
}

def scale_dict(data_dict, scale_factor):
    return {key: value * scale_factor for key, value in data_dict.items()}


def prep_data(args, inputs, target_path):
    print("Wrangling the input data")

    # Extract sets
    T = inputs["times"] # [1, 2, 3, ... 8760] ---> 8760
    N = args["N"]
    G = args["G"]
    L = args["L"]

    if not (N or G or L):
        G = inputs["generators"] # [('Country1', 'EnergySource1'), ...] ---> 107
        L = inputs["transmission_lines"] # [('Country1', 'Country2'), ...] ---> 44
        N = inputs["nodes"] # ['Country1', 'Country2', ...] ---> 20
    else:
        # Convert to tuples
        # N = [tuple(pair) for pair in N]
        G = [tuple(pair) for pair in G]
        L = [tuple(pair) for pair in L]

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
    pWeight = inputs["representative_period_weight"] / (args["sample_duration"] / 8760)

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

    if args["scale_problem"]:
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

    if not os.path.exists(target_path):
        time_ranges = [range(i, i + args["sample_duration"], 1) for i in range(1, len(T), args["sample_duration"])]
        save_opt_targets(args, inputs, target_path, T, N, G, L, pDemand, pGenAva, pVOLL, pWeight, pRamping, pInvCost, pVarCost, pUnitCap, pExpCap, pImpCap, time_ranges)


    print("Creating problem instance")
    if args["operational"]:
        data = GEPOperationalProblemSet(args, T, N, G, L, pDemand, pGenAva, pVOLL, pWeight, pRamping, pInvCost, pVarCost, pUnitCap, pExpCap, pImpCap, target_path=target_path)
    else:
        data = GEPProblemSet(args, T, N, G, L, pDemand, pGenAva, pVOLL, pWeight, pRamping, pInvCost, pVarCost, pUnitCap, pExpCap, pImpCap, target_path=target_path)

    return data

if __name__ == "__main__":
    import json

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
            save_dir = os.path.join('outputs', 'PDL',
                run_name + "-" + str(time.time()).replace('.', '-'))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(os.path.join(save_dir, 'args.dict'), 'wb') as f:
                pickle.dump(args, f)

            target_path = f"outputs/Gurobi/Operational={args['operational']}_T={args['sample_duration']}_{args['G']}"

            # Prep proble data:
            data = prep_data(args=args, inputs=experiment_instance, target_path=target_path)

            # Run PDL
            trainer = PrimalDualTrainer(data, args, save_dir)
            primal_net, dual_net, stats = trainer.train_PDL()

            primal_net, dual_net = load(data, save_dir)

            data.plot_balance(primal_net, dual_net)
            data.plot_decision_variable_diffs(primal_net, dual_net)