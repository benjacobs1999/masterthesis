from gep_config_parser import *
from data_wrangling import dataframe_to_dict
import pyomo.environ as pyo

CONFIG_FILE_NAME        = "config.toml"
VISUALIZATION_FILE_NAME = "visualization.toml"

HIGHS  = "HiGHS"
GUROBI = "Gurobi"

## Step 0: Activate environment - ensure consistency accross computers
# print("Reading the data")
# print("Activating the environment")

## Step 1: parse the input data
print("Parsing the config file")

data = parse_config(CONFIG_FILE_NAME)
experiment = data["experiment"]
outputs_config = data["outputs_config"]

print("Initializing the solver")
optimizer_name = data["optimizer_config"]["solver"]

# Determine the optimizer
if optimizer_name == HIGHS:
    raise NotImplementedError(f"{optimizer_name}: Not implemented")
elif optimizer_name == GUROBI:
    
    print(f"Using {GUROBI}")
    optimizer = "gurobi_direct"
else:
    raise ValueError(f"{optimizer_name}: Not implemented")

def run_model(inputs):
    
    if outputs_config["terminal"]["input_plots"]:
        # print("Input data statistics")
        # visualization_data = get_visualization_data(VISUALIZATION_FILE_NAME)
        # print_input_statistics(inputs, visualization_data)
        pass

    print("Wrangling the input data")

    # Extract sets
    T = inputs["times"] # [1, 2, 3, ... 8761] ---> 8761
    G = inputs["generators"] # [('Country1', 'EnergySource1'), ...] ---> 107
    L = inputs["transmission_lines"] # [('Country1', 'Country2'), ...] ---> 44
    N = inputs["nodes"] # ['Country1', 'Country2', ...] ---> 20

    original_len_T = len(T)
    samples = 10
    # 10 samples
    T = range(1, 1+samples)
    print(T)

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
    pWeight = inputs["representative_period_weight"] / (samples / original_len_T)
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
    
    # Extract optimizer attributes
    attributes = data["optimizer_config"][optimizer_name]

    # Add the log file attribute
    attributes["LogFile"] = inputs["output_log"]

    # Check the crossover setting
    if inputs["crossover"] != "gurobi":
        attributes["Crossover"] = 0

    # Initialize the model with the optimizer and attributes
    model, solver = initialize_model(optimizer, attributes)

    print("Populating the model")

    # Create variables
    print("Adding model variables")

    # Investment cost variable (non-negative)
    model.vInvCost = pyo.Var(within=pyo.NonNegativeReals, initialize=0)

    # Operational cost variable (non-negative)
    model.vOpeCost = pyo.Var(within=pyo.NonNegativeReals, initialize=0)

    # Generator investment variables
    if inputs["relaxed"] == "true":
        model.vGenInv = pyo.Var(G, within=pyo.NonNegativeReals, initialize=0)
    else:
        model.vGenInv = pyo.Var(G, within=pyo.Integers, bounds=(0, None), initialize=0)

    # Generator production variables (non-negative)
    model.vGenProd = pyo.Var(G, T, within=pyo.NonNegativeReals, initialize=0)

    # Ensure lineflow is within max import and export capacities
    # 3.1d & 3.1e
    # Since L has a tuple of two countries as index, pyomo unpacks it
    def lineFlowBounds(model, l0, l1, t):
        return (-pImpCap[(l0, l1)], pExpCap[(l0, l1)])

    # Transmission line flow variables with bounds    
    model.vLineFlow = pyo.Var(L, T, bounds=lineFlowBounds)

    # Loss of load variables with bounds based on demand
    model.vLossLoad = pyo.Var(
        N, T, 
        initialize=0, 
        bounds=lambda model, n, t: (0, pDemand[(n, t)])
    )

    # Formulate objective
    print("Formulating the objective")
    model.obj = pyo.Objective(expr=model.vInvCost + model.vOpeCost, sense=pyo.minimize)

    # Constraints
    print("Adding model constraints")

    # Investment costs
    # Sum_{g in G} IC_g * UCAP_g * ui_g
    # Investment cost (€/MW) * capacity of a unit (MW) * nr of units = Investment cost at g (€)
    def eInvCost_rule(model):
        return model.vInvCost == sum(
            pInvCost[g] * pUnitCap[g] * model.vGenInv[g] for g in G
        )
    model.eInvCost = pyo.Constraint(rule=eInvCost_rule)

    # Operating costs
    # WOP * (Sum_{g in G, t in T} (PC_g * p_{g, t}) + Sum_{n in N, t in T} (MDC * md_{n,t}))
    # Period weight * (variable production cost (€/MWh) * energy generation (MW) + 
    #                  cost of missed demand (€/MW) * missed demand (€/MW))
    def eOpeCost_rule(model):
        return model.vOpeCost == pWeight * (
            sum(pVarCost[g] * model.vGenProd[g, t] for g in G for t in T)
            + sum(pVOLL * model.vLossLoad[n, t] for n in N for t in T)
        )
    model.eOpeCost = pyo.Constraint(rule=eOpeCost_rule)

    # (3.1c)
    # Ensure energy balance at each node
    # Demand = generation + transmissionB->A - transmissionA-B + missed_demand
    def eNodeBal_rule(model, n, t):
        return (
            sum(model.vGenProd[g, t] for g in G if g[0] == n)
            + sum(model.vLineFlow[l, t] for l in L if l[1] == n)
            - sum(model.vLineFlow[l, t] for l in L if l[0] == n)
            + model.vLossLoad[n, t]
            == pDemand[(n, t)]
        )
    model.eNodeBal = pyo.Constraint(N, T, rule=eNodeBal_rule)

    # (3.1b)
    # Ensure production never exceeds capacity
    def eMaxProd_rule(model, g0, g1, t):
        availability = pGenAva.get((g0, g1, t), 1.0)  # Default availability to 1.0
        return model.vGenProd[(g0, g1), t] <= availability * pUnitCap[(g0, g1)] * model.vGenInv[(g0, g1)]
    
    model.eMaxProd = pyo.Constraint(G, T, rule=eMaxProd_rule)

    # Ramping constraints
    # No large changes in production between timesteps
    if inputs["ramping"] == "true":
        # Ramping up (3.1g)
        def eRampingUp_rule(model, g0, g1, t):
            if t == T[0]:  # Skip the first time step for ramping constraints
                return pyo.Constraint.Skip
            return (
                model.vGenProd[(g0, g1), t] - model.vGenProd[(g0, g1), T[T.index(t) - 1]]
                <= pRamping * pUnitCap[(g0, g1)] * model.vGenInv[(g0, g1)]
            )
        model.eRampingUp = pyo.Constraint(G, T, rule=eRampingUp_rule)

        # Ramping down (3.1f)
        def eRampingDown_rule(model, g0, g1, t):
            if t == T[0]:  # Skip the first time step for ramping constraints
                return pyo.Constraint.Skip
            return (
                -pRamping * pUnitCap[(g0, g1)] * model.vGenInv[(g0, g1)]
                <= model.vGenProd[(g0, g1), t] - model.vGenProd[(g0, g1), T[T.index(t) - 1]]
            )
        model.eRampingDown = pyo.Constraint(G, T, rule=eRampingDown_rule)

        ## Step 4: Solve
        print("Solving the optimization problem")
        results = solver.solve(model, tee=True)

        print(f"Objective Value: {model.obj()}")

    
if __name__ == "__main__":
    for i, experiment_instance in enumerate(experiment["experiments"]):
        # Setup output dataframe
        df_res = pd.DataFrame(columns=["setup_time", "presolve_time", "barrier_time", "crossover_time", "restore_time", "objective_value"])

        for j in range(experiment["repeats"]):
            # Run one experiment for j repeats
            res, setup_time, restore_time = run_model(experiment_instance)

            # # Check symmetry and append results
            # if experiment_instance["symmetry"] == "s2":
            #     df_res = pd.concat([
            #         df_res,
            #         pd.DataFrame({
            #             "setup_time": [setup_time],
            #             "presolve_time": ["-"],
            #             "barrier_time": ["-"],
            #             "crossover_time": ["-"],
            #             "restore_time": [restore_time],
            #             "objective_value": [res.invCost + res.opeCost]
            #         })
            #     ], ignore_index=True)
            # else:
            #     df_res = pd.concat([
            #         df_res,
            #         pd.DataFrame({
            #             "setup_time": [setup_time],
            #             "presolve_time": ["-"],
            #             "barrier_time": ["-"],
            #             "crossover_time": ["-"],
            #             "restore_time": [restore_time],
            #             "objective_value": [objective_value(res)]
            #         })
            #     ], ignore_index=True)

        # Write DataFrame to CSV
        df_res.to_csv(experiment_instance["output_file"], index=False)
    

    