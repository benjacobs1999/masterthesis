import numpy as np
from gep_config_parser import parse_config
import json
import os
import time
import pickle

from primal_dual import PrimalDualTrainer
from create_gep_dataset import create_gep_ed_dataset
from create_QP_dataset import create_QP_dataset, create_nonconvex_QP_dataset, create_varying_G_dataset, create_varying_Q_dataset
CONFIG_FILE_NAME = "config.toml"
ARGS_FILE_NAME = "config.json"


if __name__ == "__main__":
    # Load the arguments
    with open(ARGS_FILE_NAME, "r") as file:
        args = json.load(file)

    QP_args = args["QP_args"]

    assert args["problem_type"] in ["ED", "GEP", "QP"], "Problem type must be either 'ED', 'GEP', or 'QP'"


    run_name = f"train:{args['train']}_rho:{args['rho']}_rhomax:{args['rho_max']}_alpha:{args['alpha']}_L:{args['alpha']}"
    save_dir = os.path.join('outputs', 'PDL', args['problem_type'], run_name + "-" + str(time.time()).replace('.', '-'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'args.dict'), 'wb') as f:
        pickle.dump(args, f)
            
    if args["problem_type"] == "QP":
        if QP_args['random_hyperparams']:
            tau = np.random.choice([0.5, 0.6, 0.7, 0.8, 0.9], size=QP_args['repeats'])
            rho = np.random.choice([0.1, 0.5, 1, 10], size=QP_args['repeats'])
            rho_max = np.random.choice([1000, 5000, 10000, 50000], size=QP_args['repeats'])
            alpha = np.random.choice([1, 1.5, 2, 5, 10], size=QP_args['repeats'])
        else:
            rho = args['rho']
            rho_max = args['rho_max']
            alpha = args['alpha']
        for QP_type in QP_args['type']:
            data_save_path = f"data/QP_data/QP_type:{QP_type}_var:{QP_args['var']}_ineq:{QP_args['ineq']}_eq:{QP_args['eq']}_num_samples:{QP_args['num_samples']}.pkl"
            curr_type_save_dir = os.path.join(save_dir, QP_type)
            
            # Create dataset if it doesn't exist:
            if not os.path.exists(data_save_path):
                directory = os.path.dirname(data_save_path)
                os.makedirs(directory, exist_ok=True)
                if QP_type == "simple":
                    create_QP_dataset(QP_args['var'], QP_args['ineq'], QP_args['eq'], QP_args['num_samples'], data_save_path)
                elif QP_type == "row":
                    create_varying_G_dataset(QP_args['var'], QP_args['ineq'], QP_args['eq'], QP_args['num_samples'], 'row', data_save_path)
                elif QP_type == "column":
                    create_varying_G_dataset(QP_args['var'], QP_args['ineq'], QP_args['eq'], QP_args['num_samples'], 'column', data_save_path)
                elif QP_type == "random":
                    create_varying_G_dataset(QP_args['var'], QP_args['ineq'], QP_args['eq'], QP_args['num_samples'], 'random', data_save_path)
                elif QP_type == "obj":
                    create_varying_Q_dataset(QP_args['var'], QP_args['ineq'], QP_args['eq'], QP_args['num_samples'], data_save_path)
                elif QP_type == "nonconvex":
                    create_nonconvex_QP_dataset(QP_args['var'], QP_args['ineq'], QP_args['eq'], QP_args['num_samples'], data_save_path)
                else:
                    raise ValueError(f"QP type {QP_type} not supported")
        
            # Load data:
            with open(data_save_path, 'rb') as file:
                data = pickle.load(file)
            for repeat in range(QP_args['repeats']):
                if QP_args['random_hyperparams']:
                    curr_repeat_save_dir = os.path.join(curr_type_save_dir, f"tau_{tau[repeat]}_rho_{rho[repeat]}_rhomax_{rho_max[repeat]}_alpha_{alpha[repeat]}_repeat:{repeat}")
                    args['tau'] = tau[repeat]
                    args['rho'] = rho[repeat]
                    args['rho_max'] = rho_max[repeat]
                    args['alpha'] = alpha[repeat]
                else:
                    curr_repeat_save_dir = os.path.join(curr_type_save_dir, f"repeat:{repeat}")
                if not os.path.exists(curr_repeat_save_dir):
                    os.makedirs(curr_repeat_save_dir)
                trainer = PrimalDualTrainer(data, args, curr_repeat_save_dir)
                primal_net, dual_net, stats = trainer.train_PDL()
    
    else:
        ED_args = args["ED_args"]
        input_data = parse_config(CONFIG_FILE_NAME)
        gep_ed_data = input_data["experiment"]["experiments"][0] # Take first experiment, we don't change the inputs here.
        if args["problem_type"] == "ED":
            #! TODO: not all configs are correctly parsed here. E.g. when first running BEL and GER with both coal generators, is the same as with both gas generators.
            # For nodes, just use first letters: ['BEL', 'GER', 'NED'] → 'B-G-N'
            nodes_str = "-".join([n[0] for n in ED_args['N']])
            
            # For generators, count per node: [['BEL', 'WindOn'], ['BEL', 'Gas'],...] = 'B3-G2-N2'
            gen_counts = {}
            for g in ED_args['G']:
                node = g[0]
                gen_counts[node] = gen_counts.get(node, 0) + 1
            gens_str = "-".join([f"{node[0]}{count}" for node, count in gen_counts.items()])
            
            # For lines, just count: [['BEL', 'GER'], ['BEL', 'NED'], ['GER', 'NED']] → 'L3'
            lines_str = f"L{len(ED_args['L'])}"
            
            # Create a shortened filename
            data_save_path = (f"data/ED_data/ED_N{nodes_str}_G{gens_str}_{lines_str}"
                            f"_c{int(ED_args['benders_compact'])}"
                            f"_s{int(ED_args['scale_problem'])}"
                            f"_p{int(ED_args['perturb_operating_costs'])}"
                            f"_smp{ED_args['2n_synthetic_samples']}.pkl")
            
        elif args["problem_type"] == "GEP":
            data_save_path = f"data/GEP_data/N:{ED_args['N']}_G:{ED_args['G']}_L:{ED_args['L']}_scale-prob:{ED_args['scale_problem']}.pkl"
        
        if not os.path.exists(data_save_path):
            directory = os.path.dirname(data_save_path)
            os.makedirs(directory, exist_ok=True)
            data = create_gep_ed_dataset(args=ED_args, inputs=gep_ed_data, problem_type=args["problem_type"], save_path=data_save_path)

        # Load data:
        with open(data_save_path, 'rb') as file:
            data = pickle.load(file)

        for repeat in range(ED_args["repeats"]):
            curr_repeat_save_dir = os.path.join(save_dir, f"repeat:{repeat}")
            os.makedirs(curr_repeat_save_dir, exist_ok=True)
            # Run PDL
            trainer = PrimalDualTrainer(data, args, curr_repeat_save_dir)
            primal_net, dual_net, stats = trainer.train_PDL()

        # primal_net, dual_net = load(data, save_dir)
        # data.plot_balance(primal_net, dual_net)
        # data.plot_decision_variable_diffs(primal_net, dual_net)