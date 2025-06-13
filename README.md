# Self-Supervised Learning with Formal Guarantees for Energy Systems Optimization: Primal-Dual Solutions, Objective Bounds, and Benders Cuts

### Environment
To create an environment with the correct packages, type the following commands in the terminal:

1. `conda create -n {env_name} python=3.9.20`
2. `conda activate {env_name}`
3. `pip install -r requirements.txt`

### File explanations

- `config.json`: Config file for problem instances, benders, and training parameters.
- `config.toml`: Config file reading inputs (code from thesis Matthijs Arnoldus).
- `create_gep_dataset.py`: Script for creating generation expansion planning and economic dispatch datasets.
- `create_QP_dataset.py`: Script for creating quadratic programming benchmark datasets.
- `data_wrangling.py`: Code for reading input data into a format we can use.
- `gep_benders.py`: Main script for Benders decomposition.
- `gep_config_parser.py`: Parses the config.toml.
- `gep_exact_solver.py`: Methods for solving the problem with Gurobi.
- `gep_problem_operational.py`: Class of economic dispatch problem instances.
- `gep_problem.py`: Class of generation expansion planning problem instances.
- `logger.py`: Class used for training logs in Tensorboard.
- `main.py`: Main script for training and saving neural networks.
- `networks.py`: Class with neural network code.
- `primal_dual.py`: Class with training code.
- `QP_problem.py`: Class of quadratic programming benchmark instances.

### Experiments
All experiment data, as well as the code used to generate the figures throughout the thesis, are located in the folder `experiment-output`.
