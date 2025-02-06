# masterthesis

### Environment
To create an environment with the correct packages, type the following commands in the terminal:

1. `conda create -n {env_name} python=3.9.20`
2. `conda activate {env_name}`
3. `pip install -r requirements.txt`

### File explanations

- `config.json`: Config file for problem instances and PDL parameters.
- `config.toml`: Config file for solver (old code from thesis Matthijs Arnoldus).
- `data_wrangling.py`: Code for reading input data into a format we can use.
- `gep_config_parser.py`: Parses the config.toml.
- `gep_main.py`: Methods for solving the problem with Gurobi.
- `gep_primal_dual_main.py`: Main script for predicting problem instances with PDL.
- `gep_problem_operational.py`: Class creating operational problem instances
- `gep_problem.py`: Class creating GEP problem instances
- `get_gurobi_vars.py`: Class for extracting optimal primal and dual variables from Gurobi
- `primal_dual.py`: Class with PDL (and all deep learning) code.

