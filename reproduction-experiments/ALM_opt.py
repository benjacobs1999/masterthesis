import operator
from functools import reduce
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pickle
import time
from setproctitle import setproctitle
import os
import argparse
import torch

from old_gep_code.utils import my_hash, str_to_bool
import default_args
import cProfile

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# args = {"corrEps": ...}

filepath = "datasets/simple/random_simple_dataset_var100_ineq50_eq50_ex10000"
prob_type = "simple"


def get_opt_results(data, Yvalid, Ytest, Yvalid_precorr=None, Ytest_precorr=None):
    # eps_converge = args["corrEps"]
    eps_converge = 1e-4
    results = {}
    results["valid_eval"] = data.obj_fn(Yvalid).detach().cpu().numpy()
    results["valid_ineq_max"] = (
        torch.max(data.ineq_dist(data.validX, Yvalid), dim=1)[0].detach().cpu().numpy()
    )
    results["valid_ineq_mean"] = (
        torch.mean(data.ineq_dist(data.validX, Yvalid), dim=1).detach().cpu().numpy()
    )
    results["valid_ineq_num_viol_0"] = (
        torch.sum(data.ineq_dist(data.validX, Yvalid) > eps_converge, dim=1)
        .detach()
        .cpu()
        .numpy()
    )
    results["valid_ineq_num_viol_1"] = (
        torch.sum(data.ineq_dist(data.validX, Yvalid) > 10 * eps_converge, dim=1)
        .detach()
        .cpu()
        .numpy()
    )
    results["valid_ineq_num_viol_2"] = (
        torch.sum(data.ineq_dist(data.validX, Yvalid) > 100 * eps_converge, dim=1)
        .detach()
        .cpu()
        .numpy()
    )
    results["valid_eq_max"] = (
        torch.max(torch.abs(data.eq_resid(data.validX, Yvalid)), dim=1)[0]
        .detach()
        .cpu()
        .numpy()
    )
    results["valid_eq_mean"] = (
        torch.mean(torch.abs(data.eq_resid(data.validX, Yvalid)), dim=1)
        .detach()
        .cpu()
        .numpy()
    )
    results["valid_eq_num_viol_0"] = (
        torch.sum(torch.abs(data.eq_resid(data.validX, Yvalid)) > eps_converge, dim=1)
        .detach()
        .cpu()
        .numpy()
    )
    results["valid_eq_num_viol_1"] = (
        torch.sum(
            torch.abs(data.eq_resid(data.validX, Yvalid)) > 10 * eps_converge, dim=1
        )
        .detach()
        .cpu()
        .numpy()
    )
    results["valid_eq_num_viol_2"] = (
        torch.sum(
            torch.abs(data.eq_resid(data.validX, Yvalid)) > 100 * eps_converge, dim=1
        )
        .detach()
        .cpu()
        .numpy()
    )

    if Yvalid_precorr is not None:
        results["valid_correction_dist"] = (
            torch.norm(Yvalid - Yvalid_precorr, dim=1).detach().cpu().numpy()
        )
    results["test_eval"] = data.obj_fn(Ytest).detach().cpu().numpy()
    results["test_ineq_max"] = (
        torch.max(data.ineq_dist(data.testX, Ytest), dim=1)[0].detach().cpu().numpy()
    )
    results["test_ineq_mean"] = (
        torch.mean(data.ineq_dist(data.testX, Ytest), dim=1).detach().cpu().numpy()
    )
    results["test_ineq_num_viol_0"] = (
        torch.sum(data.ineq_dist(data.testX, Ytest) > eps_converge, dim=1)
        .detach()
        .cpu()
        .numpy()
    )
    results["test_ineq_num_viol_1"] = (
        torch.sum(data.ineq_dist(data.testX, Ytest) > 10 * eps_converge, dim=1)
        .detach()
        .cpu()
        .numpy()
    )
    results["test_ineq_num_viol_2"] = (
        torch.sum(data.ineq_dist(data.testX, Ytest) > 100 * eps_converge, dim=1)
        .detach()
        .cpu()
        .numpy()
    )
    results["test_eq_max"] = (
        torch.max(torch.abs(data.eq_resid(data.testX, Ytest)), dim=1)[0]
        .detach()
        .cpu()
        .numpy()
    )
    results["test_eq_mean"] = (
        torch.mean(torch.abs(data.eq_resid(data.testX, Ytest)), dim=1)
        .detach()
        .cpu()
        .numpy()
    )
    results["test_eq_num_viol_0"] = (
        torch.sum(torch.abs(data.eq_resid(data.testX, Ytest)) > eps_converge, dim=1)
        .detach()
        .cpu()
        .numpy()
    )
    results["test_eq_num_viol_1"] = (
        torch.sum(
            torch.abs(data.eq_resid(data.testX, Ytest)) > 10 * eps_converge, dim=1
        )
        .detach()
        .cpu()
        .numpy()
    )
    results["test_eq_num_viol_2"] = (
        torch.sum(
            torch.abs(data.eq_resid(data.testX, Ytest)) > 100 * eps_converge, dim=1
        )
        .detach()
        .cpu()
        .numpy()
    )
    if Ytest_precorr is not None:
        results["test_correction_dist"] = (
            torch.norm(Ytest - Ytest_precorr, dim=1).detach().cpu().numpy()
        )
    return results


# Modifies stats in place
def dict_agg(stats, key, value, op="concat"):
    if key in stats.keys():
        if op == "sum":
            stats[key] += value
        elif op == "concat":
            stats[key] = np.concatenate((stats[key], value), axis=0)
        else:
            raise NotImplementedError
    else:
        stats[key] = value


def main():
    with open(filepath, "rb") as f:
        data = pickle.load(f)
        for attr in dir(data):
            var = getattr(data, attr)
            if not callable(var) and not attr.startswith("__") and torch.is_tensor(var):
                try:
                    setattr(data, attr, var.to(DEVICE))
                except AttributeError:
                    pass
        data._device = DEVICE

        ## Run ALM optimization baseline
        if prob_type == "simple":
            solvers = ["alm"]

        for solver in solvers:
            save_dir = os.path.join(
                "results",
                str(data),
                "baselineOpt-{}".format(solver),
                "run",
                str(time.time()).replace(".", "-"),
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            Yvalid_opt, valid_time_total, valid_time_parallel = data.ALM_solve(
                data.validX
            )
            Ytest_opt, test_time_total, test_time_parallel = data.ALM_solve(data.testX)
            opt_results = get_opt_results(
                data,
                torch.tensor(Yvalid_opt).to(DEVICE),
                torch.tensor(Ytest_opt).to(DEVICE),
            )
            print("ALM")
            print(f"test eval: {opt_results['test_eval']}")
            # print(opt_results['test_ineq_max'])
            # print(opt_results['test_eq_max'])

            # print(opt_results['test_ineq_mean'])
            ### ALPAQA ######
            # Yvalid_opt, valid_time_total, valid_time_parallel = data.alpaqa_solve(
            #     data.validX
            # )
            # Ytest_opt, test_time_total, test_time_parallel = data.alpaqa_solve(data.testX)
            # print("alpaqa")
            # opt_results = get_opt_results(
            #     data,
            #     torch.tensor(Yvalid_opt).to(DEVICE),
            #     torch.tensor(Ytest_opt).to(DEVICE),
            # )
            # print(Ytest_opt.shape)
            # print(f"test eval: {opt_results['test_eval']}")
            # print(f"test eval mean: {np.mean(opt_results['test_eval'])}")
            # print(opt_results['test_ineq_max'])
            # print(opt_results['test_eq_max'])
            # print(opt_results['test_ineq_mean'])
            # print(opt_results['test_eq_mean'])

            #### ALM code again ####
            opt_results.update(
                dict(
                    [
                        ("test_time", test_time_parallel),
                        ("valid_time", valid_time_parallel),
                        ("train_time", 0),
                        ("test_time_total", test_time_total),
                        ("valid_time_total", valid_time_total),
                        ("train_time_total", 0),
                    ]
                )
            )
            with open(os.path.join(save_dir, "results.dict"), "wb") as f:
                pickle.dump(opt_results, f)


if __name__ == "__main__":
    main()
