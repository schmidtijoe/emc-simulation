"""
framweork for optimization
"""
import numpy as np
import pathlib
import typing
from emc_sim import options as eo
from emc_sim import simulations as es
import logging
import multiprocessing as mp
import tqdm
import scipy.optimize as spo
import json
import time

logModule = logging.getLogger(__name__)


class Optimizer:
    def __init__(self,
                 config_path: typing.Union[str, pathlib.Path] = "",
                 multiprocessing: bool = False, mp_num_cpus: int = 4):
        logModule.info("Init Optimizer")
        self.config_path: pathlib.Path = pathlib.Path(config_path).absolute()
        if self.config_path.is_file():
            self.emc_params: eo.SimulationParameters = eo.SimulationParameters.load(config_path)
        else:
            self.emc_params: eo.SimulationParameters = eo.SimulationParameters()
        self.emc_tmp_data: eo.SimulationData = eo.SimulationData()

        # bounds and initial guess
        self.bounds_fa: tuple = (90.0, 180.0)
        self.bounds_phase: tuple = (-180.0, 180.0)
        self.bounds: list = [
            *[self.bounds_fa]*self.emc_params.sequence.ETL,
            *[self.bounds_phase]*self.emc_params.sequence.ETL
        ]

        self.rf_0: np.ndarray = np.zeros(2*self.emc_params.sequence.ETL)
        self.rf_0[:self.emc_params.sequence.ETL] = np.array(self.emc_params.sequence.refocusAngle)

        # for multiprocessing setup
        self.mp: bool = multiprocessing
        max_cpus = np.max([mp.cpu_count() - 8, 4])        # leave at least 8 threads, take at least 4
        self.num_cpus = np.min([mp_num_cpus, max_cpus])

        # for optimization
        self.opt_max_iter: int = 50
        self.opt_popsize: int = 15

        # result
        self.result: spo.OptimizeResult = spo.OptimizeResult()

    def _func_to_optimize(self, x: np.ndarray, verbose=True):
        self.emc_params.refocusAngle = x[:self.emc_params.sequence.ETL].tolist()
        self.emc_params.refocusPhase = x[self.emc_params.sequence.ETL:].tolist()

        # calculate database
        param_list = self.emc_params.settings.get_complete_param_list()
        idx_list = np.arange(len(param_list))

        def wrap_for_mp(idx):
            self.emc_tmp_data.set_run_params(*param_list[idx])
            emcAmplitude, _ = es.simulate_mese(
                simParams=self.emc_params,
                simData=self.emc_tmp_data
            )
            return emcAmplitude.emcSignal
        logModule.debug("Simulate")

        results = []
        if verbose:
            for idx in tqdm.trange(idx_list.__len__(), desc="processing sim"):
                results.append(wrap_for_mp(idx))
        else:
            for idx in range(idx_list.__len__()):
                results.append(wrap_for_mp(idx))

        signal_curves = np.zeros((len(results), self.emc_params.sequence.ETL))

        for idx in range(len(results)):
            signal_curves[idx] = results[idx]

        # calculate correlation
        corr_matrix = np.corrcoef(signal_curves)
        # set diagonal and upper half 0, square
        obj_matrix = np.square(np.tril(corr_matrix, -1))
        objective = np.sum(obj_matrix) / len(results)**2  # want as little correlation
        return objective

    def optimize(self):
        popsize = self.opt_popsize
        max_iter = self.opt_max_iter
        N = self.rf_0.shape[0]
        logModule.info("start optimization")
        logModule.info(f"heavy compute: using {self.num_cpus} workers.")
        logModule.info(f"involving N * popsize * max_iter computation steps.")
        logModule.info(f"N = {N}; popsize = {popsize}; max_iter = {max_iter}")
        logModule.info(f"total: {N * popsize * max_iter}")
        logModule.info(f"evaluate single step")
        start = time.time()
        _ = self.test_obj_func()
        total = time.time() - start
        logModule.info(f"single run time: {total:.2f} sec; ({total / 60:.1f} min)")
        projected_t = total / self.num_cpus * N * popsize * max_iter * 1.2
        logModule.info(f"projected time: {projected_t:.2f} sec; ({projected_t / 60:.1f} min)")
        if self.mp:
            workers = self.num_cpus
            verbose = [False]
        else:
            workers = 1
            verbose = [True]
        self.result = spo.differential_evolution(
            self._func_to_optimize, popsize=popsize, args=verbose, bounds=self.bounds,
            maxiter=max_iter, x0=self.rf_0, workers=workers, disp=True
        )
        logModule.info(f"finished!")

    def get_optimal_pulse_specs(self):
        return self.result

    def test_obj_func(self):
        return self._func_to_optimize(self.rf_0)

    def save(self, path):
        rf_fa = self.result.x[:self.emc_params.sequence.ETL]
        rf_phase = self.result.x[self.emc_params.sequence.ETL:]
        if isinstance(rf_fa, np.ndarray):
            rf_fa = rf_fa.tolist()
        if isinstance(rf_phase, np.ndarray):
            rf_phase = rf_phase.tolist()
        save_dict = {
            "function_value": self.result.fun,
            "opt_rf_fa": rf_fa,
            "opt_rf_phase": rf_phase
        }
        save_path = pathlib.Path(path).absolute()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        logModule.info(f"writing file - {save_path}")
        with open(save_path, "w") as j_file:
            json.dump(save_dict, j_file, indent=2)


if __name__ == '__main__':
    abo_optimizer = Optimizer(config_path="./emc_abo/config/emc_config.json", multiprocessing=False)
    test_objective = abo_optimizer.test_obj_func()
    print(test_objective)
    abo_optimizer.optimize()
    test_res = abo_optimizer.get_optimal_pulse_specs()
    print(test_res)
    abo_optimizer.save("./emc_abo/result/optim_fa.json")
