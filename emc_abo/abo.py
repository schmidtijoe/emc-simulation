"""
framweork for optimization
"""
import numpy as np
import pathlib
from emc_sim import options as eo
from emc_sim import simulations as es
from emc_abo import options
import logging
import multiprocessing as mp
import tqdm
import scipy.optimize as spo
import json
import time
import yabox as yb
import matplotlib.pyplot as plt
import itertools

logModule = logging.getLogger(__name__)


class Optimizer:
    def __init__(self, config: options.Config):
        logModule.info("Init Optimizer")
        self.sim_config_path: pathlib.Path = pathlib.Path(config.emcSimulationConfiguration).absolute()
        if self.sim_config_path.is_file():
            self.emc_params: eo.SimulationParameters = eo.SimulationParameters.load(config.emcSimulationConfiguration)
        else:
            self.emc_params: eo.SimulationParameters = eo.SimulationParameters()
        self.emc_tmp_data: eo.SimulationData = eo.SimulationData()

        # bounds and initial guess
        self.bounds_fa: tuple = (90.0, 180.0)
        self.bounds_phase: tuple = (-180.0, 180.0)
        self.bounds: list = [*[self.bounds_fa] * self.emc_params.sequence.ETL,
                             *[self.bounds_phase] * self.emc_params.sequence.ETL]

        self.rf_0: np.ndarray = np.zeros(2 * self.emc_params.sequence.ETL)
        self.rf_0[:self.emc_params.sequence.ETL] = np.array(self.emc_params.sequence.refocusAngle)

        # for multiprocessing setup
        self.mp: bool = config.multiProcess
        max_cpus = np.max([mp.cpu_count() - config.mpHeadroom, 4])  # leave at least 8 threads, take at least 4
        self.num_cpus = np.min([config.mpNumCpus, max_cpus])

        # for optimization
        self.yabox: bool = config.useYabox
        self.opt_max_iter: int = config.optimMaxIter
        self.opt_popsize: int = config.optimPopsize

        # result
        self.result: spo.OptimizeResult = spo.OptimizeResult()
        self.history_objective: list = []
        self.history_time: list = []

        # for timing
        self.t_start: float = time.time()
        self.t_max: float = config.maxTime

        # for calculations
        # calculate database
        self.param_list: list = self.emc_params.settings.get_complete_param_list()
        # params: [t1, t2, b1, d]
        self.num_param_pairs: int = self.param_list.__len__()
        self.weight_matrix: np.ndarray = np.tril(np.ones((self.num_param_pairs, self.num_param_pairs)), -1)

        self._set_deweight_inter_t2_contrib_matrix()

    def _set_deweight_inter_t2_contrib_matrix(self):
        t2_list = self.emc_params.settings.t2_array
        for t2 in t2_list:
            t2_idxs = []
            for p_idx in range(self.num_param_pairs):
                _, t2p, _, _ = self.param_list[p_idx]
                if t2p == t2:
                    t2_idxs.append(p_idx)
            for o1_idx, o2_idx in itertools.product(t2_idxs, t2_idxs):
                self.weight_matrix[o1_idx, o2_idx] = 0

    def _func_to_optimize(self, x: np.ndarray, verbose=False):
        self.emc_params.sequence.refocusAngle = x[:self.emc_params.sequence.ETL].tolist()
        self.emc_params.sequence.refocusPhase = x[self.emc_params.sequence.ETL:].tolist()

        signal_curves = np.zeros((self.num_param_pairs, self.emc_params.sequence.ETL))

        def wrap_for_mp(idx):
            self.emc_tmp_data.set_run_params(*self.param_list[idx])
            emcAmplitude, _ = es.simulate_mese(
                simParams=self.emc_params,
                simData=self.emc_tmp_data
            )
            return emcAmplitude.emcSignal

        logModule.debug("Simulate")
        if verbose:
            for idx in tqdm.trange(self.num_param_pairs, desc="processing sim"):
                signal_curves[idx] = wrap_for_mp(idx)
        else:
            for idx in range(self.num_param_pairs):
                signal_curves[idx] = wrap_for_mp(idx)

        # calculate correlation
        corr_matrix = np.corrcoef(signal_curves)
        # set diagonal and upper half 0, square
        obj_matrix = np.square(np.tril(corr_matrix, -1))
        # additionally we are not interested in the correlation within a t2 value, between different b1 effectivities
        obj_matrix *= self.weight_matrix
        objective = np.sum(obj_matrix) / self.num_param_pairs ** 2  # want as little correlation
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
        logModule.info(f"evaluate single N step")
        start = time.time()
        _ = self.test_obj_func()
        total = time.time() - start
        logModule.info(f"single run time: {total:.2f} sec; ({total / 60:.1f} min)")
        projected_t = total / self.num_cpus * popsize * max_iter * 2.2
        logModule.info(f"projected time: {projected_t:.2f} sec; ({projected_t / 60:.1f} min)")
        logModule.info(f"will time out after: {self.t_max / 60:.1f} min")
        # reset start time
        self.t_start = time.time()
        if self.yabox:
            self._yabox_optimize()
        else:
            self._scipy_optimize()

        logModule.info(f"total compute time: {(time.time() - self.t_start) / 60:.1f} min")
        logModule.info(f"finished!")

    def _scipy_optimize(self):
        strategy = 'randtobest1bin'
        mutation = 1.3
        logModule.info(f"using scipy optimize: strategy {strategy}")
        if self.mp:
            workers = self.num_cpus
            verbose = [False]
            updating = 'deferred'
        else:
            workers = 1
            verbose = [True]
            updating = 'immediate'
        self.result = spo.differential_evolution(
            self._func_to_optimize, popsize=self.opt_popsize, args=verbose, bounds=self.bounds,
            maxiter=self.opt_max_iter, x0=self.rf_0, workers=workers, disp=True,
            strategy=strategy, mutation=mutation, callback=self._callback, updating=updating
        )

    def _yabox_optimize(self):
        logModule.info(f"using Yabox version: {yb.__version__}")
        pde = yb.algorithms.PDE(
            self._func_to_optimize, bounds=self.bounds, maxiters=self.opt_max_iter, popsize=self.opt_popsize,
            processes=self.num_cpus, mutation=1.0
        )
        pbar = tqdm.tqdm(pde.geniterator(), total=self.opt_max_iter, desc="de iterations",
                         postfix={"objective": 1.0})
        for step in pbar:
            # update result
            idx = step.best_idx
            norm_vector = step.population[idx]
            pbar.set_postfix_str(f"objective: {step.best_fitness:.3f}")
            self.result.x = np.squeeze(pde.denormalize([norm_vector]))
            if self._cb_yb(step.best_fitness):
                # leave when callback true
                break

    def _cb_yb(self, xk) -> bool:
        return self._callback(xk, 0)

    def _callback(self, xk, convergence) -> bool:
        t_curr = time.time() - self.t_start
        self.history_objective.append(xk)
        self.history_time.append(t_curr)
        if self.history_objective.__len__() > 10:
            if np.diff(self.history_objective)[-10:].mean() < 1e-8:
                logModule.info(f"callback: objective change small over last 5 iterations. exiting")
                return True
        if t_curr > self.t_max:
            logModule.info(f"callback: maximum compute time reached. timeout.")
            return True
        return False

    def get_optimal_pulse_specs(self):
        return self.result

    def test_obj_func(self):
        return self._func_to_optimize(self.rf_0, verbose=True)

    def save(self, path):
        rf_fa = self.result.x[:self.emc_params.sequence.ETL]
        rf_phase = self.result.x[self.emc_params.sequence.ETL:]
        if isinstance(rf_fa, np.ndarray):
            rf_fa = rf_fa.tolist()
        if isinstance(rf_phase, np.ndarray):
            rf_phase = rf_phase.tolist()
        save_dict = {
            "function_value": self._func_to_optimize(self.result.x),
            "opt_rf_fa": rf_fa,
            "opt_rf_phase": rf_phase,
            "history_objective": self.history_objective,
            "history_time": self.history_time
        }
        save_path = pathlib.Path(path).absolute()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        logModule.info(f"writing file - {save_path}")
        with open(save_path, "w") as j_file:
            json.dump(save_dict, j_file, indent=2)

    def plot(self):
        plt.style.use('ggplot')
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot()
        ax.plot(self.history_time, self.history_objective)
        plt.show()


if __name__ == '__main__':
    conf = options.Config()
    abo_optimizer = Optimizer(config=conf)
    test_objective = abo_optimizer.test_obj_func()
    print(test_objective)
    abo_optimizer.optimize()
    test_res = abo_optimizer.get_optimal_pulse_specs()
    print(test_res)
    abo_optimizer.save("./emc_abo/result/optim_fa.json")
