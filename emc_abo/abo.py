"""
framework for optimization
"""
import numpy as np
import pathlib
import emc_sim
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
            self.emc_params: emc_sim.options.SimulationParameters = emc_sim.options.SimulationParameters.load(
                config.emcSimulationConfiguration)
        else:
            self.emc_params: emc_sim.options.SimulationParameters = emc_sim.options.SimulationParameters()
        self.emc_tmp_data: emc_sim.options.SimulationData = emc_sim.options.SimulationData()
        self.vary_phase: bool = config.varyPhase

        # bounds and initial guess
        self.bounds_fa: tuple = (60.0, 180.0)
        self.bounds_phase: tuple = (-180.0, 180.0)
        self.bounds: list = [*[self.bounds_fa] * self.emc_params.sequence.ETL]
        self.rf_0: np.ndarray = np.zeros(self.emc_params.sequence.ETL)
        if self.vary_phase:
            self.bounds.extend([*[self.bounds_phase] * self.emc_params.sequence.ETL])
            self.rf_0: np.ndarray = np.zeros(2 * self.emc_params.sequence.ETL)
        self.rf_0[:self.emc_params.sequence.ETL] = self.emc_params.sequence.refocusAngle

        # for multiprocessing setup
        self.mp: bool = config.multiProcess
        max_cpus = np.max([mp.cpu_count() - config.mpHeadroom, 4])  # leave at least 8 threads, take at least 4
        self.num_cpus = np.min([config.mpNumCpus, max_cpus])
        # catch configuration errors - we disable multiprocessing in emc calls here,
        # even though they shouldnt be used in the implementation as is
        if self.mp:
            # if we want to mutiprocess the optimization we cant also multiprocess the simulation that is called.
            # Otherwise we end up with a number of workers x trying to spawn a number of workers y on every call
            self.emc_params.config.multiprocessing = False
        else:
            # at least log to user that its happening inside the simulation config
            if self.emc_params.config.multiprocessing:
                logModule.info(f"EMC simulation function call is using multiprocessing under the hood."
                               f"ABO implementation is ignoring EMC multiprocessing settings")
        # for optimization
        self.yabox: bool = config.useYabox
        self.opt_max_iter: int = config.optimMaxIter
        self.opt_popsize: int = config.optimPopsize
        self.opt_mutation: float = config.optimMutation
        self.opt_crossover: float = config.optimCrossover
        self.opt_lambda: float = config.optimLambda

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

        self._set_inter_t2_contrib_weight_matrix()

    def _set_inter_t2_contrib_weight_matrix(self):
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
        if self.vary_phase:
            self.emc_params.sequence.refocusPhase = x[self.emc_params.sequence.ETL:].tolist()

        signal_curves = np.zeros((self.num_param_pairs, self.emc_params.sequence.ETL))

        def wrap_for_mp(idx):
            self.emc_tmp_data.set_run_params(*self.param_list[idx])
            emcAmplitude, _ = emc_sim.simulations.simulate_mese(
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
        # normalize -> as in dictionary fit
        norm = np.linalg.norm(signal_curves, axis=-1, keepdims=True)
        # usually norm in emc  below 0.1 dependent on num samples
        signal_curves = np.divide(
            signal_curves,
            norm,
            where=norm > 1e-9,
            out=np.zeros_like(signal_curves)
        )
        # objective snr -> want to maximize the norm (minimize negative), i.e possible signal for all curves
        snr_objective = - np.sum(norm)

        # calculate correlation
        corr_matrix = np.corrcoef(signal_curves)
        # set diagonal and upper half 0, square
        obj_matrix = np.square(np.tril(corr_matrix, -1))
        # additionally we are not interested in the correlation within a t2 value, between different b1 effectivities
        obj_matrix *= self.weight_matrix
        corr_objective = np.sum(obj_matrix) / self.num_param_pairs  # want as little correlation
        # objective --> introduce some kind of weighting between snr and correlations, test with little params
        # -> norm component around 10 times smaller than correlation component
        objective = self.opt_lambda * corr_objective + (1 - self.opt_lambda) * snr_objective
        return 1e2 * objective      # just nicer tracking in output

    def optimize(self):
        popsize = self.opt_popsize
        max_iter = self.opt_max_iter
        N = self.rf_0.shape[0]
        logModule.info("start optimization")
        logModule.info(f"heavy compute: assigned {self.num_cpus} workers.")
        logModule.info(f"involving N * popsize * max_iter computation steps.")
        logModule.info(f"N = {N}; popsize = {popsize}; max_iter = {max_iter}; total: {N * popsize * max_iter}")
        logModule.info(f"evaluate single N step")
        start = time.time()
        _ = self.test_obj_func()
        total = time.time() - start
        logModule.info(f"single run time: {total:.2f} sec; ({total / 60:.1f} min)")
        projected_t = total / self.num_cpus * popsize * max_iter * 3
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
        strategy = 'best1bin'
        logModule.info(f"using scipy optimize: strategy {strategy}")
        if self.mp:
            if self.num_cpus > self.opt_popsize * self.rf_0.shape[0]:
                self.num_cpus = self.opt_popsize * self.rf_0.shape[0]
                logModule.info(f"Scipy uses population of N ({self.rf_0.shape[0]}) x "
                               f"popsize ({self.opt_popsize}) per step.")
                logModule.info(f"adopted number of workers: {self.num_cpus}")
            workers = self.num_cpus
            verbose = [False]
            updating = 'deferred'
        else:
            workers = 1
            verbose = [True]
            updating = 'immediate'
        self.result = spo.differential_evolution(
            self._func_to_optimize, popsize=self.opt_popsize, args=verbose, bounds=self.bounds,
            maxiter=self.opt_max_iter, x0=self.rf_0, workers=workers, disp=True, recombination=self.opt_crossover,
            strategy=strategy, mutation=self.opt_mutation, callback=self._callback, updating=updating, polish=False
        )

    def _yabox_optimize(self):
        logModule.info(f"using Yabox version: {yb.__version__}")
        # parallelization with yabox allows spawning of population in parallel, ie. we are only using #popsize
        # workers at once, no need to allocate more
        if self.opt_popsize % self.rf_0.shape[0] > 0:
            self.opt_popsize = np.max([self.opt_popsize - self.opt_popsize % self.rf_0.shape[0], self.rf_0.shape[0]])
            logModule.info(f"Yabox uses population independent of length of variable. "
                           f"Adopting popsize to {self.opt_popsize}")
        if self.num_cpus > self.opt_popsize:
            self.num_cpus = self.opt_popsize
            logModule.info(f"Yabox allows popsize ({self.opt_popsize}) evaluations per step")
            logModule.info(f"adopted number of workers: {self.num_cpus}")
        pde = yb.algorithms.PDE(
            self._func_to_optimize, bounds=self.bounds, maxiters=self.opt_max_iter, popsize=self.opt_popsize,
            processes=self.num_cpus, mutation=self.opt_mutation, crossover=self.opt_crossover
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
        if isinstance(xk, np.ndarray):
            xk = self._func_to_optimize(xk)
        self.history_objective.append(xk)
        self.history_time.append(t_curr)
        if self.history_objective.__len__() > 10:
            if np.abs(np.diff(self.history_objective)[-8:].mean()) < 1e-8:
                logModule.info(f"callback: objective change small over last 8 iterations. exiting")
                return True
        if t_curr > self.t_max:
            logModule.info(f"callback: maximum compute time reached. timemc_sim.optionsut.")
            return True
        return False

    def get_optimal_pulse_specs(self):
        return self.result

    def test_obj_func(self):
        return self._func_to_optimize(self.rf_0, verbose=True)

    class SetJEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                for idx in range(obj.__len__()):
                    if isinstance(obj[idx], float):
                        obj[idx] = np.round(obj[idx], 1)
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    def save(self, path):
        rf_fa = self.result.x[:self.emc_params.sequence.ETL]
        rf_phase = self.result.x[self.emc_params.sequence.ETL:]
        if not self.vary_phase:
            # if phase wasn't varied copy original setting
            rf_phase = self.emc_params.sequence.refocusPhase
        save_dict = {
            "function_value": self.history_objective[-1],
            "opt_rf_fa": rf_fa,
            "opt_rf_phase": rf_phase,
            "history_objective": self.history_objective,
            "history_time": self.history_time
        }
        # set encoder
        j_enc = self.SetJEncoder
        save_path = pathlib.Path(path).absolute()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        logModule.info(f"writing file - {save_path}")
        with open(save_path, "w") as j_file:
            json.dump(save_dict, j_file, indent=2, cls=j_enc)

    def plot(self):
        plt.style.use('ggplot')
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot()
        x_ax = np.array(self.history_time) / 60.0
        y_ax = np.array(self.history_objective)
        ax.set_xlabel("time [min]")
        ax.set_ylabel("objective function [a.u.]")
        ax.plot(x_ax, y_ax)
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
