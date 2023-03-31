"""
Main script for optimization
"""
import pathlib
import numpy as np
import scipy.optimize as spo
from emc_sim import options as e_opt
from emc_sim import simulations as e_sim
import tqdm
import inspect


# function for optimization
def func_to_optimize(x: np.ndarray,
                     emc_params: e_opt.SimulationParameters,
                     tmpSimData: e_opt.SimulationData = e_opt.SimulationData()):
    # set rf flip angle and phase to the params object
    rf_fa_array = x[:etl]
    rf_phase_array = x[etl:]
    emc_params.sequence.refocusAngle = rf_fa_array.tolist()
    emc_params.sequence.refocusPhase = rf_phase_array.tolist()

    # calculate database
    param_list = emc_params.settings.get_complete_param_list()
    emcAmplitude_resultlist = []

    # logging.info("Simulate")
    for item in tqdm.tqdm(param_list, ncols=20):
        tmpSimData.set_run_params(*item)
        emcAmplitude, _ = e_sim.simulate_mese(
            simParams=emc_params,
            simData=tmpSimData
        )
        emcAmplitude_resultlist.append(emcAmplitude.emcSignal)

    # extract curves from db
    signal_curves = np.zeros((len(emcAmplitude_resultlist), etl))
    for k_idx in range(len(emcAmplitude_resultlist)):
        signal_curves[k_idx] = emcAmplitude_resultlist[k_idx]

    # calculate correlation
    corr_matrix = np.corrcoef(signal_curves)
    objective = np.sum(corr_matrix)  # diagonals always sum to 1
    return objective


if __name__ == '__main__':
    module = inspect.getmodule(func_to_optimize)
    print("Function", func_to_optimize.__name__, "is defined in module", module.__name__)

    # __initializing__
    # first we need to get a function that takes the rf fa and phases as inputs and etl
    etl = 7

    # set bounds for optimization, first half are flip angles, second are phase
    bounds = [*[(90.0, 180.0)] * etl, *[(-180.0, 180.0)] * etl]

    # pick starting point semc
    rf_0 = np.zeros(2 * etl)
    rf_0[:etl] = 180.0

    # load in emc_configuration
    config_path = pathlib.Path(__name__).absolute()
    config_path = config_path.parent.joinpath("config/emc_config.json")

    emc_opts = e_opt.SimulationParameters.load(config_path)
    emc_sim_data = e_opt.SimulationData()

    # test = func_to_optimize(rf_0, emc_opts, emc_sim_data)
    # print(test)

    # find global minimum of multivariate function with differential evolution
    result = spo.differential_evolution(
        func_to_optimize, bounds=bounds, args=[emc_opts, emc_sim_data],
        maxiter=20, x0=rf_0, workers=4
    )

    print(result)



