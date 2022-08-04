import logging
from pathlib import Path
import pandas as pd
from emc_sim import options, simulations, utils, prep
import multiprocessing as mp
import tqdm
import time
from itertools import chain
import pprint
logging.getLogger('matplotlib.font_manager').disabled = True


def simulate_single(
        simParams: options.SimulationParameters,
        simData: options.SimulationData,
        save: bool = False) -> (pd.DataFrame, options.SimulationParameters):
    # prep pulse gradient data
    # globals and sample are initiated within the SimulationParameters class
    prep.init_prep_for_visualization(simParams=simParams)

    # estimate single process
    simData, simParams = simulations.simulate_mese(
        simParams=simParams,
        simData=simData
    )

    logging.info(f'projected time: '
                 f'{simData.time * simParams.settings.total_num_sim / 3600 / simParams.config.mpNumCpus:.2f} h\t\t'
                 f'({simData.time * simParams.settings.total_num_sim / 60 / simParams.config.mpNumCpus:.1f} min)\n'
                 f'for {simParams.settings.total_num_sim} curves')

    param_list = simParams.settings.get_complete_param_list()
    emcAmplitude_resultlist = []

    logging.info("Simulate")
    for item in tqdm.tqdm(param_list, ncols=20):
        simData.set_run_params(*item)
        emcAmplitude, _ = simulations.simulate_mese(
            simParams=simParams,
            simData=simData
        )
        emcAmplitude_resultlist.append(emcAmplitude.to_dict())
    dataBase = pd.DataFrame(emcAmplitude_resultlist)

    if save:
        utils.save_database(database=dataBase, simParams=simParams)
    return dataBase, simParams


def simulate_multi(
        simParams: options.SimulationParameters,
        simData: options.SimulationData,
        save=True):
    """
        This function creates and simulates the curves with multiprocessing, clocks full cpu power (check num_cpus)
        make sure simulation values are created accordingly

        :return: array of curves and corresponding simulation values
        """
    # prep pulse gradient data
    # globals and sample are initiated within the SimulationParameters class
    prep.init_prep_for_visualization(simParams=simParams)

    # estimate single process
    _, simParams = simulations.simulate_mese(
        simParams=simParams,
        simData=simData
    )

    # ---- using multiprocessing ---

    logging.info(f"Number of CPUs to use: {simParams.config.mpNumCpus}")
    logging.info(f'projected time: '
                 f'{simData.time * simParams.settings.total_num_sim / 3600 / simParams.config.mpNumCpus:.2f} h\t\t'
                 f'({simData.time * simParams.settings.total_num_sim / 60 / simParams.config.mpNumCpus:.1f} min)\n'
                 f'for {simParams.settings.total_num_sim} curves')

    logging.info("Simulate")
    # divide lists in as many parts as we have processes available (cpus)
    param_list = simParams.settings.get_complete_param_list()
    mp_lists = [(simParams, simData, param_list[i::simParams.config.mpNumCpus])
                for i in range(simParams.config.mpNumCpus)]

    start = time.time()

    pool = mp.Pool(simParams.config.mpNumCpus)
    results = pool.map(wrapSimulateForMP, mp_lists)

    result_list_of_dict = list(chain(*results))
    dataBase = pd.DataFrame(result_list_of_dict)

    end = time.time()

    logging.info(f'Finished simulation! Total time: {((end - start) / 3600):.2} h \t\t ({((end - start) /60):.1f} min)')
    # df = pd.DataFrame(results)

    if save:
        utils.save_database(database=dataBase, simParams=simParams)


def wrapSimulateForMP(args) -> list:
    """
    When using multiprocessing we want to distribute lists of single run parameters to multiple processes.
    """
    simParams = args[0]
    simData = args[1]
    mp_list = args[2]
    emcAmplitude_resultlist = []
    for item in mp_list:
        simData.set_run_params(*item)
        emcAmplitude, _ = simulations.simulate_mese(
            simParams=simParams,
            simData=simData)
        emcAmplitude_resultlist.append(emcAmplitude.to_dict())
    return emcAmplitude_resultlist


def main():
    """
    Main function that runs training/prediction defined by command line arguments
    """
    parser, prog_args = options.createCommandlineParser()

    simParams = options.SimulationParameters.from_cmd_args(prog_args)
    # set logging level after possible config file read
    if simParams.config.debuggingFlag:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%I:%M:%S', level=level)
    simData = options.SimulationData.from_cmd_args(prog_args)

    logging.info("starting simulation")
    logging.info("___ sequence dependent configuration of timing and pulses! ___")
    logging.info("Configuration")
    logging.info(f"spatial resolution set: {simParams.settings.lengthZ * 2 / simParams.settings.sampleNumber * 1e6:.3f}"
                 f" um")
    logging.debug(pprint.pformat(simParams.to_dict()))
    try:
        if simParams.config.multiprocessing:
            simulate_multi(simParams, simData)
        else:
            simulate_single(simParams, simData, save=True)
    except AttributeError as e:
        print(e)
        parser.print_usage()


def test():
    logging.basicConfig(level=logging.DEBUG)
    parser, prog_args = options.createCommandlineParser()
    simParams = options.SimulationParameters.from_cmd_args(prog_args)
    simData = options.SimulationData.from_cmd_args(prog_args)
    try:
        simulate_single(simParams, simData, save=True)
    # exit(0)
    except AttributeError as e:
        print(e)
        parser.print_usage()


if __name__ == '__main__':
    main()
