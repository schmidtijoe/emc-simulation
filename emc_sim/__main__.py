import logging
from pathlib import Path

import numpy as np
import pandas as pd
from emc_sim import options, simulations, utils
import multiprocessing as mp
import time
from itertools import chain
import pprint


def simulate_single(
        simParams: options.SimulationParameters,
        simData: options.SimulationData,
        save: bool = False):

    param_list = simParams.settings.get_complete_param_list()
    emcAmplitude_resultlist = []
    for item in param_list:
        simData.set_run_params(*item)
        emcAmplitude, _ = simulations.simulate_mese(simParams, simData)
        emcAmplitude_resultlist.append(emcAmplitude.to_dict())
    dataBase = pd.DataFrame(emcAmplitude_resultlist)

    if save:
        path = Path(simParams.config.savePath)
        utils.create_folder_ifn_exist(path)
        dataBase.to_json(path.joinpath(simParams.config.saveFile), indent=2)
        simParams.save(path.joinpath("SimulationConfiguration.json"), indent=2, separators=(',', ':'))
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
    # estimate single process
    _, simParams = simulations.simulate_mese(simParams=simParams, simData=simData)

    # ---- using multiprocessing ---

    print("cpu number: {}".format(simParams.config.mpNumCpus))
    logging.warning(f'projected time: '
                    f'{simData.time * simParams.settings.total_num_sim / 360 / simParams.config.mpNumCpus:.2f} h')

    # divide lists in as many parts as we have processes available (cpus)
    param_list = simParams.settings.get_complete_param_list()
    mp_lists = [(simParams, simData, param_list[i::simParams.config.mpNumCpus])
                for i in range(simParams.config.mpNumCpus)]

    logging.basicConfig(level="INFO")

    start = time.time()

    pool = mp.Pool(simParams.config.mpNumCpus)
    results = pool.map(wrapSimulateForMP, mp_lists)

    result_list_of_dict = list(chain(*results))
    dataBase = pd.DataFrame(result_list_of_dict)

    end = time.time()

    logging.warning(f'Finished simulation! Total time: {((end - start) / 3600):.2} h')
    # df = pd.DataFrame(results)

    if save:
        path = Path(simParams.config.savePath)
        utils.create_folder_ifn_exist(path)
        dataBase.to_json(path.joinpath(simParams.config.saveFile), indent=2)
        simParams.save(path.joinpath("SimulationConfiguration.json"), indent=2, separators=(',', ':'))


def wrapSimulateForMP(args):
    """
    When using multiprocessing we want to distribute lists of single run parameters to multiple processes.
    """
    simParams = args[0]
    simData = args[1]
    mp_list = args[2]
    emcAmplitude_resultlist = []
    for item in mp_list:
        simData.set_run_params(*item)
        emcAmplitude, _ = simulations.simulate_mese(simParams, simData)
        emcAmplitude_resultlist.append(emcAmplitude.to_dict())
    return emcAmplitude_resultlist


def main():
    """
    Main function that runs training/prediction defined by command line arguments
    """
    parser, prog_args = options.createCommandlineParser()

    logging.basicConfig(level=logging.INFO)
    logging.info("starting simulation")
    logging.warning("___ sequence dependent configuration of timing and pulses! ___")

    simParams = options.SimulationParameters.from_cmd_args(prog_args)
    simData = options.SimulationData.from_cmd_args(prog_args)

    logging.info("Configuration")
    pprint.pprint(simParams.to_dict())
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
