import pprint

import numpy as np
import logging
from emc_sim.options import SimulationParameters, SimulationTempData, SimulationData
from emc_sim import functions
import time

logModule = logging.getLogger(__name__)


def simulate_mese(simParams: SimulationParameters, simData: SimulationData,
                  gradientPulseData: dict, arrayTiming: np.ndarray) -> (
        SimulationData, SimulationParameters):
    """
    For a single combination of T1, T2 and B1 value the sequence response is simulated iteratively,
    depending on the sequence scheme.
    This is the main function that needs to be addressed when putting in new sequence parameters.
    Also check out pulse profile files when using verse or other pulse schemes.

    :return: simData, simParams
    """

    logModule.debug(f"Start Simulation: params {pprint.pformat(simData.get_run_params())}")
    # ----- running ----- #
    t_start = time.time()
    # globals and sample are initiated within the SimulationParameters class
    tempData = SimulationTempData(simParams)
    # we take the parameters of the specific run by assigning directly to the run obj of temp
    tempData.run = simData

    # ----- Starting Calculations ----- #
    logModule.debug('run 1')

    tempData = functions.propagateGradientPulseTime(
        dictGradPulse=gradientPulseData["excitation"],
        simParams=simParams,
        simTempData=tempData
    )

    # first refocus is different
    tempData = functions.propagateRelaxation(
        deltaT=arrayTiming[0, 0],
        simTempData=tempData,
        simParams=simParams
    )

    tempData = functions.propagateGradientPulseTime(
        dictGradPulse=gradientPulseData["refocus_1"],
        simParams=simParams,
        simTempData=tempData
    )

    tempData = functions.propagateRelaxation(
        deltaT=arrayTiming[0, 1],
        simParams=simParams,
        simTempData=tempData
    )

    for acqIdx in range(simParams.settings.acquisitionNumber):
        tempData = functions.propagateGradientPulseTime(
            dictGradPulse=gradientPulseData["acquisition"],
            simParams=simParams,
            simTempData=tempData,
            append=False
        )

        tempData.signalArray[0, acqIdx] = (np.sum(tempData.magnetizationPropagation[-1][0])
                                           + 1j * np.sum(tempData.magnetizationPropagation[-1][1])) \
                                           * 100 * simParams.settings.lengthZ / simParams.settings.sampleNumber
        # signal scaled by distance between points (not entirely sure if this makes a difference

    # ----- refocusing loop - echo train -----
    for loopIdx in np.arange(1, simParams.sequence.ETL):
        logModule.debug(f'run {loopIdx + 1}')

        tempData = functions.propagateRelaxation(
            deltaT=arrayTiming[loopIdx, 0],
            simTempData=tempData,
            simParams=simParams
        )

        tempData = functions.propagateGradientPulseTime(
            dictGradPulse=gradientPulseData["refocus"],
            simParams=simParams,
            simTempData=tempData
        )

        tempData = functions.propagateRelaxation(
            deltaT=arrayTiming[loopIdx, 1],
            simTempData=tempData,
            simParams=simParams
        )

        for acqIdx in range(simParams.settings.acquisitionNumber):
            tempData = functions.propagateGradientPulseTime(
                dictGradPulse=gradientPulseData["acquisition"],
                simParams=simParams,
                simTempData=tempData,
                append=False
            )

            tempData.signalArray[loopIdx, acqIdx] = (np.sum(tempData.magnetizationPropagation[-1][0])
                                                     + 1j * np.sum(tempData.magnetizationPropagation[-1][1])) \
                                                     * 100 * simParams.settings.lengthZ / \
                                                     simParams.settings.sampleNumber
            # signal scaled by distance between points (not entirely sure if this makes a difference

    # ----- finished loop -----

    logModule.debug('Signal array processing fourier')
    imageArray = np.fft.fftshift(np.fft.fft(np.fft.fftshift(tempData.signalArray)))
    simData.emcSignal = 2 * np.sum(np.abs(imageArray), axis=1) / simParams.settings.acquisitionNumber
    # factor 2 not necessary, stems from Noams version, ultimately want some normalization here!
    simData.time = time.time() - t_start

    # for debugging
    # np.save(f"mag_prop.npy", tempData.magnetizationPropagation)
    return simData, simParams
