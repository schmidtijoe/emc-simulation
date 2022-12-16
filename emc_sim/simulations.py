import pprint

import numpy as np
import logging
from emc_sim.options import SimulationParameters, SimulationTempData, SimulationData
from emc_sim import functions, plotting, prep
import time

logModule = logging.getLogger(__name__)


def simulate_mese(simParams: SimulationParameters, simData: SimulationData) -> (
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
    # prep pulse gradient data

    # globals and sample are initiated within the SimulationParameters class
    tempData = SimulationTempData(simParams)
    # we take the parameters of the specific run by assigning directly to the run obj of temp
    tempData.run = simData

    # ----- prep sequence ----- #
    gp_excitation, gps_refocus, timing, acquisition = prep.gradientPulsePreparationSEMC(
        simParams=simParams, simTempData=tempData)

    # ----- Starting Calculations ----- #
    logModule.debug('excitation')

    tempData = functions.propagateGradientPulseTime(
        grad_pulse=gp_excitation,
        simParams=simParams,
        simTempData=tempData
    )

    # if simParams.config.debuggingFlag and simParams.config.visualize:
    #     # for debugging
    #     plotting.plotMagnetization(tempData)

    for loopIdx in np.arange(0, simParams.sequence.ETL):
        # ----- refocusing loop - echo train -----
        logModule.debug(f'run {loopIdx + 1}')

        # delay before pulse
        tempData = functions.propagateRelaxation(deltaT=timing.time_pre_pulse[loopIdx], simTempData=tempData)

        # pulse
        tempData = functions.propagateGradientPulseTime(
            grad_pulse=gps_refocus[loopIdx],
            simParams=simParams,
            simTempData=tempData
        )

        # if simParams.config.debuggingFlag and simParams.config.visualize:
        #     # for debugging
        #     plotting.plotMagnetization(tempData)

        # delay after pulse
        tempData = functions.propagateRelaxation(deltaT=timing.time_post_pulse[loopIdx], simTempData=tempData)

        # acquisition
        for acqIdx in range(simParams.settings.acquisitionNumber):
            tempData = functions.propagateGradientPulseTime(
                grad_pulse=acquisition,
                simParams=simParams,
                simTempData=tempData,
                append=False
            )

            tempData.signalArray[loopIdx, acqIdx] = (np.sum(tempData.magnetizationPropagation[-1][0])
                                               + 1j * np.sum(tempData.magnetizationPropagation[-1][1])) \
                                               * 100 * simParams.settings.lengthZ / simParams.settings.sampleNumber
            # signal scaled by distance between points (not entirely sure if this makes a difference
        # if simParams.config.debuggingFlag and simParams.config.visualize:
        #     # for debugging
        #     plotting.plotMagnetization(tempData)
    # ----- finished loop -----

    logModule.debug('Signal array processing fourier')
    imageArray = np.fft.fftshift(np.fft.fft(np.fft.fftshift(tempData.signalArray)))
    simData.emcSignal = 2 * np.sum(np.abs(imageArray), axis=1) / simParams.settings.acquisitionNumber
    if simParams.sequence.ETL % 2 > 0:
        # for some reason we get a shift from the fft when used with odd array length.
        simData.emcSignal = np.roll(simData.emcSignal, 1)
    # factor 2 not necessary, stems from Noams version, ultimately want some normalization here!
    simData.time = time.time() - t_start

    if simParams.config.debuggingFlag and simParams.config.visualize:
        # for debugging
        plotting.visualizeSignalResponse(simData.emcSignal, (simData.t2, simData.b1))
    return simData, simParams
