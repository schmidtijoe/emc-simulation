import numpy as np
import logging
from emc_sim.options import SimulationParameters, SimulationTempData, SimulationData
from emc_sim import functions
from emc_sim import prep
from emc_sim import plotting
import time


def simulate_mese(simParams: SimulationParameters, simData: SimulationData):
    """
    For a single combination of T1, T2 and B1 value the sequence response is simulated iteratively,
    depending on the sequence scheme.
    This is the main function that needs to be addressed when putting in new sequence parameters.
    Also check out pulse profile files when using verse or other pulse schemes.

    :return: simData, simParams, time_total
    """
    # ----- running ----- #
    t_start = time.time()
    # globals and sample are initiated within the SimulationParameters class
    tempData = SimulationTempData(simParams)
    # we take the parameters of the specific run by assigning directly to the run obj of temp
    tempData.run = simData

    # ----- defining pulses ----- #
    logging.debug('Pulse preparation - Timing')
    GradientPulseData = prep.gradientPulsePreparation(simParams=simParams, simTempData=tempData)

    # ----- setup timing ----- #
    arrayTiming = prep.buildFillTiming_mese(simParams)

    # visualize
    if simParams.config.visualize:
        logging.debug('Visualization ON - Turn off when processing multiple instances!')
        plotting.visualizeAllGradientPulses(GradientPulseData)
        plotting.visualizeSequenceScheme(GradientPulseData, arrayTiming, simParams)

    # ----- Starting Calculations ----- #
    logging.debug('Simulation main calculation')

    logging.debug('propagating excitation pulse')
    tempData = functions.propagateGradientPulseTime(
        dictGradPulse=GradientPulseData["excitation"],
        simParams=simParams,
        simTempData=tempData
    )

    # first refocus is different
    logging.debug('fill time before pulse')
    tempData = functions.propagateRelaxation(
        deltaT=arrayTiming[0, 0],
        simTempData=tempData,
        simParams=simParams
    )

    logging.debug('apply pulse')
    tempData = functions.propagateGradientPulseTime(
        dictGradPulse=GradientPulseData["refocus_1"],
        simParams=simParams,
        simTempData=tempData
    )

    logging.debug('fill time after pulse')
    tempData = functions.propagateRelaxation(
        deltaT=arrayTiming[0, 1],
        simParams=simParams,
        simTempData=tempData
    )

    logging.debug('acquisition')
    for acqIdx in range(simParams.settings.acquisitionNumber):
        tempData = functions.propagateGradientPulseTime(
            dictGradPulse=GradientPulseData["acquisition"],
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
        logging.debug(f'run {loopIdx + 1}')

        logging.debug('fill time before pulse')
        tempData = functions.propagateRelaxation(
            deltaT=arrayTiming[loopIdx, 0],
            simTempData=tempData,
            simParams=simParams
        )

        logging.debug('apply pulse')
        tempData = functions.propagateGradientPulseTime(
            dictGradPulse=GradientPulseData["refocus"],
            simParams=simParams,
            simTempData=tempData
        )

        logging.debug('fill time after pulse')
        tempData = functions.propagateRelaxation(
            deltaT=arrayTiming[loopIdx, 1],
            simTempData=tempData,
            simParams=simParams
        )

        logging.debug('acquisition')
        for acqIdx in range(simParams.settings.acquisitionNumber):
            tempData = functions.propagateGradientPulseTime(
                dictGradPulse=GradientPulseData["acquisition"],
                simParams=simParams,
                simTempData=tempData,
                append=False
            )

            tempData.signalArray[loopIdx, acqIdx] = (np.sum(tempData.magnetizationPropagation[-1][0])
                                                     + 1j * np.sum(tempData.magnetizationPropagation[-1][1])) \
                                                    * 100 * simParams.settings.lengthZ / simParams.settings.sampleNumber
            # signal scaled by distance between points (not entirely sure if this makes a difference

    # ----- finished loop -----

    logging.debug('Signal array processing fourier')
    imageArray = np.fft.fftshift(np.fft.fft(np.fft.fftshift(tempData.signalArray)))
    simData.emcSignal = 2 * np.sum(np.abs(imageArray), axis=1) / simParams.settings.acquisitionNumber
    # factor 2 not necessary, stems from Noams version, ultimately want some normalization here!
    simData.time = time.time() - t_start

    np.save(f"mag_prop.npy", tempData.magnetizationPropagation)
    return simData, simParams
