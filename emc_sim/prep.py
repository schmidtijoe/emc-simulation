import numpy as np
import os
import logging
from emc_sim import plotting
from emc_sim.options import SimulationParameters, SimulationTempData
import emc_sim.functions as fns
logModule = logging.getLogger(__name__)


def init_prep_for_visualization(simParams: SimulationParameters):
    # globals and sample are initiated as variables of the SimulationParameters class
    tempData = SimulationTempData(simParams)
    # ----- defining pulses ----- #
    gradientPulseData = gradientPulsePreparation(simParams=simParams, simTempData=tempData)

    # ----- setup timing ----- #
    arrayTiming = buildFillTiming_mese(simParams)

    # visualize
    if simParams.config.visualize:
        logging.debug('Visualization ON - Turn off when processing multiple instances!')
        plotting.visualizeAllGradientPulses(gradientPulseData)
        plotting.visualizeSequenceScheme(gradientPulseData, arrayTiming, simParams)
        plotting.plotMagnetization(tempData)


def gradientPulsePreparation(simParams: SimulationParameters, simTempData: SimulationTempData):
    logModule.debug('pulse preparation')
    # excitation
    simTempData.excitation_flag = True

    # read file
    path = os.path.abspath(os.path.join(simParams.config.pathToExternals, simParams.config.pulseFileExcitation))
    pulseExcitationShape, pulseExcitationNumber = fns.readPulseFile(path)

    # calculate and normalize
    dtExcitationPulse = simParams.sequence.durationExcitation / pulseExcitationNumber
    pulseExcitation = fns.pulseCalibrationIntegral(
        pulseExcitationShape,
        dtExcitationPulse,
        simParams=simParams,
        simTempData=simTempData,
        phase=-np.pi / 2)

    # build verse pulse gradient
    gVerseExcitation, pVerseExcitation, durationVerseExcitation, areaGradientVerseExcitation = fns.buildGradientVerse(
        amplitudePulse=pulseExcitation,
        simParams=simParams,
        simTempData=simTempData,
        gradCrushRephase=simParams.sequence.gradientExcitationRephase,
        durationCrushRephase=simParams.sequence.durationExcitationRephase
    )

    # Simulation is based on moving the acquisition process (hence gradient) artificially to z-axis along the slice
    # Therefore we need to do a couple of things artificially:

    # when acquiring in k-space along the slice we need to move the k-space start to the corner of k-space
    # i.e.: prephase half an acquisition gradient moment, put it into the rephase timing
    gradientPrePhase = np.divide(
        simParams.sequence.gradientAcquisition * simParams.sequence.durationAcquisition,
        (2 * simParams.sequence.durationExcitationRephase)
    )

    # the crushers are placed symmetrically about the refocusing pulses, hence are cancelling each others k-space
    # phase. We need to make sure that the crushers are balanced. For timing reasons there is no crusher before the
    # first refocusing pulse in the sequence. We move one into the rephase space of the excitation
    gradientExcitationCrush = np.divide(
        simParams.sequence.gradientCrush * simParams.sequence.durationCrush,
        simParams.sequence.durationExcitationRephase
    )

    # When exciting with a slice selective gradient the gradient creates phase offset along the slice axis.
    # We want to rephase this phase offset (as is the original use of the gradient in the acquisition scheme).
    # However, the rephasing gradient is usually used with half the gradient moment area (at 90° pulses), which
    # is not quite accurate.
    # After investigation a manual correction term can be put in here for accuracy * 1.038
    gradientExcitationPhaseRewind = - areaGradientVerseExcitation / (2 * simParams.sequence.durationExcitationRephase)

    # The gradient pulse scheme needs to be re-done with accommodating those changes in the rephase gradient of
    # the excitation
    gVerseExcitation, pVerseExcitation, durationVerseExcitation, areaGradientVerseExcitation = fns.buildGradientVerse(
        amplitudePulse=pulseExcitation,
        simParams=simParams,
        simTempData=simTempData,
        gradCrushRephase=gradientPrePhase + gradientExcitationCrush + gradientExcitationPhaseRewind,
        durationCrushRephase=simParams.sequence.durationExcitationRephase
    )

    # refocusing
    simTempData.excitation_flag = False
    # read in pulse
    path = os.path.abspath(os.path.join(simParams.config.pathToExternals, simParams.config.pulseFileRefocus))
    pulseRefocusShape, pulseRefocusNumber = fns.readPulseFile(path)

    dtPRefocusing = simParams.sequence.durationRefocus / pulseRefocusNumber

    # calculate and normalize
    pulseRefocus = fns.pulseCalibrationIntegral(
        pulseRefocusShape,
        dtPRefocusing,
        simParams=simParams,
        simTempData=simTempData
    )

    # build first verse refocus pulse gradient
    gVerseRefocusFirst, pVerseRefocusFirst, durationVerseRefocusFirst, _ = fns.buildGradientVerse(
        amplitudePulse=pulseRefocus,
        simParams=simParams,
        simTempData=simTempData,
        gradCrushRephase=simParams.sequence.gradientCrush,
        durationCrushRephase=simParams.sequence.durationCrush
    )

    # built other verse refocus pulse gradients
    gVerseRefocus, pVerseRefocus, durationVerseRefocus, _ = fns.buildGradientVerse(
        amplitudePulse=pulseRefocus,
        simParams=simParams,
        simTempData=simTempData,
        gradCrushRephase=simParams.sequence.gradientCrush,
        durationCrushRephase=simParams.sequence.durationCrush,
        gradPre=simParams.sequence.gradientCrush,
        durationPre=simParams.sequence.durationCrush
    )

    # built list of dictionaries containing values for each gradient pulses
    # dependent on sequence scheme
    # excitation
    dictGradientPulse = {
        "excitation": {
            'pulseType': 'Excitation',
            'temporalSamplingSteps': dtExcitationPulse,
            'pulseNumber': 0,
            'numberOfSamplingPoints': gVerseExcitation.shape[0],
            'gradientData': gVerseExcitation,
            'pulseData': pVerseExcitation
        },
        "refocus_1": {
            'pulseType': 'Refocusing',
            'temporalSamplingSteps': dtPRefocusing,
            'pulseNumber': 1,
            'numberOfSamplingPoints': gVerseRefocusFirst.shape[0],
            'gradientData': gVerseRefocusFirst,
            'pulseData': pVerseRefocusFirst
        },
        "refocus": {
            'pulseType': 'Refocusing',
            'temporalSamplingSteps': dtPRefocusing,
            'pulseNumber': "2+",
            'numberOfSamplingPoints': gVerseRefocus.shape[0],
            'gradientData': gVerseRefocus,
            'pulseData': pVerseRefocus
        },
        "acquisition": {
            'pulseType': 'Acquisition',
            'temporalSamplingSteps': simParams.sequence.durationAcquisition / simParams.settings.acquisitionNumber,
            'pulseNumber': "None",
            'numberOfSamplingPoints': 1,
            'gradientData': np.linspace(simParams.sequence.gradientAcquisition, simParams.sequence.gradientAcquisition,
                                        1),
            'pulseData': np.linspace(0, 0, 1)
        },
    }
    return dictGradientPulse


def buildFillTiming_mese(simParams: SimulationParameters):
    """
    Create a timing scheme: save time in [us] in array[2] -> [0] before pulse, [1] after pulse.
    For all refocusing pulses, i.e. ETL times
    Highly Sequence scheme dependent!
    :return: timing array
    """
    # all in [us]
    arrayTime = np.zeros([simParams.sequence.ETL, 2])

    # after excitation - before first refocusing:
    arrayTime[0, 0] = 1000 * simParams.sequence.ESP / 2 - (
            simParams.sequence.durationExcitation / 2 + simParams.sequence.durationExcitationRephase
            + simParams.sequence.durationRefocus / 2
    )
    # refocusing pulse...
    # after first refocusing
    arrayTime[0, 1] = 1000 * simParams.sequence.ESP / 2 - (
            simParams.sequence.durationRefocus / 2 + simParams.sequence.durationCrush + simParams.sequence.durationAcquisition / 2
    )
    # in this scheme, equal for all pulses, should incorporate some kind of "menu" for different sequence flavors:
    for pulseIdx in np.arange(1, simParams.sequence.ETL):
        for modeIdx in range(2):
            arrayTime[pulseIdx, modeIdx] = arrayTime[0, 1]
    return arrayTime


def buildFillTiming_se(simParams: SimulationParameters):
    """
    Create a timing scheme: save time in [us] in array[2] -> [0] before pulse, [1] after pulse.
    For SE sequence
    :return: timing array
    """
    # all in [us]
    arrayTime = np.zeros(2)

    # after excitation - before refocusing (check for prephaser):
    arrayTime[0] = 1000 * simParams.sequence.ESP / 2 - (
            simParams.sequence.durationExcitation / 2 + simParams.sequence.durationExcitationRephase
            + simParams.sequence.durationRefocus / 2
    )
    # refocusing pulse...
    # after refocusing
    arrayTime[1] = 1000 * simParams.sequence.ESP / 2 - (
            simParams.sequence.durationRefocus / 2 + simParams.sequence.durationCrush + simParams.sequence.durationAcquisition / 2
    )
    return arrayTime
