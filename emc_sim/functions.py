import numpy as np
import logging
from emc_sim.options import SimulationParameters, SimulationTempData
from typing import Union


def readPulseFile(filename: str) -> (np.ndarray, tuple):
    """
    if pulse profile is provided, read in
    :param filename: name of file (txt or pta) of pulse
    :return: pulse array, pulse length
    """
    with open(filename, "r") as f:
        temp = np.array([x.strip().split('\t') for x in f], dtype=float).transpose()
    pulseShape = temp[0] * np.exp(1j * temp[1])
    return pulseShape, pulseShape.shape[0]


def pulseCalibrationIntegral(pulse: np.ndarray,
                             deltaT: float,
                             simParams: SimulationParameters,
                             simTempData: SimulationTempData,
                             phase: float = 0.0) -> np.ndarray:
    """
    Calibrates pulse waveform for given flip angle, adds phase if given

    :param simParams: Simulation parameters
    :param pulse: array of complex pulse data
    :param deltaT: temporal sampling steps of pulse events [us]
    :param phase: phase offset from x axis [°]
    :return: calibrated b1 profile, complex
    """
    # normalize
    b1Pulse = pulse / np.linalg.norm(pulse)
    # integrate (discrete steps) total flip angle achieved with the normalized pulse
    flipAngleNormalizedB1 = np.sum(b1Pulse * simParams.sequence.gammaPi) * deltaT * 1e-6
    if simTempData.excitation_flag:
        angleFlip = simParams.sequence.excitationAngle
    else:
        angleFlip = simParams.sequence.refocusAngle
    angleFlip *= np.pi / 180 * simTempData.run.b1  # calculate with applied actual flip angle offset
    b1PulseCalibrated = b1Pulse * (angleFlip / flipAngleNormalizedB1) * np.exp(1j * phase)
    return b1PulseCalibrated


def propagateRotationRelaxationDiffusion(ux: Union[np.ndarray, float],
                                         uy: Union[np.ndarray, float],
                                         uz: Union[np.ndarray, float],
                                         mx: Union[np.ndarray, float],
                                         my: Union[np.ndarray, float],
                                         mz: Union[np.ndarray, float],
                                         me: Union[np.ndarray, float],
                                         dtInSec: float,
                                         simTempData: SimulationTempData,
                                         phiRadians: float = 0.0,
                                         b: float = 0.0) -> np.ndarray:
    """
    Calculates the effect of rotation and relaxation matrices without dot and matrix array creation (much quicker
    and memory efficient!) -> iterative computation relies on hard pulse approximation
    Diffusion term is added through the effect of a gradient (switched off by default),
        needs to be precomputed from gradient strength,
    as needs to be the normal vector and rotation angle from gradients and pulses.
    can be used to facilitate relaxation (default)

    :param ux: x component of vector about which to rotate
    :param uy: y component of vector about which to rotate
    :param uz: z component of vector about which to rotate
    :param mx: x component of magnetization vector to rotate
    :param my: y component of magnetization vector to rotate
    :param mz: z component of magnetization vector to rotate
    :param me: e component of magnetization vector to rotate
    :param phiRadians: angle of rotation in radians
    :param dtInSec: time step in seconds
    :param b: gradient generated b - value in s / mm^2

    :return: magnetization vector after propagation
    """
    # want Diffusion coefficient in SI [m²/s] given [mm²/s]
    D = simTempData.run.d * 1e-6
    e1 = np.exp(-dtInSec / simTempData.run.t1)
    e2 = np.exp(-dtInSec / simTempData.run.t2 - b * D)
    co = np.cos(phiRadians)
    si = np.sin(phiRadians)
    a = 1.0 - co
    return np.array([
        (mx * (co + a * ux ** 2) + my * (a * ux * uy - uz * si) + mz * (a * ux * uz + uy * si)) * e2,
        (mx * (a * ux * uy + uz * si) + my * (co + a * uy ** 2) + mz * (a * uy * uz - ux * si)) * e2,
        (mx * (a * ux * uz - uy * si) + my * (a * uy * uz + ux * si) + mz * (co + a * uz ** 2)) * e1 + (1 - e1) * me,
        me
    ])


def toggle_excitation_refocus(simParams: SimulationParameters, simTempData: SimulationTempData) -> (
        float, float, float, float, float):
    """ Decide which gradients and durations to take from the dataclass, dependent on grad mode and excitation """
    if simParams.sequence.gradMode == "Verse":
        # verse gradients
        if simTempData.excitation_flag:
            # excitation
            duration = simParams.sequence.durationExcitation
            gradVerse1 = simParams.sequence.gradientExcitationVerse1
            gradVerse2 = simParams.sequence.gradientExcitationVerse2
            durationVerse1 = simParams.sequence.durationExcitationVerse1
            durationVerse2 = simParams.sequence.durationExcitationVerse2
        else:
            # refocus
            duration = simParams.sequence.durationRefocus
            gradVerse1 = simParams.sequence.gradientRefocusVerse1
            gradVerse2 = simParams.sequence.gradientRefocusVerse2
            durationVerse1 = simParams.sequence.durationRefocusVerse1
            durationVerse2 = simParams.sequence.durationRefocusVerse2
    else:
        # normal gradient
        if simTempData.excitation_flag:
            # excitation
            duration = simParams.sequence.durationExcitation
            gradVerse1 = 0
            gradVerse2 = simParams.sequence.gradientExcitation
            durationVerse1 = 0
            durationVerse2 = simParams.sequence.durationExcitation
        else:
            # refocus
            duration = simParams.sequence.durationRefocus
            gradVerse1 = 0
            gradVerse2 = simParams.sequence.gradientRefocus
            durationVerse1 = 0
            durationVerse2 = simParams.sequence.durationRefocus
    # due to calculations in gradient preparation phase we could not include the crusher/rephaser gradients,
    # even though it would make the code neater
    return duration, gradVerse1, durationVerse1, gradVerse2, durationVerse2


def buildGradientVerse(amplitudePulse: np.ndarray, simParams: SimulationParameters, simTempData: SimulationTempData,
                       gradCrushRephase: float, durationCrushRephase: float,
                       gradPre: float = 0.0, durationPre: float = 0.0) -> (
        np.ndarray, np.ndarray, float, float):
    """
    build pulse gradient information array, stepwise values of pulse and gradient amplitudes.
    Prephasers (generally ramp times not required from siemens timing)
    time steps in us calculated from pulse/gradient length and durations.
    gradients in mT/m

    :param durationCrushRephase: duration of crusher / rephaser gradient
    :param gradCrushRephase: duration of crusher / rephaser gradient
    :param amplitudePulse: pulse data per time step, complex
    :param simParams: Simulation Parameters
    :param gradPre: amplitude of prephasing gradient [mT/m]
    :param durationPre: duration of gradient applied before pulse [us]

    returns gradient amplitude sampled at dt = durationPulse/len(pulse),
            pulse vector sampled at dt basically input pulse but adjusted to same pre and crusher positions,
            total time of gradient sampling,
            total gradient area for pulse duration = sum(gradientAmplitude(t) dt)
    :return: [gradientArray, pulseArray, totalTime, gradientPulseArea]
    """
    duration, gradVerse1, durationVerse1, gradVerse2, durationVerse2 = toggle_excitation_refocus(simParams, simTempData)
    pulseN = amplitudePulse.shape[0]
    deltaT = duration / pulseN
    if 2 * durationVerse1 + durationVerse2 != duration:
        logging.error("pulse sampling temporal mismatch")
        exit(-1)
    # calculate total number of gradient sampling points
    preN = round(durationPre / deltaT)
    crushN = round(durationCrushRephase / deltaT)
    totalGradientSamplingPoints = preN + crushN + pulseN
    verse1N = round(durationVerse1 / deltaT)
    verse2N = pulseN - 2 * verse1N

    # allocate array
    gradientAmplitude = np.zeros([totalGradientSamplingPoints])
    gradientAmplitude[:preN] = gradPre
    gradientAmplitude[preN:preN + verse1N] = np.linspace(gradVerse1, gradVerse2, verse1N)
    gradientAmplitude[preN + verse1N:preN + verse1N + verse2N] = gradVerse2
    gradientAmplitude[preN + verse1N + verse2N:preN + pulseN] = np.linspace(gradVerse2,
                                                                            gradVerse1, verse1N)

    gradientAmplitude[preN + pulseN:totalGradientSamplingPoints] = gradCrushRephase
    pulseAmplitude = np.zeros_like(gradientAmplitude, dtype=complex)
    pulseAmplitude[preN:preN + pulseN] = amplitudePulse
    totalTime = totalGradientSamplingPoints * deltaT
    return gradientAmplitude, pulseAmplitude, totalTime, sum(gradientAmplitude[preN:preN + pulseN]) * deltaT


def propagateGradientPulseTime(dictGradPulse: dict,
                               simParams: SimulationParameters,
                               simTempData: SimulationTempData,
                               append: bool = True):
    """
    calculate effect of pulse and gradient combinations or relaxation only per time step
    on the magnetization vectors of all isochromats spread across the slice (z-direction).
    Assumed hard-pulse approximation. gradient and pulse value are assumed constant
    for a time step.

    :param append:
    :param dictGradPulse:
    :param simTempData: simulation temporary parameter class
    :param simParams: simulation parameter class
    :return: magnetization vector population after iteration
    """
    deltaT = dictGradPulse["temporalSamplingSteps"]
    pulseT = dictGradPulse["pulseData"]
    gradT = dictGradPulse["gradientData"]

    # pick last element from list
    tempMagVec = simTempData.magnetizationPropagation[-1].copy()
    tInSec = deltaT * 1e-6
    incrGamma = simParams.sequence.gammaPi * tInSec
    # checkup
    if isinstance(pulseT, complex) ^ isinstance(gradT, float):
        # logical xor, i.e. if only one is a float
        logging.error("pulse sampling temporal mismatch")
        exit(-1)
    elif isinstance(pulseT, complex) and isinstance(gradT, float):
        pulseT = np.array([pulseT])
        gradT = np.array([gradT])
    elif pulseT.shape[0] != gradT.shape[0]:
        logging.error("pulse sampling temporal mismatch")
        exit(-1)
    # allocate space
    rotationAxisVector = np.ones([3, simParams.settings.sampleNumber])
    normedRotationVectorAngle = np.zeros([1, simParams.settings.sampleNumber])
    # iterate through time steps, extended along slice dimension
    for idxT in range(gradT.shape[0]):
        # xyz direction calculated by pulse effect along transverse dirs (xy), and gradient effect along slice (z)
        rotationAxisVector[0] = np.real(pulseT[idxT])
        rotationAxisVector[1] = np.imag(pulseT[idxT])
        rotationAxisVector[2] = gradT[
                                    idxT] * simTempData.sampleAxis * 1e-3  # bz facilitated along z: [mT/m] = 1e-3 [T/m]
        # calculate norm of rotational vector
        normedRotationVectorAngle[0] = np.linalg.norm(rotationAxisVector, axis=0)
        # normalize the original vector
        rotationAxisVector = np.divide(rotationAxisVector, normedRotationVectorAngle)
        # use norm to calculate the angle of rotation about the vector
        normedRotationVectorAngle *= incrGamma

        # calculate gradient strength dependent diffusion attenuation
        if simParams.config.d_flag:
            b = np.square(simParams.sequence.gammaHz) * np.square(gradT[idxT] * 1e-3) * tInSec ** 3
        else:
            b = 0
        # so far most efficient way of propagating vector dimensions (much quicker than matrix/vector variant)
        # can still be made more efficient
        tempMagVec = propagateRotationRelaxationDiffusion(
            rotationAxisVector[0],
            rotationAxisVector[1],
            rotationAxisVector[2],
            tempMagVec[0],
            tempMagVec[1],
            tempMagVec[2],
            tempMagVec[3],
            tInSec,
            simTempData=simTempData,
            phiRadians=normedRotationVectorAngle[0],
            b=b
        )
    # update list
    if append:
        simTempData.magnetizationPropagation.append(tempMagVec)
    else:
        simTempData.magnetizationPropagation[-1] = tempMagVec
    return simTempData


def propagateRelaxation(deltaT, simTempData: SimulationTempData, simParams: SimulationParameters):
    """
    If we dont have any gradients or pulses affecting the spin system we can call this function
    to get the relaxation components of the matrix only

    :param simTempData: temporary simulation parameters
    :param simParams: Simulation parameters
    :param deltaT: Time to propagate the system for [us]
    :return: propagated magnetization vector

    formerly did this with a numpy matrix multiplication of a simple relaxation matrix,
    need some time checking between the methods
    """
    # pick last element from list
    tempMagVec = simTempData.magnetizationPropagation[-1].copy()
    tInSec = deltaT * 1e-6
    normalizedRotAxisVector = np.zeros([2, tempMagVec.shape[1]])
    normalizedRotAxisVector[1] = np.ones(normalizedRotAxisVector.shape[1])  # dummy rotation vector
    tempMagVec = propagateRotationRelaxationDiffusion(
        normalizedRotAxisVector[0],
        normalizedRotAxisVector[0],
        normalizedRotAxisVector[1],
        tempMagVec[0],
        tempMagVec[1],
        tempMagVec[2],
        tempMagVec[3],
        tInSec,
        simTempData=simTempData,
        phiRadians=0.0,
        b=0.0
    )
    # update same element
    simTempData.magnetizationPropagation[-1] = tempMagVec
    return simTempData
