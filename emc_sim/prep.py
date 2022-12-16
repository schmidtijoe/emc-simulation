import numpy as np
import logging
from emc_sim import plotting, events, options, functions
logModule = logging.getLogger(__name__)


def init_prep_for_visualization(simParams: options.SimulationParameters):
    # globals and sample are initiated as variables of the SimulationParameters class
    tempData = options.SimulationTempData(simParams)
    # ----- defining sequence ----- #
    gp_excitation, gps_refocusing, timing, acquisition = gradientPulsePreparationSEMC(
        simParams=simParams, simTempData=tempData)

    # visualize
    if simParams.config.visualize:
        logging.debug('Visualization ON - Turn off when processing multiple instances!')
        plotting.visualizeAllGradientPulses([gp_excitation, *gps_refocusing])
        plotting.visualizeSequenceScheme([gp_excitation, *gps_refocusing], timing, acquisition)
        plotting.plotMagnetization(tempData)


def gradientPulsePreparationSEMC(
        simParams: options.SimulationParameters,
        simTempData: options.SimulationTempData) -> (events.GradPulse, list, events.Timing, events.GradPulse):
    logModule.debug('pulse preparation')
    gp_excitation = events.GradPulse.prep_grad_pulse(
        pulse_type='Excitation',
        pulse_number=0,
        sym_spoil=False,
        params=simParams,
        sim_temp_data=simTempData)

    gp_refocus_1 = events.GradPulse.prep_grad_pulse(
        pulse_type='Refocusing_1',
        pulse_number=1,
        sym_spoil=False,
        params=simParams,
        sim_temp_data=simTempData
    )
    # built list of grad_pulse events, acquisition and timing
    grad_pulses = [gp_refocus_1]
    for r_idx in np.arange(2, simParams.sequence.ETL + 1):
        gp_refocus = events.GradPulse.prep_grad_pulse(
            pulse_type='Refocusing',
            pulse_number=r_idx,
            sym_spoil=True,
            params=simParams,
            sim_temp_data=simTempData
        )
        grad_pulses.append(gp_refocus)

    acquisition = events.GradPulse.prep_acquisition(params=simParams)

    logModule.debug(f"calculate timing")
    timing = events.Timing.buildFillTiming_mese(simParams)

    return gp_excitation, grad_pulses, timing, acquisition

