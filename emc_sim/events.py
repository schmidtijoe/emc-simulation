import numpy as np
from emc_sim import options, functions
from rf_pulse_files import rfpf
import pathlib as plib
import typing
import logging

logModule = logging.getLogger(__name__)


class GradPulse:
    def __init__(self,
                 pulse_type: str = 'Excitation',
                 pulse_number: int = 0,
                 num_sampling_points: int = 512,
                 dt_sampling_steps: float = 5.0,
                 duration: float = 0.0,
                 params: options.SimulationParameters = options.SimulationParameters(),
                 sim_temp_data: options.SimulationTempData = None):
        self.pulse_number = pulse_number
        self.pulse_type = pulse_type
        self.num_sampling_points = num_sampling_points
        self.dt_sampling_steps = dt_sampling_steps
        self.data_grad = np.zeros(0)
        self.data_pulse = np.zeros(0)
        self.excitation_flag: bool = True
        self.params = params
        if sim_temp_data is not None:
            self.temp_data = sim_temp_data
        else:
            self.temp_data = options.SimulationTempData(simParams=params)
        self.duration = duration

        # set and check
        self._set_exci_flag()

    def _set_exci_flag(self):
        if self.pulse_type == 'Excitation':
            self.excitation_flag = True
        else:
            self.excitation_flag = False

    @classmethod
    def prep_grad_pulse(cls, pulse_type: str = 'Excitation', pulse_number: int = 0, sym_spoil: bool = True,
                        params: options.SimulationParameters = options.SimulationParameters(),
                        sim_temp_data: options.SimulationTempData = None):
        # -- prep pulse
        logModule.debug(f'prep pulse {pulse_type}; # {pulse_number}')
        grad_pulse = cls(pulse_type=pulse_type, params=params, pulse_number=pulse_number, sim_temp_data=sim_temp_data)
        # read file
        if grad_pulse.excitation_flag:
            sim_temp_data.excitation_flag = True
            path = plib.Path(params.config.pathToExternals).absolute().joinpath(params.config.pulseFileExcitation)
            grad_crush_rephase = params.sequence.gradientExcitationRephase
            duration_crush_rephase = params.sequence.durationExcitationRephase
            duration_pulse = params.sequence.durationExcitation
        else:
            path = plib.Path(params.config.pathToExternals).absolute().joinpath(params.config.pulseFileRefocus)
            sim_temp_data.excitation_flag = False
            grad_crush_rephase = params.sequence.gradientCrush
            duration_crush_rephase = params.sequence.durationCrush
            duration_pulse = params.sequence.durationRefocus
        # change to rfpf object here
        rf = rfpf.RF.load(path)

        if np.abs(rf.duration_in_us - duration_pulse) > 1e-5:
            # resample pulse
            rf.resample_to_duration(duration_in_us=int(duration_pulse))
        pulse = rf.amplitude*np.exp(1j*rf.phase)

        # calculate and normalize
        pulse = functions.pulseCalibrationIntegral(
            pulse=pulse,
            deltaT=rf.get_dt_sampling_in_us(),
            pulseNumber=pulse_number,
            simParams=params,
            simTempData=sim_temp_data)

        if sym_spoil:
            grad_prewind = grad_crush_rephase
            duration_prewind = duration_crush_rephase
        else:
            grad_prewind = 0.0
            duration_prewind = 0.0

        # build verse pulse gradient
        grad_verse, pulse_verse, duration, area_gradient_verse = functions.buildGradientVerse(
            amplitudePulse=pulse,
            simParams=params,
            simTempData=sim_temp_data,
            gradCrushRephase=grad_crush_rephase,
            durationCrushRephase=duration_crush_rephase,
            gradPre=grad_prewind,
            durationPre=duration_prewind
        )

        if grad_pulse.excitation_flag:
            # Simulation is based on moving the acquisition process (hence gradient) artificially to
            # z-axis along the slice
            # Therefore we need to do a couple of things artificially:

            # when acquiring in k-space along the slice we need to move the k-space start to the corner of k-space
            # i.e.: prephase half an acquisition gradient moment, put it into the rephase timing
            gradient_pre_phase = np.divide(
                params.sequence.gradientAcquisition * params.sequence.durationAcquisition,
                (2 * params.sequence.durationExcitationRephase)
            )

            # the crushers are placed symmetrically about the refocusing pulses,
            # hence are cancelling each others k-space phase. We need to make sure that the crushers are balanced.
            # For timing reasons there is no crusher before the first refocusing pulse in the sequence.
            # We move one into the rephase space of the excitation
            gradient_excitation_crush = np.divide(
                params.sequence.gradientCrush * params.sequence.durationCrush,
                params.sequence.durationExcitationRephase
            )

            # When exciting with a slice selective gradient the gradient creates phase offset along the slice axis.
            # We want to rephase this phase offset (as is the original use of the gradient in the acquisition scheme).
            # However, the rephasing gradient is usually used with half the gradient moment area (at 90° pulses), which
            # is not quite accurate.
            # After investigation a manual correction term can be put in here for accuracy * 1.038
            gradient_excitation_phase_rewind = - area_gradient_verse / (
                    2 * params.sequence.durationExcitationRephase)

            # The gradient pulse scheme needs to be re-done with accommodating those changes in the rephase gradient of
            # the excitation
            grad_verse, pulse_verse, duration, area_gradient_verse = functions.buildGradientVerse(
                amplitudePulse=pulse,
                simParams=params,
                simTempData=sim_temp_data,
                gradCrushRephase=gradient_pre_phase + gradient_excitation_crush + gradient_excitation_phase_rewind,
                durationCrushRephase=params.sequence.durationExcitationRephase
            )

        # assign vars
        grad_pulse.num_sampling_points = rf.num_samples
        grad_pulse.dt_sampling_steps = rf.get_dt_sampling_in_us()
        grad_pulse.data_grad = grad_verse
        grad_pulse.data_pulse = pulse_verse
        grad_pulse.duration = rf.duration_in_us

        return grad_pulse

    @classmethod
    def prep_single_grad_pulse(cls, params: options.SimulationParameters = options.SimulationParameters(),
                               sim_temp_data: options.SimulationTempData = None,
                               grad_rephase_factor: float = 1.0):
        # -- prep pulse
        pulse_type: str = 'Excitation'  # just set it
        logModule.debug(f'prep pulse {pulse_type}; # {0}')
        grad_pulse = cls(pulse_type=pulse_type, params=params, pulse_number=0, sim_temp_data=sim_temp_data)
        # read file
        sim_temp_data.excitation_flag = True

        path = plib.Path(params.config.pathToExternals).absolute().joinpath(params.config.pulseFileExcitation)

        duration_pulse = params.sequence.durationExcitation
        rf: rfpf.RF = rfpf.RF.load(path)

        if np.abs(rf.duration_in_us - duration_pulse) > 1e-5:
            # resample pulse
            rf.resample_to_duration(duration_in_us=int(duration_pulse))

        # calculate and normalize
        pulse = functions.pulseCalibrationIntegral(
            pulse=rf.amplitude * np.exp(1j * rf.phase),
            deltaT=rf.get_dt_sampling_in_us(),
            pulseNumber=0,
            simParams=params,
            simTempData=sim_temp_data)

        # build verse pulse gradient
        grad_verse, pulse_verse, duration, area_gradient_verse = functions.buildGradientVerse(
            amplitudePulse=pulse,
            simParams=params,
            simTempData=sim_temp_data,
            gradCrushRephase=0.0,
            durationCrushRephase=0.0,
            gradPre=0.0,
            durationPre=0.0
        )

        # When exciting with a slice selective gradient the gradient creates phase offset along the slice axis.
        # We want to rephase this phase offset (as is the original use of the gradient in the acquisition scheme).
        # However, the rephasing gradient is usually used with half the gradient moment area (at 90° pulses), which
        # is not quite accurate.
        # After investigation a manual correction term can be put in here for accuracy * 1.038
        gradient_excitation_phase_rewind = - area_gradient_verse / (
                grad_rephase_factor * 2 * params.sequence.durationExcitationRephase)

        # The gradient pulse scheme needs to be re-done with accommodating those changes in the rephase gradient of
        # the excitation
        grad_verse, pulse_verse, duration, area_gradient_verse = functions.buildGradientVerse(
            amplitudePulse=pulse,
            simParams=params,
            simTempData=sim_temp_data,
            gradCrushRephase=gradient_excitation_phase_rewind,
            durationCrushRephase=params.sequence.durationExcitationRephase
        )

        # assign vars
        grad_pulse.num_sampling_points = rf.num_samples
        grad_pulse.dt_sampling_steps = rf.get_dt_sampling_in_us()
        grad_pulse.data_grad = grad_verse
        grad_pulse.data_pulse = pulse_verse
        grad_pulse.duration = rf.duration_in_us

        return grad_pulse

    @classmethod
    def prep_acquisition(cls, params: options.SimulationParameters = options.SimulationParameters()):
        logModule.debug("prep acquisition")
        dt_sampling_steps = params.sequence.durationAcquisition / params.settings.acquisitionNumber
        grad_pulse = cls(pulse_type='Acquisition', pulse_number=0, num_sampling_points=1,
                         params=params, dt_sampling_steps=dt_sampling_steps)
        # assign data
        grad_pulse.data_grad = np.linspace(
            params.sequence.gradientAcquisition,
            params.sequence.gradientAcquisition,
            1)
        grad_pulse.data_pulse = np.linspace(0, 0, 1)
        grad_pulse.duration = params.sequence.durationAcquisition
        return grad_pulse


class Timing:
    def __init__(self, time_pre_pulse: typing.Union[float, np.ndarray] = 0.0,
                 time_post_pulse: typing.Union[float, np.ndarray] = 0.0):
        self.time_post_pulse = time_post_pulse
        self.time_pre_pulse = time_pre_pulse

    @classmethod
    def buildFillTiming_mese(cls, params: options.SimulationParameters = options.SimulationParameters()):
        """
        Create a timing scheme: save time in [us] in array[2] -> [0] before pulse, [1] after pulse.
        For all refocusing pulses, i.e. ETL times
        Highly Sequence scheme dependent!
        :return: timing array
        """
        # all in [us]
        time_pre_pulse = np.zeros([params.sequence.ETL])
        time_post_pulse = np.zeros([params.sequence.ETL])

        # after excitation - before first refocusing:
        time_pre_pulse[0] = 1000 * params.sequence.ESP / 2 - (
                params.sequence.durationExcitation / 2 + params.sequence.durationExcitationRephase
                + params.sequence.durationRefocus / 2
        )
        # refocusing pulse...
        # after first refocusing
        time_post_pulse[0] = 1000 * params.sequence.ESP / 2 - (
                params.sequence.durationRefocus / 2 + params.sequence.durationCrush +
                params.sequence.durationAcquisition / 2
        )

        # in this scheme, equal for all pulses, should incorporate some kind of "menu" for different sequence flavors:
        for pulseIdx in np.arange(1, params.sequence.ETL):
            time_pre_pulse[pulseIdx] = time_post_pulse[0]
            time_post_pulse[pulseIdx] = time_post_pulse[0]
        return cls(time_pre_pulse=time_pre_pulse, time_post_pulse=time_post_pulse)

    @classmethod
    def buildFillTiming_se(cls, params: options.SimulationParameters = options.SimulationParameters()):
        """
        Create a timing scheme: save time in [us] in array[2] -> [0] before pulse, [1] after pulse.
        For SE sequence
        :return: timing array
        """
        # all in [us]
        # after excitation - before refocusing (check for prephaser):
        time_pre_pulse = 1000 * params.sequence.ESP / 2 - (
                params.sequence.durationExcitation / 2 + params.sequence.durationExcitationRephase
                + params.sequence.durationRefocus / 2
        )
        # refocusing pulse...
        # after refocusing
        time_post_pulse = 1000 * params.sequence.ESP / 2 - (
                params.sequence.durationRefocus / 2 + params.sequence.durationCrush +
                params.sequence.durationAcquisition / 2
        )
        return cls(time_pre_pulse=time_pre_pulse, time_post_pulse=time_post_pulse)
