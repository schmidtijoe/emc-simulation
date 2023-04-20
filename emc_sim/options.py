import numpy as np
from dataclasses import dataclass, field
from simple_parsing import Serializable, ArgumentParser, choice
from simple_parsing.helpers.serialization import register_decoding_fn, encode
import multiprocessing as mp
from typing import List
import logging
from scipy import stats
from emc_sim import utils
import pandas as pd
import pathlib as plib

logModule = logging.getLogger(__name__)


@dataclass
class SimulationData(Serializable):
    emcSignal: np.ndarray = np.zeros(10)
    t1: float = 1.5
    t2: float = 0.04
    b1: float = 1.0
    d: float = 0.007
    time: float = 0

    def set_run_params(self, t1, t2, b1, d):
        self.t1 = t1
        self.t2 = t2
        self.b1 = b1
        self.d = d

    def get_run_params(self) -> dict:
        ret = {
            "T1": self.t1,
            "T2": self.t2,
            "B1": self.b1,
            "D": self.d
        }
        return ret

    @classmethod
    def from_cmd_args(cls, args: ArgumentParser.parse_args):
        simData = SimulationData(args.run)
        simData.emcSignal = np.zeros(args.sequence.ETL)
        return simData


@dataclass
class SimulationConfig(Serializable):
    """
    Configuration for simulation
    """
    # provide Configuration file (.json)
    configFile: str = ""
    # provide separate sequence params
    emcSeqConfig: str = ""
    # set path to save database and used config
    savePath: str = "./data"
    # set filename of database
    databaseName: str = "database_name"
    # set filepath to external pulse-files (pkl or json)
    pathToExternals: str = "./external"

    # name of external pulse file for excitation - assumed to be rf_pulse_files compatible.
    # See rf_pulse_files to convert from .txt or .pta
    pulseFileExcitation: str = 'slr_fa90_dur2001_lin-phase.pkl'
    # name of external pulse file for refocussing - assumed to be rf_pulse_files compatible.
    # See rf_pulse_files to convert from .txt or .pta
    pulseFileRefocus: str = 'slfrank_se_pulse_linear_phase.pkl'

    # set flag to visualize pulse profiles and sequence scheme
    visualize: bool = True
    # toggle diffusion calculations
    diffusionFlag: bool = False
    # toggle debugging log
    debuggingFlag: bool = True
    # toggle multithreading
    multiprocessing: bool = False
    # give number of CPUs to leave unused when multiprocessing
    mpHeadroom: int = 16
    # give desired number of CPUs to use
    mpNumCpus: int = 1

    def __post_init__(self):
        if self.multiprocessing:
            # we take at most the maximum free cpus but leave some headroom for other users
            self.mpNumCpus = mp.cpu_count() - self.mpHeadroom
            self.mpNumCpus = np.max([mp.cpu_count() - self.mpHeadroom, 4])
            # we take at least 4 cpus (kind of catches misconfiguration of the headroom parameter)
        else:
            self.mpNumCpus = 1
        self.mpNumCpus = int(self.mpNumCpus)


@dataclass
class SequenceParams(Serializable):
    """
    Parameters related to Sequence simulation
    """
    # global parameter gamma [Hz/t]
    gammaHz: float = 42577478.518

    # echo train length
    ETL: int = 16
    # echo spacing [ms]
    ESP: float = 9.0
    # bandwidth [Hz/px]
    bw: float = 349
    # gradient mode
    gradMode: str = choice("Rect", "Normal", "Verse", default="Verse")

    # Excitation, Flip Angle [°]
    excitationAngle: float = 90.0
    # Excitation, Phase [°]
    excitationPhase: float = 90.0
    # Excitation, gradient if rectangular/trapezoid [mt/m]
    gradientExcitation: float = 0.0
    # Excitation, duration of pulse [us]
    durationExcitation: float = 2560.0

    gradientExcitationRephase: float = -10.51  # [mT/m], rephase
    durationExcitationRephase: float = 1080.0  # [us], rephase

    gradientExcitationVerse1: float = -42.0  # [mT/m], verse
    gradientExcitationVerse2: float = -23.92  # [mt/m], verse
    durationExcitationVerse1: float = 770.0  # [us], verse
    durationExcitationVerse2: float = 1020.0  # [us], verse

    # Refocussing, Flip Angle [°]
    refocusAngle: List = field(default_factory=lambda: [140.0])
    # Refocussing, Phase [°]
    refocusPhase: List = field(default_factory=lambda: [0.0])
    # Refocussing, gradient strength if rectangular/trapezoid [mt/m]
    gradientRefocus: float = 0.0
    # Refocussing, duration of pulse [us]
    durationRefocus: float = 3584.0

    gradientCrush: float = -38.70  # [mT/m], crusher
    durationCrush: float = 1000.0  # [us], crusher

    gradientRefocusVerse1: float = -22.0  # [mT/m], verse
    gradientRefocusVerse2: float = -20.53  # [mt/m], verse
    durationRefocusVerse1: float = 1080.0  # [us], verse
    durationRefocusVerse2: float = 1424.0  # [us], verse

    gradientAcquisition: float = 0.0     # set automatically after settings init
    # time for acquisition (of one pixel) * 1e6 <- [(px)s] * 1e6

    def __post_init__(self):
        self.gammaPi: float = self.gammaHz * 2 * np.pi
        self.durationAcquisition: float = 1e6 / self.bw  # [us]
        if self.refocusPhase.__len__() != self.refocusAngle.__len__():
            err = f"provide same amount of refocusing pulse angle ({self.refocusAngle.__len__()}) " \
                  f"and phases ({self.refocusPhase.__len__()})"
            logModule.error(err)
            raise AttributeError(err)
        # check for phase values
        for l_idx in range(self.refocusPhase.__len__()):
            while np.abs(self.refocusPhase[l_idx]) > 180.0:
                self.refocusPhase[l_idx] = self.refocusPhase[l_idx] - np.sign(self.refocusPhase[l_idx]) * 180.0
            while np.abs(self.refocusAngle[l_idx]) > 180.0:
                self.refocusAngle[l_idx] = self.refocusAngle[l_idx] - np.sign(self.refocusAngle[l_idx]) * 180.0
        while self.refocusAngle.__len__() < self.ETL:
            # fill up list with last value
            self.refocusAngle.append(self.refocusAngle[-1])
            self.refocusPhase.append(self.refocusPhase[-1])


@dataclass
class SimulationSettings(Serializable):
    """
    Optional settings for simulation eg. spatial resolution
    """
    sampleNumber: int = 1000  # no of sampling points along slice profile
    lengthZ: float = 0.005  # [m] length extension of z-axis spanned by sample -> total length 2*lengthZ (-:+)
    acquisitionNumber: int = 50  # number of bins across slice sample -> effectively sets spatial resolution
    # resolution = lengthZ / acquisitionNumber

    t1_list: List = field(default_factory=lambda: [1.5])  # T1 to simulate [s]
    t2_list: List = field(default_factory=lambda: [[25, 30, 0.5], [30, 35, 1]])  # T2 to simulate [ms]
    b1_list: List = field(default_factory=lambda: [0.6, 1.0])  # B1 to simulate
    d_list: List = field(default_factory=lambda: [700.0])
    # diffusion values to use if flag in config is set [mm²/s]
    total_num_sim: int = 4

    def __post_init__(self):
        array = np.empty(0)
        for item in self.t2_list:
            if type(item) == str:
                item = [float(i) for i in item[1:-1].split(',')]
            array = np.concatenate((array, np.arange(*item)))

        array = [t2 / 1000.0 for t2 in array]  # cast to [s]
        self.t2_array = array
        # sanity checks
        if max(self.t2_array) > min(self.t1_list):
            logModule.error('T1 T2 mismatch (T2 > T1)')
            exit(-1)
        if max(self.t2_array) < 1e-4:
            logModule.error('T2 value range exceeded, make sure to post T2 in ms')
            exit(-1)
        else:
            self.total_num_sim = len(self.t1_list) * len(self.t2_array) * len(self.b1_list) * len(self.d_list)

    def get_complete_param_list(self):
        return [(t1, t2, b1, d) for t1 in self.t1_list
                for t2 in self.t2_array for b1 in self.b1_list for d in self.d_list]


@dataclass
class SimulationParameters(Serializable):
    config: SimulationConfig = SimulationConfig()
    sequence: SequenceParams = SequenceParams()
    settings: SimulationSettings = SimulationSettings()

    def __post_init__(self):
        self.sequence.gradientAcquisition = - self.settings.acquisitionNumber * self.sequence.bw \
                                            / (self.sequence.gammaHz * 2 * self.settings.lengthZ) * 1000

    def set_acquisition_gradient(self):
        self.__post_init__()

    @classmethod
    def from_cmd_args(cls, args: ArgumentParser.parse_args):
        simParams = SimulationParameters(config=args.config, settings=args.settings, sequence=args.sequence)

        nonDefaultConfig, nonDefaultSettings, nonDefaultSequence = simParams._checkNonDefaultVars()

        if args.config.configFile:
            simParams = SimulationParameters.load(args.config.configFile)
            # overwrite non default input args
            for key, value in nonDefaultConfig.items():
                simParams.config.__setattr__(key, value)
            for key, value in nonDefaultSettings.items():
                simParams.settings.__setattr__(key, value)
            for key, value in nonDefaultSequence.items():
                simParams.sequence.__setattr__(key, value)

        # we check parsed arguments for explicit cmd line input assuming explicit input means "different from default".
        # Since everytime the cmd is parsed all values not given explicitly are parsed with respective
        # dataclass defaults.
        # Hence, if explicit input is coincidentally a default value this value will be ignored:
        # eg. to overwrite an instance (containing non-default values) loaded by a configFile
        # and explicitly trying to change entries to default via cmd input.
        # ToDo: Fix explicit cmd line input
        if args.config.emcSeqConfig or simParams.config.emcSeqConfig:
            emcSeqConfig = simParams.config.emcSeqConfig
            if args.config.emcSeqConfig:
                emcSeqConfig = args.config.emcSeqConfig
            simParams.sequence = SequenceParams.load(emcSeqConfig)
        simParams.set_acquisition_gradient()
        return simParams

    def _checkNonDefaultVars(self) -> (dict, dict, dict):
        defConfig = SimulationConfig()
        nonDefaultConfig = {}
        for key, value in vars(self.config).items():
            if self.config.__getattribute__(key) != defConfig.__getattribute__(key):
                nonDefaultConfig.__setitem__(key, value)

        defSettings = SimulationSettings()
        nonDefaultSettings = {}
        for key, value in vars(self.settings).items():
            if self.settings.__getattribute__(key) != defSettings.__getattribute__(key):
                nonDefaultSettings.__setitem__(key, value)

        defSequence = SequenceParams()
        nonDefaultSequence = {}
        for key, value in vars(self.sequence).items():
            # catch post init attribute
            if key == 'gradientAcquisition':
                continue
            if self.sequence.__getattribute__(key) != defSequence.__getattribute__(key):
                nonDefaultSequence.__setitem__(key, value)
        return nonDefaultConfig, nonDefaultSettings, nonDefaultSequence

    def save_database(self, database: pd.DataFrame) -> None:
        base_path = plib.Path(self.config.savePath).absolute()
        # create parent folder ifn existent
        plib.Path.mkdir(base_path, exist_ok=True, parents=True)

        db_path = base_path.joinpath(self.config.saveFile)
        config_path = base_path.joinpath(f"{db_path.stem}_config.json")

        logModule.info(f"writing file {db_path}")
        # mode dependent on file ending given
        save_fn = {
            ".pkl": database.to_pickle,
            ".json": lambda obj: database.to_json(obj, indent=2)
        }
        assert save_fn.get(db_path.suffix), f"Database save path{db_path}: type not recognized;" \
                                            f"Supported: {list(save_fn.keys())}"
        save_fn.get(db_path.suffix)(db_path.__str__())
        # save used config
        logModule.info(f"writing file {config_path}")
        self.save(config_path, indent=2, separators=(',', ':'))


@dataclass
class SimulationTempData:
    """
    Carrying data through simulation
    """
    sample: np.ndarray
    sampleAxis: np.ndarray
    signalArray: np.ndarray
    magnetizationPropagation: [np.ndarray]
    excitation_flag: bool = True  # flag to toggle between excitation and refocus
    run: SimulationData = SimulationData()

    def __init__(self, simParams: SimulationParameters):
        self.sampleAxis = np.linspace(-simParams.settings.lengthZ, simParams.settings.lengthZ,
                                      simParams.settings.sampleNumber)
        # self.sample = np.exp(-8 * ((self.sampleAxis * 1e2) ** 16)) + 1e-6
        self.sample = stats.gennorm(24).pdf(self.sampleAxis/simParams.settings.lengthZ*1.1) + 1e-6
        self.sample = utils.normalize_array(self.sample, max_factor=1.0, normalization="max")
        mInit = np.zeros([4, simParams.settings.sampleNumber])
        mInit[2, :] = self.sample
        mInit[3, :] = self.sample
        self.signalArray = np.zeros((simParams.sequence.ETL, simParams.settings.acquisitionNumber), dtype=complex)
        self.magnetizationPropagation = [mInit]


@encode.register
def encode_ndarray(obj: np.ndarray):
    if len(obj.flatten()) > 200:
        return []
    return obj.tolist()


register_decoding_fn(np.ndarray, np.array)


def createCommandlineParser():
    """
    Build the parser for arguments
    Parse the input arguments.
    """
    parser = ArgumentParser(prog='emc_sim')
    parser.add_arguments(SimulationConfig, dest="config")
    parser.add_arguments(SimulationSettings, dest="settings")
    parser.add_arguments(SequenceParams, dest="sequence")
    parser.add_arguments(SimulationData, dest="run")

    args = parser.parse_args()

    return parser, args
