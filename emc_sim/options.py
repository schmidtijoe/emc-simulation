import numpy as np
from dataclasses import dataclass, field
from simple_parsing import Serializable, ArgumentParser, choice
from simple_parsing.helpers.serialization import register_decoding_fn, encode
import multiprocessing as mp
from typing import List
import logging
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
    configFile: str = ""
    savePath: str = "./data"
    saveFile: str = "database_name.pkl"
    pathToExternals: str = "./external"

    pulseFileExcitation: str = 'gauss_shape.txt'
    pulseFileRefocus: str = 'gauss_shape.txt'

    visualize: bool = True  # visualize different checkpoints
    d_flag: bool = False  # toggle diffusion calculations

    multiprocessing: bool = False
    mpHeadroom: int = 16
    mpNumCpus: int = 1

    def __post_init__(self):
        if self.multiprocessing:
            # we take at most the maximum free cpus but leave some headroom for other users
            self.mpNumCpus = mp.cpu_count() - self.mpHeadroom
            self.mpNumCpus = np.max([mp.cpu_count() - self.mpHeadroom, 4])
            # we take at least 4 cpus (kind of catches misconfiguration of the headroom parameter)
        self.mpNumCpus = int(self.mpNumCpus)


@dataclass
class SequenceParams(Serializable):
    """
    Parameters related to Sequence simulation
    """
    gammaHz: float = 42577478.518  # [Hz/t], global parameter
    gammaPi: float = gammaHz * 2 * np.pi

    ETL: int = 16  # echo train length
    ESP: float = 9.0  # echo spacing [ms]
    bw: float = 349  # bandwidth [Hz/px]
    durationAcquisition: float = 1e6 / bw  # [us] time for acquisition (of one pixel) * 1e6 <- [(px)s] * 1e6
    gradMode: str = choice("Rect", "Normal", "Verse", default="Verse")

    excitationAngle: float = 90.0  # [°]

    gradientExcitation: float = 0.0  # [mt/m], rectangular
    durationExcitation: float = 2560.0  # [us], rectangular

    gradientExcitationRephase: float = -10.51  # [mT/m], rephase
    durationExcitationRephase: float = 1080.0  # [us], rephase

    gradientExcitationVerse1: float = -42.0  # [mT/m], verse
    gradientExcitationVerse2: float = -23.92  # [mt/m], verse
    durationExcitationVerse1: float = 770.0  # [us], verse
    durationExcitationVerse2: float = 1020.0  # [us], verse

    refocusAngle: float = 135  # [°]

    gradientRefocus: float = 0.0  # [mT/m], rectangular
    durationRefocus: float = 3584.0  # [us], rectangular

    gradientCrush: float = -38.70  # [mT/m], crusher
    durationCrush: float = 1000.0  # [us], crusher

    gradientRefocusVerse1: float = -22.0  # [mT/m], verse
    gradientRefocusVerse2: float = -20.53  # [mt/m], verse
    durationRefocusVerse1: float = 1080.0  # [us], verse
    durationRefocusVerse2: float = 1424.0  # [us], verse

    gradientAcquisition: float = 0  # [mT/m]
    # gradient is set along Z axis to sample the magnetization into signal along the slice profile (artificial gradient)


@dataclass
class SimulationSettings(Serializable):
    """
    Optional settings for simulation eg. spatial resolution
    """
    sampleNumber: int = 2000  # no of sampling points along slice profile
    lengthZ: float = 0.01  # length extension of z-axis spanned by sample -> total length 2*lengthZ (-:+)
    acquisitionNumber: int = 50  # number of bins across slice sample -> effectively sets spatial resolution
    # resolution = lengthZ / acquisitionNumber

    t1_list: List = field(default_factory=lambda: [1.5])  # T1 to simulate [s]
    t2_list: List = field(default_factory=lambda: [[1, 20, 1]])  # T2 to simulate [ms]
    b1_list: List = field(default_factory=lambda: [0.9, 1.0])  # B1 to simulate
    d_list: List = field(default_factory=lambda: [700.0])
    # diffusion values to use if flag in config is set [mm²/s]
    total_num_sim: int = 4

    def __post_init__(self):
        array = np.empty(0)
        for item in self.t2_list:
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

    @classmethod
    def from_cmd_args(cls, args: ArgumentParser.parse_args):
        simParams = SimulationParameters(
            config=args.config,
            settings=args.settings,
            sequence=args.sequence,
        )

        if args.config.configFile:
            simParams = SimulationParameters.load(args.config.configFile)

        return simParams


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
        self.sample = np.exp(-8 * ((self.sampleAxis * 100) ** 16)) + 1e-6
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
