from simple_parsing import ArgumentParser, helpers, choice
from dataclasses import dataclass
from emc_sim.utils import create_folder_ifn_exist
from pathlib import Path
import json


@dataclass
class FileConfiguration(helpers.Serializable):
    ConfigFile: str = ""
    NiiDataPath: str = ""
    DatabasePath: str = ""
    ResampledDataOutputPath: str = ""
    FitDataOutputPath: str = ""


@dataclass
class FitParameters(helpers.Serializable):
    ResamplingOption: str = choice("re_db", "re_data", "keep", default="re_data")
    FitMetric: str = choice("threshold", "pearson", "mle", "l2", default="pearson")
    NoiseBackgroundEstimateCornerFraction: float = 8.0
    NoiseBackgroundEstimateVisualize: bool = False
    Multiprocessing: bool = False
    ProcessingNumBlocks: int = 50
    ProcessingHeadroomMultiprocessing: int = 20
    TestingFlag: bool = False


@dataclass
class FitOptions:
    config: FileConfiguration = FileConfiguration()
    opts: FitParameters = FitParameters()

    @classmethod
    def load(cls, path):
        with open(path, "r") as j_file:
            j_dict = json.load(j_file)
            config = FileConfiguration().from_dict(j_dict["config"])
            opts = FitParameters.from_dict(j_dict["opts"])
        return cls(config=config, opts=opts)

    @classmethod
    def fromCmdLine(cls, args: ArgumentParser.parse_args):
        """
        Configuration File overwrites cmd line input!
        :param args: parsed arguments from cmd
        :return: FitSettings instance
        """
        fitSet = cls(config=args.config, opts=args.opts)
        if fitSet.config.ConfigFile:
            path = Path(fitSet.config.ConfigFile).absolute()
            fitSet = FitOptions.load(path)
        return fitSet


def createCmdLineParser():
    """
    Build the parser for arguments
    Parse the input arguments.
    :return: parser, parsed arguments
    """
    parser = ArgumentParser(prog='emc_fit')
    parser.add_arguments(FileConfiguration, dest="config")
    parser.add_arguments(FitParameters, dest="opts")
    args = parser.parse_args()

    return parser, args
