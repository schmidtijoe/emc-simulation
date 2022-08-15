import numpy as np
from simple_parsing import ArgumentParser, helpers, choice
from dataclasses import dataclass
from emc_sim import utils
from pathlib import Path
import json
import nibabel as nib
import logging

logModule = logging.getLogger(__name__)


@dataclass
class FileConfiguration(helpers.Serializable):
    ConfigFile: str = ""
    NiiDataPath: str = ""
    DatabasePath: str = ""
    ResampledDataOutputPath: str = ""
    FitDataOutputPath: str = ""


@dataclass
class FitParameters(helpers.Serializable):
    ResamplingOption: str = choice("re_db", "re_data", "keep", "load_resampled", default="re_data")
    ResampleDataNumIterations: int = 3
    ResampleDataSimple: bool = False
    ResampleDataRegularizationLambda: float = 0.1
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

    def saveFit(self, fitArray: np.ndarray, niiImg: nib.Nifti1Image, name: str):
        shape = fitArray.shape
        if shape.__len__() != niiImg.shape:
            fitArray = np.reshape(fitArray, niiImg.shape[:-1])
        else:
            fitArray = np.reshape(fitArray, niiImg.shape)

        # save
        path = Path(self.config.FitDataOutputPath).absolute()
        utils.create_folder_ifn_exist(path)
        save_path = path.joinpath(f"{self.opts.FitMetric}_{name}_map.nii")
        logModule.info(f"Saving File: {save_path}")

        niiImg = nib.Nifti1Image(fitArray, niiImg.affine)
        nib.save(niiImg, save_path)


def createCmdLineParser() -> (ArgumentParser, ArgumentParser.parse_args):
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
