import numpy as np
import simple_parsing as sp
import dataclasses as dc
import pathlib as plib
import json
import nibabel as nib
import logging
import typing

logModule = logging.getLogger(__name__)


@dc.dataclass
class FileConfiguration(sp.helpers.Serializable):
    ConfigFile: str = sp.field(alias=["-c"], default="")
    NiiDataPath: str = sp.field(alias=["-i"], default="")
    DatabasePath: str = sp.field(alias=["-db"], default="")
    OutputPath: str = sp.field(alias=["-o"], default="")
    NameId: str = ""
    Multiprocessing: bool = True
    HeadroomMultiprocessing: int = 16

    def __post_init__(self):
        op = plib.Path(self.OutputPath).absolute()
        if not op.is_dir():
            self.OutputPath = op.parent.__str__()


@dc.dataclass
class FitParameters(sp.helpers.Serializable):
    Mode: str = sp.choice("Denoize", "d", "Fit", "f", "Both", "df", default="Both")
    FitB1Weighting: bool = True
    FitB1WeightingInput: str = ""
    FitB1WeightingLambda: float = 0.1
    FitB1PriorVoxelDims: typing.List = dc.field(default_factory=lambda: [0.7, 0.7, 0.7])
    DenoizeMaxNumIterations: int = 1
    DenoizeSave: bool = True
    Visualize: bool = True
    TestingFlag: bool = False


@dc.dataclass
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
    def fromCmdLine(cls, args: sp.ArgumentParser.parse_args):
        """
        Configuration File overwrites cmd line input!
        :param args: parsed arguments from cmd
        :return: FitSettings instance
        """
        fitSet = cls(config=args.config, opts=args.opts)
        # catch non defaults
        def_config = FileConfiguration().to_dict()
        def_opts = FitParameters().to_dict()
        non_def = {}
        for key, value in args.config.to_dict().items():
            if def_config[key] != value:
                non_def[key] = value
        for key, value in args.opts.to_dict().items():
            if def_opts[key] != value:
                non_def[key] = value
        # read in config file if provided
        if fitSet.config.ConfigFile:
            path = plib.Path(fitSet.config.ConfigFile).absolute()
            fitSet = FitOptions.load(path)
        # fill in non defaults, aka additional provided
        for key, value in non_def.items():
            if key in fitSet.config.to_dict().keys():
                fitSet.config.__setattr__(key, value)
            else:
                fitSet.opts.__setattr__(key, value)
        # catch empty outut path and use input path
        if not fitSet.config.OutputPath or fitSet.config.OutputPath == plib.Path(__file__).absolute().parent.__str__():
            fitSet.config.__setattr__("OutputPath", plib.Path(fitSet.config.NiiDataPath).absolute().parent.__str__())
        return fitSet

    def saveFit(self, fitArray: np.ndarray, niiImg: nib.Nifti1Image, name: str):
        shape = fitArray.shape
        if shape.__len__() != niiImg.shape:
            fitArray = np.reshape(fitArray, niiImg.shape[:-1])
        else:
            fitArray = np.reshape(fitArray, niiImg.shape)

        # save
        path = plib.Path(self.config.OutputPath).absolute()
        if not path.stem == "fit":
            path = path.joinpath("fit/")
        path.mkdir(parents=True, exist_ok=True)
        save_path = path.joinpath(f"{name}_map.nii")
        logModule.info(f"Saving File: {save_path}")

        niiImg = nib.Nifti1Image(fitArray, niiImg.affine)
        nib.save(niiImg, save_path)


def createCmdLineParser() -> (sp.ArgumentParser, sp.ArgumentParser.parse_args):
    """
    Build the parser for arguments
    Parse the input arguments.
    :return: parser, parsed arguments
    """
    parser = sp.ArgumentParser(prog='emc_fit')
    parser.add_arguments(FileConfiguration, dest="config")
    parser.add_arguments(FitParameters, dest="opts")
    args = parser.parse_args()

    return parser, args
