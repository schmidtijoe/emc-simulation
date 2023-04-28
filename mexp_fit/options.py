import dataclasses as dc
import logging
import pathlib as plib
import numpy as np
from simple_parsing import ArgumentParser, helpers, field
import typing
import nibabel as nib
logModule = logging.getLogger(__name__)


@dc.dataclass
class Configuration(helpers.Serializable):
    ConfigFile: str = field(alias=["-c"], default="")
    NiiDataPath: str = field(alias=["-i"], default="")
    OutputPath: str = field(alias=["-o"], default="")
    NameId: str = ""
    Denoize: bool = False
    DenoizeMaxNumIterations: int = 5
    DenoizeUseMp: bool = True
    DenoizeMpHeadroom: int = 12
    Visualize: bool = True
    EchoTimes: typing.List = dc.field(default_factory=lambda: [10.0, 20.0, 30.0])

    def __post_init__(self):
        self.ArrEchoTimes = np.array(self.EchoTimes) / 1e3

    @classmethod
    def from_cmd_args(cls, args: ArgumentParser.parse_args):
        # create default_dict
        default_instance = cls()
        instance = cls()
        if args.config.ConfigFile:
            confPath = plib.Path(args.config.ConfigFile).absolute()
            instance = cls.load(confPath)
            # might contain defaults
        for key, item in default_instance.__dict__.items():
            parsed_arg = args.config.__dict__.get(key)
            # catch lists arrays
            if isinstance(parsed_arg, (np.ndarray, list)):
                p_arr = np.array(parsed_arg)
                d_arr = np.array(default_instance.__dict__.get(key))
                for n_idx in range(len(p_arr)):
                    if p_arr[n_idx] != d_arr[n_idx]:
                        instance.__setattr__(key, parsed_arg)
                        break
            else:
                # if parsed arguments are not defaults
                if parsed_arg != default_instance.__dict__.get(key):
                    # update instance even if changed by config file -> that way prioritize cmd line input
                    instance.__setattr__(key, parsed_arg)
        return instance


def save_nii(path: typing.Union[str, plib.Path], nii_data: np.ndarray, nii_img: nib.nifti1.Nifti1Image, name: str):
    # save
    path = plib.Path(path).absolute()
    if path.suffixes:
        path = path.parent

    path.mkdir(parents=True, exist_ok=True)
    save_path = path.joinpath(f"{name}_map.nii")
    logModule.info(f"Saving File: {save_path}")

    nii_img = nib.Nifti1Image(nii_data, nii_img.affine)
    nib.save(nii_img, save_path)


def createCmdLineParser() -> (ArgumentParser, ArgumentParser.parse_args):
    """
    Build the parser for arguments
    Parse the input arguments.
    :return: parser, parsed arguments
    """
    parser = ArgumentParser(prog='mexp_fit')
    parser.add_arguments(Configuration, dest="config")
    args = parser.parse_args()

    return parser, args
