
import sys
import pathlib
emc_path = pathlib.Path(__file__).absolute().parent.parent.joinpath("emc-simulation/")
sys.path.append(emc_path.as_posix())
