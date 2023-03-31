import sys
import pathlib
from . import options
from . import simulations
from .utils import normalize_array, niiDataLoader, load_database
pulse_path = pathlib.Path(__file__).absolute().parent.parent.joinpath("rf_pulse_files/")
sys.path.append(pulse_path.as_posix())
