import sys
import pathlib

pulse_path = pathlib.Path(__file__).absolute().parent.parent.joinpath("rf_pulse_files/")
sys.path.append(pulse_path.as_posix())
