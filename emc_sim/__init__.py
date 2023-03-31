import sys
import pathlib

pulse_path = pathlib.Path(__name__).absolute().parent.joinpath("rf_pulse_files/")
sys.path.append(pulse_path.__str__())
