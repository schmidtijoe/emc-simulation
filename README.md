# EMC simulation package for python

A Python Adaption of the Echo Modulation Curve Algorithm, a Bloch Simulation based Reconsturction for Multi-Echo Spin-Echo Data (
[Ben-Eliezer et al. 2015](https://doi.org/10.1002/mrm.25156) ).

#### When to use
- Reconstruction of Multi-Echo Spin-Echo (MESE) slice selective MRI Sequences (possible extension to other slice selective modalities)
- Evaluation of Slice Profiles and Stimulated Echoes
- Generation of a lookup dictionary database of simulated sequence response curves for fitting of MESE data
- quantitative T<sub>2</sub> estimation with evaluation of B<sub>1</sub> transmit field and perspectively diffusion bias.

#### What is needed
- conda or venv python environment specified by *environment.yml*
- Sequence specs:
  - pulses & gradients with exakt timings.
  - Simulation uses hard pulse approximation but is tailored to the sequence parameters. i.e. Siemens *IDEA* sequence simulation for a Siemens scanner measurement is needed

#### How to use
###### Command Line Interface:

- activate environment, navigate to directory eg:

```
conda activate emc-simulation
cd /path/to/emc_sim
```

- show help / usage options; run script
```
python -m emc_sim --help
python -m emc_sim
```

- This would refer to default settings, for easy interfacing it is best to save your sequence parameters within a configFile (see examples), which is just a .json file with all the parameters
```
python -m emc_sim --configFile /path/to/config.json
```
- exemplary config and run: see [examples](https://github.com/schmidtijoe/emc-simulation/tree/master/examples) folder
