# Example usage

Provided is an exemplary configuration File *SimulationConfiguration.json*,
with
```
python -m emc_sim --configFile /examples/SimulationConfiguration.json
```

You can run a simulation which would generate 38 curves.
Note the T~2~ values simulated range from 20 - 35 ms. However from 20 - 25 in steps of 0.5 ms. From 25 - 35 in steps of 2 ms. The equivalent definition of this range is given in the config (line 51) via:
```
"t2_list": [ [ 20, 25, 0.1 ], [ 25, 35, 2 ] ],
```
Thus a wide range of T~2~ values with varying step size can be created.

### Pulse Files
- The easiest way providing pulse shapes is via a (.txt) file consisting of tab delimited amplitude and phase values (see *external/* folder).
- If the visualization flag is set in the configuration it is easy to check the correct pulse input from the plotting of the pulse gradient forms:

[![pgform](https://github.com/schmidtijoe/emc-simulation/blob/master/examples/pulsegrad_visual.png)]

for the given gaussian shape pulse

- Pulses can also be created via python (eg Gauss or Sync shapes) and fed as array into the `functions.pulseCalibrationIntegral` function

### Sample & Pulse Profile
- The created sample is also displayed when the visualization flag is turned on.
- Additionally a small code change would yield plots of the pulse profiles during the simulation to evaluate the magnitude evolution (not recommended simulations with high number of curves or ETL)
- ToDo: add magnetization propagation evaluation visuals

| Sample Initialization         | Excitation Profile          |
| ---------------------         | ------------------          |
| [![sample](https://github.com/schmidtijoe/emc-simulation/blob/master/examples/sample_visual.png)]   | [![exci](https://github.com/schmidtijoe/emc-simulation/blob/master/examples/profile_visual.png)]  |
|    | parameters: B1: 1.0, T2: 0.02  |
