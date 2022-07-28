# EMC simulation package for python

## Simulation Code for creating a database for multi - eccho spin - echo sequences
- the main part does exactly this and is usable via commandline
- serializability was added, hence one can also provide .json files with the configurations (todo: see examplefiles)
- .json files can be saved (enabled by default) to keep track of the parameters for the created databases

## Watch out
- the simulation is based on knowing the sequence slice selective gradients and pulses.
- The magnetization response to the sequence is simulated for a variety of relaxation values across the slice profile.
- The sequence scheme has to be provided (e.g. through prior sequence simulations) in detail. The fields that need to be provided are found in the .sequence class, currently VERSE and rectangular gradients are implemented, pulse files might need to be loaded in and be provided externally (/external)
- The simulation might be extended to other scanning modalities, check simulations or define your own scheme there
- It is HIGHLY sequence dependent and might need to be modified, currently it is tailored to the sequence product se_mc sequence.

