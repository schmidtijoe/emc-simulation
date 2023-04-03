# Agent Based Optimization for MESE sequence parameter selection based on EMC signal response prediction

### Problem
- the emc framework allows computation of signal response curve dictionaries given sequence parameters of a MESE sequence
- this in turn gives the opportunity to track changes in signal based on sequence parameter or setup influences.
- i.e. B1 transmit inhomogeneity leads to changes in the flip angles and to different signal responses in MESE acquisitions
- We want to solve the reverse problem: Can we change the sequence parameters (e.g. refocusing flip angles) of a MESE sequence in order to separate signal responses from different underlying tissue
- i.e. optimize for minimum correlation of signal prediction curves coming from selected voxels / tissue contributions.

### Selected T2 characteristics
- Want to separate short contributions, Subcortical GM, GM, WM, and CSF tissues.
- For that we choose:
- T2 = 5, 20, 35, 45, 200 ms

### Ensure B1 bias insensitivity
- Want B1+ bias to not influence separation capabilities of the objective function
- Different T2 contributions need to be uncorrelated irrespective of given B1+ bias
- Choose B1+ offsets:
- B1 = 0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.4

### Optimisation Aim
- MESE sequence with ETL = 7, and minimum echo spacing (ESP ~ 9ms).
- First objective: randomize Refocusing flip angles following a 90 ° excitation pulse, between 90 and 180 °.