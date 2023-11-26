# Damage variability papers

Text and accompanying code used for write ups on modeling damage variability in FFXIV. Each directory contains the final written version, along with any notebooks/code/LaTeX files used to create the write up. Most notebooks will be sparsely documented since they are largely a proof of concept and used to create figures. Most notebooks generally hard-code in parameters for specific levels, which may be outdated too. If you would like to use the code based on the theory in these papers, check out the [ffxiv_stats repo](https://github.com/ffxiv-acerola/ffxiv_stats) for the most maintained and code.

## Brief description of each papers:

1. [**Variability in damage calculations**](/01_variability_in_damage_calculations/Variability%20in%20damage%20calculations.pdf): Framework for exactly computing the mean, variance, and skewness of a rotation using moment generating functions. How the addition of direct and critical-direct hits affects variance is analyzed.

2. [**Damage distributions for deterministic and stochastic rotations**](/02_damage_distributions_deterministic_stochastic/Damage%20distributions%20for%20deterministic%20and%20stochastic%20rotations.pdf): Damage distributions are exactly computed by convolving the 1-hit damage distribution multiple times. How to calculate damage distributions for stochastic rotations, where the number of action usages or rotations themselves are randomly distributed is also discussed.

3. [**Sampling damage distributions**](/03_sampling_damage_distributions/sampling-damage-distributions.ipynb): An application of previously developed theory. Views performing a rotation as sampling a damage distribution and discusses applications for speed kills and parsing. How to generate rotations via the FFLogs API is also shown.

