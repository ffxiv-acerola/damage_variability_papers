# Damage variability papers

Text and accompanying code used for write ups on modeling damage variability in FFXIV. Each directory contains the final written version, along with any notebooks/code/LaTeX files used to create the write up. Most notebooks will be sparsely documented since they are largely a proof of concept and used to create figures. If you would like to use the code based on the theory in these papers, check out the [ffxiv_stats repo](https://github.com/ffxiv-acerola/ffxiv_stats).

## Brief description of each papers:

1. **Variability in damage calculations**: Framework for exactly computing the mean, variance, and skewness of a rotation using moment generating functions. How the addition of direct and critical-direct hits affects variance is analyzed.

2. **Damage distributions for deterministic and stochastic rotations**: Damage distributions are exactly computed by convolving the 1-hit damage distribution multiple times. How to calculate damage distributions for stochastic rotations, where the number of action usages or rotations themselves are randomly distributed is also discussed.

3. **???**