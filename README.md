This is a version of the Non-Envy Equlibrium Solver (NEES) algorithm.

The algorithm implementation is mainly in [allocate.rs](src/solver/allocate.rs).

It solves for non-envy equilibrium prices in a setup with agents with heterogeneous preferences to be allocated to a set of non-divisible goods at price p.
This only allows for one dimension of heterogeneity for each item. Therefore utility functions are parameterised with money and single value for item quality.

This solver uses an envelope based system to determine ordering, where an envelope is a group of allocations that have certain properties.
