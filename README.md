This is a prototype version of the Non-Envy Equlibrium Solver (NEES) algorithm, using the 'envelope' approach to overcoming the single-crossing restriction.

The algorithm implementation is mainly in [allocate.rs](src/solver/allocate.rs).

It solves for non-envy equilibrium prices in a setup with agents with heterogeneous preferences to be allocated to a set of non-divisible goods at price p.
This only allows for one dimension of heterogeneity for each item. Therefore utility functions are parameterised with money and single value for item quality.

This solver uses an envelope based system to determine ordering, where an envelope is a group of allocations that have certain properties.

A higher dimensional version of the solution algorithm has beem developed which uses a graph based approach, which is fundamentally capturing the same thing, but expressing it in a more general way that can handle an arbitrary number of item quality dimensions.

Future work is planned on allowing endogenous income (based on agents being able to own items).
