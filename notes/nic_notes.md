# Package Architecture

This document is a work-in-progress. Use it to start our documentation of the `RobustNeuralNetworks.jl` package. I'm using this document to keep track of how I'm writing the package. It is not necessarily up-to-date or correct.

## TODO Lists (and other useful things)

### General:
- Write tests for basic functionality
- Add documentation and improve speed for `Base/acyclic_ren_solver.jl` code taken from Max's work

### Bugs:
- Cholesky initialisation of `DirectRENParams`

### Functionality
- How to include different initialisation methods?
- Write full support for GPU/CUDA arrays

### Conversion questions:
- `D22` terms in `output.jl` initialised as 0. Good idea in general?
- `sample_ff_ren` had `randn(2nx + nv, 2nx + nv) / sqrt(2nx + nv)`, not `.../ sqrt(2*(2nx + nv))`. Why?
- Why have `bx_scale` and `bv_scale` in constructor for `DirectRENParams`?
- Constructor for `implicit_ff_cell` has zeros for some params, random for others. Why?
- Construction of S1 (now `Y1`) in `ffREN.jl` line 130/136 divides by 2 an extra time. Not necessary, right? Does this change anything?
