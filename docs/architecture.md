# Package Architecture

Some notes from our team meeting on 19/09/22. This document is a work-in-progress. Use it to start our documentation of the `RecurrentEquilibriumNetworks.jl` package.

## Description
The package is structured around an `REN` type. An object of type `REN` should have the following attributes:
- output layer
- explicit model struct
- in/out, state/nl sizes
- nonlinearity

and functions to build/use it as follows:
- constructor
- self-call method
- init state
- update params...? (possibly not)

Each `REN` is constructed from a direct (implicit) parameterisation of the REN architecture. Each variation of REN (eg: contracting, passive, Lipschitz bounded) will be a subtype of `DirectREN`, an abstract type. They must all include the following attributes:
- in/out, state/nl sizes `nu, ny, nx, nv`
- nonlinearity eg: `Flux.relu`
- output layer `Output`
- direct (implicit) parameters `DirectParams`

The output layer and implicit parameters are structs defined in `src/output.jl` and `direct_params.jl` (respectively).

## TODO:
- Decide what to include in `DirectParams` struct
- How best to import packages. In each file, or centrally?
- Write full support for GPU/CUDA arrays