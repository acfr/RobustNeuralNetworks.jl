# Package Architecture

Some notes from our team meeting on 19/09/22. This document is a work-in-progress. Use it to start our documentation of the `RecurrentEquilibriumNetworks.jl` package.

## Description
The package is structured around the `REN` type. An object of type `REN` should have the following attributes:
- output layer
- explicit model struct
- in/out, state/nl sizes
- nonlinearity

and functions to build/use it as follows:
- constructor
- self-call method
- init state
- update params...? (possibly not)

Each `REN` is constructed from a direct (implicit) parameterisation of the REN architecture. Each variation of REN (eg: contracting, passive, Lipschitz bounded) will be a subtype of `AbstractRENParams`, an abstract type. They must all include the following attributes:
- in/out, state/nl sizes `nu, ny, nx, nv`
- output layer `Output`
- direct (implicit) parameters `DirectParams`

The output layer and implicit parameters are structs defined in `src/output.jl` and `direct_params.jl` (respectively).

## Code Structure
The main file is `src/RecurrentEquilibriumNetworks.jl`. This imports all relevant packages, defines abstract types, includes code from other files, and exports the necessary components of our package. All `using PackageName` statements should be included in this file. As a general guide:
- Only import packages you really need
- If you only need one function from a package, import it explicitly (not the whole package)

When including files in our `src/` folder, the order often matters. I have tried to structure the `include` statements in `RecurrentEquilibriumNetworks.jl` so that we only ever have to include code once, in the main file. Please follow the conventioned outlined in the comments.

## TODO Lists (and other useful things)

### Changes from Max's code:
- I've included `D22` as a trainable parameter in the output layer. This will break some older code in which we ignored `D22`
- Currently no method to construct a REN without specifying type. Seems good to force it

### Bugs:
- Cholesky initialisation of `DirectParams`

### Functionality
- How to include different initialisation methods?
- Write full support for GPU/CUDA arrays

### Conversion questions:
- `D22` terms in `output.jl` initialised as 0. Why not with `glorot_normal()`?
- `sample_ff_ren` had `randn(2nx + nv, 2nx + nv) / sqrt(2nx + nv)`, not `.../ sqrt(2*(2nx + nv))`. Why?
- Why have `bx_scale` and `bv_scale` in constructor for `DirectParams`?
- Constructor for `implicit_ff_cell` has zeros for some params, random for others. Why?
