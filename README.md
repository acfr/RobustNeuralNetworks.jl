# RecurrentEquilibriumNetworks.jl

## Status
[![Build Status](https://github.com/nic-barbara/RecurrentEquilibriumNetworks.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/nic-barbara/RecurrentEquilibriumNetworks.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Package Description

Julia package for Recurrent Equilibrium Networks.

[NOTE] This package is a work-in-progress. For now, you may find the following links useful:
- Tutorial on [developing Julia packages](https://julialang.org/contribute/developing_package/) by Chris Rackauckas (MIT)
- Documentation on [managing Julia packages](https://pkgdocs.julialang.org/v1/managing-packages/) and developing unregistered packages with `Pkg.jl`


## Installation for Development

To install the package for development, clone the repository into your Julia dev folder:
- For Linux/Mac, use `git clone git@github.com:acfr/RecurrentEquilibriumNetworks.jl.git RecurrentEquilibriumNetworks` inside your `~/.julia/dev/` directory.
- Note that the repo is `RecurrentEquilibriumNetworks.jl`, but the folder is `RecurrentEquilibriumNetworks`. This is convention for Julia packages.

Navigate to the repository directory, start the Julia REPL, and type `] activate .` to activate the package. You can now test out some basic functionality:

```julia
using Random
using RecurrentEquilibriumNetworks

batches = 50
nu, nx, nv, ny = 4, 10, 20, 2

contracting_ren_ps = ContractingRENParams{Float64}(nu, nx, nv, ny)
contracting_ren = REN(contracting_ren_ps)

x0 = init_states(contracting_ren, batches)
u0 = randn(contracting_ren.nu, batches)

x1, y1 = contracting_ren(x0, u0)  # Evaluates the REN over one timestep

println(x1)
println(y1)
```


## Contributing to the Package

The main file is `src/RecurrentEquilibriumNetworks.jl`. This imports all relevant packages, defines abstract types, includes code from other files, and exports the necessary components of our package. All `using PackageName` statements should be included in this file. As a general guide:
- Only import packages you really need
- If you only need one function from a package, import it explicitly (not the whole package)

When including files in our `src/` folder, the order often matters. I have tried to structure the `include` statements in `RecurrentEquilibriumNetworks.jl` so that we only ever have to include code once, in the main file. Please follow the conventioned outlined in the comments.

The source files for our package are al in the `src/` folder, and are split between `src/Base/` and `src/ParameterTypes/`. The `Base/` folder should contain code relevant to the core functionality of this package. The `ParameterTypes/` is where to add different versions of REN (eg: contracting REN, Lipschitz-bounded REN, etc.). See below for further documentation.

Once you have written any code for this package, be sure to test it thoroughly. Write testing scripts for the package in `test/`:
- See [`Test.jl`](https://docs.julialang.org/en/v1/stdlib/Test/) documentation for writing tests
- Run all tests for the package with `] test`

Use git to pull/push changes to the package as normal while developing it.


## Using the package

Once the package is functional, it can be used in other Julia workspaces like any normal package by following these instructions:

- Add a development version of the package with: `] dev git@github.com:acfr/RecurrentEquilibriumNetworks.jl.git`
- This is instead of the usual `] add` command. We also have to use the git link not the package name because it is an unregistered Julia package.
- Whenever you use the package, it will access the latest version in your `.julia/dev/` folder rather than the stable release in the `main` branch. This is easiest for development while we frequently change the package.
- To use the code on the current main branch of the repo (latest stable release), instead type `] add git@github.com:acfr/RecurrentEquilibriumNetworks.jl.git`. You will have to manually update the package as normal with `] update RecurrentEquilibriumNetworks`.


## Some Early Documentation
The package is structured around the `REN <: AbstractREN` type. An object of type `REN` has the following attributes:
- explicit model struct
- in/out, state/nl sizes
- nonlinearity

and functions to build/use it as follows:
- A constructor
- A self-call method
- A function to initialise a state vector, `x0 = init_state(batches)`
- A function to set the output to zero, `set_output_zero!(ren)`

Each `REN` is constructed from a direct (implicit) parameterisation of the REN architecture. Each variation of REN (eg: contracting, passive, Lipschitz bounded) is a subtype of `AbstractRENParams`, an abstract type. This encodes all information required to build a `REN` satisfying some set of behavioural constraints. Each subtype must include the following attributes:
- in/out, state/nl sizes `nu, ny, nx, nv`
- output layer of type `OutputLayer`
- direct (implicit) parameters of type `DirectParams`
- Any other attributes relevant to the parameterisation. Eg: `Q, S, R, alpha_bar` for a general REN

The output layer and implicit parameters are structs defined in `src/Base/output_layer.jl` and `src/Base/direct_params.jl` (respectively). Each subtype of `AbstractRENParams` must also have the following methods:
- A constructor
- A definition of `Flux.trainable()` specifying the trainable parameters
- A definition of `direct_to_explicit()` to convert the direct paramterisation to its explicit form

See `src/ParameterTypes/general_ren.jl` for an example.


### Non-differentiable REN Wrapper

There are many ways to train a REN, some of which do not involve differentiating the model. In these cases, it is convenient to have a wrapper `WrapREN <: AbstractREN` for the `REN` type that does not need to be destroyed an recreated whenever the direct parameters change. `WrapREN` is structured exactly the same as `REN`, but also holds the `AbstractRENParams` used to construct its explicit model. The explicit model can be updated in-place following any changes to the direct parameters. See below for an example.

```julia
using Random
using RecurrentEquilibriumNetworks

batches = 50
nu, nx, nv, ny = 4, 10, 20, 2

ren_ps = GeneralRENParams{Float64}(nu, nx, nv, ny)
ren = WrapREN(ren_ps)

x0 = init_states(ren, batches)
u0 = randn(ren.nu, batches)

x1, y1 = ren(x0, u0)  # Evaluates the REN over one timestep

# Update the model after changing a parameter
ren.params.direct.B2 .*= rand(size(ren.params.direct.B2)...)
update_explicit!(ren)
```
[NOTE] This operation is not compatible with Flux differentiation because the explicit parameters are mutated during the update.

## Contact
Nic Barbara (nicholas.barbara@sydney.edu.au) for any questions/concerns.
