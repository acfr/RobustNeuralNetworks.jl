# Getting Started

## Installation

`RobustNeuralNetworks.jl` is written in Julia and can be installed with the in-built package manager. It is not currently a registered package and must be installed directly from our GitHub repository.

Start a new Julia session and type the following into the REPL.
```julia
using Pkg
Pkg.add("git@github.com:acfr/RobustNeuralNetworks.jl.git")
```

## Basic Usage

You should now be able to construct robust neural network models. The following example constructs a contracting REN and evalutates it given a batch of random initial states `x0` and inputs `u0`.

```jldoctest
using Random
using RobustNeuralNetworks

# Setup
rng = MersenneTwister(42)
batches = 10
nu, nx, nv, ny = 4, 2, 20, 1

# Construct a REN
contracting_ren_ps = ContractingRENParams{Float64}(nu, nx, nv, ny; rng=rng)
ren = REN(contracting_ren_ps)

# Some random inputs
x0 = init_states(ren, batches; rng=rng)
u0 = randn(rng, ren.nu, batches)

# Evaluate the REN over one timestep
x1, y1 = ren(x0, u0)

println(round.(y1;digits=2))

# output

[-31.41 0.57 -0.55 -3.56 -35.0 -18.28 -25.48 -7.49 -4.14 15.31]
```