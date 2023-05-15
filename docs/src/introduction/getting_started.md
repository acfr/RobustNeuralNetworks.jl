# Getting Started

## Installation

`RobustNeuralNetworks.jl` is written in Julia and can be installed with the in-built package manager. To add the package, type the following into the REPL.

```
] add RobustNeuralNetworks
```

## Basic Usage

You should now be able to construct robust neural network models. The following example constructs a Lipschitz-bounded REN and evalutates it given a batch of random initial states and inputs.

```jldoctest
using Random
using RobustNeuralNetworks

# Setup
rng = MersenneTwister(42)
batches = 10
nu, nx, nv, ny = 4, 10, 20, 1
γ = 1

# Construct a REN
lipschitz_ren_ps = LipschitzRENParams{Float64}(nu, nx, nv, ny, γ; rng=rng)
ren = REN(lipschitz_ren_ps)

# Some random inputs
x0 = init_states(ren, batches; rng=rng)
u0 = randn(rng, ren.nu, batches)

# Evaluate the REN over one timestep
x1, y1 = ren(x0, u0)

# Print results for testing
println(round.(y1; digits=2))

# output

[0.23 -0.01 -0.06 0.15 -0.03 -0.11 0.0 0.42 0.24 0.22]
```

See [Package Overview](@ref) for a detailed walkthrough of this example. For a detailed example of training models from `RobustNeuralNetworks.jl`, we recommend starting with [Fitting a Curve with LBDN](@ref).
