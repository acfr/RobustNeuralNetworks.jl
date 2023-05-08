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
nu, nx, nv, ny = 4, 10, 20, 2
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
yout = round.(y1; digits=2)
println(yout[1,:])
println(yout[2,:])

# output

[0.73, 0.72, -0.53, 0.25, 0.84, 0.97, 0.96, 1.13, 0.87, 1.07]
[1.13, 1.07, 1.44, 0.83, 0.94, 1.26, 0.86, 0.8, 0.96, 0.86]
```

See [Package Overview](@ref) for a detailed walkthrough of this example.