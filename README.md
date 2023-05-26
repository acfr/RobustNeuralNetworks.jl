# RobustNeuralNetworks.jl

[![Build Status](https://github.com/acfr/RobustNeuralNetworks.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/acfr/RobustNeuralNetworks.jl/actions/workflows/CI.yml?query=branch%3Amain)

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://acfr.github.io/RobustNeuralNetworks.jl/stable/)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://acfr.github.io/RobustNeuralNetworks.jl/dev)


A Julia package for robust neural networks built from the [Recurrent Equilibrium Network (REN)](https://arxiv.org/abs/2104.05942) and [Lipschitz-Bounded Deep Network (LBDN)](https://arxiv.org/abs/2301.11526) model classes. Please visit [the docs page](https://acfr.github.io/RobustNeuralNetworks.jl/dev/) for detailed documentation

## Installation

To install the package, type the following into the REPL.

```
] add RobustNeuralNetworks
```

You should now be able to construct robust neural network models. The following example constructs a contracting REN and evalutates it given a batch of random initial states `x0` and inputs `u0`.

```julia
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
```

The output should be:

```julia
[-1.1 0.32 0.27 0.14 -1.23 -0.4 -0.7 0.01 0.19 0.81]
```

## Contact
Please contact Nic Barbara (nicholas.barbara@sydney.edu.au) with any questions.
