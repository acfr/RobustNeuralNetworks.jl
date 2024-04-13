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
rng = Xoshiro(42)
batches = 10
nu, nx, nv, ny = 4, 2, 20, 1

# Construct a REN
contracting_ren_ps = ContractingRENParams{Float64}(nu, nx, nv, ny; rng)
ren = REN(contracting_ren_ps)

# Some random inputs
x0 = init_states(ren, batches; rng)
u0 = randn(rng, ren.nu, batches)

# Evaluate the REN over one timestep
x1, y1 = ren(x0, u0)

println(round.(y1;digits=2))
```

The output should be:

```julia
[-1.49 0.75 1.34 -0.23 -0.84 0.38 0.79 -0.1 0.72 0.54]
```

## Citing the Package

If you use `RobustNeuralNetworks.jl` for any research or publications, please cite our work as necessary.
```bibtex
@article{barbara2023robustneuralnetworksjl,
   title   = {RobustNeuralNetworks.jl: a Package for Machine Learning and Data-Driven Control with Certified Robustness},
   author  = {Nicholas H. Barbara and Max Revay and Ruigang Wang and Jing Cheng and Ian R. Manchester},
   journal = {arXiv preprint arXiv:2306.12612},
   month   = {6},
   year    = {2023},
   url     = {https://arxiv.org/abs/2306.12612v1},
}
```

## Contact
Please contact Nic Barbara (nicholas.barbara@sydney.edu.au) with any questions.
