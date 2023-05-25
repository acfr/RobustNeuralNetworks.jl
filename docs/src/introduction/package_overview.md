# Package Overview

The key difference between the REN and LBDN models in `RobustNeuralNetworks.jl` and models in other popular machine learning libraries like [`Flux.jl`](https://fluxml.ai/) is our separation of the direct and explicit parameters. Before reading this page, we recommend reading at least the first two sections of the [Background Theory](@ref).

## Separating model parameterisations

The REN models are defined by two fundamental types. Any subtype of [`AbstractRENParams`](@ref) holds all the information required to directly parameterise a REN satisfying some user-defined behavioural constraints. Any subtype of [`AbstractREN`](@ref) represents the REN in its explicit form so that it can be called and evaluated. The same is true for [`AbstractLBDNParams`](@ref) and [`AbstractLBDN`](@ref) regarding LBDN models.

Our main reason for separating the direct and explicit parameterisations is to address the computational bottleneck when converting from direct to explicit (mapping ``\theta \mapsto \bar{\theta}``).

- In some applications (eg: reinforcement learning or system identification), a model is called many times before its learnable parameters are updated.

- To save on computation time, it is convenient to convert to the explicit parameters, evaluate the model many times, and only update the explicit parameters when the learnable parameters are changed. 

- Have to differentiate through the mapping with backpropagation. Can't store explicit params, or they'd be mutated. Flux not happy!

- Instead, we create a new explicit model each time the params are updated.

I'll need some pseudo code or a small example for this though! It's not really very clear yet. 


