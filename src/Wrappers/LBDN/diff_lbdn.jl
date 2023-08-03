# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

mutable struct DiffLBDN{T} <: AbstractLBDN{T}
    nl::Function
    nu::Int
    nh::Vector{Int}
    ny::Int
    params::AbstractLBDNParams{T}
end

"""
    DiffLBDN(ps::AbstractLBDNParams{T}) where T

Construct a differentiable LBDN from its direct parameterisation.

`DiffLBDN` is an alternative to [`LBDN`](@ref) that computes the explicit parameterisation [`ExplicitLBDNParams`](@ref) each time the model is called, rather than storing it in the `LBDN` object.

This model wrapper is easiest to use if you plan to update the model parameters after every call of the model. It can be trained just like any other [`Flux.jl`](http://fluxml.ai/Flux.jl/stable/) model and does not need to be re-created if the trainable parameters are updated (unlike [`LBDN`](@ref)).
    
However, it is slow and computationally inefficient if the model is called many times before updating the parameters (eg: in reinforcement learning). 

# Examples

The syntax to construct a `DiffLBDN` is identical to that of an [`LBDN`](@ref). 

``` julia
using RobustNeuralNetworks

nu, nh, ny, γ = 1, [10, 20], 1
lbdn_params = DenseLBDNParams{Float64}(nu, nh, ny, γ)
model = DiffLBDN(lbdn_params)
```

See also [`AbstractLBDN`](@ref), [`LBDN`](@ref), [`SandwichFC`](@ref).
"""
function DiffLBDN(ps::AbstractLBDNParams{T}) where T
    return DiffLBDN{T}(ps.nl, ps.nu, ps.nh, ps.ny, ps)
end

function (m::DiffLBDN)(u::AbstractVecOrMat)
    explicit = direct_to_explicit(m.params)
    return m(u, explicit)
end

@functor DiffLBDN (params, )

function set_output_zero!(m::DiffLBDN)
    set_output_zero!(m.params)
end