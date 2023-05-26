mutable struct DiffREN{T} <: AbstractREN{T}
    nl::Function
    nu::Int
    nx::Int
    nv::Int
    ny::Int
    params::AbstractRENParams{T}
end

"""
    DiffREN(ps::AbstractRENParams{T}) where T

Construct a differentiable REN from its direct parameterisation.

`DiffREN` is an alternative to [`REN`](@ref) and [`WrapREN`](@ref) that computes the explicit parameterisation [`ExplicitRENParams`](@ref) every time the model is called, rather than storing it in the `REN` object.

This model wrapper is easiest to use if you plan to update the model parameters after every call of the model. It an be trained just like any other [`Flux.jl`](http://fluxml.ai/Flux.jl/stable/) model (unlike [`WrapREN`](@ref)) and does not need to be re-created if the trainable parameters are updated (unlike [`REN`](@ref)).
    
However, it is slow and computationally inefficient if the model is called many times before updating the parameters (eg: in reinforcement learning). 

# Examples

The syntax to construct a `DiffREN` is identical to that of a [`REN`](@ref). 

``` julia
using RobustNeuralNetworks

nu, nx, nv, ny = 1, 10, 20, 1
ren_params = ContractingRENParams{Float64}(nu, nx, nv, ny)
model = DiffREN(ren_params)
```

See also [`AbstractREN`](@ref), [`REN`](@ref), and [`WrapREN`](@ref).
"""
function DiffREN(ps::AbstractRENParams{T}) where T
    return DiffREN{T}(ps.nl, ps.nu, ps.nx, ps.nv, ps.ny, ps)
end

Flux.@functor DiffREN (params, )

function (m::DiffREN)(xt::AbstractVecOrMat, ut::AbstractVecOrMat)
    explicit = direct_to_explicit(m.params)
    return m(xt, ut, explicit) 
end