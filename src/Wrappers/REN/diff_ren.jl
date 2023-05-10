mutable struct DiffREN{T} <: AbstractREN{T}
    nl
    nu::Int
    nx::Int
    nv::Int
    ny::Int
    params::AbstractRENParams
end

"""
    DiffREN(ps::AbstractRENParams{T}) where T

Construct a differentiable REN from its direct parameterisation.

`DiffREN` is an alternative to [`REN`](@ref) and [`WrapREN`](@ref) that computes the explicit parameterisation `ExplicitRENParams`](@ref) every time the model is called, rather than storing it in the `REN` object.

This is slow and computationally inefficient if the model is called many times before updating the parameters (eg: in reinforcement learning). However, it can be trained just like any other [`Flux.jl`](http://fluxml.ai/Flux.jl/stable/) model (unlike [`WrapREN`](@ref)) and does not need to re-created if the trainable parameters are updated (unlike [`REN`](@ref)).

See also [`AbstractREN`](@ref), [`REN`](@ref), and [`WrapREN`](@ref).
"""
function DiffREN(ps::AbstractRENParams{T}) where T
    return DiffREN{T}(ps.nl, ps.nu, ps.nx, ps.nv, ps.ny, ps)
end

Flux.trainable(m::DiffREN) = Flux.trainable(m.params)

function (m::DiffREN)(xt::AbstractVecOrMat, ut::AbstractVecOrMat)
    explicit = direct_to_explicit(m.params)
    return m(xt, ut, explicit) 
end