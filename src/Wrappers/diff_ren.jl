"""
    mutable struct DiffREN <: AbstractREN

Wrapper for `REN` type which automatically re-computes 
explicit parameters every time the model is called.

This is slow, but is compatible with `Flux.jl`
"""
mutable struct DiffREN <: AbstractREN
    nl
    nu::Int
    nx::Int
    nv::Int
    ny::Int
    params::AbstractRENParams
    T::DataType
end

"""
    DiffREN(ps::AbstractRENParams{T}) where T

Construct a differentiable REN from its direct parameterisation.

`DiffREN` is an alternative to [`REN`](@ref) that computes the explicit parameterisation every time the model is called. This is slow and computationally inefficient. However, it can be used with [`Flux.jl`](http://fluxml.ai/Flux.jl/stable/) just like any other `Flux` model to do machine learning.

The difference to [`REN`](@ref) and [`WrapREN`](@ref) is that the `ExplicitParams` struct is never stored, so an instance of `DiffREN` never has to be mutated or re-defined after it is created, even when learnable parameters are updated.

See also [`REN`](@ref) and [`WrapREN`](@ref).
"""
function DiffREN(ps::AbstractRENParams{T}) where T
    return DiffREN(ps.nl, ps.nu, ps.nx, ps.nv, ps.ny, ps, T)
end

Flux.trainable(m::DiffREN) = Flux.trainable(m.params)

function (m::DiffREN)(xt::VecOrMat, ut::VecOrMat)

    explicit = direct_to_explicit(m.params)

    b = explicit.C1 * xt + explicit.D12 * ut .+ explicit.bv
    wt = tril_eq_layer(m.nl, explicit.D11, b)
    xt1 = explicit.A * xt + explicit.B1 * wt + explicit.B2 * ut .+ explicit.bx
    yt = explicit.C2 * xt + explicit.D21 * wt + explicit.D22 * ut .+ explicit.by

    return xt1, yt
    
end