"""
$(TYPEDEF)

Wrapper for Recurrent Equilibrium Network type which
automatically re-computes explicit parameters every
time the model is called.

Compatible with Flux.jl
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
    DiffREN(ps::AbstractRENParams)

Construct DiffREN wrapper from direct parameterisation
"""
function DiffREN(ps::AbstractRENParams{T}) where T
    return DiffREN(ps.nl, ps.nu, ps.nx, ps.nv, ps.ny, ps, T)
end

"""
    Flux.trainable(m::DiffREN)

Define trainable parameters for `DiffREN` type
"""
Flux.trainable(m::DiffREN) = Flux.trainable(m.params)

"""
    (m::DiffREN)(xt::VecOrMat, ut::VecOrMat)

Call the REN given internal states xt and inputs ut. If 
function arguments are matrices, each column must be a 
vector of states or inputs (allows batch simulations).

Computes explicit parameterisation each time. This may
be slow if called many times!
"""
function (m::DiffREN)(xt::VecOrMat, ut::VecOrMat)

    explicit = direct_to_explicit(m.params)

    b = explicit.C1 * xt + explicit.D12 * ut .+ explicit.bv
    wt = tril_eq_layer(m.nl, explicit.D11, b)
    xt1 = explicit.A * xt + explicit.B1 * wt + explicit.B2 * ut .+ explicit.bx
    yt = explicit.C2 * xt + explicit.D21 * wt + explicit.D22 * ut .+ explicit.by

    return xt1, yt
    
end