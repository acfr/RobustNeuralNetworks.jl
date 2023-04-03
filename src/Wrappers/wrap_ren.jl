"""
$(TYPEDEF)

Wrapper for Recurrent Equilibrium Network type combining
direct parameters and explicit model into one struct.

Requires user to manually update explicit params when
direct params are changed. Faster/more efficient than
computing explicit parameterisation at each model call.
"""
mutable struct WrapREN <: AbstractREN
    nl
    nu::Int
    nx::Int
    nv::Int
    ny::Int
    explicit::ExplicitParams
    params::AbstractRENParams
    T::DataType
end

"""
    WrapREN(ps::AbstractRENParams{T}) where T

Construct REN wrapper from direct parameterisation
"""
function WrapREN(ps::AbstractRENParams{T}) where T
    explicit = direct_to_explicit(ps)
    return WrapREN(ps.nl, ps.nu, ps.nx, ps.nv, ps.ny, explicit, ps, T)
end

"""
    update_explicit!(m::WrapREN)

Update explicit model using the current direct parameters
"""
function update_explicit!(m::WrapREN)
    m.explicit = direct_to_explicit(m.params)
    return nothing
end

"""
    Flux.trainable(m::WrapREN)

Define trainable parameters for `WrapREN` type
"""
Flux.trainable(m::WrapREN) = Flux.params(m.params)
