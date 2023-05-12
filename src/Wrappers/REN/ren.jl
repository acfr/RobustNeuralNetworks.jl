mutable struct REN{T} <: AbstractREN{T}
    nl::Function
    nu::Int
    nx::Int
    nv::Int
    ny::Int
    explicit::ExplicitRENParams{T}
end

"""
    REN(ps::AbstractRENParams{T}) where T

Construct a REN from its direct parameterisation.

This constructor takes a direct parameterisation of REN
(eg: a [`GeneralRENParams`](@ref) instance) and converts it to a
**callable** explicit parameterisation of the REN.

See also [`AbstractREN`](@ref), [`WrapREN`](@ref), and [`DiffREN`](@ref).
"""
function REN(ps::AbstractRENParams{T}) where T
    explicit = direct_to_explicit(ps)
    return REN{T}(ps.nl, ps.nu, ps.nx, ps.nv, ps.ny, explicit)
end

"""
    abstract type AbstractREN end

Explicit parameterisation for recurrent equilibrium networks.

    (m::AbstractREN)(xt::AbstractVecOrMat, ut::AbstractVecOrMat)

Call an  `AbstractREN` model given internal states `xt` and inputs `ut`. 

If arguments are matrices, each column must be a vector of states or inputs (allows batch simulations).

# Examples

This example creates a contracting [`REN`](@ref) using [`ContractingRENParams`](@ref) and calls the model with some randomly generated inputs. 

```jldoctest
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

# output

[-31.41 0.57 -0.55 -3.56 -35.0 -18.28 -25.48 -7.49 -4.14 15.31]
```

See also [`REN`](@ref), [`WrapREN`](@ref), and [`DiffREN`](@ref).
"""
function (m::AbstractREN)(xt::AbstractVecOrMat, ut::AbstractVecOrMat)
    return m(xt, ut, m.explicit)
end

function (m::AbstractREN{T})(
    xt::AbstractVecOrMat{T}, 
    ut::AbstractVecOrMat{T},
    explicit::ExplicitRENParams{T}
) where T

    b = explicit.C1 * xt + explicit.D12 * ut .+ explicit.bv
    wt = tril_eq_layer(m.nl, explicit.D11, b)
    xt1 = explicit.A * xt + explicit.B1 * wt + explicit.B2 * ut .+ explicit.bx
    yt = explicit.C2 * xt + explicit.D21 * wt + explicit.D22 * ut .+ explicit.by

    return xt1, yt
end

"""
    init_states(m::AbstractREN, nbatches; rng=nothing)

Return matrix of (nbatches) state vectors of a REN initialised as zeros.
"""
function init_states(m::AbstractREN{T}, nbatches; rng=nothing) where T
    return zeros(T, m.nx, nbatches)
end

function init_states(m::AbstractREN{T}; rng=nothing) where T
    return zeros(T, m.nx)
end

"""
    set_output_zero!(m::AbstractREN)

Set output map of a REN to zero.

If the resulting model is called with `x1,y = ren(x,u)` then `y = 0` for any `x` and `u`.
"""
function set_output_zero!(m::AbstractREN)
    m.explicit.C2 .*= 0
    m.explicit.D21 .*= 0
    m.explicit.D22 .*= 0
    m.explicit.by .*= 0

    return nothing
end
