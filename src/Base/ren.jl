"""
$(TYPEDEF)

Struct containing explicit REN parameters
"""
mutable struct ExplicitParams{T}
    A::Matrix{T}
    B1::Matrix{T}
    B2::Matrix{T}
    C1::Matrix{T}
    C2::Matrix{T}
    D11::Matrix{T}
    D12::Matrix{T}
    D21::Matrix{T}
    D22::Matrix{T}
    bx::Vector{T}
    bv::Vector{T}
    by::Vector{T}
end

"""
$(TYPEDEF)

Type for Recurrent Equilibrium Networks
"""
mutable struct REN <: AbstractREN
    nl
    nu::Int
    nx::Int
    nv::Int
    ny::Int
    explicit::ExplicitParams
    T::DataType
end

"""
    REN(ps::AbstractRENParams{T}) where T

Construct a REN from its direct parameterisation.

This constructor takes a direct parameterisation of REN
(eg: a `GeneralRENParams` instance) and converts it to an
explicit parameterisation of the REN for implementation.
"""
function REN(ps::AbstractRENParams{T}) where T
    explicit = direct_to_explicit(ps)
    return REN(ps.nl, ps.nu, ps.nx, ps.nv, ps.ny, explicit, T)
end

"""
    (m::AbstractREN)(xt::VecOrMat, ut::VecOrMat)

Call a REN given internal states `xt` and inputs `ut`. If 
function arguments are matrices, each column must be a 
vector of states or inputs (allows batch simulations).
"""
function (m::AbstractREN)(xt::VecOrMat, ut::VecOrMat)

    b = m.explicit.C1 * xt + m.explicit.D12 * ut .+ m.explicit.bv
    wt = tril_eq_layer(m.nl, m.explicit.D11, b)
    xt1 = m.explicit.A * xt + m.explicit.B1 * wt + m.explicit.B2 * ut .+ m.explicit.bx
    yt = m.explicit.C2 * xt + m.explicit.D21 * wt + m.explicit.D22 * ut .+ m.explicit.by

    return xt1, yt
end

"""
    init_states(m::AbstractREN; rng=nothing)

Return state vector of a REN initialised as zeros
"""
function init_states(m::AbstractREN; rng=nothing)
    return zeros(m.T, m.nx)
end

"""
    init_states(m::AbstractREN, nbatches; rng=nothing)

Return matrix of (nbatches) state vectors of a REN initialised as zeros
"""
function init_states(m::AbstractREN, nbatches; rng=nothing)
    return zeros(m.T, m.nx, nbatches)
end

"""
    set_output_zero!(m::AbstractREN)

Set output map of a REN to zero such that `x1,y = ren(x,u)`
then `y = 0` for any `x` and `u`.
"""
function set_output_zero!(m::AbstractREN)
    m.explicit.C2 .*= 0
    m.explicit.D21 .*= 0
    m.explicit.D22 .*= 0
    m.explicit.by .*= 0
end
