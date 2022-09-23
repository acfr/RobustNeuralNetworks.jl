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

Main Recurrent Equilibrium Network type
"""
mutable struct REN
    nl
    nu::Int
    nx::Int
    nv::Int
    ny::Int
    explicit::ExplicitParams
end

"""
    REN(ps::AbstractRENParams)

Construct REN from direct parameterisation
"""
function REN(ps::AbstractRENParams)
    explicit = direct_to_explicit(ps)
    return REN(ps.nl, ps.nu, ps.nx, ps.nv, ps.ny, explicit)
end

"""
    (m::REN)(xt::VecOrMat, ut::VecOrMat)

Call the REN given internal states xt and inputs ut. If 
function arguments are matrices, each column must be a 
vector of states or inputs (allows batch simulations).
"""
function (m::REN)(xt::VecOrMat, ut::VecOrMat)

    b = m.explicit.C1 * xt + m.explicit.D12 * ut .+ m.explicit.bv
    wt = tril_eq_layer(m.nl, m.explicit.D11, b)
    xt1 = m.explicit.A * xt + m.explicit.B1 * wt + m.explicit.B2 * ut .+ m.explicit.bx
    yt = m.explicit.C2 * xt + m.explicit.D21 * wt + m.explicit.D22 * ut .+ m.explicit.by

    return xt1, yt
end

"""
    init_states(m::REN)

Return state vector initialised as zeros
"""
function init_states(m::REN)
    return typeof(m.explicit.A)(zeros(m.nx))
end

"""
    init_states(m::REN, nbatches)

Return matrix of (nbatches) state vectors initialised as zeros
"""
function init_states(m::REN, nbatches)
    return typeof(m.explicit.A)(zeros(m.nx, nbatches))
end

"""
    set_output_zero!(m::REN)

Set output map of REN to zero
"""
function set_output_zero!(m::REN)
    m.explicit.C2 .*= 0
    m.explicit.D21 .*= 0
    m.explicit.D22 .*= 0
    m.explicit.by .*= 0
end
