"""
$(TYPEDEF)

Wrapper for Recurrent Equilibrium Network type which
automatically re-computes explicit parameters whenever
the direct parameters are edited.

Not compatible with Flux.jl
"""
mutable struct WrapREN2 <: AbstractREN
    nl
    nu::Int
    nx::Int
    nv::Int
    ny::Int
    params::AbstractRENParams
    explicit::ExplicitParams
    old_params::AbstractRENParams
    T::DataType
end

"""
    WrapREN2(ps::AbstractRENParams)

Construct WrapREN2 wrapper from direct parameterisation
"""
function WrapREN2(ps::AbstractRENParams{T}) where T
    explicit = direct_to_explicit(ps)
    old_ps = deepcopy(ps)
    return WrapREN2(ps.nl, ps.nu, ps.nx, ps.nv, ps.ny, ps, explicit, old_ps, T)
end

"""
    Flux.trainable(m::WrapREN2)

Define trainable parameters for `WrapREN2` type
"""
Flux.trainable(m::WrapREN2) = Flux.trainable(m.params)

"""
    (m::WrapREN2)(xt::VecOrMat, ut::VecOrMat)

Call the REN given internal states xt and inputs ut. If 
function arguments are matrices, each column must be a 
vector of states or inputs (allows batch simulations).

Updates the explicit parameterisation if direct parameters 
have been updated.
"""
function (m::WrapREN2)(xt::VecOrMat, ut::VecOrMat)

    # Compute explicit parameterisation
    if !( m.params.direct == m.old_params.direct )
        m.explicit = direct_to_explicit(m.params)
        m.old_params = deepcopy(m.params)
    end

    # Compute update
    b = m.explicit.C1 * xt + m.explicit.D12 * ut .+ m.explicit.bv
    wt = tril_eq_layer(m.nl, m.explicit.D11, b)
    xt1 = m.explicit.A * xt + m.explicit.B1 * wt + m.explicit.B2 * ut .+ m.explicit.bx
    yt = m.explicit.C2 * xt + m.explicit.D21 * wt + m.explicit.D22 * ut .+ m.explicit.by

    return xt1, yt
    
end