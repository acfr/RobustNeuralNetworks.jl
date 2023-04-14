@doc raw"""
    mutable struct ExplicitParams{T}

Explicit REN parameter struct.

These parameters define a recurrent equilibrium network with model inputs and outputs ``u_t, y_t``, neuron inputs and outputs ``v_t,w_t``, and states `x_t`.

```math
\begin{equation*}
\begin{bmatrix}
x_{t+1} \\ v_t \\ y_t
\end{bmatrix}
= 
\begin{bmatrix}
A & B_1 & B_2 \\
C_1 & D_{11} & D_{12} \\
C_2 & D_{21} & D_{22} \\
\end{bmatrix}
\begin{bmatrix}
x_t \\ w_t \\ u_t
\end{bmatrix}
+ 
\begin{bmatrix}
b_x \\ b_v \\ b_y
\end{bmatrix}
\end{equation*}
```

See [Revay et al. (2021)](https://arxiv.org/abs/2104.05942) for more details on explicit parameterisations of REN.
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
(eg: a [`GeneralRENParams`](@ref) instance) and converts it to a
**callable** explicit parameterisation of the REN.

See also [`AbstractREN`](@ref), [`WrapREN`](@ref), and [`DiffREN`](@ref).
"""
function REN(ps::AbstractRENParams{T}) where T
    explicit = direct_to_explicit(ps)
    return REN(ps.nl, ps.nu, ps.nx, ps.nv, ps.ny, explicit, T)
end

"""
    abstract type AbstractREN end

Explicit parameterisation for recurrent equilibrium networks.

    (m::AbstractREN)(xt::VecOrMat, ut::VecOrMat)

Call an  `AbstractREN` model given internal states `xt` and inputs `ut`. 

If arguments are matrices, each column must be a vector of states or inputs (allows batch simulations).

# Examples

This example creates a contracting [`REN`](@ref) using [`ContractingRENParams`](@ref) and calls the model with some randomly generated inputs. 

```jldoctest; output = false
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
function (m::AbstractREN)(xt::VecOrMat, ut::VecOrMat)

    b = m.explicit.C1 * xt + m.explicit.D12 * ut .+ m.explicit.bv
    wt = tril_eq_layer(m.nl, m.explicit.D11, b)
    xt1 = m.explicit.A * xt + m.explicit.B1 * wt + m.explicit.B2 * ut .+ m.explicit.bx
    yt = m.explicit.C2 * xt + m.explicit.D21 * wt + m.explicit.D22 * ut .+ m.explicit.by

    return xt1, yt
end

"""
    init_states(m::AbstractREN, nbatches; rng=nothing)

Return matrix of (nbatches) state vectors of a REN initialised as zeros.
"""
function init_states(m::AbstractREN, nbatches; rng=nothing)
    return zeros(m.T, m.nx, nbatches)
end

function init_states(m::AbstractREN; rng=nothing)
    return zeros(m.T, m.nx)
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
end
