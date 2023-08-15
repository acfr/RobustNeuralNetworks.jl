# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

mutable struct LBDN{T, L} <: AbstractLBDN{T, L}
    nl::Function
    nu::Int
    nh::NTuple{L, Int}
    ny::Int
    explicit::ExplicitLBDNParams{T}
end

"""
    LBDN(ps::AbstractLBDNParams)

Construct an LBDN from its direct parameterisation.

This constructor takes a direct parameterisation of LBDN (eg: a [`DenseLBDNParams`](@ref) instance) and converts it to a **callable** explicit parameterisation of the LBDN. An example can be found in the docs for [`AbstractLBDN`](@ref).

See also [`AbstractLBDN`](@ref), [`DiffLBDN`](@ref).
"""
function LBDN(ps::AbstractLBDNParams{T, L}) where {T, L}
    explicit = direct_to_explicit(ps)
    return LBDN{T, L}(ps.nl, ps.nu, ps.nh, ps.ny, explicit)
end

# No trainable params
@functor LBDN
trainable(m::LBDN) = (; )

"""
    abstract type AbstractLBDN{T, L} end

Explicit parameterisation for Lipschitz-bounded deep networks.

    (m::AbstractLBDN)(u::AbstractVecOrMat)

Call and `AbstractLBDN` model given inputs `u`.

If arguments are matrices, each column must be a vector of inputs (allows batch simulations).

# Examples

This example creates a dense [`LBDN`](@ref) using [`DenseLBDNParams`](@ref) and calls the model with some randomly generated inputs.

```jldoctest
using Random
using RobustNeuralNetworks

# Setup
rng = Xoshiro(42)
batches = 10
γ = 20.0

# Model with 4 inputs, 1 ouput, 4 hidden layers
nu, ny = 4, 1
nh = [5, 10, 5, 15]

lbdn_ps = DenseLBDNParams{Float64}(nu, nh, ny, γ; rng)
lbdn = LBDN(lbdn_ps)

# Evaluate model with a batch of random inputs
u = 10*randn(rng, nu, batches)
y = lbdn(u)

println(round.(y; digits=2))

# output

[-1.11 -1.01 -0.07 -2.25 -4.22 -1.76 -3.82 -1.13 -11.85 -3.01]
```
"""
function (m::AbstractLBDN)(u::AbstractVecOrMat)
    return m(u, m.explicit)
end

function (m::AbstractLBDN{T})(u::AbstractVecOrMat, explicit::ExplicitLBDNParams{T,N,M}) where {T,N,M}

    # Extract explicit params
    σ   = m.nl
    A_T = explicit.A_T
    B   = explicit.B
    Ψd  = explicit.Ψd
    b   = explicit.b

    sqrt2 = T(√2)
    sqrtγ = explicit.sqrtγ

    # Evaluate LBDN (extracting Ψd[k] is faster for backprop)
    # Note: backpropagation is similarly fast with for loops as with Flux chains (tested)
    h = sqrtγ .* u
    for k in 1:M
        Ψdk = Ψd[k]
        h = sqrt2 * (A_T[k] .* Ψdk') * σ.(sqrt2 * (B[k] ./ Ψdk) * h .+ b[k])
    end
    return sqrtγ .* B[N] * h .+ b[N]
end

function set_output_zero!(m::AbstractLBDN)
    m.explicit.B[end] .= 0
    m.explicit.b[end] .= 0

    return nothing
end
