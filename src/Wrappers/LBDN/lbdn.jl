mutable struct LBDN{T} <: AbstractLBDN{T}
    nl::Function
    nu::Int
    nh::Vector{Int}
    ny::Int
    sqrt_γ::T
    explicit::ExplicitLBDNParams{T}
end

"""
    LBDN(ps::AbstractLBDNParams{T}) where T

Construct an LBDN from its direct parameterisation.

This constructor takes a direct parameterisation of LBDN (eg: a [`DenseLBDNParams`](@ref) instance) and converts it to a **callable** explicit parameterisation of the LBDN. An example can be found in the docs for [`AbstractLBDN`](@ref).

See also [`AbstractLBDN`](@ref), [`DiffLBDN`](@ref).
"""
function LBDN(ps::AbstractLBDNParams{T}) where T
    sqrt_γ = sqrt(ps.γ)
    explicit = direct_to_explicit(ps)
    return LBDN{T}(ps.nl, ps.nu, ps.nh, ps.ny, sqrt_γ, explicit)
end

"""
    abstract type AbstractLBDN{T} end

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
rng = MersenneTwister(42)
batches = 10
γ = 20.0

# Model with 4 inputs, 1 ouput, 4 hidden layers
nu, ny = 4, 1
nh = [5, 10, 5, 15]

lbdn_ps = DenseLBDNParams{Float64}(nu, nh, ny, γ; rng=rng)
lbdn = LBDN(lbdn_ps)

# Evaluate model with a batch of random inputs
u = 10*randn(rng, nu, batches)
y = lbdn(u)

println(round.(y; digits=2))

# output

[-0.69 -1.89 -9.68 3.47 -11.65 -4.48 -4.53 3.61 1.37 -0.68]
```
"""
function (m::AbstractLBDN)(u::AbstractVecOrMat)
    return m(u, m.explicit)
end

function (m::AbstractLBDN{T})(u::AbstractVecOrMat{T}, explicit::ExplicitLBDNParams{T,N,M}) where {T,N,M}

    # Extract explicit params
    σ   = m.nl
    A_T = explicit.A_T
    B   = explicit.B
    Ψd  = explicit.Ψd
    b   = explicit.b

    sqrt2 = T(√2)
    sqrtγ = m.sqrt_γ

    # Evaluate LBDN (extracting Ψd[k] is faster for backprop)
    # Note: backpropagation is similarly fast with for loops as with Flux chains (tested)
    h = sqrtγ * u
    for k in 1:M
        Ψdk = Ψd[k]
        h = sqrt2 * (A_T[k] .* Ψdk') * σ.(sqrt2 * (B[k] ./ Ψdk) * h .+ b[k])
    end
    return sqrtγ * B[N] * h .+ b[N]
end

"""
    set_output_zero!(m::AbstractLBDN)

Set output map of an LBDN to zero.

If the resulting model is called with `y = lbdn(u)` then `y = 0` for any `u`.
"""
function set_output_zero!(m::AbstractLBDN)
    m.explicit.B[end] .*= 0
    m.explicit.b[end] .*= 0

    return nothing
end
