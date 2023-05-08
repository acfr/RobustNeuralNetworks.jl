mutable struct LBDN{T} <: AbstractLBDN{T}
    nl::Function
    nu::Int
    nh::Vector{Int}
    ny::Int
    sqrt_γ::T
    explicit::ExplicitLBDNParams{T}
end

# Constructor
function LBDN(ps::AbstractLBDNParams{T}) where T
    sqrt_γ = T(sqrt(ps.γ))
    explicit = direct_to_explicit(ps)
    return LBDN{T}(ps.nl, ps.nu, ps.nh, ps.ny, sqrt_γ, explicit)
end

# Call the model
# Note: backpropagation is similarly fast with for loops as with Flux chains (tested)
function (m::AbstractLBDN)(u::AbstractVecOrMat{T}, explicit::ExplicitLBDNParams{T,N,M}) where {T,N,M}

    # Extract explicit params
    σ   = m.nl
    A_T = explicit.A_T
    B   = explicit.B
    Ψd  = explicit.Ψd
    b   = explicit.b

    sqrt2 = T(√2)
    sqrtγ = m.sqrt_γ

    # Evaluate LBDN (extracting Ψd[k] is faster for backprop)
    h = sqrtγ * u
    for k in 1:M
        Ψdk = Ψd[k]
        h = (sqrt2 * Ψdk) .* A_T[k] * σ.(sqrt2 ./ Ψdk .* (B[k] * h) .+ b[k])
    end
    return sqrtγ * B[N] * h .+ b[N]
end

function (m::AbstractLBDN)(u::AbstractVecOrMat) 
    return m(u, m.explicit)
end
