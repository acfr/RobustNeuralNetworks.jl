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
function (m::AbstractLBDN)(u::AbstractVecOrMat{T}, explicit::ExplicitLBDNParams{T,N,M}) where {T,N,M}

    sqrt2 = T(√2)
    h = m.sqrt_γ * u

    for k in 1:M
        h = sqrt2 * explicit.A_T[k] .* explicit.Ψd[k] * m.nl.(
            sqrt2 ./explicit.Ψd[k] .* (explicit.B[k] * h) .+ explicit.b[k]
        )
    end

    return m.sqrt_γ * explicit.B[N] * h .+ explicit.b[N]
end

function (m::AbstractLBDN)(u::AbstractVecOrMat) 
    return m(u, m.explicit)
end
