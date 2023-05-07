mutable struct LBDN{T} <: AbstractLBDN{T}
    nl::Function
    L ::Int                         # Number of hidden layers
    nu::Int
    ny::Int
    sqrt_γ::T
    explicit::ExplicitLBDNParams{T}
end

# Constructor
function LBDN(ps::AbstractLBDNParams{T}) where T
    sqrt_γ = T(sqrt(ps.γ))
    explicit = direct_to_explicit(ps)
    return LBDN{T}(ps.nl, length(ps.nh), ps.nu, ps.ny, sqrt_γ, explicit)
end

# Call the model
# TODO: Improve efficiency
function (m::AbstractLBDN)(u::AbstractVecOrMat{T}, explicit::ExplicitLBDNParams{T}) where T

    sqrt2 = T(√2)
    h = m.sqrt_γ * u

    for k in 1:m.L
        h = sqrt2 * explicit.A_T[k] .* explicit.Ψd[k] * m.nl.(
            sqrt2 ./explicit.Ψd[k] .* explicit.B[k] * h .+ explicit.b[k]
        )
    end

    return m.sqrt_γ * explicit.B[m.L+1] * h .+ explicit.b[m.L+1]
end

function (m::AbstractLBDN)(u::AbstractVecOrMat) 
    return m(u, m.explicit)
end
