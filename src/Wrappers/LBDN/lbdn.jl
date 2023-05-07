mutable struct LBDN <: AbstractLBDN
    nl::Function
    L ::Int                         # Number of hidden layers
    nu::Int
    ny::Int
    sqrt_γ::Number                  # TODO: More explicit type setting here
    explicit::ExplicitLBDNParams
end

# Constructor
function LBDN(ps::AbstractLBDNParams{T}) where T
    sqrt_γ = T(sqrt(ps.γ))
    explicit = direct_to_explicit(ps)
    return LBDN(ps.nl, length(ps.nh), ps.nu, ps.ny, sqrt_γ, explicit)
end

# Call the model
# TODO: Improve efficiency
function (m::AbstractLBDN)(u::AbstractVecOrMat{T}, explicit::ExplicitLBDNParams) where T

    r2 = m.T(√2)
    h = m.sqrt_γ * u

    for k in 1:m.L
        h = r2 * explicit.A_T[k] .* explicit.Ψd[k] * m.nl.(
            r2 ./explicit.Ψd[k] .* explicit.B[k] * h .+ explicit.b[k]
        )
    end

    return m.sqrt_γ * explicit.B[m.L+1] * h .+ explicit.b[m.L+1]
end

function (m::AbstractLBDN)(u::AbstractVecOrMat) 
    return m(u, m.explicit)
end
