mutable struct LBDN{T} <: AbstractLBDN{T}
    nl::Function
    nu::Int
    nh::Vector{Int}
    ny::Int
    sqrt_γ::T
    explicit::Chain
end

# Constructor
function LBDN(ps::AbstractLBDNParams{T}) where T
    sqrt_γ = T(sqrt(ps.γ))
    explicit = direct_to_explicit(ps)
    return LBDN{T}(ps.nl, ps.nu, ps.nh, ps.ny, sqrt_γ, explicit)
end

# Call the model
function (m::AbstractLBDN)(u::AbstractVecOrMat) 
    return m.explicit(u)
end
