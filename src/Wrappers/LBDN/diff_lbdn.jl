mutable struct DiffLBDN{T} <: AbstractLBDN{T}
    nl::Function
    nu::Int
    nh::Vector{Int}
    ny::Int
    sqrt_γ::T
    params::AbstractLBDNParams{T}
end

# Constructor
function DiffLBDN(ps::AbstractLBDNParams{T}) where T
    sqrt_γ = T(sqrt(ps.γ))
    return DiffLBDN{T}(ps.nl, ps.nu, ps.nh, ps.ny, sqrt_γ, ps)
end

# Call the model
function (m::DiffLBDN)(u::AbstractVecOrMat)
    explicit = direct_to_explicit(m.params)
    return explicit(u)
end

Flux.trainable(m::DiffLBDN) = Flux.trainable(m.params)
