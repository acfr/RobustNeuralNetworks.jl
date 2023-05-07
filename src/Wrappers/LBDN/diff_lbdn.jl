mutable struct DiffLBDN{T} <: AbstractLBDN{T}
    nl::Function
    L ::Int                         # Number of hidden layers
    nu::Int
    ny::Int
    sqrt_γ::T                  # TODO: More explicit type setting here
    params::AbstractLBDNParams{T}
end

# Constructor
function DiffLBDN(ps::AbstractLBDNParams{T}) where T
    sqrt_γ = T(sqrt(ps.γ))
    return DiffLBDN{T}(ps.nl, length(ps.nh), ps.nu, ps.ny, sqrt_γ, ps)
end

# Call the model
function (m::DiffLBDN)(u::AbstractVecOrMat)
    explicit = direct_to_explicit(m.params)
    return m(u, explicit)
end

Flux.trainable(m::DiffLBDN) = Flux.trainable(m.params)