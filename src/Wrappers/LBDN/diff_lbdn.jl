mutable struct DiffLBDN <: AbstractLBDN
    nl::Function
    L ::Int                         # Number of hidden layers
    nu::Int
    ny::Int
    sqrt_γ::Number                  # TODO: More explicit type setting here
    params::AbstractLBDNParams
    T::DataType
end

# Constructor
function DiffLBDN(ps::AbstractLBDNParams{T}) where T
    sqrt_γ = T(sqrt(ps.γ))
    return DiffLBDN(ps.nl, length(ps.nh), ps.nu, ps.ny, sqrt_γ, ps, T)
end

# Call the model
function (m::DiffLBDN)(u::AbstractVecOrMat)
    explicit = direct_to_explicit(m.params)
    return m(u, explicit)
end

Flux.trainable(m::DiffLBDN) = Flux.trainable(m.params)