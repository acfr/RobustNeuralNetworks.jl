mutable struct DiffLBDN <: AbstractLBDN
    nl::Function
    L ::Int                         # Number of hidden layers
    nu::Int
    ny::Int
    sqrt_γ::Number                       # TODO: More explicit type setting here
    params::AbstractLBDNParams
    T::DataType
end

# Constructor
function DiffLBDN(ps::AbstractLBDNParams{T}) where T
    sqrt_γ = T(sqrt(ps.γ))
    return DiffLBDN(ps.nl, length(ps.nh), ps.nu, ps.ny, sqrt_γ, ps, T)
end

# Call the model
# TODO: See if you can re-use code from previous wrapper lbdn.jl
function (m::DiffLBDN)(u::VecOrMat)

    explicit = direct_to_explicit(m.params)

    r2 = m.T(√2)
    h = m.sqrt_γ * u

    for k in 1:m.L
        h = r2 * explicit.A[k] .* explicit.Ψ[k] * m.nl.(
            r2 ./explicit.Ψ[k] .* explicit.B[k] * h + explicit.b[k]
        )
    end

    return m.sqrt_γ * explicit.B[m.L+1] * h + explicit.b[m.L+1]
end