mutable struct LBDN <: AbstractLBDN
    nl::Function
    L ::Int                         # Number of hidden layers
    nu::Int
    ny::Int
    sqrt_γ::Number                       # TODO: More explicit type setting here
    explicit::ExplicitLBDNParams
    T::DataType
end

# Constructor
function LBDN(ps::AbstractLBDNParams{T}) where T
    sqrt_γ = T(sqrt(ps.γ))
    explicit = direct_to_explicit(ps)
    return LBDN(ps.nl, length(ps.nh), ps.nu, ps.ny, sqrt_γ, explicit, T)
end

# Call the model
# TODO: Improve efficiency
function (m::AbstractLBDN)(u::Union{T,AbstractVecOrMat{T}}) where T

    r2 = m.T(√2)
    h = m.sqrt_γ * u

    for k in 1:m.L
        h = r2 * m.explicit.A[k] .* m.explicit.Ψ[k] * m.nl.(
            r2 ./m.explicit.Ψ[k] .* m.explicit.B[k] * h + m.explicit.b[k]
        )
    end

    return m.sqrt_γ * m.explicit.B[m.L+1] * h + m.explicit.b[m.L+1]
end

# TODO: Can I just use this function for forward pass so I have the same code across the different implementations? Would be nice.
# function (m::AbstractLBDN)(u::VecOrMat, explicit::ExplicitLBDNParams)
#     r2 = m.T(√2)
#     h = m.sqrt_γ * u

#     for k in 1:m.L
#         h = r2 * explicit.A[k] .* explicit.Ψ[k] * m.nl.(
#             r2 ./explicit.Ψ[k] .* explicit.B[k] * h + explicit.b[k]
#         )
#     end

#     return m.sqrt_γ * explicit.B[m.L+1] * h + explicit.b[m.L+1]
# end
