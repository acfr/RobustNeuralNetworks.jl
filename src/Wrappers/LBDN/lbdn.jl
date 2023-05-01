mutable struct LBDN <: AbstractLBDN
    nl::Function
    nu::Int
    L ::Int
    ny::Int
    sqrt_γ::Number                       # TODO: More explicit type setting here
    explicit::ExplicitLBDNParams
    T::DataType
end

# Constructor
function LBDN(ps::AbstractLBDNParams{T}) where T
    sqrt_γ = T(sqrt(ps.γ))
    explicit = direct_to_explicit(ps)
    return LBDN(ps.nl, ps.nu, length(ps.nh), ps.ny, sqrt_γ, explicit, T)
end

# Call the model
# TODO: Improve efficiency
function (m::AbstractLBDN)(u::VecOrMat)

    r2 = m.T(√2)
    h = m.sqrt_γ * u

    for k in 1:m.L
        h = r2 * m.explicit.A[k] .* m.explicit.Ψ[k] * m.nl.(
            r2 ./m.explicit.Ψ[k] .* m.explicit.B[k] * h + m.explicit.b[k]
        )
    end

    return m.sqrt_γ * m.explicit.B[m.L+1] * h + m.explicit.b[m.L+1]
end