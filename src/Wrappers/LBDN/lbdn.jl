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
# function (m::AbstractLBDN)(u::AbstractVecOrMat{T}, explicit::ExplicitLBDNParams) where T

#     r2 = m.T(√2)
#     h = m.sqrt_γ * u

#     for k in 1:m.L
#         h = r2 * explicit.A_T[k] .* explicit.Ψd[k] * m.nl.(
#             r2 ./explicit.Ψd[k] .* explicit.B[k] * h .+ explicit.b[k]
#         )
#     end

#     return m.sqrt_γ * explicit.B[m.L+1] * h .+ explicit.b[m.L+1]
# end

function (m::AbstractLBDN)(u::AbstractVecOrMat) 
    return m(u, m.explicit)
end





function (m::AbstractLBDN)(u::AbstractVecOrMat{T}, explicit::ExplicitLBDNParams) where T

    ps = m.params
    L = length(ps.nh)
    ns = [ps.nh..., ps.ny]

    r2 = m.T(√2)
    h = m.sqrt_γ * u

    # Ψd = fill(zeros(T,0), L)
    # for k in 1:L
    #     Ψd = setindex(Ψd, exp.(ps.direct.d[k]), k)
    # end

    # TODO: THIS ACTUALLY WORKED MAYBE....?
    buf = Buffer([zeros(T,0)], L)
    for k in 1:L
        buf[k] = exp.(ps.direct.d[k])
    end
    Ψd = copy(buf)

    for k in 1:L

        XY_k = ps.direct.XY[k]
        α_k = ps.direct.α[k]
        n_k = ns[k]

        A_k, B_k = cayley(XY_k, α_k, n_k)
        B_k = B_k'

        # Ψd_k = exp.(ps.direct.d[k])
        Ψd_k = Ψd[k]

        b_k = ps.direct.b[k]

        h = r2 * A_k .* Ψd_k * m.nl.(
            r2 ./Ψd_k .* B_k * h .+ b_k
        )
    end

    XY_L1 = ps.direct.XY[L+1]
    α_L1 = ps.direct.α[L+1]
    n_L1 = ns[L+1]

    _, B_L1 = cayley(XY_L1, α_L1, n_L1)
    B_L1 = B_L1'    
    b_L1 = ps.direct.b[L+1]

    return m.sqrt_γ * B_L1 * h .+ b_L1
end



# """
#     setindex(A,X,inds...)
# `setindex` with copy. An alternative for `setindex!`
# that allows automatic differentation with Zygote.
# Example:
# ```
# A = setindex(A,X,1:2,1:2)
# ```
# Replaces `A[1:2,1:2] = X`
# """
# setindex(A,X,inds...) = setindex!(copy(A),X,inds...)

# ## Adjoint
# @adjoint setindex(A,X,inds...) = begin
#   B = setindex(A,X,inds...)
#   adj = function(Δ)
#     bA = copy(Δ)
#     bA[inds...] .*= 0 #zero(eltype(A))
#     bX = similar(X)
#     bX[:] = Δ[inds...]
#     binds = fill(nothing,length(inds))
#     return bA,bX,binds...
#   end
#   B,adj
# end

# import Base.zero
# zero(x::AbstractVecOrMat{T}) where T<:AbstractVecOrMat = zero.(x)
# zero(x::Tuple) = zero.(x)
