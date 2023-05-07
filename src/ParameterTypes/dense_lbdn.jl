mutable struct DenseLBDNParams{T} <: AbstractLBDNParams{T}
    nl::Function                    # Sector-bounded nonlinearity
    nu::Int
    nh::Vector{Int}
    ny::Int
    γ::T
    direct::DirectLBDNParams{T}
end

# Constructor
function DenseLBDNParams{T}(
    nu::Int, nh::Vector{Int}, ny::Int, γ::Number = T(1);
    nl::Function = Flux.relu, 
    initW::Function = Flux.glorot_normal,
    initb::Function = Flux.glorot_normal,
    rng::AbstractRNG = Random.GLOBAL_RNG
) where T

    direct = DirectLBDNParams{T}(nu, nh, ny; initW=initW, initb=initb, rng=rng)
    return DenseLBDNParams{T}(nl, nu, nh, ny, T(γ), direct)

end

Flux.trainable(m::DenseLBDNParams) = Flux.trainable(m.direct)

function direct_to_explicit(ps::DenseLBDNParams{T}) where T

    b = get_b(ps.direct.b)
    Ψd = get_Ψ(ps.direct.d)
    A_T, B = get_AB(ps.direct.XY, ps.direct.α, (ps.nh..., ps.ny))

    # Faster to backpropagate with tuples than vectors
    L = length(ps.nh)
    return ExplicitLBDNParams{T,L+1,L}(tuple(A_T...), tuple(B...), tuple(Ψd...), tuple(b...))

end

# TODO: Improve speed
function cayley(XY, α, n)

    XY = α[1] .* XY ./ norm(XY)

    X = XY[1:n, :]
    Y = XY[(n+1):end, :]

    Z = X - X' + Y'*Y
    IZ = (I + Z)
    A_T = IZ \ (I - Z)
    B_T = -2Y / IZ

    return A_T, B_T

end

# Vector operations
# TODO: Can I speed these up? See: https://discourse.julialang.org/t/how-to-use-initialize-zygote-buffer/87653
function get_b(b::NTuple{N, AbstractVector{T}}) where {T, N}

    buf = Buffer([zeros(T,0)], N)
    for k in 1:N
        buf[k] = b[k]
    end
    return copy(buf)

end

function get_Ψ(d::NTuple{N, AbstractVector{T}}) where {T, N}

    buf = Buffer([zeros(T,0)], N)
    for k in 1:N
        buf[k] = exp.(d[k])
    end
    return copy(buf)

end

function get_AB(
    XY::NTuple{N, AbstractMatrix{T}}, 
    α ::NTuple{N, AbstractVector{T}},
    n ::NTuple{N, Int}
) where {T, N}

    buf_A = Buffer([zeros(T,0,0)], N)
    buf_B = Buffer([zeros(T,0,0)], N)
    for k in 1:N
        AB_k = cayley(XY[k], α[k], n[k])
        buf_A[k] = AB_k[1]
        buf_B[k] = AB_k[2]'
    end
    return copy(buf_A), copy(buf_B)
    
end


# TODO: Add GPU compatibility
# Flux.cpu() ...
# Flux.gpu() ...