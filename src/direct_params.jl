abstract type DirectREN end

mutable struct DirectParams{T}
    ϕ                                   # nonlinearity
    polar_param::Bool                   # Whether or not to use polar param
    ρ::Union{Vector{T},CuVector{T}}     # used in polar param
    V::Union{Matrix{T},CuMatrix{T}}
    S_1::Union{Matrix{T},CuMatrix{T}}
    B2::Union{Matrix{T},CuMatrix{T}}
    D12::Union{Matrix{T},CuMatrix{T}}
    bx::Union{Vector{T},CuVector{T}}
    bv::Union{Vector{T},CuVector{T}}
    ϵ::T
end

Flux.trainable(L::DirectParams) = filter(p -> length(p) !=0, [L.ρ, L.V, L.S_1, L.B2, L.D12, L.bx, L.bv])