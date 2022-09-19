using CUDA
using Flux
import Flux.gpu, Flux.cpu
using LinearAlgebra

# Linear output layer of the REN
mutable struct Output{T}
    C2::Union{Matrix{T},CuMatrix{T}}
    D21::Union{Matrix{T},CuMatrix{T}}
    D22::Union{Matrix{T},CuMatrix{T}}
    by::Union{Vector{T},CuVector{T}}
end

# Constructor
function Output{T}(nu::Int, nx::Int, nv::Int, ny::Int) where T
    C2 = randn(T, ny, nx) / convert(T, sqrt(ny + nx))
    D21 = randn(T, ny, nv) / convert(T, sqrt(ny + nv))
    D22 = zeros(T, ny, nu)
    by = randn(T, ny) / convert(T, sqrt(ny))
    return Output{T}(C2, D21, D22, by)
end

# Trainable params
Flux.trainable(layer::Output) = (layer.C2, layer.D21, layer.D22, layer.by)

# Call output layer
(layer::Output)(x, w, u) = layer.C2 * x + layer.D21 * w + layer.D22 * u .+ layer.by

# GPU/CPU compatibility
function Flux.gpu(M::Output{T}) where T
    if T != Float32
        println("Moving type: ", T, " to gpu may not be supported. Try Float32!")
    end
    return Output{T}(gpu(M.C2), gpu(M.D21), gpu(M.D22), gpu(M.by))
end

function Flux.cpu(M::Output{T}) where T
    return Output{T}(cpu(M.C2), cpu(M.D21), cpu(M.D22), cpu(M.by))
end
