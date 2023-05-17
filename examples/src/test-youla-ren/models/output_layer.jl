using LinearAlgebra
using Flux
import Flux.gpu, Flux.cpu

# Output layer
mutable struct output{T}
    C2::AbstractMatrix{T}
    D21::AbstractMatrix{T}
    D22::AbstractMatrix{T}
    by::AbstractVector{T}
end

function output{T}(nu::Int, nx::Int, nv::Int, ny::Int) where T
    C2 = randn(T, ny, nx) / convert(T, sqrt(ny + nx))
    D21 = randn(T, ny, nv) / convert(T, sqrt(ny + nv))
    D22 = zeros(T, ny, nu)
    by = randn(T, ny) / convert(T, sqrt(ny))
    return output{T}(C2, D21, D22, by)
end

(layer::output)(x, w, u) = layer.C2 * x + layer.D21 * w + layer.D22 * u .+ layer.by
Flux.trainable(layer::output) = (layer.C2, layer.D21, layer.by)

function Flux.gpu(M::output{T}) where T
    if T != Float32
        println("Moving type: ", T, " to gpu may not be supported. Try Float32!")
    end
    return output{T}(gpu(M.C2), gpu(M.D21), gpu(M.D22), gpu(M.by))
end

function Flux.cpu(M::output{T}) where T
    return output{T}(cpu(M.C2), cpu(M.D21), cpu(M.D22), cpu(M.by))
end
