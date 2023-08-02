# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 
cd(@__DIR__)
using Pkg
Pkg.activate(".")

using BenchmarkTools
using BSON
using Flux
using Random
using RobustNeuralNetworks

rng = Xoshiro(42)


################################################################################
# New stuff

using LinearAlgebra
using ChainRulesCore: NoTangent, @non_differentiable, rrule
import ChainRulesCore: rrule

# Actually solve the layer
# TODO: Speed up with array mutation
function solve_tril_layer(ϕ::F, W::Matrix, b::VecOrMat) where F
    z_eq = similar(b)
    for i in axes(b,1)
        Wi = @view W[i:i, 1:i - 1]
        zi = @view z_eq[1:i-1,:]
        bi = @view b[i:i, :]
        z_eq[i:i,:] .= ϕ.(Wi * zi .+ bi)       
    end
    return z_eq
end
@non_differentiable solve_tril_layer(ϕ, W, b)

# To define the custom backwards pass
function tril_layer_back(σ::F, D11::Matrix, b::VecOrMat{T}, w_eq::VecOrMat{T}) where {F,T}
    return w_eq
end

# Main function to call
function tril_eq_layer(σ::F, D11::Matrix, b::VecOrMat) where F

    # Solve the equilibirum layer
    w_eq = solve_tril_layer(σ, D11, b)

    # Run the equation for auto-diff to get grads: ∂σ/∂(.) * ∂(D₁₁w + b)/∂(.)
    # By definition, w_eq1 = w_eq so this doesn't change the forward pass.
    w_eq1 = σ.(D11 * w_eq + b)
    return tril_layer_back(σ, D11, b, w_eq1)
end

# The backwards pass
function rrule(::typeof(tril_layer_back), 
               σ::F, D11::Matrix, b::VecOrMat{T}, w_eq::VecOrMat{T}) where {F,T}

    # Forwards pass
    y = tril_layer_back(σ, D11, b, w_eq)

    # Write pullback
    function tril_layer_back_pullback(ȳ)

        # Only w_eq actually gets used
        f̄ = NoTangent()
        σ̄ = NoTangent()
        D̄11 = NoTangent()
        b̄ = NoTangent()

        # Get gradient of σ(v) wrt v evaluated at v = D₁₁w + b
        # TODO: Pass in v, or do the w_eq1 pullback myself?
        v = D11 * w_eq + b
        j = similar(b)
        for i in eachindex(j)
            j[i] = rrule(σ, v[i])[2](one(T))[2]
        end

        # Compute gradient from implicit function theorem
        eval_grad(t) = (I - (j[:, t] .* D11))' \ ȳ[:, t]
        w̄_eq = reduce(hcat, eval_grad(t) for t in 1:size(b, 2))
        return f̄, σ̄, D̄11, b̄, w̄_eq
    end
    return y, tril_layer_back_pullback
end


################################################################################

function (m::AbstractREN{T})(
    xt::AbstractVecOrMat, 
    ut::AbstractVecOrMat,
    explicit::ExplicitRENParams{T}
) where T

    bv = (m.nv == 0) ? 0 : explicit.bv
    bx = (m.nx == 0) ? 0 : explicit.bx

    b = explicit.C1 * xt + explicit.D12 * ut .+ bv
    wt = tril_eq_layer(m.nl, explicit.D11, b)
    xt1 = explicit.A * xt + explicit.B1 * wt + explicit.B2 * ut .+ bx
    yt = explicit.C2 * xt + explicit.D21 * wt + explicit.D22 * ut .+ explicit.by

    return xt1, yt
end


################################################################################

T = Float32
batches = 10
nu, nx, nv, ny = 4, 5, 10, 2
model_ps = ContractingRENParams{T}(nu, nx, nv, ny; rng)

# Dummy data
us = randn(rng, T, nu, batches)
ys = randn(rng, T, ny, batches)

# Dummy loss function just for testing
function loss(model_ps, u, y)
    m = REN(model_ps)
    y0 = deepcopy(y)
    x = init_states(m, size(u,2))
    for _ in 1:5
        x, y = m(x, u)
    end
    return Flux.mse(y, y0) + sum(x.^2)
end

# Load the test data
data = BSON.load("testdata_relu.bson")
l1, g1 = data["loss"], data["grads"]

# Run it forwards
l2 = loss(model_ps, us, ys)
g2 = Flux.gradient(loss, model_ps, us, ys)

println("Losses match? ", l1 == l2)
println("Grads match?  ", g1 == g2)

# Time the forwards and backwards passes
# @btime loss(model_ps, us, y?s)
@btime Flux.gradient(loss, model_ps, us, ys)
println()
