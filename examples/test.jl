# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

using BenchmarkTools
using Flux
using Random
using RobustNeuralNetworks

rng = Xoshiro(42)


################################################################################

using LinearAlgebra
using Zygote: @adjoint, pullback

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

function tril_layer_calculate_gradient(Δz, ϕ, W, b, zeq; tol=1E-9)
    one_vec = typeof(b)(ones(size(b)))
    v = W * zeq + b
    j = pullback(z -> ϕ.(z), v)[2](one_vec)[1]

    eval_grad(t) = (I - (j[:, t] .* W))' \ Δz[:, t]
    gn = reduce(hcat, eval_grad(t) for t in 1:size(b, 2))

    return nothing, nothing, nothing, gn
end
tril_layer_backward(ϕ, W, b, zeq) = zeq

@adjoint solve_tril_layer(ϕ, W, b) = solve_tril_layer(ϕ, W, b), Δz -> (nothing, nothing, nothing)
@adjoint tril_layer_backward(ϕ, W, b, zeq) = tril_layer_backward(ϕ, W, b, zeq), Δz -> tril_layer_calculate_gradient(Δz, ϕ, W, b, zeq)

function tril_eq_layer(ϕ::F, W::Matrix, b::VecOrMat) where F
    weq  = solve_tril_layer(ϕ, W, b)
    weq1 = ϕ.(W * weq + b)  # Run forward and track grads
    return tril_layer_backward(ϕ, W, b, weq1)
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
data = BSON.load("testdata.bson")
l1, g1 = data["loss"], data["grads"]

# Run it forwards
l2 = loss(model_ps, us, ys)
g2 = Flux.gradient(loss, model_ps, us, ys)

println("Losses match? ", l1 == l2)
println("Grads match?  ", g1 == g2)

# Time the forwards and backwards passes
# @btime loss(model_ps, us, ys)
# @btime Flux.gradient(loss, model_ps, us, ys)
println()
