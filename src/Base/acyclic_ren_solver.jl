# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

"""
    solve_tril_layer(ϕ, W::Matrix, b::VecOrMat)

Solves z = ϕ.(W*z .+ b) for lower-triangular W, where
ϕ is a generic static nonlinearity (eg: relu, tanh).
"""
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

# TODO: Tidy up/speed up the code below. Needs docs too

"""
    tril_layer_calculate_gradient(Δz, ϕ, W, b, zeq; tol=1E-9)

Calculate gradients for solving lower-triangular equilibirum
network layer.
"""
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
