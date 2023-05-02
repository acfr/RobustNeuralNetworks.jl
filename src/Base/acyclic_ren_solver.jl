"""
    solve_tril_layer(ϕ, W::Matrix, b::VecOrMat)

Solves z = ϕ.(W*z .+ b) for lower-triangular W, where
ϕ is a either a ReLU or tanh activation function.
"""
function solve_tril_layer(ϕ::Union{typeof(Flux.relu), typeof(Flux.tanh)}, W::Matrix, b::VecOrMat)
    z_eq = similar(b)
    Wi_zi=similar(z_eq,1,size(b,2))
    for i::Int64 in (1:size(b,1))
        Wi = @view W[i:i, 1:i - 1]
        zi = @view z_eq[1:i-1,:]
        bi = @view b[i:i, :]
        LinearAlgebra.mul!(Wi_zi,Wi,zi)
        @inbounds z_eq[i:i,:] .= ϕ.(Wi_zi .+ bi)       
    end
    return z_eq
end

"""
    solve_tril_layer(ϕ, W::Matrix, b::VecOrMat)

Solves z = ϕ.(W*z .+ b) for lower-triangular W, where
ϕ is a generic static nonlinearity.
"""
function solve_tril_layer(ϕ, W::Matrix, b::VecOrMat)

    # Slower to not specify typeof(ϕ), which is why this is separate
    println("Using non-ReLU/tanh version of solve_tril_layer()")
    z_eq = similar(b)
    Wi_zi=similar(z_eq,1,size(b,2))
    for i::Int64 in (1:size(b,1))
        Wi = @view W[i:i, 1:i - 1]
        zi = @view z_eq[1:i-1,:]
        bi = @view b[i:i, :]
        LinearAlgebra.mul!(Wi_zi,Wi,zi)
        @inbounds z_eq[i:i,:] .= ϕ.(Wi_zi .+ bi)       
    end
    return z_eq
end

# TODO: Can also speed things up by specifying function argument types
# TODO: This code needs documentation and tidying up. Has not been touched since Max's thesis

"""
    tril_layer_calculate_gradient(Δz, ϕ, W, b, zeq; tol=1E-9)

Calculate gradients for solving lower-triangular equilibirum
network layer.
"""
function tril_layer_calculate_gradient(Δz, ϕ, W, b, zeq; tol=1E-9)
    one_vec = typeof(b)(ones(size(b)))
    v = W * zeq + b
    j = pullback(z -> ϕ.(z), v)[2](one_vec)[1]
    # J = Diagonal(j[:])

    eval_grad(t) = (I - (j[:, t] .* W))' \ Δz[:, t]
    gn = reduce(hcat, eval_grad(t) for t in 1:size(b, 2))

    return nothing, nothing, nothing, gn
end
tril_layer_backward(ϕ, W, b, zeq) = zeq

@adjoint solve_tril_layer(ϕ, W, b) = solve_tril_layer(ϕ, W, b), Δz -> (nothing, nothing, nothing)
@adjoint tril_layer_backward(ϕ, W, b, zeq) = tril_layer_backward(ϕ, W, b, zeq), Δz -> tril_layer_calculate_gradient(Δz, ϕ, W, b, zeq)

function tril_eq_layer(ϕ, W, b)
    weq = solve_tril_layer(ϕ, W, b)
    # TODO: return weq if not differentiating anything
    weq1 = ϕ.(W * weq + b)  # Run forward and track grads
    return tril_layer_backward(ϕ, W, b, weq1)
end
