using Flux
using Zygote
using Zygote:@adjoint
# using NLsolve
using Distributions
using LinearAlgebra
using FiniteDifferences
# using Printf
using BenchmarkTools
using Revise
# using IterativeSolvers

# includet("solvers.jl")
##
"""
We need to include a custom backwards pass for the fixed point layer. 
We can do this by decompose the fixed point layer into two functions, 
eq and fp, with the adjoints defined as follows:

        x |>        eq        |>         fp            |>  ...
Forward: x |>  zeq = f(zeq, x) |>    zeq -> zeq         |>  loss(zeq)
Back:   Δx <|     Δx -> Δx     <| Δx = Δz'(I-df/dz)^-1  <|  Δz= dl/dz*

"""
    

# ## Define equilibrium layer
# function eq_solve(solver::Solver, f, x)
#     feq(z) = f(z, x)
#     z0 = typeof(x)(zeros(size(x)))
#     zeq = solver(feq, z0)
#     return zeq
# end

# function eq_backward(Δf, solver::Solver, f, x)
#     return nothing, nothing, nothing
# end

# ## Define fixed point layer
# function fp_forward(solver, f, zeq, x)
#     return zeq
# end

# function fp_backward(Δz, solver::Solver, f, zeq, x)

#     # f(z, x, W) = W * (nl.(z)) + x
#     jvp_z(y) = Zygote.pullback(z -> f(z, x), zeq)[2](y)[1]
    
#     # Solve for Δz'(I - ∂f/∂z)^{-1}
#     func(g) = jvp_z(g) + Δz
    
#     z0 = typeof(x)(zeros(size(x))) 
#     g = solver(func, z0)
    
#     return nothing, nothing, g, nothing
# end

# @adjoint eq_solve(solver, f, x) = eq_solve(solver, f, x), Δf -> eq_backward(Δf, solver::Solver, f, x)
# @adjoint fp_forward(solver::Solver, f, zeq, x) = fp_forward(solver, f, zeq, x), Δz -> fp_backward(Δz, solver, f, zeq, x)

#  # combine the two above layers to make a fixed point layer
# function fixed_point_layer(solver::Solver, f::Function, x)
#     z1 = eq_solve(solver, f, x)
#     z2 = f(z1, x)  # run forward pass and track gradients
#     z3 = fp_forward(solver, f, z2, x)
#     return z3
# end


# # Operator Splitting Layer
# function eq_solve(solver::OperatorSplitting, ϕ, x, V)

#     # cg is not batched unfortunately
#     # RB(u) = solver.cg ? bicgstabl(V, (u + solver.α * x)) : V \ (u + solver.α * x)

#     RB(u) = V \ (u + solver.α * x)

#     RA(z) = ϕ.(z)
#     z0 = typeof(x)(zeros(size(x)))
#     zeq = solver(RA, RB, z0)
#     return zeq
# end
# function eq_backward(Δf, solver::OperatorSplitting, ϕ, x, V)
#     return nothing, nothing, nothing, nothing
# end

# function fp_forward(solver::OperatorSplitting, ϕ, W, zeq, x)
#     return zeq
# end

# function fp_backward(Δz, solver::OperatorSplitting, ϕ, W, zeq, x)

#     one_vec = typeof(x)(ones(size(x)))
#     v = W * zeq + x
#     j = Zygote.pullback(z -> ϕ.(z), v)[2](one_vec)[1]

#     eval_grad(t) = (I - (j[:, t] .* W))' \ Δz[:, t]
#     gn = reduce(hcat, eval_grad(t) for t in 1:size(x, 2))

#     return nothing, nothing, nothing, gn, nothing
# end



# @adjoint eq_solve(solver::OperatorSplitting, ϕ, x, V) = eq_solve(solver, ϕ, x, V), Δf -> eq_backward(Δf, solver::OperatorSplitting, ϕ, x, V)
# @adjoint fp_forward(solver::OperatorSplitting, ϕ, W, zeq, x) = fp_forward(solver::OperatorSplitting, ϕ, W, zeq, x), Δz -> fp_backward(Δz, solver::OperatorSplitting, ϕ, W, zeq, x) 

# function fixed_point_layer(solver::OperatorSplitting, ϕ::Function, W, x, V)
#     z1 = eq_solve(solver, ϕ, x, V) # Solves for z = ϕ(Wz + x), records no grad
#     z2 = ϕ.(W * z1 + x)   # run forward pass and track gradients w.r.t D11 and b
#     z3 = fp_forward(solver, ϕ, W, z2, x)
#     return z3 
# end





function solve_tril_layer(ϕ, W, b)
    function eval_row(z_last, i)
        zi = ϕ.(W[i:i, 1:i - 1] * z_last + b[i:i, :])
        return vcat(z_last, zi)
    end
    z0 = typeof(b)(zeros(0, size(b, 2)))
    return reduce(eval_row, 1:size(b, 1), init=z0)
end


function tril_layer_calculate_gradient(Δz, ϕ, W, b, zeq; tol=1E-9)
    one_vec = typeof(b)(ones(size(b)))
    v = W * zeq + b
    j = Zygote.pullback(z -> ϕ.(z), v)[2](one_vec)[1]
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
    # return weq
    weq1 = ϕ.(W * weq + b)  # Run forward and track grads
    return tril_layer_backward(ϕ, W, b, weq1)
end
