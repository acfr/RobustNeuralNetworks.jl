using NLsolve
using Distributions
using LinearAlgebra
using Printf
using BenchmarkTools

# Need to figure out how to a type system for solvers.
# E.g. Potential solvers inlcude forward solvers, splitting methods and ode solvers
abstract type Solver end
abstract type ForwardSolver <: Solver end
abstract type OperatorSplitting <: Solver end


# Forward Iteration
struct forward_iteration <: ForwardSolver
    tol
    maxIter
    verbose
    forward_iteration(;tol=1E-4, maxIter=1000, verbose=false) = new(tol, maxIter, verbose)
end
function (solver::forward_iteration)(f, z0)
    zp, z = z0, f(z0)
    for iter in range(1, stop=solver.maxIter)
        zp, z = z, f(z)
        err = norm(z - zp)
        if err < solver.tol
            if solver.verbose
                @printf("forward solve: \tIters = %d\terror=%1.2E\n", iter, err)
            end
            return z
        end
    end
    
    @printf("Solver did not converge\tError = %1.2E\n", norm(z - zp))
    return z
end

# Newton method
struct newton <: ForwardSolver
    tol
    maxIter
    verbose
    autodiff
    newton(;tol=1E-4, maxIter=1000, verbose=false, autodiff=:forward) = new(tol, maxIter, verbose, autodiff)
end
function (solver::newton)(f::Function, z0)
    function f!(F, z)
        F .= f(z) - z
    end
    res = nlsolve(f!, z0; ftol=solver.tol, xtol=solver.tol, method=:newton, autodiff=solver.autodiff)
    if solver.verbose
        println(res)
    end
    return res.zero
end

# Solver using Anderson Acceleration
struct anderson <: ForwardSolver
    tol
    maxIter
    verbose
    autodiff
    anderson(;tol=1E-4, maxIter=1000, verbose=false, autodiff=:forward) = new(tol, maxIter, verbose, autodiff)
end
function (solver::anderson)(f::Function, z0)
    function f!(F, z)
        F .= f(z) - z
    end
    res = nlsolve(f!, z0; ftol=solver.tol, xtol=solver.tol, method=:anderson, autodiff=solver.autodiff)
    if solver.verbose
        println(res)
    end
    if !res.x_converged && !res.f_converged 
        println("Solver may not have converged.")
    end
    return res.zero
end


# Trust Region Solver
struct trust_region <: ForwardSolver
    tol
    maxIter
    verbose
    autodiff
    trust_region(;tol=1E-4, maxIter=1000, verbose=false, autodiff=:forward) = new(tol, maxIter, verbose, autodiff)
end
function (solver::trust_region)(f::Function, z0)
    function f!(F, z)
        F .= f(z) - z
    end
    res = nlsolve(f!, z0; ftol=solver.tol, xtol=solver.tol, method=:trust_region, autodiff=solver.autodiff)
    if solver.verbose
        println(res)
    end
    return res.zero
end


## Splitting methods
mutable struct PeacemanRachford <: OperatorSplitting
    tol
    maxIter
    α
    verbose
    cg
    PeacemanRachford(;tol=1E-4, maxIter=1000, α=1, verbose=false, cg=true) = new(tol, maxIter, α, verbose, cg)
end
function (solver::PeacemanRachford)(RA, RB, z0) 
    # Takes in the resolvent operators of the operators A and B
    error = 1E10
    uk, zk, u_12, z_12, zn, un = zero(z0), zero(z0), zero(z0), zero(z0), zero(z0), zero(z0)

    error_log = []
    for k in (1:solver.maxIter)
        u_12 = 2zk - uk
        z_12 = RB(u_12)
        un = 2z_12 - u_12
        zn = RA(un)
        error = norm(zn - zk)
        zk = zn
        uk = un
        # println(norm(error))

        if isnan(error)
            break
        end
        append!(error_log, error)
        if error < solver.tol
            if solver.verbose
                println(k, ",\t", error)
            end
            return zk
        end
    end
    println("PeacemanRachford did not converge. Error: ", error)
    return zk
end


mutable struct DouglasRachford <: OperatorSplitting
    tol
    maxIter
    α
    verbose
    DouglasRachford(;tol=1E-4, maxIter=1000, α=1, verbose=false) = new(tol, maxIter, α, verbose)
end
function (solver::DouglasRachford)(RA, RB, z0) 
    # Takes in the resolvent operators of the operators A and B
    error = 1E10
    uk, zk = z0, z0
    for k in (1:solver.maxIter)
        x_12 =  RB(zk)
        println("x_12 = ", norm(x_12))
        z_12 = 2x_12 - zk
        xn = RA(z_12)
        println("x_n = ", norm(x_n))
        zn = zk + xn - x_12
        error = norm(zn - zk) / (norm(zk) + 1E-7)
        if error < solver.tol
            println("\tError ", error, "\tIter ", k)
            return zk
        end
        zk, xk = zn, xn
    end
    println("DouglasRachford did not converge. Error: ", error)
    return zk
end


function test_solvers()
    
    f(x) = sin.(x)
    solver = anderson()

    z0 = [0.1]
    res = solver(f, z0)

    z = res.zero
end
