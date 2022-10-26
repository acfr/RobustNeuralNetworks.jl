cd(@__DIR__)
using Pkg
Pkg.activate("./..")

using BenchmarkTools
using LinearAlgebra
using RecurrentEquilibriumNetworks
# using Distributions
"""
Johnny's random testing script: 
1. Passive ren test: similar to gitlab repo: passiveren/experiment/test_passive_ren.jl
2. passivity type test (incremental input? output?)
"""

nu = 4
nx = 10
nv = 20
ny = 4
T = 100

# Test constructors
Params = PassiveRENParams(nu, nx, nv, ny; init=:random)



u = [randn(nu, 1) for t in 1:T]

# TODO: randomly initialise passiveREN
Q = passive_ren(nuy,nx,nv)
x0 = init_state(Q,batches)
y = Q(x0, u)

function passive_index(u, y, T)
  s = zeros(1,T)
  J = 0;
  for t in 1:T
      du = u[t][:,1] - u[t][:,1]
      dy = y[t][:,1] - y[t][:,1]
      J += sum(du .* dy)
      s[t] = J
  end

  return s 
end

# Check signs
passive_index(u, y, T)


# ==================================
# Test more general REN construction with Q,S,R matrices
Q = zeros(Float64, ny, ny)
R = zeros(Float64, nu, nu)
# general size of S := (nu,ny)
S = Matrix{Float64}(-I(ny)) 

gren_ps = GeneralRENParams{Float64}(nu, nx, nv, ny, Q, S, R)

