cd(@__DIR__)
using Pkg
Pkg.activate("../")

using BenchmarkTools
using Flux
using Random
using LinearAlgebra
using RobustNeuralNetworks

Random.seed!(0)

# Sizes
nu = 2
nh = [5, 10]
ny = 1
γ = 1

ps = DenseLBDNParams{Float64}(nu, nh, ny, γ)
lbdn = LBDN(ps)

u = rand(2)
y = lbdn(u)
