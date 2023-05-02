cd(@__DIR__)
using Pkg
Pkg.activate("../")

using BenchmarkTools
using Flux
using Random
using LinearAlgebra
using RobustNeuralNetworks

Random.seed!(0)

# Set up model params
nu, ny   = 1, 1
nh       = [10,5,5,15]
γ        = 1
model_ps = DenseLBDNParams{Float64}(nu, nh, ny, γ)
ps       = Flux.params(model_ps)

# Function to estimate
f(x) = sin(x)+(1/N)*sin.(N*x)

# Training data
N  = 10
dx = 0.05
xs = -π:dx:π
ys = f.(xs)
T  = length(xs)
data = zip(xs,ys)

# Loss function
function loss(x, y) 
    m = LBDN(model_ps)
    Flux.mse(m([x]),[y])
    # return loss(m,x,y)
end
# loss(m, x, y)  = Flux.mse(m([x]),[y])



# Set up training loop
num_epochs = 5
lrs = [1e-3] #, 1e-4, 1e-5]
# for k in eachindex(lrs)
k = 1

    function evalcb() 
        m         = LBDN(model_ps)
        fit_error = sqrt(sum(loss.((m,), xs, ys)) / length(xs))
        slope     = maximum(abs.(diff(m(xs'),dims=2)))/dx
        @show lrs[k] fit_error slope
        println()
    end

    opt = NADAM(lrs[k])
    for _ in 1:num_epochs
        Flux.train!(loss, ps, data, opt, cb = Flux.throttle(evalcb, 10))
        # for d in data
        #     dLdm = Flux.gradient(loss, ps, d...)[1]
        #     Flux.update!(opt, ps, dLdm)
        # end
    end

# end

# # TODO: Add plotting to check it all works
# # ŷ = map(x -> m(x)[1], xs)
# # p =plot(xs,ys, label = "data", lw = 3)
# # plot!(p,xs,ŷ, label = "LBDN", lw = 3, la = 0.8)
# # display(p)

# Empirical_Lipschitz = maximum(abs.(diff(m(xs'),dims=2)))/dx
