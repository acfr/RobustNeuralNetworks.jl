using Flux
using RobustNeuralNetworks
# using Plots


# training data
xmin = -π
xmax = π
xi = 0.05
xs = xmin:xi:xmax
N = 10
f(x) = sin(x)+(1/N)*sin.(N*x)
ys = f.(xs)
T = length(xs)
data = zip(xs,ys)


# model and loss
m = LBFN{Float64}(1,[10,5,5,15], 1)
ps = Flux.params(m)
loss(x,y) = Flux.mse(m(x),y) 

fit_error() = sqrt(sum(loss.(xs,ys))/length(xs))
slope() = maximum(abs.(diff(m(xs'),dims=2)))/xi

num_epochs = 50
lrs = [1e-3, 1e-4, 1e-5]
for k in eachindex(lrs)
    evalcb() = @show lrs[k] fit_error() slope()
    opt = NADAM(lrs[k])
    for _ in 1:num_epochs
        Flux.train!(loss, ps, data, opt, cb = Flux.throttle(evalcb, 10))
    end
end

# ŷ = map(x -> m(x)[1], xs)
# p =plot(xs,ys, label = "data", lw = 3)
# plot!(p,xs,ŷ, label = "LBDN", lw = 3, la = 0.8)
# display(p)

Empirical_Lipschitz = maximum(abs.(diff(m(xs'),dims=2)))/xi
