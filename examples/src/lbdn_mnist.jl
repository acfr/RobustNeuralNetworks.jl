# TODO: This is an old demo from our previous implementation of LBDN.
# No good for the current version!

# Import packages (iincluding our REN package)
using Flux
using Flux:onehotbatch, @epochs, crossentropy,onecold,throttle, OneHotMatrix
using MLDatasets: MNIST
using RobustNeuralNetworks
using Statistics

# Get MNIST training and test data
train_x, train_y = MNIST.traindata(Float64)
test_x, test_y = MNIST.testdata(Float64)

# Reshape as appropriate for training
train_x = Flux.flatten(train_x)
test_x = Flux.flatten(test_x)

train_y, test_y = onehotbatch(train_y, 0:9), onehotbatch(test_y, 0:9)


# Define a model with LBDN
nu = 28*28      # Inputs
nh = [60]       # 1 hidden layer of size 60 (has to be in a vector)
ny = 10         # Outputs
γ = 5.0         # Lipschitz bound upper limit (must be a float)

lbfn = LBFN{Float64}(nu, nh, ny, γ)
m = Chain(lbfn, softmax)

# Could instead use a fully-connected network (see below)
#m =  Chain(Dense(28*28,60, relu), Dense(60, 10), softmax)

# Define loss function, optimiser, and get params
loss(x, y) = crossentropy(m(x), y) 
opt = ADAM(1e-3)
ps = Flux.params(m)

# Comparison functions
compare(y::OneHotMatrix, y′) = maximum(y′, dims = 1) .== maximum(y .* y′, dims = 1)
accuracy(x, y::OneHotMatrix) = mean(compare(y, m(x)))

# To check progrress while training
progress = () -> @show(loss(train_x, train_y), accuracy(test_x, test_y) ) # callback to show loss

# Train model with two different leaning rates
opt = ADAM(1e-3)
@epochs 200 Flux.train!(loss, ps,[(train_x,train_y)], opt, cb = throttle(progress, 10))
opt = ADAM(1e-4)
@epochs 400 Flux.train!(loss, ps,[(train_x,train_y)], opt, cb = throttle(progress, 10))

# Show results
accuracy(train_x,train_y)
accuracy(test_x,test_y)