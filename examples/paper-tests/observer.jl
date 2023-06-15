cd(@__DIR__)
using Pkg
Pkg.activate("../")

m = 1                   # Mass (kg)
k = 5                   # Spring constant (N/m)
μ = 0.5                 # Viscous damping (kg/m)
nx = 2                  # Number of states

# Continuous and discrete dynamics and measurements
f(x::Matrix,u::Matrix) = [x[2:2,:]; (u[1:1,:] - 
    k*x[1:1,:] - μ*x[2:2,:].^2)/m]
fd(x,u) = x + dt*f(x,u)
gd(x::Matrix) = x[1:1,:]

Tmax = 10               # Simulation horizon
dt = 0.01               # Time-step (s)
ts = 1:Int(Tmax/dt)     # Time array indices

nbatch = 200
u = fill(zeros(1, nbatch), length(ts)-1)
X = fill(zeros(1, nbatch), length(ts))
X[1] = 0.5*(2*rand(nx, nbatch) .- 1)

for t in ts[1:end-1]
    X[t+1] = fd(X[t],u[t])
end

# Current/next state, measurements
Xt = X[1:end-1]
Xn = X[2:end]
y  = gd.(Xt)

# Store training data
obsv_data = [[ut; yt] for (ut,yt) in zip(u, y)]
indx = shuffle(1:length(obsv_data))
data = zip(Xn[indx], Xt[indx], obsv_data[indx])