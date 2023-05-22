# (Convex) Nonlinear Control with REN

*This example was first presented in Section IX of [Revay, Wang & Manchester (2021)](https://doi.org/10.48550/arXiv.2104.05942).*


We can use RENs and LBDNs for a lot more than just learning-based problems. In this example, we'll see how RENs can be used to design nonlinear feedback controllers with stability guarantees for linear dynamical systems with constraints. Introducing constraints (eg: minimum/maximum control inputs) often means that nonlinear controllers perform better than linear policies. A common approach is to use *Model Predictive Control* ([MPC](https://en.wikipedia.org/wiki/Model_predictive_control)). In our case, we'll use convex optimisation to design a nonlinear controller with stability guarantees. The controller will include a contracting REN, which we'll treat as an [*echo state network*](https://en.wikipedia.org/wiki/Echo_state_network), alongside the [*Youla-Kucera parameterisation*](https://www.sciencedirect.com/science/article/pii/S1367578820300249).

For a detailed explanation of the theory behind this example, please read the original [paper](https://doi.org/10.48550/arXiv.2104.05942). For more on using RENs in the Youla parameterisation, see [Wang et al. (2022)](https://ieeexplore.ieee.org/abstract/document/9802667) or [Barbara et al. (2023)](https://doi.org/10.48550/arXiv.2304.06193).


## 1. Background theory

### Stabilising a linear system

We'll start with some background on the structure of linear systems and output-feedback controllers. Consider a discrete-time linear dynamical system with state vector ``x_t``, control signal ``u_t``, external inputs ``d_t``, measured output ``y_t,`` and some performance variable to be kept small ``z_t``.

```math
\begin{aligned}
x_{t+1} &= \mathbb{A}x_t + \mathbb{B_1} d_t + \mathbb{B_2} u_t \\
z_t &= \mathbb{C_1} x_t + \mathbb{D_{11}} d_t + \mathbb{D_{12}} u_t \\
y_t &= \mathbb{C_2} x_t + \mathbb{D_{21}} d_t
\end{aligned}
```

A typical choice of stabilising controller is an output-feedback structure with state estimate ``\hat{x}_t`` and observer/controller gain matrices ``L`` and ``K``, respectively.

```math
\begin{aligned}
\hat{x}_{t+1} &= \mathbb{A}\hat{x}_t + \mathbb{B_2} u_t + L \tilde{y}_t \\
\tilde{y}_t &= y_t - \mathbb{C_2} \hat{x}_t \\
u_t &= -K\hat{x}_t + \tilde{u}_t
\end{aligned}
```

We have also included an additional signal ``\tilde{u}_t`` to augment the control signal. The closed-loop dynamics of the system can be written in the following form, where ``\mathcal{T}_0, \mathcal{T}_1, \mathcal{T}_2`` are linear systems.

```math
\begin{bmatrix}
z \\ \tilde{y}
\end{bmatrix}
= 
\begin{bmatrix}
\mathcal{T}_0 & \mathcal{T}_1 \\ \mathcal{T}_2 & 0
\end{bmatrix}
\begin{bmatrix}
d \\ \tilde{u}
\end{bmatrix}
```

Notice that there is no coupling between ``\tilde{y}`` and ``\tilde{u}``. 

### Controller augmentation

The linear controller will stabilise our linear dynamical system in the absence of any constraints. But what if we want to shape the closed-loop response to meet some user-defined design criteria without losing stability? For example, what if we want to keep the control signal in some safe range ``u_\mathrm{min} < u_t < u_\mathrm{max}`` at all times?

It turns out that if we augment the original controller with ``\tilde{u} = \mathcal{Q}(\tilde{y})`` where ``\mathcal{Q}`` is a *contracting system* then the closed-loop system is still guaranteed to be stable. This is incredibly useful for optimal control design. For example, we could use a contracting REN as our parameter ``\mathcal{Q}`` and optimise it to meet some performance specifications (like control constrains), knowing that final closed-loop system is guaranteed to be stable. The closed-loop response can be written as follows.

```math
z = \mathcal{T}_0 d + \mathcal{T}_1 \mathcal{Q}(\mathcal{T}_2 d)
```

This is an old idea in linear control theory called the Youla-Kucera parameterisation. We extended it to nonlinear models (like RENs) and nonlinear dynamical systems in [Wang et al. (2022)](https://ieeexplore.ieee.org/abstract/document/9802667) and [Barbara et al. (2023)](https://doi.org/10.48550/arXiv.2304.06193).


### Echo state networks with REN

Now that we've decided on a structure for our control framework, we need a way to optimise our contracting REN ``\mathcal{Q}`` to meet design criteria like control saturation. We could use something like reinforcement learning to train the REN, thereby learning over the space of [all stabilising controllers](https://ieeexplore.ieee.org/abstract/document/9802667)  for this linear system. While it's very useful to have this as an option, sometimes we might want a faster solution. Enter convex optimisation with echo state networks.

Let's say our REN has learnable parameters ``\theta``. Suppose our problem is to minimise some convex objective function ``J(z)`` subject to a set of convex constraints. I.e:

```math
\min_\theta J(z) \quad \text{s.t.} \quad c(z) \le 0
```

An [*echo state network*](https://en.wikipedia.org/wiki/Echo_state_network) is a dynamic model with randomly sampled but *fixed* dynamics and a learnable output map. We can create contracting echo state networks with contracting RENs. When a REN model is called, it can be viewed as a system with [`ExplicitRENParams`](@ref) in the following form.

```math
\begin{equation*}
\begin{bmatrix}
\bar{x}_{t+1} \\ v_t \\ \bar{y}_t
\end{bmatrix}
= 
\begin{bmatrix}
A & B_1 & B_2 \\
C_1 & D_{11} & D_{12} \\
C_2 & D_{21} & D_{22} \\
\end{bmatrix}
\begin{bmatrix}
\bar{x}_t \\ w_t \\ \bar{u}_t
\end{bmatrix}
+ 
\begin{bmatrix}
b_x \\ b_v \\ b_y
\end{bmatrix}
\end{equation*}
```

The inputs and outputs of the REN are ``\bar{u}_t`` and ``\bar{y}_t``, respectively. We can therefore create an echo state network by randomly initialising a REN whose outputs are ``\bar{x}_t, w_t, \bar{u}_t``, and then optimising the output layer 
```math
\bar{y}_t = C_2 \bar{x}_t + D_{21} w_t + D_{22} \bar{u}_t + b_y
```
separately, where the learnable parameters are ``\theta = [C_2 \ D_{21} \ D_{22} \ b_y].``


## 2. Problem setup