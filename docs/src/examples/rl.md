# Reinforcement Learning with LBDN

*[Example coming soon. Some examples of RL with REN can be found in [Barbara, Wang & Manchester (2023)](https://doi.org/10.48550/arXiv.2304.06193).]*

One of the original motivations for developing `RobustNeuralNetworks.jl` was to learn robust models for applications in control engineering. Some of our recent research (eg: [Wang et al. (2022)](https://ieeexplore.ieee.org/abstract/document/9802667) and [Barbara et al. (2023)](https://doi.org/10.48550/arXiv.2304.06193)) has shown that, with the right controller architecture, we can learn over the space of all stabilising controllers for all linear or nonlinear systems using standard reinforcement learning techniques, so long as our control policy is parameterised by a REN (see also [(Convex) Nonlinear Control with REN](@ref)).

In this example, we'll demonstrate how to train an LBDN controller with *Reinforcement Learnign* (RL) for a simple nonlinear dynamical system. Note that this controller will not have any stability guarantees. However, it will still showcase all the steps required to set up RL experiments for more complex systems with RENs and LBDNs.

## 1. Problem setup

Let's consider the simple mechanical system shown below: a box of mass ``m`` sits in a tub of fluid, held between the walls of the tub by two springs, each with spring constant ``k/2.`` We can push the box with force ``u.`` Its dynamics are
```math
m\ddot{q} = u - kq - \mu \dot{q}^2
```
where ``\mu`` is the viscous friction coefficient due to the box moving through the fluid.

```@example
@html_str """<p align="center"> <object type="image/png" data=$(joinpath(Main.buildpath, "../assets/lbdn-rl/mass_rl.png")) width="35%"></object> </p>""" #hide
```

We can write this as a (nonlinear) state-space model with state ``x = (q,\dot{q}),`` control input ``u,`` and dynamics
```math
\dot{x} = f(x,u) = \begin{bmatrix}
\dot{q} \\ (u - kq - \mu \dot{q}^2)/m
\end{bmatrix}.
```
Our aim is to learn a controller ``u = \mathcal{K}_\theta(x),`` defined by some learnable parameters ``\theta,`` that can push the box to any goal position we choose. Specifically, we want the box to:
- Reach a (stationary) goal position ``q_\mathrm{ref}``
- Within a time ``T``
- Using minimal control force ``u``

Note that the force required to reach an equilibrium position ``q_\mathrm{ref}`` is simply ``u_\mathrm{ref} = k q_\mathrm{ref}`` (set the derivative terms in the dynamics to zero and re-arrange for ``u``). We can encode these objectives in a cost function, and write our RL problem as
```math
\min_\theta
```