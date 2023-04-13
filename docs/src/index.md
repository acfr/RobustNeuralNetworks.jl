# RobustNeuralNetworks.jl Documentation

*Bringing robust machine learning to Julia.*

Welcome to the documentation for `RobustNeuralNetworks.jl`! This package contains neural network models that are constructed to naturally satisfy robustness constraints, all in native Julia.

## Why Robustness?

Modern machine learning relies heavily on rapidly training and evaluating neural networks in problems ranging from image classification to robotic control. Most existing neural network architectures have no robustness certificates, making them sensitive to poor data quality, adversarial attacks, and other input perturbations. The few neural network architectures proposed in recent years that offer solutions to this brittle behaviour rely on explicitly enforcing constraints during training to “smooth” the network response. These methods are computationally expensive, making them slow and difficult to scale up to complex real-world problems.

[*TODO: Add a comments on LBDN here.*]
Recently, we proposed the Recurrent Equilibrium Network (REN) architecture as computationally efficient solutions to these problems. The REN architecture is flexible in that it includes all commonly used neural network models, such as fully-connected networks, convolutional neural networks, and recurrent neural networks. The weight matrices and bias vectors in a REN are directly parameterised to **naturally satisfy** behavioural constraints chosen by the user. For example, the user can build a REN with a given Lipschitz constant to ensure the output of the network is quantifiably less sensitive to unexpected input perturbations. 

The direct parameterisation of RENs means that we can train RENs with standard, unconstrained optimization methods (such as gradient descent) while also guaranteeing their robustness. Achieving the “best of both worlds” in this way is unique to our REN/LBDN model classes, and allows us to freely train them for common machine learning problems as well as more difficult applications where safety and robustness are critical.


## Introduction

```@contents
Pages = ["man/introduction.md", "man/layout.md"]
Depth = 1
```

## Examples

```@contents
Pages = ["examples/lbdn.md", "examples/rl.md", "examples/nonlinear_ctrl.md", "examples/pde_obsv.md"]
Depth = 1
```

## Library

```@contents
Pages = ["lib/lbdn.md", "lib/ren.md", "lib/ren_params.md", "lib/functions.md"]
Depth = 1
```

## Research Papers

For more information on REN and LBDN, please see the following papers.

[*Add these in tomorrow*]
