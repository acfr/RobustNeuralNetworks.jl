# RobustNeuralNetworks.jl Documentation

*Bringing robust machine learning to Julia.*

Welcome to the documentation for `RobustNeuralNetworks.jl`! This package contains neural network models that are constructed to naturally satisfy robustness constraints, all in native Julia.

## Why Robust Models?

Modern machine learning relies heavily on rapidly training and evaluating neural networks in problems ranging from image classification to robotic control. Most existing neural network architectures have no robustness certificates, making them sensitive to poor data quality, adversarial attacks, and other input perturbations. The few neural network architectures proposed in recent years that offer solutions to this brittle behaviour rely on explicitly enforcing constraints during training to “smooth” the network response. These methods are computationally expensive, making them slow and difficult to scale up to complex real-world problems.

Recently, we proposed the Recurrent Equilibrium Network (REN) architecture as computationally efficient solutions to these problems. The REN architecture is flexible in that it includes all commonly used neural network models, such as fully-connected networks, convolutional neural networks, and recurrent neural networks. The weight matrices and bias vectors in a REN are directly parameterised to **naturally satisfy** behavioural constraints chosen by the user. For example, the user can build a REN with a given Lipschitz constant to ensure the output of the network is quantifiably less sensitive to unexpected input perturbations. 

The direct parameterisation of RENs means that we can train RENs with standard, unconstrained optimization methods (such as gradient descent) while also guaranteeing their robustness. Achieving the “best of both worlds” in this way is the main advantage of our REN/LBDN model classes, and allows us to freely train them for common machine learning problems as well as more difficult applications where safety and robustness are critical.

[*TODO: Add comments on LBDN when properly added to the package.*]


## Introduction

```@contents
Pages = ["introduction/getting_started.md", "introduction/layout.md", "introduction/developing.md"]
Depth = 1
```

## Examples

```@contents
Pages = ["examples/lbdn.md", "examples/rl.md", "examples/nonlinear_ctrl.md", "examples/pde_obsv.md"]
Depth = 1
```

## Library

```@contents
Pages = ["lib/models.md", "lib/model_params.md", "lib/functions.md"]
Depth = 1
```

## Research Papers

`RobustNeurlaNetworks.jl` is built on the REN and LBDN model parameterisations described in the following two papers (respectively):

> M. Revay, R. Wang, and I. R. Manchester, "Recurrent equilibrium networks: Flexible dynamic models with guaranteed stability and robustness," April 2021. doi: [https://doi.org/10.48550/arXiv.2104.05942](https://doi.org/10.48550/arXiv.2104.05942).

> R. Wang and I. R. Manchester, "Direct parameterization of Lipschitz-bounded deep networks," January 2023. doi: [https://doi.org/10.48550/arXiv.2301.11526](https://doi.org/10.48550/arXiv.2301.11526).

The REN parameterisation was extended to continuous-time systems in:

> D. Martinelli, C. L. Galimberti, I. R. Manchester, L. Furieri, and G. Ferrari-Trecate, "Unconstrained Parametrization of Dissipative and Contracting Neural Ordinary Differential Equations," April 2023. doi: [https://doi.org/10.48550/arXiv.2304.02976](https://doi.org/10.48550/arXiv.2304.02976).

See below for a collection of projects and papers using `RobustNeuralNetworks.jl`.

> N. H. Barbara, R. Wang, and I. R. Manchester, "Learning Over All Contracting and Lipschitz Closed-Loops for Partially-Observed Nonlinear Systems," April 2023. doi: [https://doi.org/10.48550/arXiv.2304.06193](https://doi.org/10.48550/arXiv.2304.06193).