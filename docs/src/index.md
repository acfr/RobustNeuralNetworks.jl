# RobustNeuralNetworks.jl Documentation

*A Julia package for robust neural networks.*

Welcome to the documentation for `RobustNeuralNetworks.jl`! This package contains neural network models that are constructed to naturally satisfy robustness constraints, all in native Julia. Check out our GitHub repository [here](https://github.com/acfr/RobustNeuralNetworks.jl).

## Why Robust Models?

Modern machine learning relies heavily on rapidly training and evaluating neural networks in problems ranging from image classification to robotic control. Most neural network architectures have no robustness certificates, and can be sensitive to adversarial attacks, poor data quality, and other input perturbations. Many solutions that address this brittle behaviour rely on explicitly enforcing constraints during training to smooth or stabilise the network response. While effective on small-scale problems, these methods are computationally expensive, making them slow and difficult to scale up to complex real-world problems.

Recently, we proposed the *Recurrent Equilibrium Network* (REN) and *Lipschitz-Bounded Deep Network* (LBDN) or *sandwich layer* model classes as computationally efficient solutions to these problems. The REN architecture is flexible in that it includes many common neural network models, such as multi-layer-perceptrons (MLPs), convolutional neural networks (CNNs), and recurrent neural networks (RNNs). The weights and biases in RENs are directly parameterised to **naturally satisfy** behavioural constraints chosen by the user. For example, we can build a REN with a given Lipschitz constant to ensure its output is quantifiably less sensitive to input perturbations. LBDNs are specializations of RENs with the specific feed-forward structure of deep neural networks like MLPs or CNNs and built-in guarantees on the Lipschitz bound.

The direct parameterisation of RENs and LBDNs means that we can train models with standard, unconstrained optimisation methods (such as stochastic gradient descent) while also guaranteeing their robustness. Achieving the "best of both worlds" in this way is the main advantage of the REN and LBDN model classes, and allows the user to freely train robust models for many common machine learning problems, as well as for more challenging real-world applications where safety is critical.


## Introduction

```@contents
Pages = ["introduction/getting_started.md", "introduction/package_overview.md", "introduction/developing.md"]
Depth = 1
```

## Examples

```@contents
Pages = ["examples/lbdn_curvefit.md", "examples/lbdn_mnist.md", "examples/rl.md", "examples/box_obsv.md", "examples/echo_ren.md"]
Depth = 1
```

## Library

```@contents
Pages = ["lib/models.md", "lib/model_params.md", "lib/functions.md"]
Depth = 1
```

## Citing the Package

If you use `RobustNeuralNetworks.jl` for any research or publications, please cite our work as necessary.
```bibtex
@article{barbara2023robustneuralnetworksjl,
   title   = {RobustNeuralNetworks.jl: a Package for Machine Learning and Data-Driven Control with Certified Robustness},
   author  = {Nicholas H. Barbara and Max Revay and Ruigang Wang and Jing Cheng and Ian R. Manchester},
   journal = {arXiv preprint arXiv:2306.12612},
   month   = {6},
   year    = {2023},
   url     = {https://arxiv.org/abs/2306.12612v1},
}
```


## Research Papers

`RobustNeurlaNetworks.jl` is built on the REN and LBDN model classes described in the following two papers (respectively):

> M. Revay, R. Wang, and I. R. Manchester, "Recurrent Equilibrium Networks: Flexible Dynamic Models with Guaranteed Stability and Robustness" *IEEE Trans Automat Contr* 1–16 (2023) [doi:10.1109/TAC.2023.3294101](https://ieeexplore.ieee.org/document/10179161).

> R. Wang and I. R. Manchester, "Direct parameterization of Lipschitz-bounded deep networks" in *Proceedings of the 40th International Conference on Machine Learning* (PMLR, 2023) [202:36093-36110](https://proceedings.mlr.press/v202/wang23v.html).

The REN parameterisation was extended to continuous-time systems in [yet to be implemented]:

> D. Martinelli, C. L. Galimberti, I. R. Manchester, L. Furieri, and G. Ferrari-Trecate, "Unconstrained Parametrization of Dissipative and Contracting Neural Ordinary Differential Equations," April 2023. doi: [https://doi.org/10.48550/arXiv.2304.02976](https://doi.org/10.48550/arXiv.2304.02976).

See below for a collection of projects and papers using `RobustNeuralNetworks.jl`.

> N. H. Barbara, R. Wang, and I. R. Manchester, "Learning Over Contracting and Lipschitz Closed-Loops for Partially-Observed Nonlinear Systems," April 2023. doi: [https://arxiv.org/abs/2304.06193v2](https://arxiv.org/abs/2304.06193v2).