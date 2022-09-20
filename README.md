# RecurrentEquilibriumNetworks.jl

## Status
[![Build Status](https://github.com/nic-barbara/RecurrentEquilibriumNetworks.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/nic-barbara/RecurrentEquilibriumNetworks.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Description

Julia package for Recurrent Equilibrium Networks.

[NOTE] This package is a work-in-progress. For now, you may find the following links useful:
- Tutorial on [developing Julia packages](https://julialang.org/contribute/developing_package/) by Chris Rackauckas (MIT)
- Documentation on [managing Julia packages](https://pkgdocs.julialang.org/v1/managing-packages/) and developing unregistered packages with `Pkg.jl`

So far, the package only contains some of the basic data types and methods for RENs.

## How to use 

To install the package for development, clone the repository into your Julia dev folder:
- For Linux/Mac, use `git clone git@github.com:acfr/RecurrentEquilibriumNetworks.jl.git RecurrentEquilibriumNetworks` inside your `~/.julia/dev/` directory.
- Note that the repo is `RecurrentEquilibriumNetworks.jl`, but the folder is `RecurrentEquilibriumNetworks`. This is convention for Julia packages.

Navigate to the repo directory, start the Julia REPL, and type `] activate .` to activate the package. You can now test out some basic functionality:
- Type `using RecurrentEquilibriumNetworks` in the REPL to add the package to your current session.
- Try to print out some documentation. Eg: type `? DirectParams` into the REPL.

## To contribute to the package:

Edit source files in `src/`:
- `RecurrentEquilibriumNetworks.jl` is the main file. Include dependencies and export types/functions here.
- Add other source files for new functionality (eg: `src/functions.jl`).

Write testing scripts for the package in `test/`:
- See [`Test.jl`](https://docs.julialang.org/en/v1/stdlib/Test/) documentation for writing tests
- Run tests with `] test`

Use git to pull/push changes to the package as normal

## To use the package in a separate Julia workspace:


- Add development version of the package with: `] dev git@github.com:acfr/RecurrentEquilibriumNetworks.jl.git`
- This is instead of the usual `] add` command. We also have to use the git link not the package name because it is an unregistered Julia package.
- Whenever you use the package, it will access the latest version in your `.julia/dev/` folder rather than the stable release in the `main` branch. This is easiest for development while we frequently change the package.
- To use the code on the current main branch of the repo (latest stable release), instead type `] add git@github.com:acfr/RecurrentEquilibriumNetworks.jl.git`. You will have to manually update the package as normal with `] update RecurrentEquilibriumNetworks`.

### Contact
Nic Barbara (nicholas.barbara@sydney.edu.au) for any questions/concerns. 