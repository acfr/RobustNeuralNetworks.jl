# Contributing to the Package

All contributors welcome! Please contact [nicholas.barbara@sydney.edu.au](mailto:nicholas.barbara@sydney.edu.au) with any questions.

## Installation for Development

If you would like to contribute the package, please clone the repository into your `~/.julia/dev/` directory with

    git clone git@github.com:acfr/RobustNeuralNetworks.jl.git RobustNeuralNetworks

!!! info "The name matters"
    Note that the repository is named `RobustNeuralNetworks.jl` but your folder should be named `RobustNeuralNetworks` (no `.jl`). This is convention for Julia packages, and will ensure the package manager knows where to look.

    Also make sure you have cloned the repo inside your `~/.julia/dev/` folder for the following instructions to work. If the `dev/` directory does not exist, create it. 


Navigate to `~/.julia/dev/RobustNeuralNetworks/`, start a Julia session, and type the following in the REPL to activate the package.

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Check that the example in [Getting Started](@ref) runs without errors and matches the given output before continuing.


## Git Workflow

The package is just another git repository, so the development workflow is similar to most projects.

- When developing features, please create a new branch labelled `feature/<some_descriptive_name>`. For example, the branch `feature/documentation` is where this documentation was first written and tested.

- If you notice a bug, please create a git issue and a new branch associated with that issue to let others know what you are working on. The branch should be deleted once the issue is closed.

- Submit pull requests once you have completed a new feature and tested it thorougly. All pull requests must be approved by at least one other developer of the package, and must pass the continuous integration pipeline.


## Package Structure

### Layout

The package is divided into four main directories:

- The `src/` folder contains all source code required for users of the package.
- The `examples/` folder contains complete examples that are referenced in the documentation, and any assets or results required or produced by the examples.
- The `docs/` folder contains all files related to the documentation.
- The `test/` folder contains the main `runtests.jl` file and all other test files loaded within it.

!!! info "Separate projects"
    Note that the `examples/` and `docs/` folders each have their own `Project.toml` (and `Manifest.toml`). You can activate them by typing
    ```
    ] activate docs
    ```
    (or similar for `examples`) into the REPL from the home directory of the repository. Please do not add packages to the main `Project.toml` if they are not required by `RobustNeuralNetworks.jl` itself.

### Main file

The main file is `src/RobustNeuralNetworks.jl`. This imports all relevant packages, defines abstract types, includes code from other files, and exports the necessary components of our package. 

- All `using PackageName` statements should be included in this file.
- Only import packages that are absolutely required.
- If you only need one function from a package, import it explicitly (not the whole package). For example:
```julia
using Flux: relu
```

When including files in the `src/` folder, the order matters. Code should only ever be included with a single `include` statement in the main file. Please follow the convention outlined in the comments.


### Source files

The source files for our package are all in the `src/` directory, and are split into the following sub-directories.

- `src/Base/`: Contains code relevant to the core package functionality.
- `src/ParameterTypes/`: Contains the various REN and LBDN *direct* parameterisations, which are all subtypes of [`AbstractRENParams`](@ref) or [`AbstractLBDNParams`](@ref).
- `src/Wrappers/`: Contains wrappers used to define *explicit* (callable) REN and LBDN models (eg: subtypes of [`AbstractREN`](@ref) and [`AbstractLBDN`](@ref)).


### Writing tests

Once you have written any code for this package, be sure to test it thoroughly. You should also consider adding test scripts to the `test/` directory.
- See the documentation for [`Test.jl`](https://docs.julialang.org/en/v1/stdlib/Test/) for help with writing good package tests.
- Run all tests for the package with `] test` in the REPL.
- All tests will be run by the CI client when submitting pull requests to the `main` git branch.

Currently, the tests in place check that the various model parameterisations satisfy the constraints they should, and that the models are differentiable with `Flux.jl`.

### Writing documentation

If you would like to contribute the docs, [this page](https://m3g.github.io/JuliaNotes.jl/stable/publish_docs/) provides a great outline of the required workflow. In summary:

- All documentation should be written in markdown (`.md` files) within the `docs/` folder.
- Please add documentation in the `docs/src/` folder.
- To build the docs locally from the `~/.julia/dev/RobustNeuralNetworks/` directory, activate the Julia REPL and enter the following.

```julia
using Pkg
Pkg.activate("./docs")

using LiveServer
servedocs()
```
