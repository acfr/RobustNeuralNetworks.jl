# Contributing to the Package

All contributors welcome! Please contact [nicholas.barbara@sydney.edu.au](mailto:nicholas.barbara@sydney.edu.au) with any questions.

## Installation for Development

If you would like to contribute the package, clone the repository into your `~/.julia/dev/` directory with

    git clone git@github.com:acfr/RobustNeuralNetworks.jl.git RobustNeuralNetworks

Note that the repo is `RobustNeuralNetworks.jl` but the folder name is `RobustNeuralNetworks`. This is convention for Julia packages. Navigate to the repository directory, start a Julia session, and type the following in the REPL to activate the package.

```julia
using Pkg
Pkg.instantiate()
Pkg.activate(".")
```

Check that the example in [Getting Started](@ref) runs without errors and matches the given output before continuing.

## Package Structure

The main file is `src/RobustNeuralNetworks.jl`. This imports all relevant packages, defines abstract types, includes code from other files, and exports the necessary components of our package. 

- All `using PackageName` statements should be included in this file.
- Only import the packages you really need
- If you only need one function from a package, import it explicitly (not the whole package)

When including files in our `src/` folder, the order often matters. Code should only ever be included with a single `include` statement in the main file. Please follow the convention outlined in the comments.

The source files for our package are all in the `src/` directory, and are split into the following sub-directories.

- `src/Base/`: Contains code relevant to the core package functionality.
- `src/ParameterTypes/`: Contains the various REN parameterisations, all of type [`AbstractRENParams`](@ref).
- `src/LBDN/`: Contains code exclusively used for [`AbstractLBDN`](@ref) models

Once you have written any code for this package, be sure to test it thoroughly. Write testing scripts in the `test/` directory.
- See [`Test.jl`](https://docs.julialang.org/en/v1/stdlib/Test/) documentation for help with writing good package tests.
- Run all tests for the package with `] test` in the REPL.
- All tests will be run by the CI client when submitting pull requests to the main git branch.

If you would like to contribute the docs, [this page](https://m3g.github.io/JuliaNotes.jl/stable/publish_docs/) provides a great outline of the required workflow.

## Git Workflow

The package can be treated as an independent git repository for devlopment.

- Please feel free to submit git issues, pull requests, etc. as usual.

- Always develop new features in a new branch labelled `feature/<some_descriptive_words>`. For example, the branch `feature/documentation` is where this documentation was first written and tested.

- Submit pull requests once you have completed a new feature and tested it thorougly. Pull requests without thorough testing will be rejected.
