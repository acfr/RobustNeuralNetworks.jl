# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

using Documenter
using RobustNeuralNetworks

const buildpath = haskey(ENV, "CI") ? ".." : ""

makedocs(
    sitename = "RobustNeuralNetworks.jl", 
    modules = [RobustNeuralNetworks],
    format = Documenter.HTML(prettyurls = haskey(ENV, "CI")),

    pages = [
        "Home" => "index.md",
        "Introduction" => Any[
            "Getting Started" => "introduction/getting_started.md",
            "Package Overview" => "introduction/package_overview.md",
            "Contributing to the Package" => "introduction/developing.md",
        ],
        "Examples" => Any[
            "Fitting a Curve" => "examples/lbdn_curvefit.md",
            "Image Classification" => "examples/lbdn_mnist.md",
            "Reinforcement Learning" => "examples/rl.md",
            "Observer Design" => "examples/box_obsv.md",
            "(Convex) Nonlinear Control" => "examples/echo_ren.md",
        ],
        "Library" => Any[
            "Model Wrappers" => "lib/models.md",
            "Model Parameterisations" => "lib/model_params.md",
            "Functions" => "lib/functions.md",
        ],
        "API" => "api.md"
    ],

    doctest = true,
    checkdocs=:exports
)

deploydocs(repo = "github.com/acfr/RobustNeuralNetworks.jl.git")
