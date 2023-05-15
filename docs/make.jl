using Documenter
using RobustNeuralNetworks

makedocs(
    sitename = "RobustNeuralNetworks.jl", 
    modules = [RobustNeuralNetworks],
    format = Documenter.HTML(prettyurls = haskey(ENV, "CI")),
    
    # Need to format this nicely
    pages = [
        "Home" => "index.md",
        "Introduction" => Any[
            "Getting Started" => "introduction/getting_started.md",
            "Package Overview" => "introduction/layout.md",
            "Contributing to the Package" => "introduction/developing.md",
        ],
        "Examples" => Any[
            "Fitting a Curve" => "examples/lbdn_curvefit.md",
            "Image Classification" => "examples/lbdn_mnist.md",
            "Reinforcement Learning" => "examples/rl.md",
            "PDE Observer" => "examples/pde_obsv.md",
            "Nonlinear Control" => "examples/ren_ctrl.md",
        ],
        "Library" => Any[
            "Model Wrappers" => "lib/models.md",
            "Model Parameterisations" => "lib/model_params.md",
            "Functions" => "lib/functions.md",
        ],
        "API" => "api.md"
    ],

    doctest = true, # Can set to false while testing!
    checkdocs=:exports
)

deploydocs(repo = "github.com/acfr/RobustNeuralNetworks.jl.git")

# To generate the documentation locally:
# ] activate docs
# using LiverServer
# servedocs()
