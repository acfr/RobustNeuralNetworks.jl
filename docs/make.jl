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
            "Image Classification" => "examples/lbdn.md",
            "Reinforcement Learning" => "examples/rl.md",
            "Nonlinear Control" => "examples/nonlinear_ctrl.md",
            "PDE Observer" => "examples/pde_obsv.md",
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
