# push!(LOAD_PATH,"../src/")

using Documenter
using RobustNeuralNetworks

makedocs(
    sitename = "RobustNeuralNetworks.jl", 
    modules = [RobustNeuralNetworks],
    format = Documenter.HTML(prettyurls = false),
    
    # Need to format this nicely
    pages = [
        "Home" => "index.md"
    ],

    doctest = true, # Can set to false while testing!
    checkdocs=:exports
)

deploydocs(repo = "github.com/acfr/RobustNeuralNetworks.jl.git")


# ] activate docs
# using LiverServer
# servedocs()