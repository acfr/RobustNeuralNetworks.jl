push!(LOAD_PATH,"../src/")

using Documenter
using RecurrentEquilibriumNetworks

makedocs(
    sitename = "RecurrentEquilibriumNetworks.jl", 
    modules = [RecurrentEquilibriumNetworks],
    format = Documenter.HTML(prettyurls = false),
    
    # Need to format this nicely
    pages = [
        "Home" => "index.md"
    ],

    doctest = true # Can set to false while testing!
)

deploydocs(
    repo = "github.com/acfr/RecurrentEquilibriumNetworks.jl.git",
    versions = nothing # Remove this line when the package has versions!
)