using Documenter
import Pkg

# Get Trixi root directory
trixi_root_dir = dirname(@__DIR__)

# Install dependencies and import modules...
# ...Trixi
Pkg.activate(trixi_root_dir)
Pkg.instantiate()
import Trixi

# Define module-wide setups such that the respective modules are available in doctests
DocMeta.setdocmeta!(Trixi,
                    :DocTestSetup,
                    :(push!(LOAD_PATH, ".."); using Trixi);
                    recursive=true)

# Make documentation
makedocs(
    # Specify modules for which docstrings should be shown
    modules = [Trixi],
    # Set sitename to Trixi
    sitename="Trixi",
    # Provide additional formatting options
    format = Documenter.HTML(
        # Disable pretty URLs during manual testing
        prettyurls = get(ENV, "CI", nothing) == "true",
        # Explicitly add favicon as asset
        assets = ["assets/favicon.ico"],
        # Set canonical URL to GitHub pages URL
        canonical = "https://trixi-framework.github.io/Trixi.jl/stable"
    ),
    # Explicitly specify documentation structure
    pages = [
        "Home" => "index.md",
        "Development" => "development.md",
        "Visualization" => "visualization.md",
        "Style guide" => "styleguide.md",
        "GitHub & Git" => "github-git.md",
        "Reference" => "reference.md",
        "Authors" => "authors.md",
        "Contributing" => "contributing.md",
        "License" => "license.md"
    ]
)

deploydocs(
    repo = "github.com/trixi-framework/Trixi.jl",
)
