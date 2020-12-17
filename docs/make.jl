using Documenter
import Pkg
using Trixi
using Trixi2Vtk
using Trixi2Img

# Get Trixi root directory
trixi_root_dir = dirname(@__DIR__)

# Copy list of authors to not need to synchronize it manually
authors_text = read(joinpath(trixi_root_dir, "AUTHORS.md"), String)
authors_text = replace(authors_text, "in the [LICENSE.md](LICENSE.md) file" => "under [License](@ref)")
write(joinpath(@__DIR__, "src", "authors.md"), authors_text)

# Define module-wide setups such that the respective modules are available in doctests
DocMeta.setdocmeta!(Trixi,     :DocTestSetup, :(using Trixi);     recursive=true)
DocMeta.setdocmeta!(Trixi2Vtk, :DocTestSetup, :(using Trixi2Vtk); recursive=true)
DocMeta.setdocmeta!(Trixi2Img, :DocTestSetup, :(using Trixi2Img); recursive=true)

# Make documentation
makedocs(
    # Specify modules for which docstrings should be shown
    modules = [Trixi, Trixi2Vtk, Trixi2Img],
    # Set sitename to Trixi
    sitename="Trixi.jl",
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
        "Visualization" => "visualization.md",
        "Conventions" => "conventions.md",
        "Time integration" => "time_integration.md",
        "Development" => "development.md",
        "Performance" => "performance.md",
        "Parallelization" => "parallelization.md",
        "Testing" => "testing.md",
        "Style guide" => "styleguide.md",
        "GitHub & Git" => "github-git.md",
        "Reference" => [
                        "Trixi.jl" => "reference-trixi.md",
                        "Trixi2Vtk.jl" => "reference-trixi2vtk.md",
                        "Trixi2Img.jl" => "reference-trixi2img.md",
                       ],
        "Authors" => "authors.md",
        "Contributing" => "contributing.md",
        "License" => "license.md"
    ]
)

deploydocs(
    repo = "github.com/trixi-framework/Trixi.jl",
)
