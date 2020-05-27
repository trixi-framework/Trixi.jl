using Documenter
import Pkg

# Get Trixi root directory
trixi_root_dir = dirname(@__DIR__)

# Install dependencies and import modules...
# ...Trixi
Pkg.activate(trixi_root_dir)
Pkg.instantiate()
import Trixi

# ...Trixi2Img
Pkg.activate(joinpath(trixi_root_dir, "postprocessing", "pkg", "Trixi2Img"))
Pkg.instantiate()
import Trixi2Img

# ...Trixi2Vtk
Pkg.activate(joinpath(trixi_root_dir, "postprocessing", "pkg", "Trixi2Vtk"))
Pkg.instantiate()
import Trixi2Vtk

# Set paths based on availability of CI variables
canonical = get(ENV, "CI_PAGES_URL", "https://numsim.gitlab-pages.sloede.com/code/Trixi.jl/")
repo_url = get(ENV, "CI_PROJECT_URL", "https://gitlab.mi.uni-koeln.de/numsim/code/Trixi.jl")

# Define module-wide setups such that the respective modules are available in doctests
DocMeta.setdocmeta!(Trixi,
                    :DocTestSetup,
                    :(push!(LOAD_PATH, ".."); using Trixi);
                    recursive=true)
DocMeta.setdocmeta!(Trixi2Img,
                    :DocTestSetup,
                    :(push!(LOAD_PATH, "../postprocessing/pkg/Trixi2Img"); using Trixi2Img);
                    recursive=true)
DocMeta.setdocmeta!(Trixi2Vtk,
                    :DocTestSetup,
                    :(push!(LOAD_PATH, "../postprocessing/pkg/Trixi2Vtk"); using Trixi2Vtk);
                    recursive=true)

# Make documentation
makedocs(
    # Specify modules for which docstrings should be shown
    modules = [Trixi, Trixi2Img, Trixi2Vtk],
    # Set sitename to Trixi
    sitename="Trixi",
    # Provide additional formatting options
    format = Documenter.HTML(
        # Disable pretty URLs during manual testing
        prettyurls = get(ENV, "CI", nothing) == "true",
        # Explicitly add favicon as asset
        assets = ["assets/favicon.ico"],
        # Set canonical URL to GitLab pages URL
        canonical = canonical
    ),
    # Explicitly specify documentation structure
    pages = [
        "Home" => "index.md",
        "Development" => "development.md",
        "Visualization" => "visualization.md",
        "Style guide" => "styleguide.md",
        "GitLab & Git" => "gitlab-git.md",
        "Reference" => [
            "Trixi" => "reference/trixi.md",
            "Trixi2Img" => "reference/trixi2img.md",
            "Trixi2Vtk" => "reference/trixi2vtk.md",
        ],
        "Authors" => "authors.md",
        "Contributing" => "contributing.md",
        "License" => "license.md"
    ],
    # Set repo to GitLab
    repo = "$(repo_url)/blob/{commit}{path}#{line}"
)
