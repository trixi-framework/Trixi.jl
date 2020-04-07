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
        canonical = "https://numsim.gitlab-pages.sloede.com/code/Trixi.jl/"
    ),
    # Explicitly specify documentation structure
    pages = [
        "Home" => "index.md",
        "Development" => "development.md",
        "Visualization" => "visualization.md",
        "Style guide" => "styleguide.md",
        "APIs" => [
            "Trixi" => "api/trixi.md",
            "Trixi2Img" => "api/trixi2img.md",
            "Trixi2Vtk" => "api/trixi2vtk.md",
        ]
    ],
    # Set repo to GitLab
    repo = "https://gitlab.mi.uni-koeln.de/numsim/code/Trixi.jl/blob/{commit}{path}#{line}"
)
