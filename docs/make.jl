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

# Generate markdown and notebook files for tutorials
trixi_include("src/tutorials/make.jl")

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
        "Getting started" => [
            "Overview" => "overview.md",
            "Visualization" => "visualization.md",
        ],
        "Tutorials" => [
            "Introduction" => "tutorials/pages/index.md",
            "Adding a new equation" => "tutorials/pages/t1_adding_a_new_equation.md",
            "Differentiable programming" => "tutorials/pages/t2_differentiable_programming.md",
            "Testing" => "tutorials/pages/t3_testing_repository.md",
        ],
        "Basic building blocks" => [
            "Meshes" => [
                "Tree mesh" => joinpath("meshes", "tree_mesh.md"),
                "Structured mesh" => joinpath("meshes", "structured_mesh.md"),
                "Unstructured mesh" => joinpath("meshes", "unstructured_quad_mesh.md"),
            ],
            "Time integration" => "time_integration.md",
            "Callbacks" => "callbacks.md",
        ],
        "Advanced topics & developers" => [
            "Conventions" =>"conventions.md",
            "Development" => "development.md",
            "GitHub & Git" => "github-git.md",
            "Style guide" => "styleguide.md",
            "Testing" => "testing.md",
            "Performance" => "performance.md",
            "Parallelization" => "parallelization.md",
        ],
        "Troubleshooting and FAQ" => "troubleshooting.md",
        "Reference" => [
                        "Trixi.jl" => "reference-trixi.md",
                        "Trixi2Vtk.jl" => "reference-trixi2vtk.md",
                        "Trixi2Img.jl" => "reference-trixi2img.md",
                       ],
        "Authors" => "authors.md",
        "Contributing" => "contributing.md",
        "License" => "license.md"
    ],
    strict = true # to make the GitHub action fail when doctests fail, see https://github.com/neuropsychology/Psycho.jl/issues/34
)

# Copy Project.toml to binder folder
Sys.cp(joinpath(@__DIR__, "Project.toml"), joinpath(@__DIR__, "../binder/Project.toml"))

deploydocs(
    repo = "github.com/trixi-framework/Trixi.jl",
    devbranch = "main",
    push_preview = true
)
