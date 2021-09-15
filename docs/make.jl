using Documenter
import Pkg

# Fix for https://github.com/trixi-framework/Trixi.jl/issues/668
if (get(ENV, "CI", nothing) != "true") && (get(ENV, "TRIXI_DOC_DEFAULT_ENVIRONMENT", nothing) != "true")
    push!(LOAD_PATH, dirname(@__DIR__))
end

using Trixi
using Trixi2Vtk

# Get Trixi root directory
trixi_root_dir = dirname(@__DIR__)

# Copy list of authors to not need to synchronize it manually
authors_text = read(joinpath(trixi_root_dir, "AUTHORS.md"), String)
authors_text = replace(authors_text, "in the [LICENSE.md](LICENSE.md) file" => "under [License](@ref)")
write(joinpath(@__DIR__, "src", "authors.md"), authors_text)

# Define module-wide setups such that the respective modules are available in doctests
DocMeta.setdocmeta!(Trixi,     :DocTestSetup, :(using Trixi);     recursive=true)
DocMeta.setdocmeta!(Trixi2Vtk, :DocTestSetup, :(using Trixi2Vtk); recursive=true)

# Make documentation
makedocs(
    # Specify modules for which docstrings should be shown
    modules = [Trixi, Trixi2Vtk],
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
            "Adding a new equation" => [
                "Scalar conservation law" => joinpath("adding_new_equations", "cubic_conservation_law.md"),
                "Nonconservative equation" => joinpath("adding_new_equations", "nonconservative_advection.md")
            ],
            "Differentiable programming" => "differentiable_programming.md",
            "Unstructured meshes with HOHQMesh.jl" => "hohqmesh_tutorial.md",
        ],
        "Basic building blocks" => [
            "Meshes" => [
                "Tree mesh" => joinpath("meshes", "tree_mesh.md"),
                "Structured mesh" => joinpath("meshes", "structured_mesh.md"),
                "Unstructured mesh" => joinpath("meshes", "unstructured_quad_mesh.md"),
                "P4est-based mesh" => joinpath("meshes", "p4est_mesh.md"),
                "Simplicial mesh" => joinpath("meshes", "mesh_data_meshes.md"),
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
                        "Trixi2Vtk.jl" => "reference-trixi2vtk.md"
                       ],
        "Authors" => "authors.md",
        "Contributing" => "contributing.md",
        "License" => "license.md"
    ],
    strict = true # to make the GitHub action fail when doctests fail, see https://github.com/neuropsychology/Psycho.jl/issues/34
)

deploydocs(
    repo = "github.com/trixi-framework/Trixi.jl",
    devbranch = "main",
    push_preview = true
)
