using Documenter
import Pkg
using Changelog: Changelog

# Fix for https://github.com/trixi-framework/Trixi.jl/issues/668
if (get(ENV, "CI", nothing) != "true") &&
   (get(ENV, "TRIXI_DOC_DEFAULT_ENVIRONMENT", nothing) != "true")
    push!(LOAD_PATH, dirname(@__DIR__))
end

using Trixi
using Trixi2Vtk
using TrixiBase

# Get Trixi.jl root directory
trixi_root_dir = dirname(@__DIR__)

include(joinpath(trixi_root_dir, "docs", "literate", "make.jl"))

# Copy list of authors to not need to synchronize it manually
authors_text = read(joinpath(trixi_root_dir, "AUTHORS.md"), String)
authors_text = replace(authors_text,
                       "in the [LICENSE.md](LICENSE.md) file" => "under [License](@ref)")
write(joinpath(@__DIR__, "src", "authors.md"), authors_text)

# Define module-wide setups such that the respective modules are available in doctests
DocMeta.setdocmeta!(Trixi, :DocTestSetup, :(using Trixi); recursive = true)
DocMeta.setdocmeta!(Trixi2Vtk, :DocTestSetup, :(using Trixi2Vtk); recursive = true)

# Copy some files from the repository root directory to the docs and modify them
# as necessary
# Based on: https://github.com/ranocha/SummationByPartsOperators.jl/blob/0206a74140d5c6eb9921ca5021cb7bf2da1a306d/docs/make.jl#L27-L41
open(joinpath(@__DIR__, "src", "code_of_conduct.md"), "w") do io
    # Point to source license file
    println(io,
            """
            ```@meta
            EditURL = "https://github.com/trixi-framework/Trixi.jl/blob/main/CODE_OF_CONDUCT.md"
            ```
            """)
    # Write the modified contents
    println(io, "# [Code of Conduct](@id code-of-conduct)")
    println(io, "")
    for line in eachline(joinpath(dirname(@__DIR__), "CODE_OF_CONDUCT.md"))
        line = replace(line, "[AUTHORS.md](AUTHORS.md)" => "[Authors](@ref)")
        println(io, "> ", line)
    end
end

open(joinpath(@__DIR__, "src", "contributing.md"), "w") do io
    # Point to source license file
    println(io,
            """
            ```@meta
            EditURL = "https://github.com/trixi-framework/Trixi.jl/blob/main/CONTRIBUTING.md"
            ```
            """)
    # Write the modified contents
    for line in eachline(joinpath(dirname(@__DIR__), "CONTRIBUTING.md"))
        line = replace(line, "[LICENSE.md](LICENSE.md)" => "[License](@ref)")
        line = replace(line, "[AUTHORS.md](AUTHORS.md)" => "[Authors](@ref)")
        println(io, line)
    end
end

# Create tutorials for the following files:
# Normal structure: "title" => "filename.jl"
# If there are several files for one topic and one folder, the structure is:
#   "title" => ["subtitle 1" => ("folder 1", "filename 1.jl"),
#               "subtitle 2" => ("folder 2", "filename 2.jl")]
files = [
    # Topic: introduction
    "First steps in Trixi.jl" => [
        "Getting started" => ("first_steps", "getting_started.jl"),
        "Create your first setup" => ("first_steps", "create_first_setup.jl"),
        "Changing Trixi.jl itself" => ("first_steps", "changing_trixi.jl")
    ],
    "Behind the scenes of a simulation setup" => "behind_the_scenes_simulation_setup.jl",
    # Topic: DG semidiscretizations
    "Introduction to DG methods" => "scalar_linear_advection_1d.jl",
    "DGSEM with flux differencing" => "DGSEM_FluxDiff.jl",
    "Shock capturing with flux differencing and stage limiter" => "shock_capturing.jl",
    "Subcell limiting with the IDP Limiter" => "subcell_shock_capturing.jl",
    "Non-periodic boundaries" => "non_periodic_boundaries.jl",
    "DG schemes via `DGMulti` solver" => "DGMulti_1.jl",
    "Other SBP schemes (FD, CGSEM) via `DGMulti` solver" => "DGMulti_2.jl",
    "Upwind FD SBP schemes" => "upwind_fdsbp.jl",
    # Topic: equations
    "Adding a new scalar conservation law" => "adding_new_scalar_equations.jl",
    "Adding a non-conservative equation" => "adding_nonconservative_equation.jl",
    "Parabolic terms" => "parabolic_terms.jl",
    "Adding new parabolic terms" => "adding_new_parabolic_terms.jl",
    # Topic: meshes
    "Adaptive mesh refinement" => "adaptive_mesh_refinement.jl",
    "Structured mesh with curvilinear mapping" => "structured_mesh_mapping.jl",
    "Unstructured meshes with HOHQMesh.jl" => "hohqmesh_tutorial.jl",
    "P4est mesh from gmsh" => "p4est_from_gmsh.jl",
    # Topic: other stuff
    "Explicit time stepping" => "time_stepping.jl",
    "Differentiable programming" => "differentiable_programming.jl",
    "Custom semidiscretizations" => "custom_semidiscretization.jl"
]
tutorials = create_tutorials(files)

# Create changelog
Changelog.generate(Changelog.Documenter(),                        # output type
                   joinpath(@__DIR__, "..", "NEWS.md"),           # input file
                   joinpath(@__DIR__, "src", "changelog_tmp.md"); # output file
                   repo = "trixi-framework/Trixi.jl",             # default repository for links
                   branch = "main",)
# Fix edit URL of changelog
open(joinpath(@__DIR__, "src", "changelog.md"), "w") do io
    for line in eachline(joinpath(@__DIR__, "src", "changelog_tmp.md"))
        if startswith(line, "EditURL")
            line = "EditURL = \"https://github.com/trixi-framework/Trixi.jl/blob/main/NEWS.md\""
        end
        println(io, line)
    end
end

# Make documentation
makedocs(
         # Specify modules for which docstrings should be shown
         modules = [Trixi, TrixiBase, Trixi2Vtk],
         # Set sitename to Trixi.jl
         sitename = "Trixi.jl",
         # Provide additional formatting options
         format = Documenter.HTML(
                                  # Disable pretty URLs during manual testing
                                  prettyurls = get(ENV, "CI", nothing) == "true",
                                  # Explicitly add favicon as asset
                                  assets = ["assets/favicon.ico"],
                                  # Set canonical URL to GitHub pages URL
                                  canonical = "https://trixi-framework.github.io/Trixi.jl/stable",
                                  size_threshold_ignore = ["reference-trixi.md"]),
         # Explicitly specify documentation structure
         pages = [
             "Home" => "index.md",
             "Getting started" => [
                 "Overview" => "overview.md",
                 "Visualization" => "visualization.md",
                 "Restart simulation" => "restart.md"
             ],
             "Tutorials" => tutorials,
             "Basic building blocks" => [
                 "Meshes" => [
                     "Tree mesh" => joinpath("meshes", "tree_mesh.md"),
                     "Structured mesh" => joinpath("meshes", "structured_mesh.md"),
                     "Unstructured mesh" => joinpath("meshes", "unstructured_quad_mesh.md"),
                     "P4est-based mesh" => joinpath("meshes", "p4est_mesh.md"),
                     "DGMulti mesh" => joinpath("meshes", "dgmulti_mesh.md")
                 ],
                 "Time integration" => "time_integration.md",
                 "Callbacks" => "callbacks.md",
                 "Coupling" => "multi-physics_coupling.md"
             ],
             "Advanced topics & developers" => [
                 "Conventions" => "conventions.md",
                 "Development" => "development.md",
                 "GitHub & Git" => "github-git.md",
                 "Style guide" => "styleguide.md",
                 "Testing" => "testing.md",
                 "Performance" => "performance.md",
                 "Parallelization" => "parallelization.md"
             ],
             "Troubleshooting and FAQ" => "troubleshooting.md",
             "Reference" => [
                 "Trixi.jl" => "reference-trixi.md",
                 "TrixiBase.jl" => "reference-trixibase.md",
                 "Trixi2Vtk.jl" => "reference-trixi2vtk.md"
             ],
             "Changelog" => "changelog.md",
             "Authors" => "authors.md",
             "Contributing" => "contributing.md",
             "Code of Conduct" => "code_of_conduct.md",
             "License" => "license.md"
         ])

deploydocs(repo = "github.com/trixi-framework/Trixi.jl",
           devbranch = "main",
           push_preview = true)
