using Literate: Literate
using Test: @testset
using HTTP: get
import Pkg

# Function to create markdown and notebook files for specific file
function create_files(title, file; folder="")
    notebook_filename = first(splitext(file)) * ".ipynb"
    if folder != ""
        notebook_filename = "$folder/$notebook_filename"
    end
    
    binder_logo   = "https://mybinder.org/badge_logo.svg"
    nbviewer_logo = "https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg"
    download_logo = "https://camo.githubusercontent.com/aea75103f6d9f690a19cb0e17c06f984ab0f472d9e6fe4eadaa0cc438ba88ada/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f646f776e6c6f61642d6e6f7465626f6f6b2d627269676874677265656e"

    # TODO: Fix the links for tutorials in Trixi, not TrixiTutorials
    binder_url   = "https://mybinder.org/v2/gh/trixi-framework/TrixiTutorials/gh-pages?filepath=dev/notebooks/$notebook_filename"
    nbviewer_url = "https://nbviewer.jupyter.org/github/trixi-framework/TrixiTutorials/blob/gh-pages/dev/notebooks/$notebook_filename"
    download_url = "https://raw.githubusercontent.com/trixi-framework/TrixiTutorials/gh-pages/dev/notebooks/$notebook_filename"

    binder_badge   = "# [![]($binder_logo)]($binder_url)"
    nbviewer_badge = "# [![]($nbviewer_logo)]($nbviewer_url)"
    download_badge = "# [![]($download_logo)]($download_url)"
    
    # Generate notebooks
    function preprocess_notebook(content)
        return string("# # $title\n\n", preprocess_links(content))
    end
    Literate.notebook(joinpath(repo_src, folder, file), joinpath(notebooks_dir, folder); execute=false, preprocess=preprocess_notebook, credit=false)

    # Generate markdowns
    function preprocess_docs(content)
        return string("# # [$title](@id $(splitext(file)[1]))\n $binder_badge\n $nbviewer_badge\n $download_badge\n\n", preprocess_links(content))
    end
    Literate.markdown(joinpath(repo_src, folder, file), joinpath(pages_dir, folder); preprocess=preprocess_docs,)

    @testset "TrixiTutorials $title" begin include(joinpath(repo_src, folder, file)) end

end

# Creating tutorials for the following files:
# Normal structure: "title" => "filename.jl"
# If there are several files for one topic and folder, the structure is:
#   "title" => ["subtitle 1" => ("folder 1", "filename 1.jl"),
#               "subtitle 2" => ("folder 2", "filename 2.jl")]
files = [
    "Adding a new equation" => ["Scalar conservation law" => ("adding_new_equations_literate", "cubic_conservation_law_literate.jl"),
                                "Nonconservative equation" => ("adding_new_equations_literate", "nonconservative_advection_literate.jl")],
    "Differentiable programming" => "differentiable_programming_literate.jl",
    ]

repo_src        = joinpath(@__DIR__, "src", "files")

pages_dir       = joinpath(@__DIR__, "..", "src", "tutorials")
notebooks_dir   = joinpath(@__DIR__, "src", "notebooks")

Sys.rm(pages_dir;       recursive=true, force=true)
Sys.rm(notebooks_dir;   recursive=true, force=true)

# Preprocess files to add reference web links automatically.
# trixi_version = Pkg.dependencies()[Pkg.project().dependencies["Trixi"]].version
trixi_link = "https://trixi-framework.github.io/Trixi.jl/v0.3.61/"

# Function that replaces `@trixi-docs` and `@trixi-ref` with the links to the Trixi documentation
function preprocess_links(content)
    # Replacing `@trixi-docs:` in `content` with the defined `trixi_link`
    content = replace(content, "@trixi-docs:" => trixi_link)
    # Searching for `[`Example`](@trixi-ref)` in content and replace it with `[`Example`](trixi_link/reference_trixi/#Trixi.Example)`.
    content = replace(content, r"\[`(?<ref>\w+)`\]\(@trixi-ref\)" => SubstitutionString("[`\\g<ref>`]($(trixi_link)reference-trixi/#Trixi.\\g<ref>)"))
end

# Generate markdown for index.jl
Literate.markdown(joinpath(repo_src, "index.jl"), pages_dir; name="introduction_literate", preprocess=preprocess_links,)
# Navigation system for makedocs
pages = Any["Introduction" => "tutorials/introduction_literate.md",]
list = ["introduction_literate.md"]

# Create markdown and notebook files for tutorials.
for (i, (title, filename)) in enumerate(files)
    # Several files of one topic are created seperately and pushed to `pages` together.
    if filename isa Vector
        vector = []
        for j in eachindex(filename)
            create_files("$i.$j: $title: $(filename[j][1])", filename[j][2][2]; folder=filename[j][2][1])

            path = "$(filename[j][2][1])/$(splitext(filename[j][2][2])[1]).md"
            push!(vector, "$i.$j $(filename[j][1])" => "tutorials/$path")
            push!(list, path)
        end
        # Add to navigation menu
        push!(pages, ("$i $title" => vector))
    else # Single files
        create_files("$i: $title", filename)
        # Add to navigation menu
        path = "$(splitext(filename)[1]).md"
        push!(pages, ("$i $title" => "tutorials/$path"))
        push!(list, path)
    end
end

# Simple version of checking links
for file in list
    content = read(joinpath(pages_dir, file), String)
    if occursin(r"\(https://[^\(\)]+\)", content)
        matches = collect(eachmatch(r"\(https://[^\(\)]+\)", content))
        for i in 1:length(matches)
            link = string(chop(matches[i].match, head=1, tail=1))
            try 
                HTTP.get(link, retry=false, connect_timeout=15)
            catch
                if get(ENV, "CI", nothing) == "true"
                    error("URL doesn't exist: ", link)
                else
                    @warn "URL doesn't exist: " link
                end
            end
        end
    end
end

return pages
