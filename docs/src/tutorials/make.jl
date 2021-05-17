using Literate
using Trixi

# Creating tutorials for these files:
files = [
    "Adding a new equation" => "adding_a_new_equation.jl",
    "Differentiable programming" => "differentiable_programming.jl",
    "Testing" => "testing_repository.jl",
    ]
pages_dir       = joinpath(@__DIR__,"pages")
notebooks_dir   = joinpath(@__DIR__,"../../../binder")

repo_src        = joinpath(@__DIR__,"files")

Sys.rm(pages_dir;       recursive=true, force=true)
Sys.rm(notebooks_dir;   recursive=true, force=true)

# Add index.md file as introduction to navigation menu
pages = ["Introduction" => "index.md"]

binder_logo = "https://mybinder.org/badge_logo.svg"
nbviewer_logo = "https://img.shields.io/badge/show-nbviewer-579ACA.svg"
binder_url = "https://mybinder.org/v2/gh/bennibolm/Trixi.jl/tutorials?filepath=binder/" # @__BINDER_ROOT_URL__
nbviewer_url = "https://nbviewer.jupyter.org/github/bennibolm/Trixi.jl/tree/tutorials/binder/" # @__NBVIEWER_ROOT_URL__


# Generate markdown
binder_badge = string("# [![](", binder_logo, ")](", binder_url, ")")
nbviewer_badge = string("# [![](", nbviewer_logo, ")](", nbviewer_url, ")")
function preprocess_docs(content)
    return string("# ## Welcome to the tutorials of Trixi.jl", "\n", binder_badge, "\n", nbviewer_badge, "\n\n", content)
end
Literate.markdown(joinpath(repo_src,"index.jl"), pages_dir; name="index", documenter=false, execute=true, codefence="```julia" => "```", preprocess=preprocess_docs)
# TODO: With `documenter=false` there is no `link to source` in html file. With `true` the link is not defined because of some `<unkown>`.

for (i, (title, filename)) in enumerate(files)
    tutorial_title = string("# # Tutorial ", i, ": ", title)
    tutorial_file = string(splitext(filename)[1])
    notebook_filename = string(tutorial_file, ".ipynb")
    
    binder_badge_ = string("# [![](", binder_logo, ")](", joinpath(binder_url, notebook_filename), ")")
    nbviewer_badge_ = string("# [![](", nbviewer_logo, ")](", joinpath(nbviewer_url, notebook_filename), ")")
    
    # Generate notebooks
    function preprocess_notebook(content)
        return string(tutorial_title, "\n\n", content)
    end
    # TODO: With `documenter=true` no references are written to notebook file. With `documenter=false` there are existing but not working right now.
    Literate.notebook(joinpath(repo_src,filename), notebooks_dir; name=tutorial_file, documenter=false, execute=true, preprocess=preprocess_notebook)

    # Generate markdown
    function preprocess_docs(content)
        return string(tutorial_title, "\n", binder_badge_, "\n", nbviewer_badge_, "\n\n", content)
    end
    Literate.markdown(joinpath(repo_src,filename), pages_dir; name=tutorial_file, documenter=false, execute=true, codefence="```julia" => "```", preprocess=preprocess_docs)

    # Generate navigation menu entries
    ordered_title = string(i, " ", title)
    path_to_markdown_file = joinpath("pages",string(tutorial_file,".md"))
    push!(pages, (ordered_title=>path_to_markdown_file))
end
