using Documenter
using Literate
using Trixi

# Creating tutorials for these files:
files = [
    "First Example" => "first_example.jl",
    "Adding a new equation" => "adding_a_new_equation.jl",
    "Differentiable programming" => "differentiable_programming.jl"]

pages_dir       = joinpath(@__DIR__,"pages")
notebooks_dir   = joinpath(@__DIR__,"notebooks")

repo_src        = joinpath(@__DIR__,"files")

Sys.rm(pages_dir;       recursive=true, force=true)
Sys.rm(notebooks_dir;   recursive=true, force=true)

# Add index.md file as introduction to navigation menu
pages = ["Introduction" => "index.md"]

Literate.markdown(joinpath(repo_src,"index.jl"), pages_dir; name="index", documenter=false, execute=true, codefence="```julia" => "```")

binder_logo = "https://mybinder.org/badge_logo.svg"
nbviewer_logo = "https://img.shields.io/badge/show-nbviewer-579ACA.svg"

for (i, (title, filename)) in enumerate(files)
    tutorial_prefix = string("t$(i)_")
    tutorial_title = string("# # Tutorial ", i, ": ", title)
    tutorial_file = string(tutorial_prefix,splitext(filename)[1])
    notebook_filename = string(tutorial_file, ".ipynb")

    binder_url = joinpath("@__BINDER_ROOT_URL__", "notebooks", notebook_filename)
    binder_badge = string("# [![](",binder_logo,")](",binder_url,")")
    nbviewer_url = joinpath("@__NBVIEWER_ROOT_URL__","notebooks", notebook_filename)
    nbviewer_badge = string("# [![](",nbviewer_logo,")](",nbviewer_url,")")

    # Generate notebooks
    function preprocess_notebook(content)
        return string(tutorial_title, "\n\n", content)
    end
    Literate.notebook(joinpath(repo_src,filename), notebooks_dir; name=tutorial_file, documenter=false, execute=false, preprocess=preprocess_notebook)

    # Generate markdown
    function preprocess_docs(content)
        return string(tutorial_title, "\n", binder_badge, "\n", nbviewer_badge, "\n\n", content)
    end
    Literate.markdown(joinpath(repo_src,filename), pages_dir; name=tutorial_file, documenter=false, execute=true, codefence="```julia" => "```", preprocess=preprocess_docs)

    # Generate navigation menu entries
    ordered_title = string(i, " ", title)
    path_to_markdown_file = joinpath("pages",string(tutorial_file,".md"))
    push!(pages, (ordered_title=>path_to_markdown_file))
end
# Run notebook with `notebook(;dir=joinpath(@__DIR__, "docs/src/tutorials/src/notebooks"))`
