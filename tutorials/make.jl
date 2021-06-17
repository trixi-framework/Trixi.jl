using Documenter
using Literate
import Pkg
using Trixi

# Creating tutorials for these files:
files = [
    "Adding a new equation" => "adding_a_new_equation.jl",
    "Differentiable programming" => "differentiable_programming.jl",
    "Testing" => "testing_repository.jl",
    ]

pages_dir       = joinpath(@__DIR__,"pages")
notebooks_dir   = joinpath(@__DIR__,"notebooks")

repo_src        = joinpath(@__DIR__,"files")

Sys.rm(pages_dir;       recursive=true, force=true)
Sys.rm(notebooks_dir;   recursive=true, force=true)

trixi_version = Pkg.dependencies()[Pkg.project().dependencies["Trixi"]].version
trixi_link = "https://trixi-framework.github.io/Trixi.jl/v$trixi_version/"
function preprocess_links(content)
    content = replace(content, "@trixi-docs:" => trixi_link)
end

binder_logo = "https://mybinder.org/badge_logo.svg"
nbviewer_logo = "https://img.shields.io/badge/show-nbviewer-579ACA.svg"

# binder_url = joinpath("@__BINDER_ROOT_URL__","notebooks")
# nbviewer_url = joinpath("@__NBVIEWER_ROOT_URL__","notebooks")

binder_url = joinpath("https://mybinder.org/v2/gh/bennibolm/Trixi.jl/tutorials?filepath=tutorials/notebooks/")
nbviewer_url = joinpath("https://nbviewer.jupyter.org/github/bennibolm/Trixi.jl/tree/tutorials/tutorials/notebooks/")

binder_badge = string("# [![](", binder_logo, ")](", binder_url, ")")
nbviewer_badge = string("# [![](", nbviewer_logo, ")](", nbviewer_url, ")")

# Introduction file (index.jl)
# Add to navigation menu
pages = ["Introduction" => "index.md"]
# Generate markdown
function preprocess_docs(content)
    return string("# ## Welcome to the tutorials of Trixi.jl", "\n", binder_badge, "\n", nbviewer_badge, "\n\n",
                  preprocess_links(content))
end
Literate.markdown(joinpath(repo_src,"index.jl"), pages_dir; name="index", documenter=false,
                  execute=true, preprocess=preprocess_docs,)
# TODO: With `documenter=false` there is no `link to source` in html file. With `true` the link is not defined because of some `<unkown>`.


for (i, (title, filename)) in enumerate(files)
    tutorial_title = string("# # Tutorial ", i, ": ", title)
    tutorial_file = string(splitext(filename)[1])
    notebook_filename = string(tutorial_file, ".ipynb")
    
    binder_badge_ = string("# [![](", binder_logo, ")](", joinpath(binder_url, notebook_filename), ")")
    nbviewer_badge_ = string("# [![](", nbviewer_logo, ")](", joinpath(nbviewer_url, notebook_filename), ")")
    
    # Generate notebooks
    function preprocess_notebook(content)
        return string(tutorial_title, "\n\n", preprocess_links(content))
    end

    Literate.notebook(joinpath(repo_src,filename), notebooks_dir; name=tutorial_file,
                      execute=true, preprocess=preprocess_notebook)

    # Generate markdown
    function preprocess_docs(content)
        return string(tutorial_title, "\n", binder_badge_, "\n", nbviewer_badge_, "\n\n",
                      preprocess_links(content))
    end
    Literate.markdown(joinpath(repo_src,filename), pages_dir; name=tutorial_file, documenter=false,
                      execute=true, preprocess=preprocess_docs,)

    # Add to navigation menu
    path_to_markdown_file = string(tutorial_file,".md")
    push!(pages, (title=>path_to_markdown_file))
end

Sys.cp(joinpath(@__DIR__, "assets"), joinpath(@__DIR__, "pages/assets"), force=true)

makedocs(
    # Set sitename to Trixi
    sitename="Trixi.jl - Tutorials",
    # # Provide additional formatting options
    format = Documenter.HTML(
        # Disable pretty URLs during manual testing
        prettyurls = get(ENV, "CI", nothing) == "true",
        # Explicitly add favicon as asset
        # assets = ["assets/favicon.ico"],
        # Set canonical URL to GitHub pages URL
        canonical = "https://bennibolm.github.io/Trixi.jl/stable"
    ),
    # Explicitly specify documentation structure
    source  = "pages",
    pages = pages,
    strict = true # to make the GitHub action fail when doctests fail, see https://github.com/neuropsychology/Psycho.jl/issues/34
)

# deploydocs(
#     repo = "github.com/bennibolm/Trixi.jl.git",
#     push_preview = true,
#     deploy_config = deployconfig,
# )
