using Documenter
using Literate
using Trixi

files = [
    "First Example"=>"first_example.jl",
    "Adding a new equation"=>"adding_a_new_equation.jl",
    "Differentiable programming"=>"differentiable_programming.jl"]


pages_dir = joinpath(@__DIR__,"src","pages")
notebooks_dir = joinpath(@__DIR__,"src","notebooks")

repo_src = joinpath(@__DIR__,"src","files")

Sys.rm(pages_dir;recursive=true,force=true)
Sys.rm(notebooks_dir;recursive=true,force=true)

# Add index.md file as introduction to navigation menu
pages = ["Introduction"=> "index.md"]

Literate.markdown(joinpath(repo_src,"index.jl"), joinpath(@__DIR__,"src"); name="index", codefence="```julia" => "```")# preprocess=preprocess_docs)

for (i, (title, filename)) in enumerate(files)
    tutorial_prefix = string("t$(i)_")
    tutorial_title = string("# # Tutorial ", i, ": ", title)
    tutorial_file = string(tutorial_prefix,splitext(filename)[1])
    notebook_filename = string(tutorial_file, ".ipynb")

    # For title, preprocessing, ... look into `make.jl` of `gridap/Tutorials`

    ## Generate notebooks
    #function preprocess_notebook(content)
    #    return string(tutorial_title, "\n\n", content)
    #end
    Literate.notebook(joinpath(repo_src,filename), notebooks_dir; name=tutorial_file, documenter=false, execute=false)#, preprocess=preprocess_notebook)

    ## Generate markdown
    # function preprocess_docs(content)
    #     return string(tutorial_title, "\n", binder_badge, "\n", nbviwer_badge, "\n\n", content)
    # end
    Literate.markdown(joinpath(repo_src,filename), pages_dir; name=tutorial_file, codefence="```julia" => "```")# preprocess=preprocess_docs)

    # Generate navigation menu entries
    ordered_title = string(i, " ", title)
    path_to_markdown_file = joinpath("pages",string(tutorial_file,".md"))
    push!(pages, (ordered_title=>path_to_markdown_file))
end

# makedocs(
#     # Specify modules for which docstrings should be shown
#     # modules = [Trixi],#, Trixi2Vtk, Trixi2Img],
#     # Set sitename to Trixi
#     sitename = "Trixi.jl",
#     format = Documenter.HTML(
#         # Disable pretty URLs during manual testing
#         prettyurls = get(ENV, "CI", nothing) == "true",
#         # Set canonical URL to GitHub pages URL
#         # canonical = "https://trixi-framework.github.io/Trixi.jl/stable"
#     ),
#     pages = pages
# )
# Run notebook with `notebook(;dir=joinpath(@__DIR__, "docs/src/tutorials/src/notebooks"))`
