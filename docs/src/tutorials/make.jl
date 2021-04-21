using Documenter
using Literate
using Trixi

# filename = first_example
# Literate.notebook(joinpath(repo_src,filename), notebooks_dir)
title = "first_example.jl"
tutorial_title = string("# # Tutorial ", 1, ": ", title)

# Generate notebooks
function preprocess_notebook(content)
  return string(tutorial_title, "\n\n", content)
end

# Generate markdown
function preprocess_docs(content)
  return string(tutorial_title, "\n\n", content)#"\n", binder_badge, "\n", nbviwer_badge, "\n\n", content)
end
Literate.notebook(joinpath(@__DIR__,"files/index.jl"), joinpath(@__DIR__,"notebooks"); preprocess=preprocess_notebook)
Literate.markdown(joinpath(@__DIR__,"files/index.jl"), joinpath(@__DIR__,"src"); preprocess=preprocess_docs, codefence="```julia" => "```")

Literate.notebook(joinpath(@__DIR__,"files/first_example.jl"), joinpath(@__DIR__,"notebooks"); preprocess=preprocess_notebook)
Literate.markdown(joinpath(@__DIR__,"files/first_example.jl"), joinpath(@__DIR__,"src"); preprocess=preprocess_docs, codefence="```julia" => "```")

# Literate.notebook(joinpath(@__DIR__,"adding_a_new_equation.jl"), joinpath(@__DIR__,"notebooks"); preprocess=preprocess_notebook)
# Literate.markdown(joinpath(@__DIR__,"adding_a_new_equation.jl"), joinpath(@__DIR__,"markdowns"); preprocess=preprocess_docs, codefence="```julia" => "```")

makedocs(sitename="My Documentation")# , root = joinpath(@__DIR__, "out", "markdowns"))
