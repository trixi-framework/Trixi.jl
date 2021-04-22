using Documenter
using Literate
using Trixi

# For title, preprocessing, ... look into `make.jl` of `gridap/Tutorials`
tutorial_file = "Start"
Literate.notebook(joinpath(@__DIR__,"files/index.jl"), joinpath(@__DIR__,"notebooks"); name=tutorial_file, execute=false)
Literate.markdown(joinpath(@__DIR__,"files/index.jl"), joinpath(@__DIR__,"src"); name=tutorial_file, codefence="```julia" => "```")

tutorial_file = "First example"
Literate.notebook(joinpath(@__DIR__,"files/first_example.jl"), joinpath(@__DIR__,"notebooks"); name=tutorial_file, execute=false)
Literate.markdown(joinpath(@__DIR__,"files/first_example.jl"), joinpath(@__DIR__,"src"); name=tutorial_file, codefence="```julia" => "```")

tutorial_file = "Adding a new equation"
Literate.notebook(joinpath(@__DIR__,"files/adding_a_new_equation.jl"), joinpath(@__DIR__,"notebooks"); name=tutorial_file, execute=false)
Literate.markdown(joinpath(@__DIR__,"files/adding_a_new_equation.jl"), joinpath(@__DIR__,"src"); name=tutorial_file, codefence="```julia" => "```")

tutorial_file = "Differentiable programming"
Literate.notebook(joinpath(@__DIR__,"files/differentiable_programming.jl"), joinpath(@__DIR__,"notebooks"); name=tutorial_file, execute=false)
Literate.markdown(joinpath(@__DIR__,"files/differentiable_programming.jl"), joinpath(@__DIR__,"src"); name=tutorial_file, codefence="```julia" => "```")

makedocs(sitename="My Documentation")# , root = joinpath(@__DIR__, "out", "markdowns"))

# Run notebook with `notebook(joinpath(@__DIR__, "docs/src/tutorials/notebooks))`
