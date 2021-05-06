using Documenter

pages = [
    "Introduction" => "index.md",
    "First Example" => "t1_first_example.md",
    "Adding a new equation" => "t2_adding_a_new_equation.md",
    "Differentiable programming" => "t3_differentiable_programming.md"]


makedocs(
    # root = "/home/benjamin/Dokumente/git/Trixi.jl/docs/src/tutorials/pages",
    repo = "https://github.com/bennibolm/Trixi.jl/tree/tutorials/docs/src/tutorials/files", # for the "link to source" feature. in github automatically
    # Specify modules for which docstrings should be shown
    modules = [Trixi],#, Trixi2Vtk, Trixi2Img],
    # Set sitename to Trixi
    sitename = "Trixi.jl",
    format = Documenter.HTML(
        # Disable pretty URLs during manual testing
        prettyurls = get(ENV, "CI", nothing) == "true",
        # Set canonical URL to GitHub pages URL
        canonical = "https://github.com/bennibolm/Trixi.jl"
    ),
    source = "pages",
    pages = pages
)