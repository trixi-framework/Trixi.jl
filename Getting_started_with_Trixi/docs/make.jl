using Documenter
using Tutorial

makedocs(
    sitename = "Tutorial",
    format = Documenter.HTML(),
    modules = [Tutorial]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
