# Add all relevant paths to LOAD_PATH

# Trixi
push!(LOAD_PATH, "..")

# Trixi2Img
push!(LOAD_PATH, joinpath("..", "postprocessing", "pkg", "Trixi2Img"))

# Trixi2Vtk
push!(LOAD_PATH, joinpath("..", "postprocessing", "pkg", "Trixi2Vtk"))


# Load required modules
using Documenter, Trixi, Trixi2Img, Trixi2Vtk

# Make documentation
makedocs(sitename="Trixi")
