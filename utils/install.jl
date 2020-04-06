#!/usr/bin/env julia

# Usage
#
# You can execute this file directly as long as the `julia` executable is
# in your PATH and if you are on a Linux/macOS system. If `julia` is not in
# your PATH or if you are on a Windows system, call Julia and explicitly
# provide this file as a command line parameter, e.g., `path/to/julia
# Trixi.jl/utils/install.jl`.

import Pkg

# Get Trixi root directory
trixi_root_dir = dirname(@__DIR__)

# Install Trixi dependencies
println("*"^80)
println("Installing dependencies for Trixi...")
Pkg.activate(trixi_root_dir)
Pkg.instantiate()

# Install Trixi2Img dependencies
println("*"^80)
println("Installing dependencies for Trixi2Img...")
Pkg.activate(joinpath(trixi_root_dir, "postprocessing", "pkg", "TrixiImg"))
Pkg.instantiate()

# Install Trixi2Vtk dependencies
println("*"^80)
println("Installing dependencies for Trixi2Vtk...")
Pkg.activate(joinpath(trixi_root_dir, "postprocessing", "pkg", "TrixiVtk"))
Pkg.instantiate()

println("*"^80)
println("Done.")
