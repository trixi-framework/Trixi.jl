module TestCUDA

using CUDA
using Test
using Trixi

include("test_trixi.jl")

# EXAMPLES_DIR = joinpath(examples_dir(), "dgmulti_1d")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

# TODO:

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive = true)

end # module
