module TestExamplesKernelAbstractions

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = examples_dir()

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
Trixi.mpi_isroot() && isdir(outdir) && rm(outdir, recursive = true)
Trixi.MPI.Barrier(Trixi.mpi_comm())

@testset "Threaded tests" begin
#! format: noindent

@testset "basic" begin
    @test Trixi._PREFERENCE_THREADING == :kernelabstractions
end
end

# Clean up afterwards: delete Trixi.jl output directory
Trixi.mpi_isroot() && isdir(outdir) && @test_nowarn rm(outdir, recursive = true)
Trixi.MPI.Barrier(Trixi.mpi_comm())

end # module
