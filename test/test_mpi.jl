module TestExamplesMPI

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
Trixi.mpi_isroot() && isdir(outdir) && rm(outdir, recursive=true)

@testset "MPI" begin
  # TreeMesh tests
  include("test_mpi_tree.jl")

  # P4estMesh tests
  include("test_mpi_p4est_2d.jl")
  include("test_mpi_p4est_3d.jl")
end # MPI

# Clean up afterwards: delete Trixi output directory
Trixi.mpi_isroot() && @test_nowarn rm(outdir, recursive=true)

end # module
