module TestExamplesMPI

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
Trixi.mpi_isroot() && isdir(outdir) && rm(outdir, recursive=true)

@testset "MPI" begin
  # TreeMesh tests
  include("test_mpi_tree.jl")

  # P4estMesh tests
  include("test_mpi_p4est_2d.jl")
  include("test_mpi_p4est_3d.jl")
end # MPI


@trixi_testset "MPI supporting functionality" begin
  using OrdinaryDiffEq

  t = 0.5
  let u = 1.0
    @test ode_norm(u, t) ≈ OrdinaryDiffEq.ODE_DEFAULT_NORM(u, t)
  end
  let u = [1.0, -2.0]
    @test ode_norm(u, t) ≈ OrdinaryDiffEq.ODE_DEFAULT_NORM(u, t)
  end
  let u = [SVector(1.0, -2.0), SVector(0.5, -0.1)]
    @test ode_norm(u, t) ≈ OrdinaryDiffEq.ODE_DEFAULT_NORM(u, t)
  end
end # MPI supporting functionality

# Clean up afterwards: delete Trixi.jl output directory
Trixi.mpi_isroot() && @test_nowarn rm(outdir, recursive=true)

end # module
