module TestExamplesMPI

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
Trixi.mpi_isroot() && isdir(outdir) && rm(outdir, recursive=true)

# CI with MPI and 3D p4est fails often on Windows. Thus, we check whether this
# is the case here. We use GitHub actions, so we can check whether we run CI
# in the cloud with Windows as follows, see also
# https://docs.github.com/en/actions/learn-github-actions/environment-variables
CI_ON_WINDOWS = (get(ENV, "GITHUB_ACTIONS", false) == "true") && Sys.iswindows()

@testset "MPI" begin
  # TreeMesh tests
  include("test_mpi_tree.jl")

  # P4estMesh tests
  include("test_mpi_p4est_2d.jl")
  if CI_ON_WINDOWS # see comment on `CI_ON_WINDOWS` above
    include("test_mpi_p4est_3d_windows.jl")
  else
    include("test_mpi_p4est_3d.jl")
  end
end # MPI

# Clean up afterwards: delete Trixi output directory
Trixi.mpi_isroot() && @test_nowarn rm(outdir, recursive=true)

end # module
