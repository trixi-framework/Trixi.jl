module TestExamplesMPI

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
Trixi.mpi_isroot() && isdir(outdir) && rm(outdir, recursive = true)
Trixi.MPI.Barrier(Trixi.mpi_comm())

# CI with MPI and some tests fails often on Windows. Thus, we check whether this
# is the case here. We use GitHub Actions, so we can check whether we run CI
# in the cloud with Windows as follows, see also
# https://docs.github.com/en/actions/learn-github-actions/environment-variables
CI_ON_WINDOWS = (get(ENV, "GITHUB_ACTIONS", false) == "true") && Sys.iswindows()

@testset "MPI" begin
    # TreeMesh tests
    include("test_mpi_tree.jl")

    # P4estMesh and T8codeMesh tests
    include("test_mpi_p4est_2d.jl")
    include("test_mpi_t8code_2d.jl")
    if !CI_ON_WINDOWS # see comment on `CI_ON_WINDOWS` above
        include("test_mpi_p4est_3d.jl")
        include("test_mpi_t8code_3d.jl")
    end
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
Trixi.mpi_isroot() && @test_nowarn rm(outdir, recursive = true)
Trixi.MPI.Barrier(Trixi.mpi_comm())

end # module
