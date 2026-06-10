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
        include("test_mpi_p4est_parabolic_2d.jl")
    end
end # MPI

@trixi_testset "MPI supporting functionality" begin
    using Trixi: Trixi, ode_norm, SVector
    t = 0.5
    let u = 1.0
        @test ode_norm(u, t) ≈ Trixi.DiffEqBase.ODE_DEFAULT_NORM(u, t)
    end
    let u = [1.0, -2.0]
        @test ode_norm(u, t) ≈ Trixi.DiffEqBase.ODE_DEFAULT_NORM(u, t)
    end
    let u = [SVector(1.0, -2.0), SVector(0.5, -0.1)]
        @test ode_norm(u, t) ≈ Trixi.DiffEqBase.ODE_DEFAULT_NORM(u, t)
    end
end # MPI supporting functionality

@trixi_testset "TrixiStateVector MPI norm and dot" begin
    using Trixi: Trixi, TrixiStateVector, mpi_comm, mpi_rank
    using LinearAlgebra: norm, dot

    # Each rank contributes rank-dependent data so that the global reduction is
    # non-trivial and differs from any single-rank local result.
    rank = mpi_rank()
    v = TrixiStateVector(Float64[rank + 1, rank + 2, rank + 3])
    w = TrixiStateVector(ones(Float64, 3))

    # Expected global norm: sqrt of sum of abs2 across all ranks.
    local_sumabs2 = sum(abs2, v.data)
    expected_norm = sqrt(Trixi.MPI.Allreduce(local_sumabs2, +, mpi_comm()))
    @test norm(v) ≈ expected_norm

    # Expected global dot product: sum of local dot products across all ranks.
    local_dot = dot(v.data, w.data)
    expected_dot = Trixi.MPI.Allreduce(local_dot, +, mpi_comm())
    @test dot(v, w) ≈ expected_dot

    # ode_norm dispatches to the AbstractArray method, which already performs
    # an MPI.Allreduce internally; verify it is callable and positive.
    @test ode_norm(v, 0.0) > 0
end # TrixiStateVector MPI norm and dot

# Clean up afterwards: delete Trixi.jl output directory
Trixi.mpi_isroot() && @test_nowarn rm(outdir, recursive = true)
Trixi.MPI.Barrier(Trixi.mpi_comm())

end # module
