module TestExamplesP4estMesh2DBlockFV

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "p4est_2d_blockfv")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "BlockFV 2D (P4estMesh)" begin
#! format: noindent

@testset "Linear scalar advection" begin
#! format: noindent

@trixi_testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        l2=[0.03374473828931446],
                        linf=[0.047702534871517654])

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "elixir_advection_basic.jl (matches flat TreeMesh result)" begin
    # On a flat (uncurved) P4estMesh, BlockFV must reproduce the TreeMesh result
    # at the same total resolution (8 trees x 4 FV cells = 32 cells per direction,
    # cf. tree_2d_blockfv/elixir_advection_basic.jl with initial_refinement_level=5
    # and n_nodes=1, or equivalently TreeMesh's `l2=[0.017295205942012868]` test).
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        tspan=(0.0, 0.5),
                        l2=[0.01729520594201313],
                        linf=[0.02444847499806957])

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end
end # Linear scalar advection

@testset "Compressible Euler equations" begin
#! format: noindent

@trixi_testset "elixir_euler_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_convergence.jl"),
                        tspan=(0.0, 0.5),
                        l2=[0.007478883579676853,
                            0.01821425474595155,
                            0.018214254745951636,
                            0.05489564575130002],
                        linf=[0.015383851800672144,
                            0.031626224897317146,
                            0.031626224897316924,
                            0.08890916386160175])

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "elixir_euler_source_term_nonperiodic.jl" begin
    # Non-periodic (Dirichlet) boundaries on a curved mesh exercise the exact
    # boundary-face normal and midpoint computed in `create_cache` (as opposed
    # to the offset FV cell center), see `get_fv_boundary_normal`/
    # `get_fv_boundary_midpoint` in `blockfv_p4est_2d.jl`.
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_term_nonperiodic.jl"),
                        tspan=(0.0, 0.5),
                        l2=[0.005822138358153528,
                            0.012273490132437661,
                            0.012273490132437665,
                            0.04292680829402406],
                        linf=[0.015377806924300241,
                            0.02904404506694558,
                            0.029044045066946023,
                            0.08809487044331288])

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end
end # Compressible Euler equations
end # BlockFV 2D (P4estMesh)

end # module
