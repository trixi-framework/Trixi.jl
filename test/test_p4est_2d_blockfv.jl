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
    # at the same total resolution (8 trees x 2^1 refinements x 4 FV cells = 64 cells
    # per direction).
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        tspan=(0.0, 0.5),
                        l2=[0.01729520594201313],
                        linf=[0.02444847499806957])

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "elixir_advection_basic.jl with fewer n_nodes and higher refinement" begin
    # Compute with more FV cells per macro element.
    # trees=(8,8), level=1, n_nodes=4 -> 16 elements × 4 = 64 FV cells per direction.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        n_nodes=4,
                        initial_refinement_level=1,
                        tspan=(0.0, 0.5))
    res1 = @inferred analysis_callback(sol)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)

    # Compute with fewer FV cells per macro element at higher mesh refinement.
    # trees=(8,8), level=2, n_nodes=2 -> 32 elements × 2 = 64 FV cells per direction.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        n_nodes=2,
                        initial_refinement_level=2,
                        tspan=(0.0, 0.5))
    res2 = @inferred analysis_callback(sol)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)

    # Both setups have exactly the same total number of FV cells.
    # On this mesh BlockFV must return the same errors (up to floating-point precision).
    @test res1.l2 ≈ res2.l2
    @test res1.linf ≈ res2.linf
end

@trixi_testset "elixir_advection_unstructured_flag.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_unstructured_flag.jl"),
                        l2=[0.03679333289248872],
                        linf=[0.14531183424293137])

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

@trixi_testset "elixir_euler_free_stream.jl" begin
    # Free-stream preservation on a curved unstructured mesh:
    # a constant initial state must stay constant up to machine precision.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream.jl"),
                        tspan=(0.0, 0.5),
                        l2=[
                            1.9312342788165835e-16,
                            1.1102787289041371e-15,
                            1.3624908968852466e-15,
                            2.7423460152751346e-15
                        ],
                        linf=[
                            2.220446049250313e-15,
                            1.6181500583911657e-14,
                            2.3453461395206432e-14,
                            2.842170943040401e-14
                        ],
                        atol=1.0e-12)

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "elixir_euler_free_stream_hybrid_mesh.jl" begin
    # Free-stream preservation on a hybrid mesh (mixed first-order and second-order
    # quadrilateral elements).
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_free_stream_hybrid_mesh.jl"),
                        tspan=(0.0, 0.5),
                        l2=[
                            1.0449504481826722e-17,
                            1.1935127706587904e-16,
                            1.2657920453908178e-16,
                            9.750392607336836e-17
                        ],
                        linf=[
                            1.1102230246251565e-16,
                            3.608224830031759e-16,
                            4.163336342344337e-16,
                            1.7763568394002505e-15
                        ],
                        atol=1.0e-12)

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end
@trixi_testset "elixir_euler_NACA6412airfoil_mach2.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_NACA6412airfoil_mach2.jl"),
                        tspan=(0.0, 0.05),
                        l2=[
                            0.12535465649604935,
                            0.2387836884109298,
                            0.12319269120809999,
                            0.5506659100710758,
                        ],
                        linf=[
                            1.8860955840906906,
                            2.1495129396086883,
                            1.890312312939321,
                            7.526922464267882,
                        ])

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end
end # Compressible Euler equations

@testset "Visualization" begin
#! format: noindent

@trixi_testset "PlotData2DTriangulated for BlockFV on P4estMesh" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        tspan=(0.0, 0.1))

    # PlotData2D must return PlotData2DTriangulated
    # for BlockFV on P4estMesh.
    pd = @test_nowarn PlotData2D(sol)
    @test pd isa Trixi.PlotData2DTriangulated

    # Variable lookup by name
    @test pd["scalar"] == Trixi.PlotDataSeries(pd, 1)

    # show must not error
    @trixi_test_nowarn show(stdout, pd)
    println(stdout)
end
end # Visualization
end # BlockFV 2D (P4estMesh)

end # module
