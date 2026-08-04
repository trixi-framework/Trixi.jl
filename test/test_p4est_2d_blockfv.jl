@testsnippet P4estMesh2DBlockFV begin
    EXAMPLES_DIR = joinpath(examples_dir(), "p4est_2d_blockfv")
end

@testitem "BlockFV 2D (P4estMesh): elixir_advection_basic.jl" setup=[
    Setup,
    P4estMesh2DBlockFV
] tags=[:p4est_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        l2=[0.03374473828931446],
                        linf=[0.047702534871517654])

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "BlockFV 2D (P4estMesh): elixir_advection_basic.jl (matches flat TreeMesh result)" setup=[
    Setup,
    P4estMesh2DBlockFV
] tags=[:p4est_part1] begin
    # On a flat (uncurved) P4estMesh, BlockFV must reproduce the TreeMesh result
    # at the same total resolution (8 trees x 2^1 refinements x 4 FV cells = 64 cells
    # per direction).
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        tspan=(0.0, 0.5),
                        l2=[0.01729520594201313],
                        linf=[0.02444847499806957])

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "BlockFV 2D (P4estMesh): elixir_advection_basic.jl with fewer n_nodes and higher refinement" setup=[
    Setup,
    P4estMesh2DBlockFV
] tags=[:p4est_part1] begin
    # Compute with more FV cells per macro element.
    # trees=(8,8), level=1, n_nodes=4 -> 16 elements × 4 = 64 FV cells per direction.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        n_nodes=4,
                        initial_refinement_level=1,
                        tspan=(0.0, 0.5))
    res1 = @inferred analysis_callback(sol)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)

    # Compute with fewer FV cells per macro element at higher mesh refinement.
    # trees=(8,8), level=2, n_nodes=2 -> 32 elements × 2 = 64 FV cells per direction.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        n_nodes=2,
                        initial_refinement_level=2,
                        tspan=(0.0, 0.5))
    res2 = @inferred analysis_callback(sol)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)

    # Both setups have exactly the same total number of FV cells.
    # On this mesh BlockFV must return the same errors (up to floating-point precision).
    @test res1.l2 ≈ res2.l2
    @test res1.linf ≈ res2.linf
end

@testitem "BlockFV 2D (P4estMesh): elixir_advection_unstructured_flag.jl" setup=[
    Setup,
    P4estMesh2DBlockFV
] tags=[:p4est_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_unstructured_flag.jl"),
                        tspan=(0.0, 0.5),
                        l2=[0.07100284030308462],
                        linf=[0.19410337900218544])

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "BlockFV 2D (P4estMesh): elixir_euler_convergence.jl" setup=[
    Setup,
    P4estMesh2DBlockFV
] tags=[:p4est_part1] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "BlockFV 2D (P4estMesh): elixir_euler_source_term_nonperiodic.jl" setup=[
    Setup,
    P4estMesh2DBlockFV
] tags=[:p4est_part1] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "BlockFV 2D (P4estMesh): elixir_euler_free_stream.jl" setup=[
    Setup,
    P4estMesh2DBlockFV
] tags=[:p4est_part1] begin
    # Free-stream preservation on a curved unstructured mesh:
    # a constant initial state must stay constant up to machine precision.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream.jl"),
                        tspan=(0.0, 0.5),
                        l2=[
                            2.201247136337667e-16,
                            1.083753952748301e-15,
                            1.3712015883493754e-15,
                            3.1945347710309807e-15
                        ],
                        linf=[
                            2.1094237467877974e-15,
                            1.6139867220488213e-14,
                            2.328692794151266e-14,
                            3.019806626980426e-14
                        ],
                        atol=1.0e-12)

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "BlockFV 2D (P4estMesh): elixir_euler_free_stream_hybrid_mesh.jl" setup=[
    Setup,
    P4estMesh2DBlockFV
] tags=[:p4est_part1] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "BlockFV 2D (P4estMesh): elixir_euler_NACA6412airfoil_mach2.jl" setup=[
    Setup,
    P4estMesh2DBlockFV
] tags=[:p4est_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_NACA6412airfoil_mach2.jl"),
                        tspan=(0.0, 0.05),
                        l2=[
                            0.1257912392758344,
                            0.23951632082414004,
                            0.12365990082471959,
                            0.5526135336754846
                        ],
                        linf=[
                            1.9012542584577492,
                            2.147231652310891,
                            1.8946722039151167,
                            7.5709758185412
                        ])

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "BlockFV 2D (P4estMesh): PlotData2DTriangulated" setup=[
    Setup,
    P4estMesh2DBlockFV
] tags=[:p4est_part1] begin
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
