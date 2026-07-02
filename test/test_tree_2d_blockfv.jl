@testsnippet TreeMesh2DBlockFV begin
    EXAMPLES_DIR = joinpath(examples_dir(), "tree_2d_blockfv")
end

@testitem "BlockFV 2D: elixir_advection_basic.jl" setup=[Setup, TreeMesh2DBlockFV] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        l2=[0.017295205942012868],
                        linf=[0.02444847499806624],
                        tspan=(0.0, 0.5))

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "BlockFV 2D: elixir_advection_basic.jl with less n_nodes and higher refinement" setup=[
    Setup,
    TreeMesh2DBlockFV
] tags=[:tree_part1] begin
    # Compute with more volumes per macro cell.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        n_nodes=4,
                        initial_refinement_level=4,
                        tspan=(0.0, 0.5))
    res1 = @inferred analysis_callback(sol)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)

    # Compute with fewer volumes per macro cell.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        n_nodes=2,
                        initial_refinement_level=5,
                        tspan=(0.0, 0.5))
    res2 = @inferred analysis_callback(sol)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)

    # Both setups have exactly the same degrees of freedom.
    # Thus, they should return the same errors (up to floating-point precision).
    @test res1.l2 ≈ res2.l2
    @test res1.linf ≈ res2.linf
end

@testitem "BlockFV 2D: elixir_euler_density_wave.jl" setup=[Setup, TreeMesh2DBlockFV] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_density_wave.jl"),
                        l2=[0.031233316749041267,
                            0.003123331674903803,
                            0.006246663349808052,
                            0.0007808329187371395],
                        linf=[0.044169344994492266,
                            0.0044169344994492215,
                            0.008833868998898514,
                            0.0011042336248863194],
                        tspan=(0.0, 0.5))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "BlockFV 2D: elixir_euler_vortex.jl" setup=[Setup, TreeMesh2DBlockFV] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_vortex.jl"),
                        l2=[0.0009462760556996494,
                            0.034845346890640956,
                            0.0349234255730328,
                            0.09387847561186147],
                        linf=[0.01522697023057773,
                            0.40428197961893275,
                            0.39638850053862995,
                            1.628539546658537],
                        tspan=(0.0, 1.0))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "BlockFV 2D: elixir_euler_convergence.jl" setup=[Setup, TreeMesh2DBlockFV] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_convergence.jl"),
                        l2=[0.003798391701194144,
                            0.009489467813506548,
                            0.00948946781350655,
                            0.02704154630948781],
                        linf=[0.005743846316061285,
                            0.013649501767585503,
                            0.013649501767585726,
                            0.03876289859195037],
                        tspan=(0.0, 0.5))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "BlockFV 2D: elixir_euler_source_term_nonperiodic.jl" setup=[
    Setup,
    TreeMesh2DBlockFV
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_term_nonperiodic.jl"),
                        l2=[
                            0.0013980788738505803,
                            0.0027151896203078626,
                            0.0027151896203078817,
                            0.008307485477336464
                        ],
                        linf=[
                            0.0028249606444796793,
                            0.005820266937670571,
                            0.005820266937670571,
                            0.016196092853339117
                        ],
                        tspan=(0.0, 0.5))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "BlockFV 2D: elixir_euler_vortex_mortar.jl with blockfv vs with dgsem with polydeg=0" setup=[
    Setup,
    TreeMesh2DBlockFV
] tags=[:tree_part1] begin
    # We explicitly pass a time step size `dt` and set the `stepsize_callback` to `nothing`
    # to avoid subtle differences coming from different time step size evaluations in the
    # two runs: The `DGSEM` solver computes the wave speeds in all directions and takes the
    # maximum over all cells, while the `BlockFV` solver first takes a maximum of the wave speeds
    # separately in each macro-cell before combining them, leading to slightly different results.

    # Compute with BlockFV solver.
    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar.jl"),
                  n_nodes = 4,
                  initial_refinement_level = 5,
                  tspan = (0.0, 0.5),
                  dt = 2.0e-3,
                  stepsize_callback = nothing)
    res1 = @inferred analysis_callback(sol)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)

    # Compute with DGSEM solver with polynomial degree = 0, i.e., a first order finite volume solver.
    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar.jl"),
                  solver = DGSEM(polydeg = 0, surface_flux = flux_hllc),
                  initial_refinement_level = 7,
                  tspan = (0.0, 0.5),
                  dt = 2.0e-3,
                  stepsize_callback = nothing)
    res2 = @inferred analysis_callback(sol)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)

    # Both setups have exactly the same degrees of freedom.
    # Thus, they should return the same errors (up to floating-point precision).
    @test res1.l2 ≈ res2.l2
    @test res1.linf ≈ res2.linf
end
