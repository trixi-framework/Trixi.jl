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

@testitem "BlockFV 2D: elixir_advection_amr.jl with even n_nodes" setup=[Setup, TreeMesh2DBlockFV] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"),
                        l2=[2.23122321e-03],
                        linf=[2.73842937e-02],
                        tspan=(0.0, 0.5))

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "BlockFV 2D: elixir_advection_amr.jl with odd n_nodes" setup=[Setup, TreeMesh2DBlockFV] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"),
                        l2=[1.79519028e-03],
                        linf=[2.31327415e-02],
                        tspan=(0.0, 0.5),
                        n_nodes=5)

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "BlockFV 2D: elixir_euler_density_wave.jl" setup=[Setup, TreeMesh2DBlockFV] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_density_wave.jl"),
                        l2=[
                            0.003190908142060662,
                            0.00031909081420601837,
                            0.0006381816284120868,
                            7.977270355347174e-5
                        ],
                        linf=[
                            0.0045107823283880855,
                            0.00045107823283929704,
                            0.000902156465677928,
                            0.00011276955822125956
                        ],
                        tspan=(0.0, 0.05))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "BlockFV 2D: elixir_euler_vortex.jl" setup=[Setup, TreeMesh2DBlockFV] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_vortex.jl"),
                        l2=[
                            0.0009463095855559644,
                            0.034840444590943494,
                            0.03491933775722927,
                            0.09388027508058767
                        ],
                        linf=[
                            0.015226079023008654,
                            0.40423179950815646,
                            0.3963363807671082,
                            1.628386806835067
                        ],
                        tspan=(0.0, 1.0))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "BlockFV 2D: elixir_euler_convergence.jl" setup=[Setup, TreeMesh2DBlockFV] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_convergence.jl"),
                        l2=[
                            0.003798391701194131,
                            0.0094894678135065,
                            0.009489467813506488,
                            0.027041546309487817
                        ],
                        linf=[
                            0.005743846316061063,
                            0.013649501767585726,
                            0.013649501767585726,
                            0.03876289859195037
                        ],
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
                            0.0013980802560727824,
                            0.0027151888286537657,
                            0.002715188828653762,
                            0.008307473062557628
                        ],
                        linf=[
                            0.002824953578837608,
                            0.0058202665506430495,
                            0.0058202665506430495,
                            0.0161960768546221
                        ],
                        tspan=(0.0, 0.5))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "BlockFV 2D: elixir_euler_vortex_mortar.jl, BlockFV vs DGSEM with polydeg=0" setup=[
    Setup,
    TreeMesh2DBlockFV
] tags=[:tree_part1] begin
    # Compute with BlockFV solver.
    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar.jl"),
                  n_nodes = 4,
                  initial_refinement_level = 5,
                  tspan = (0.0, 0.5))
    res1 = @inferred analysis_callback(sol)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)

    # Compute with DGSEM solver with polynomial degree = 0, i.e., a first order finite volume solver.
    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar.jl"),
                  solver = DGSEM(polydeg = 0, surface_flux = flux_hllc),
                  initial_refinement_level = 7,
                  tspan = (0.0, 0.5))
    res2 = @inferred analysis_callback(sol)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    # TODO: Investigate why this allocation tests fails.
    # See https://github.com/trixi-framework/Trixi.jl/pull/3096 for more details.
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test_broken (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end

    # Both setups have exactly the same degrees of freedom.
    # Thus, they should return the same errors (up to floating-point precision).
    @test res1.l2 ≈ res2.l2
    @test res1.linf ≈ res2.linf
end
