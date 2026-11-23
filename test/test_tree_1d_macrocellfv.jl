@testsnippet TreeMesh1DMacroCellFV begin
    EXAMPLES_DIR = joinpath(examples_dir(), "tree_1d_macrocellfv")

    # The kernels do not allocate. With more than one thread, Polyester.jl's `@batch`
    # allocates a fixed amount for each of the two `@threaded` loops of the RHS
    # (independent of the number of threads and elements; DGSEM allocates a similar
    # amount for its loops), so threaded runs get a higher limit.
    ALLOCATION_LIMIT = Threads.nthreads() == 1 ? 1000 : 2000

    # L2 error of the first variable against the exact subcell averages (the errors of
    # the `AnalysisCallback` compare the averages with point values, which is only
    # second-order accurate)
    function subcell_average_error(sol, semi)
        u = Trixi.wrap_array_native(sol.u[end], semi)
        u_exact = Trixi.wrap_array_native(Trixi.compute_coefficients(sol.t[end], semi),
                                          semi)
        difference = selectdim(u, 1, 1) .- selectdim(u_exact, 1, 1)
        return sqrt(sum(abs2, difference) / length(difference))
    end

    # The same for `sol` and `semi` defined in `mod` by a preceding `trixi_include`
    function subcell_average_error(mod::Module)
        sol = Base.invokelatest(getglobal, mod, :sol)
        semi = Base.invokelatest(getglobal, mod, :semi)
        return Base.invokelatest(subcell_average_error, sol, semi)
    end
end

@testitem "MacroCellFV 1D: elixir_advection_gaussian.jl" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_gaussian.jl"),
                        l2=[0.12586020153550745],
                        linf=[0.3726415651944912],
                        tspan=(0.0, 0.5))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, ALLOCATION_LIMIT)
end

@testitem "MacroCellFV 1D: ReconstructionConstant agrees with BlockFV" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # With identity moment inversion and piecewise constant reconstruction, MacroCellFV
    # performs the same operations as BlockFV.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_gaussian.jl"),
                        tspan=(0.0, 0.1))
    u_ode = compute_coefficients(0.0, semi)
    du_macrocell = similar(u_ode)
    Trixi.rhs_hyperbolic!(du_macrocell, u_ode, semi, 0.0)

    semi_blockfv = SemidiscretizationHyperbolic(mesh, equations, initial_condition,
                                                BlockFV(n_nodes = n_cells,
                                                        surface_flux = flux_lax_friedrichs);
                                                boundary_conditions = boundary_condition_periodic)
    du_blockfv = similar(u_ode)
    Trixi.rhs_hyperbolic!(du_blockfv, u_ode, semi_blockfv, 0.0)
    @test du_macrocell == du_blockfv
end

@testitem "MacroCellFV 1D: elixir_advection_sine.jl" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_sine.jl"),
                        l2=[0.0008971312291256202],
                        linf=[0.0012687102434813802],
                        tspan=(0.0, 0.25))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, ALLOCATION_LIMIT)
end

@testitem "MacroCellFV 1D: elixir_advection_sine.jl with minmod" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_sine.jl"),
                        l2=[0.007633348149705237],
                        linf=[0.02173487014838904],
                        slope_limiter=minmod,
                        tspan=(0.0, 0.25))
end

@testitem "MacroCellFV 1D: elixir_advection_sine.jl convergence" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # Second-order convergence of MUSCL with unlimited central slopes
    eocs, errors = convergence_test(joinpath(EXAMPLES_DIR,
                                             "elixir_advection_sine.jl"),
                                    3, initial_refinement_level = 2)
    @test all(eocs[:l2] .> 1.9)
    @test all(eocs[:linf] .> 1.9)
end

@testitem "MacroCellFV 1D: constant state and block decomposition" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # A constant state is preserved exactly
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_sine.jl"),
                        initial_condition=initial_condition_constant,
                        tspan=(0.0, 0.1))
    du_ode = similar(sol.u[end])
    Trixi.rhs_hyperbolic!(du_ode, sol.u[end], semi, 0.0)
    @test maximum(abs, du_ode) == 0
    @test maximum(abs, sol.u[end] .- sol.u[1]) < 1.0e-14

    # The same 64 global subcells split into different numbers of blocks give the same
    # solution up to round-off errors.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_sine.jl"),
                        slope_limiter=minmod, n_cells=32,
                        initial_refinement_level=1, tspan=(0.0, 0.1))
    u_2blocks = copy(sol.u[end])
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_sine.jl"),
                        slope_limiter=minmod, n_cells=4,
                        initial_refinement_level=4, tspan=(0.0, 0.1))
    @test isapprox(u_2blocks, sol.u[end]; atol = 1.0e-14, rtol = 0)
end

@testitem "MacroCellFV 1D: elixir_euler_sod.jl" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sod.jl"),
                        l2=[0.15705319306155355, 0.16981100736981203, 0.450050048499648],
                        linf=[0.537210827922814, 0.3971661065734396, 1.4838542544932765],
                        tspan=(0.0, 0.1))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, ALLOCATION_LIMIT)

    # Mass and energy are conserved (the waves have not reached the transmissive
    # boundaries yet); the momentum changes by the boundary pressure forces.
    integrals_0 = Trixi.integrate(sol.u[1], semi; normalize = false)
    integrals = Trixi.integrate(sol.u[end], semi; normalize = false)
    @test isapprox(integrals[1], integrals_0[1]; atol = 1.0e-14)
    @test isapprox(integrals[2] - integrals_0[2], (1 - 0.1) * 0.1; atol = 1.0e-14)
    @test isapprox(integrals[3], integrals_0[3]; atol = 1.0e-14)
end

@testitem "MacroCellFV 1D: elixir_euler_sod.jl with HLLC and van Leer" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sod.jl"),
                        l2=[0.15963946970756976, 0.1718073269196492, 0.45428878543978385],
                        linf=[0.5578053992129199, 0.39701672781070857, 1.5258586868699129],
                        surface_flux=flux_hllc, slope_limiter=vanleer,
                        tspan=(0.0, 0.1))
end

@testitem "MacroCellFV 1D: types" setup=[Setup] tags=[:tree_part1] begin
    solver = MacroCellFV(n_cells = 4, surface_flux = flux_lax_friedrichs,
                         reconstruction = ReconstructionMUSCL(vanleer))
    @test solver isa MacroCellFV
    @test solver isa BlockFV
    @test solver.volume_integral isa VolumeIntegralMacroCellFV
    @test !(BlockFV(n_nodes = 4, surface_flux = flux_lax_friedrichs) isa MacroCellFV)
    @test Trixi.nnodes(solver) == 4
    @test Trixi.nhalo(solver) == 2
    # `n_nodes` is accepted for consistency with `BlockFV`
    solver_nodes = MacroCellFV(n_nodes = 4, surface_flux = flux_lax_friedrichs,
                               reconstruction = ReconstructionMUSCL(vanleer))
    @test typeof(solver_nodes) == typeof(solver)
    @test Trixi.nnodes(solver_nodes) == 4
    @test_throws ArgumentError MacroCellFV(surface_flux = flux_lax_friedrichs)
    @test_throws ArgumentError MacroCellFV(n_cells = 4, n_nodes = 4,
                                           surface_flux = flux_lax_friedrichs)
    @test ReconstructionMUSCL().slope_limiter === minmod
    @test Trixi.nhalo(MacroCellFV(n_cells = 4, surface_flux = flux_lax_friedrichs,
                                  reconstruction = ReconstructionConstant())) == 1
    # The identity moment inversion adds no halo cells
    @test Trixi.nhalo(MacroCellFV(n_cells = 4, surface_flux = flux_lax_friedrichs,
                                  moment_inversion = MomentInversionIdentity())) == 2

    @test_throws ArgumentError MacroCellFV(n_cells = 4,
                                           surface_flux = flux_lax_friedrichs,
                                           initial_condition_nodes = 0)
    # The halos are copied from the direct neighbors, so the halo width is limited by
    # the number of subcells per macro-cell (MUSCL: 2 halo subcells)
    mesh = TreeMesh(0.0, 1.0, initial_refinement_level = 2, periodicity = true)
    @test_throws ArgumentError SemidiscretizationHyperbolic(mesh,
                                                            LinearScalarAdvectionEquation1D(1.0),
                                                            initial_condition_constant,
                                                            MacroCellFV(n_cells = 1,
                                                                        surface_flux = flux_lax_friedrichs,
                                                                        reconstruction = ReconstructionMUSCL());
                                                            boundary_conditions = boundary_condition_periodic)

    for item in (solver, solver.volume_integral)
        @test_nowarn show(IOContext(IOBuffer(), :compact => true), MIME"text/plain"(),
                          item)
        @test_nowarn show(IOContext(IOBuffer(), :compact => false),
                          MIME"text/plain"(), item)
    end
end

@testitem "MacroCellFV 1D: thread-local pads" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # The pads of the threads are reused for all elements; repeated evaluations of the
    # right-hand side give the same result (reconstruction of primitive variables,
    # physical boundaries)
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sod.jl"),
                        tspan=(0.0, 0.1))
    u_ode = copy(sol.u[end])
    du_ode = similar(u_ode)
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, 0.0)
    du_first = copy(du_ode)
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, 0.0)
    @test du_ode == du_first
    # The pads hold `n_cells + 2 nhalo` subcells
    @test all(pad -> size(pad) ==
                     (nvariables(equations), n_cells + 2 * Trixi.nhalo(solver)),
              semi.cache.pad_threaded)
    @test length(semi.cache.pad_threaded) == Threads.maxthreadid()
end
