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

@testitem "MacroCellFV 1D: elixir_euler_shu_osher.jl" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shu_osher.jl"),
                        l2=[1.1944825857548282, 4.241513259108338, 15.335780201884392],
                        linf=[3.2653242300179475, 11.460858046777963, 38.37747179449188],
                        tspan=(0.0, 0.5))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, ALLOCATION_LIMIT)
end

@testitem "MacroCellFV 1D: elixir_euler_shu_osher.jl with WENO5-JS and central inversion" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shu_osher.jl"),
                        l2=[1.190219649654911, 4.228981130067419, 15.303140294050143],
                        linf=[3.10995150700578, 11.035530536359744, 38.31271215773018],
                        reconstruction=ReconstructionPositivity(ReconstructionWENO5JS()),
                        moment_inversion=MomentInversionCentral(),
                        tspan=(0.0, 0.5))
end

@testitem "MacroCellFV 1D: positivity limiter for a strong double rarefaction" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # Double rarefaction (similar to Toro's test 2, with stronger rarefactions): the
    # density and pressure in the center approach zero. Without the limiter, the WENO
    # face states become inadmissible and the run stops with a NaN time step.
    function initial_condition_double_rarefaction(x, t,
                                                  equations::CompressibleEulerEquations1D)
        RealT = eltype(x)
        v = x[1] < 0.5f0 ? -3 * one(RealT) : 3 * one(RealT)
        return prim2cons(SVector(one(RealT), v, convert(RealT, 0.4)), equations)
    end
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sod.jl"),
                        initial_condition=initial_condition_double_rarefaction,
                        solver=MacroCellFV(n_cells = 8, surface_flux = flux_hll,
                                           reconstruction = ReconstructionPositivity(ReconstructionWENO5Z()),
                                           initial_condition_nodes = 3),
                        stepsize_callback=StepsizeCallback(cfl = 0.25),
                        tspan=(0.0, 0.1))
    u = Trixi.wrap_array_native(sol.u[end], semi)
    rho = u[1, :, :]
    p = [Trixi.pressure(Trixi.get_node_vars(u, equations, solver, i, element), equations)
         for i in Trixi.eachnode(solver), element in Trixi.eachelement(solver, semi.cache)]
    @test minimum(rho) > 0
    @test minimum(p) > 0
    # The limiter is active: the minimum density is far below the initial density
    @test minimum(rho) < 0.1
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
    # The moment inversion adds one cell to the halo of the reconstruction
    @test Trixi.nhalo(MacroCellFV(n_cells = 4, surface_flux = flux_lax_friedrichs,
                                  reconstruction = ReconstructionWENO5Z(),
                                  moment_inversion = MomentInversionWENO())) == 4

    @test_throws ArgumentError MacroCellFV(n_cells = 4,
                                           surface_flux = flux_lax_friedrichs,
                                           initial_condition_nodes = 0)
    # The moment inversions to point values support only WENO reconstructions
    @test_throws ArgumentError MacroCellFV(n_cells = 4,
                                           surface_flux = flux_lax_friedrichs,
                                           reconstruction = ReconstructionMUSCL(),
                                           moment_inversion = MomentInversionCentral())
    @test_throws ArgumentError MomentInversionWENO(linear_weight = 0.5)
    @test_throws ArgumentError ReconstructionPositivity(ReconstructionMUSCL();
                                                        omega = 0.6)

    solver_point = MacroCellFV(n_cells = 8, surface_flux = flux_lax_friedrichs,
                               reconstruction = ReconstructionPositivity(ReconstructionWENO5Z()),
                               moment_inversion = MomentInversionWENO())
    # The halos are copied from the direct neighbors, so the halo width is limited by
    # the number of subcells per macro-cell (WENO5 with moment inversion: 4 halo
    # subcells)
    mesh = TreeMesh(0.0, 1.0, initial_refinement_level = 2, periodicity = true)
    @test_throws ArgumentError SemidiscretizationHyperbolic(mesh,
                                                            LinearScalarAdvectionEquation1D(1.0),
                                                            initial_condition_constant,
                                                            MacroCellFV(n_cells = 3,
                                                                        surface_flux = flux_lax_friedrichs,
                                                                        reconstruction = ReconstructionWENO5Z(),
                                                                        moment_inversion = MomentInversionWENO());
                                                            boundary_conditions = boundary_condition_periodic)

    for item in (solver, solver.volume_integral, solver_point.volume_integral)
        @test_nowarn show(IOContext(IOBuffer(), :compact => true), MIME"text/plain"(),
                          item)
        @test_nowarn show(IOContext(IOBuffer(), :compact => false),
                          MIME"text/plain"(), item)
    end
end

@testitem "MacroCellFV 1D: elixir_advection_high_order.jl" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_high_order.jl"),
                        l2=[0.000655400913147737],
                        linf=[0.001231816024114707])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, ALLOCATION_LIMIT)
end

@testitem "MacroCellFV 1D: elixir_advection_high_order.jl with MomentInversionWENO" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_high_order.jl"),
                        l2=[0.0006552566913937183],
                        linf=[0.0012328898533147115],
                        moment_inversion=MomentInversionWENO())
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, ALLOCATION_LIMIT)
end

@testitem "MacroCellFV 1D: elixir_advection_high_order.jl convergence" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # Orders of the errors against the exact subcell averages, with a small time step
    # (fourth-order time integration): fifth order of WENO5-Z on the averages, fourth
    # order of the WENO interpolation of the point values of the moment inversions
    for (moment_inversion, eoc_min) in ((MomentInversionIdentity(), 4.8),
                                        (MomentInversionCentral(), 3.8),
                                        (MomentInversionWENO(), 3.8))
        errors = map((3, 4)) do level
            trixi_include(@__MODULE__,
                          joinpath(EXAMPLES_DIR, "elixir_advection_high_order.jl");
                          moment_inversion, initial_refinement_level = level,
                          stepsize_callback = StepsizeCallback(cfl = 0.2))
            return subcell_average_error(@__MODULE__)
        end
        @test log2(errors[1] / errors[2]) > eoc_min
    end
end

@testitem "MacroCellFV 1D: thread-local pads" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # The pads of the threads are reused for all elements; repeated evaluations of the
    # right-hand side give the same result (moment inversion with point values in the
    # pads, physical boundaries)
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shu_osher.jl"),
                        tspan=(0.0, 0.1))
    u_ode = copy(sol.u[end])
    du_ode = similar(u_ode)
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, 0.0)
    du_first = copy(du_ode)
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, 0.0)
    @test du_ode == du_first
    # The pads hold `n_cells + 2 nhalo` subcells with point values and averages
    @test all(pad -> size(pad) ==
                     (2 * nvariables(equations), n_cells + 2 * Trixi.nhalo(solver)),
              semi.cache.pad_threaded)
    @test length(semi.cache.pad_threaded) == Threads.maxthreadid()
end
