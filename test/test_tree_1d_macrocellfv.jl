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

@testitem "MacroCellFV 1D: initial condition as subcell averages" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # Maximum error of the initial data against the exact subcell averages of
    # sin(2πx) on 8 * 2^level subcells
    function initial_average_error(level, initial_condition_nodes)
        equations = LinearScalarAdvectionEquation1D(1.0)
        initial_condition(x, t, equations) = SVector(sinpi(2 * x[1]))
        solver = MacroCellFV(n_cells = 8, surface_flux = flux_lax_friedrichs,
                             initial_condition_nodes = initial_condition_nodes)
        mesh = TreeMesh(0.0, 1.0, initial_refinement_level = level, periodicity = true)
        semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                            boundary_conditions = boundary_condition_periodic)
        u = Trixi.wrap_array_native(compute_coefficients(0.0, semi), semi)
        x = semi.cache.elements.node_coordinates # subcell centers
        h = 1 / (8 * 2^level)
        u_exact = @. (cospi(2 * (x - h / 2)) - cospi(2 * (x + h / 2))) / (2 * pi * h)
        return maximum(abs, u .- u_exact)
    end

    # Point values at the subcell centers: second order
    errors = [initial_average_error(level, 1) for level in 2:3]
    @test 1.9 < log2(errors[1] / errors[2]) < 2.1
    # Gauss-Legendre quadrature with 3 nodes: sixth order
    errors = [initial_average_error(level, 3) for level in 2:3]
    @test log2(errors[1] / errors[2]) > 5.8
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

@testitem "MacroCellFV 1D: elixir_advection_high_order.jl" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_high_order.jl"),
                        l2=[0.0006554009131478032],
                        linf=[0.0012318160241164833])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, ALLOCATION_LIMIT)
end

@testitem "MacroCellFV 1D: elixir_advection_high_order.jl with WENO5-JS" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_high_order.jl"),
                        l2=[0.0008736186419141591],
                        linf=[0.002023947818738492],
                        reconstruction=ReconstructionWENO5JS())
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, ALLOCATION_LIMIT)
end

@testitem "MacroCellFV 1D: elixir_advection_high_order.jl convergence" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # Orders of the errors against the exact subcell averages, with a small time step
    # (fourth-order time integration): fifth order of WENO5-Z; WENO5-JS loses accuracy
    # at the critical points of the solution on these coarse meshes
    for (reconstruction, eoc_min) in ((ReconstructionWENO5Z(), 4.8),
                                      (ReconstructionWENO5JS(), 4.2))
        errors = map((3, 4)) do level
            trixi_include(@__MODULE__,
                          joinpath(EXAMPLES_DIR, "elixir_advection_high_order.jl");
                          reconstruction, initial_refinement_level = level,
                          stepsize_callback = StepsizeCallback(cfl = 0.2))
            return subcell_average_error(@__MODULE__)
        end
        @test log2(errors[1] / errors[2]) > eoc_min
    end
end

@testitem "MacroCellFV 1D: elixir_advection_high_order.jl block decomposition" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # The same subcells split into blocks of 3 subcells (the halo width of WENO5, so
    # that the halos span whole neighbors) and into 1 block give the same
    # solution up to round-off errors (the subcell centers, and thus the initial
    # subcell averages, differ by a few ulps between the decompositions, which the
    # nonlinear WENO weights amplify slightly; an error in the halos would give
    # differences of the order of the truncation error): periodic advection with 48
    # subcells
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_high_order.jl"),
                        n_cells=48, initial_refinement_level=0, tspan=(0.0, 0.1))
    u_1block = copy(sol.u[end])
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_high_order.jl"),
                        n_cells=3, initial_refinement_level=4, tspan=(0.0, 0.1))
    @test isapprox(u_1block, sol.u[end]; atol = 1.0e-13, rtol = 0)
end

@testitem "MacroCellFV 1D: elixir_euler_shu_osher.jl block decomposition" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    # The same 96 subcells split into 2 blocks and into blocks of 3 subcells (the halo
    # width of WENO5) give the same solution up to round-off errors, also with ghost
    # subcells at the transmissive boundaries
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shu_osher.jl"),
                        n_cells=48, initial_refinement_level=1, tspan=(0.0, 0.1))
    u_2blocks = copy(sol.u[end])
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shu_osher.jl"),
                        n_cells=3, initial_refinement_level=5, tspan=(0.0, 0.1))
    @test isapprox(u_2blocks, sol.u[end]; atol = 1.0e-13, rtol = 0)
end

@testitem "MacroCellFV 1D: elixir_euler_shu_osher.jl" setup=[
    Setup,
    TreeMesh1DMacroCellFV
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shu_osher.jl"),
                        l2=[1.1929652719920263, 4.23670862934106, 15.323415000350867],
                        linf=[3.25549904946853, 11.435937173461083, 38.31689116352348],
                        tspan=(0.0, 0.5))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, ALLOCATION_LIMIT)
end

@testitem "MacroCellFV 1D: WENO5 face values" setup=[Setup] tags=[:tree_part1] begin
    # Mirrored data give mirrored face values bitwise (e.g., at reflecting walls)
    data = ((1.0, 2.0, 4.0, 3.5, -1.0), (0.3, -0.2, 0.1, 5.0, 7.0),
            (1.0, 1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 1.0, 0.0, 0.0))
    for reconstruction in (ReconstructionWENO5JS(), ReconstructionWENO5Z()),
        values in data

        w_minus, w_plus = Trixi.weno5_cell(reconstruction, values...)
        w_minus_mirror, w_plus_mirror = Trixi.weno5_cell(reconstruction,
                                                         reverse(values)...)
        @test w_minus === w_plus_mirror
        @test w_plus === w_minus_mirror
    end
    # Fifth order of both face values for smooth data: averages of exp(x) over the
    # cells [(i - 1/2) h, (i + 1/2) h], whose face values are exp(-h / 2), exp(h / 2)
    function face_errors(reconstruction, h)
        averages = Tuple((exp((i + 0.5) * h) - exp((i - 0.5) * h)) / h for i in -2:2)
        w_minus, w_plus = Trixi.weno5_cell(reconstruction, averages...)
        return abs(w_minus - exp(-h / 2)), abs(w_plus - exp(h / 2))
    end
    for reconstruction in (ReconstructionWENO5JS(), ReconstructionWENO5Z())
        errors_coarse = face_errors(reconstruction, 0.1)
        errors_fine = face_errors(reconstruction, 0.05)
        @test all(log2.(errors_coarse ./ errors_fine) .> 4.8)
    end
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

    solver_weno = MacroCellFV(n_cells = 4, surface_flux = flux_lax_friedrichs,
                              reconstruction = ReconstructionWENO5Z())
    @test Trixi.nhalo(solver_weno) == 3
    @test Trixi.nhalo(MacroCellFV(n_cells = 4, surface_flux = flux_lax_friedrichs,
                                  reconstruction = ReconstructionWENO5JS())) == 3
    @test ReconstructionWENO5JS().epsilon == 1.0e-6
    @test ReconstructionWENO5Z().epsilon == 1.0e-40
    @test ReconstructionWENO5Z(epsilon = 1.0f-20).epsilon isa Float32
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
    # WENO5: 3 halo subcells
    @test_throws ArgumentError SemidiscretizationHyperbolic(mesh,
                                                            LinearScalarAdvectionEquation1D(1.0),
                                                            initial_condition_constant,
                                                            MacroCellFV(n_cells = 2,
                                                                        surface_flux = flux_lax_friedrichs,
                                                                        reconstruction = ReconstructionWENO5Z());
                                                            boundary_conditions = boundary_condition_periodic)

    for item in (solver, solver.volume_integral, solver_weno.volume_integral)
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
    # Only the halo cells read by the reconstruction are filled and transformed: with a
    # left neighbor, the first pad cell is neither filled nor read
    # (also the line buffers are written before they are read)
    foreach(pad -> fill!(pad, NaN), semi.cache.pad_threaded)
    foreach(buffers -> foreach(b -> fill!(b, NaN), buffers),
            semi.cache.line_buffers_threaded)
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, 0.0)
    @test du_ode == du_first
    # The pads hold `n_cells + 2 nhalo` subcells, each variable contiguously
    @test all(pad -> size(pad) ==
                     (n_cells + 2 * Trixi.nhalo(solver), nvariables(equations)),
              semi.cache.pad_threaded)
    @test length(semi.cache.pad_threaded) == Threads.maxthreadid()
    @test length(semi.cache.line_buffers_threaded) == Threads.maxthreadid()
end
