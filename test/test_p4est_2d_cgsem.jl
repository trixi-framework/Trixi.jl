@testsnippet P4estMesh2DCGSEM begin
    using Trixi: AssemblyCGSEM, assemble_and_apply_jacobian!, l2_projection!,
                 nnodes_unique, local_nodes_of, eachinterface, index_to_start_step_2d

    EXAMPLES_DIR = joinpath(examples_dir(), "p4est_2d_cgsem")

    # Maximum absolute difference between the values of `u` on the two sides of an
    # interface, over all variables, nodes, and interfaces in `cache.interfaces`. It is
    # zero if and only if `u` is continuous across all element interfaces; on a conforming
    # mesh, this implies continuity at all shared nodes, including those shared only
    # through a vertex. It does not use `AssemblyCGSEM` and thus also checks the numbering.
    # The interface nodes are iterated as in `prolong2interfaces!` for `P4estMesh{2}`.
    function max_interface_jump(u, mesh::P4estMesh{2}, equations, dg, cache)
        (; neighbor_ids, node_indices) = cache.interfaces
        index_range = eachnode(dg)

        jump = zero(eltype(u))
        for interface in eachinterface(dg, cache)
            primary_element = neighbor_ids[1, interface]
            secondary_element = neighbor_ids[2, interface]
            primary_indices = node_indices[1, interface]
            secondary_indices = node_indices[2, interface]

            i_primary, i_primary_step = index_to_start_step_2d(primary_indices[1],
                                                               index_range)
            j_primary, j_primary_step = index_to_start_step_2d(primary_indices[2],
                                                               index_range)
            i_secondary, i_secondary_step = index_to_start_step_2d(secondary_indices[1],
                                                                   index_range)
            j_secondary, j_secondary_step = index_to_start_step_2d(secondary_indices[2],
                                                                   index_range)

            for i in index_range
                for v in eachvariable(equations)
                    jump = max(jump,
                               abs(u[v, i_primary, j_primary, primary_element] -
                                   u[v, i_secondary, j_secondary, secondary_element]))
                end
                i_primary += i_primary_step
                j_primary += j_primary_step
                i_secondary += i_secondary_step
                j_secondary += j_secondary_step
            end
        end

        return jump
    end

    # Convenience method for the solution vector of a CGSEM semidiscretization
    function max_interface_jump(u_ode::AbstractVector, semi)
        mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
        u = Trixi.wrap_array(u_ode, semi)
        return max_interface_jump(u, mesh, equations, solver.dg, cache)
    end

    # Curved mapping of [-1, 1]^2 onto itself that is the identity on the boundary and
    # hence compatible with periodic boundaries, cf.
    # `examples/p4est_2d_dgsem/elixir_advection_diffusion_periodic_curved.jl`.
    function mapping_periodic_curved(xi, eta)
        x = xi + 0.1 * sin(pi * xi) * sin(pi * eta)
        y = eta + 0.1 * sin(pi * xi) * sin(pi * eta)
        return SVector(x, y)
    end

    # DGSEM discretization of the compressible Euler equations on a periodic curved mesh,
    # the `CGSEM` wrapping its solver, and the corresponding gather-scatter data
    function setup_euler_periodic_curved(initial_condition; polydeg = 3,
                                         initial_refinement_level = 1)
        equations = CompressibleEulerEquations2D(1.4)
        solver = DGSEM(polydeg = polydeg, surface_flux = flux_lax_friedrichs)
        mesh = P4estMesh((4, 3); polydeg = 3, mapping = mapping_periodic_curved,
                         initial_refinement_level = initial_refinement_level,
                         periodicity = true)
        boundary_conditions = boundary_condition_periodic
        semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                            boundary_conditions)
        cgsem = CGSEM(solver)
        assembly = AssemblyCGSEM(mesh, equations, cgsem, semi.cache)
        return semi, cgsem, assembly
    end
end

@testitem "CGSEM 2D (P4estMesh): number of unique nodes" setup=[
    Setup,
    P4estMesh2DCGSEM
] tags=[:p4est_part1] begin
    equations = LinearScalarAdvectionEquation2D((1.0, 1.0))
    trees_per_dimension = (3, 2)

    for polydeg in (1, 3), initial_refinement_level in (0, 1),
        periodicity in (true, false)

        mesh = P4estMesh(trees_per_dimension; polydeg = 1,
                         coordinates_min = (-1.0, -1.0), coordinates_max = (1.0, 1.0),
                         initial_refinement_level = initial_refinement_level,
                         periodicity = periodicity)
        solver = DGSEM(polydeg = polydeg)
        cache = Trixi.create_cache_hyperbolic(mesh, equations, solver, Float64, Float64)
        assembly = AssemblyCGSEM(mesh, equations, CGSEM(solver), cache)

        # `polydeg` distinct nodes per element and direction, plus one on a
        # nonperiodic boundary
        n_x, n_y = trees_per_dimension .* 2^initial_refinement_level .* polydeg
        expected = periodicity ? n_x * n_y : (n_x + 1) * (n_y + 1)
        @test nnodes_unique(assembly) == expected

        # Every element-local node belongs to exactly one global node
        @test length(assembly.copies) == Trixi.ndofs(mesh, solver, cache)
        @test assembly.offsets[end] == length(assembly.copies) + 1
        @test allunique(assembly.copies)

        # The rows of the global-to-local map are sorted by element id and partition
        # the element-local nodes; vertices are shared by four elements
        rows = [local_nodes_of(n, assembly) for n in 1:nnodes_unique(assembly)]
        @test all(issorted, rows)
        @test reduce(vcat, rows) == assembly.copies
        @test maximum(length, rows) == 4
        @test minimum(length, rows) == (polydeg == 1 && periodicity ? 4 : 1)
        @test_throws BoundsError local_nodes_of(nnodes_unique(assembly) + 1, assembly)

        @test occursin("unique nodes", repr(assembly))
    end
end

@testitem "CGSEM 2D (P4estMesh): L2 projection" setup=[
    Setup,
    P4estMesh2DCGSEM
] tags=[:p4est_part1] begin
    # The initial condition is evaluated pointwise, so it is single-valued at the shared
    # nodes up to roundoff even if it is discontinuous. Add a different constant in each
    # element to obtain genuinely multi-valued data.
    semi, cgsem, assembly = setup_euler_periodic_curved(initial_condition_weak_blast_wave)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    u_ode = compute_coefficients(0.0, semi)
    u = Trixi.wrap_array(u_ode, semi)
    for element in eachelement(solver, cache), j in eachnode(solver),
        i in eachnode(solver), v in eachvariable(equations)

        u[v, i, j, element] += 0.1 * sin(element)
    end
    @test max_interface_jump(u, mesh, equations, solver, cache) > 0.01
    integrals = Trixi.integrate(cons2cons, u_ode, semi)

    l2_projection!(u, assembly, mesh, equations, cgsem, cache)
    @test max_interface_jump(u, mesh, equations, solver, cache) == 0.0

    # The projection preserves the integral of every variable
    @test isapprox(Trixi.integrate(cons2cons, u_ode, semi), integrals, rtol = 1e-14)

    # The projection is idempotent up to roundoff errors
    u_ode_projected = copy(u_ode)
    l2_projection!(u, assembly, mesh, equations, cgsem, cache)
    @test maximum(abs, u_ode - u_ode_projected) <= 10 * eps() * maximum(abs, u_ode)

    # The projection changes smooth data only by roundoff
    u_ode_smooth = compute_coefficients(initial_condition_convergence_test, 0.0, semi)
    u_ode_smooth_before = copy(u_ode_smooth)
    l2_projection!(Trixi.wrap_array(u_ode_smooth, semi), assembly, mesh, equations,
                   cgsem, cache)
    @test maximum(abs, u_ode_smooth - u_ode_smooth_before) <=
          100 * eps() * maximum(abs, u_ode_smooth_before)

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities); the threading macro may allocate a few bytes
    @test (@allocated l2_projection!(u, assembly, mesh, equations, cgsem, cache)) < 1000
end

@testitem "CGSEM 2D (P4estMesh): assembly" setup=[
    Setup,
    P4estMesh2DCGSEM
] tags=[:p4est_part1] begin
    semi, cgsem, assembly = setup_euler_periodic_curved(initial_condition_weak_blast_wave)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    (; weights) = solver.basis

    # Deterministic pseudo-random element-local residuals
    du_ode = compute_coefficients(0.0, semi)
    du_ode .= sin.(eachindex(du_ode))
    du_ode_raw = copy(du_ode)
    du = Trixi.wrap_array(du_ode, semi)
    du_raw = Trixi.wrap_array(du_ode_raw, semi)

    assemble_and_apply_jacobian!(du, assembly, mesh, equations, cgsem, cache)
    @test max_interface_jump(du, mesh, equations, solver, cache) == 0.0

    # The assembled values satisfy sum_n (sum_k w J_k) du_n = -sum_k w du_k for every
    # global node, so the volume integral of the result equals minus the weighted sum of
    # the raw residuals
    integral = Trixi.integrate(cons2cons, du_ode, semi, normalize = false)
    expected = -sum(weights[i] * weights[j] *
                    Trixi.get_node_vars(du_raw, equations, solver, i, j, element)
                    for element in eachelement(solver, cache), j in eachnode(solver),
                        i in eachnode(solver))
    @test isapprox(integral, expected, rtol = 1e-13)

    # At interior nodes, the assembly is exactly `apply_jacobian!` of DGSEM
    du_ode_dg = copy(du_ode_raw)
    du_dg = Trixi.wrap_array(du_ode_dg, semi)
    Trixi.apply_jacobian!(nothing, du_dg, mesh, equations, solver, cache)
    interior = 2:(Trixi.nnodes(solver) - 1)
    @test view(du, :, interior, interior, :) == view(du_dg, :, interior, interior, :)

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities); the threading macro may allocate a few bytes
    @test (@allocated assemble_and_apply_jacobian!(du, assembly, mesh, equations, cgsem,
                                                   cache)) <
          1000
end

@testitem "CGSEM 2D (P4estMesh): assembly errors" setup=[
    Setup,
    P4estMesh2DCGSEM
] tags=[:p4est_part1] begin
    equations = LinearScalarAdvectionEquation2D((1.0, 1.0))
    mesh = P4estMesh((2, 2); polydeg = 1,
                     coordinates_min = (-1.0, -1.0), coordinates_max = (1.0, 1.0),
                     initial_refinement_level = 1, periodicity = true)

    # Nonconforming meshes are not supported: refine the bottom left quadrant of each
    # tree once more (cf. `examples/p4est_2d_dgsem/elixir_advection_nonconforming_flag.jl`)
    function refine_fn(p4est, which_tree, quadrant)
        quadrant_obj = unsafe_load(quadrant)
        if quadrant_obj.x == 0 && quadrant_obj.y == 0
            return Cint(1)
        else
            return Cint(0)
        end
    end
    refine_fn_c = @cfunction(refine_fn, Cint,
                             (Ptr{Trixi.p4est_t}, Ptr{Trixi.p4est_topidx_t},
                              Ptr{Trixi.p4est_quadrant_t}))
    Trixi.refine_p4est!(mesh.p4est, false, refine_fn_c, C_NULL)

    solver = DGSEM(polydeg = 3)
    cache = Trixi.create_cache_hyperbolic(mesh, equations, solver, Float64, Float64)
    @test Trixi.nmortars(cache.mortars) > 0
    @test_throws ArgumentError AssemblyCGSEM(mesh, equations, CGSEM(solver), cache)
end

@testitem "CGSEM 2D (P4estMesh): elixir_advection_basic.jl" setup=[
    Setup,
    P4estMesh2DCGSEM
] tags=[:p4est_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        l2=[0.0002815770105516991],
                        linf=[0.0015513438749037434])

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)

    # The solution stays continuous
    @test max_interface_jump(sol.u[end], semi) == 0.0

    # Plotting and type adaptation are forwarded to the DGSEM solver
    @test PlotData2D(sol) isa Trixi.PlotData2DTriangulated
    @test PlotData1D(sol; slice = :y, point = (0.1, 0.2)) isa Trixi.PlotData1D
    u = Trixi.wrap_array(sol.u[end], semi)
    @test ScalarPlotData2D(u[1, :, :, :], semi) isa Trixi.PlotData2DTriangulated
    semi32 = Trixi.trixi_adapt(Array, Float32, semi)
    @test real(semi32.solver) == Float32
    @test real(semi32.solver.dg) == Float32
    @test eltype(semi32.cache.assembly.inverse_sum_jacobian) == Float32
end

@testitem "CGSEM 2D (P4estMesh): elixir_advection_basic.jl (convergence)" setup=[
    Setup,
    P4estMesh2DCGSEM
] tags=[:p4est_part1] begin
    # Observed order of convergence of the central CGSEM scheme on the curved mesh
    eocs, _ = convergence_test(@__MODULE__,
                               joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"), 3;
                               initial_refinement_level = 0, tspan = (0.0, 0.5))
    mean_convergence = Trixi.calc_mean_convergence(eocs)
    @test isapprox(mean_convergence[:l2], [3.858734049173437], rtol = 0.05)
    @test isapprox(mean_convergence[:linf], [3.647081230807921], rtol = 0.05)
end

@testitem "CGSEM 2D (P4estMesh): elixir_euler_free_stream.jl" setup=[
    Setup,
    P4estMesh2DCGSEM
] tags=[:p4est_part1] begin
    # Free-stream preservation on the curved mesh with weakly imposed Dirichlet
    # boundary conditions and flux differencing: all errors are at roundoff level
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream.jl"),
                        l2=[
                            2.6102233325946257e-15,
                            1.1447479911078741e-14,
                            1.1079551626617037e-14,
                            1.16768963220328e-14
                        ],
                        linf=[
                            1.5987211554602254e-14,
                            1.2644052471699752e-13,
                            9.769962616701378e-14,
                            9.414691248821327e-14
                        ],
                        # roundoff-level values, cf. the DGSEM free-stream test
                        atol=2.0e-12)

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)

    # The solution stays continuous
    @test max_interface_jump(sol.u[end], semi) == 0.0
end

@testitem "CGSEM 2D (P4estMesh): elixir_euler_source_terms.jl" setup=[
    Setup,
    P4estMesh2DCGSEM
] tags=[:p4est_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
                        l2=[
                            0.000895262985930122,
                            0.0006580790353505263,
                            0.000658079035350117,
                            0.0014968204204355083
                        ],
                        linf=[
                            0.004949691922103039,
                            0.004873770026974311,
                            0.0048737700269609885,
                            0.008909059775469075
                        ])

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)

    # The solution stays continuous although the source terms are evaluated at node
    # coordinates that differ by roundoff between the copies of a node
    @test max_interface_jump(sol.u[end], semi) == 0.0

    # Observed order of convergence with source terms and Dirichlet boundary conditions
    eocs, _ = convergence_test(@__MODULE__,
                               joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
                               3; initial_refinement_level = 0, tspan = (0.0, 0.5))
    mean_convergence = Trixi.calc_mean_convergence(eocs)
    @test isapprox(mean_convergence[:l2],
                   [
                       3.681436079357308,
                       3.7363468691830493,
                       3.736346869177713,
                       3.812613300064779
                   ], rtol = 0.05)
end

@testitem "CGSEM 2D (P4estMesh): elixir_euler_ec.jl" setup=[
    Setup,
    P4estMesh2DCGSEM
] tags=[:p4est_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
                        l2=[
                            0.1223818734047356,
                            0.09921498279250525,
                            0.09923553122903732,
                            0.4474479179138367
                        ],
                        linf=[
                            0.3017800167995779,
                            0.2879413580175879,
                            0.29050273964712503,
                            1.085125018348565
                        ])

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)

    # The solution stays continuous
    @test max_interface_jump(sol.u[end], semi) == 0.0

    # Flux differencing with an entropy-conservative two-point flux is entropy
    # conservative in the interior, which is the whole domain on this periodic mesh
    t = sol.t[end]
    u_ode = sol.u[end]
    du_ode = similar(u_ode)
    Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, t)
    u = Trixi.wrap_array(u_ode, semi)
    du = Trixi.wrap_array(du_ode, semi)
    @test abs(Trixi.analyze(Trixi.entropy_timederivative, du, u, t, semi)) < 1e-12
end

@testitem "CGSEM 2D (P4estMesh): elixir_euler_shockcapturing.jl" setup=[
    Setup,
    P4estMesh2DCGSEM
] tags=[:p4est_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing.jl"),
                        l2=[
                            0.09143138442458294,
                            0.10434435741181056,
                            0.10434817762209272,
                            0.33607404448023165
                        ],
                        linf=[
                            0.20113968373283186,
                            0.34321185058998693,
                            0.34303766697132626,
                            0.7425324867387526
                        ])

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)

    # The solution stays continuous with the blended subcell finite volume method
    @test max_interface_jump(sol.u[end], semi) == 0.0

    # The shock-capturing indicator is available as an element variable for the
    # solution files
    element_variables = Dict{Symbol, Any}()
    Trixi.get_element_variables!(element_variables, sol.u[end], semi)
    @test haskey(element_variables, :indicator_shock_capturing)
end

@testitem "CGSEM 2D (P4estMesh): solver errors" setup=[
    Setup,
    P4estMesh2DCGSEM
] tags=[:p4est_part1] begin
    # Only Lobatto-Legendre nodes lie on the element faces
    @test_throws ArgumentError CGSEM(DGSEM(polydeg = 3,
                                           basis_type = GaussLegendreBasis))
    # The weak-form surface integral is required by the admissibility criterion
    surface_integral = SurfaceIntegralStrongForm(flux_lax_friedrichs)
    @test_throws ArgumentError CGSEM(DGSEM(polydeg = 3; surface_integral))
    # Unsupported volume integral
    @test_throws ArgumentError CGSEM(polydeg = 3,
                                     volume_integral = VolumeIntegralStrongForm())

    # Nonconservative equations are not supported yet
    equations = IdealGlmMhdEquations2D(1.4)
    mesh = P4estMesh((2, 2); polydeg = 1,
                     coordinates_min = (-1.0, -1.0), coordinates_max = (1.0, 1.0),
                     initial_refinement_level = 0, periodicity = true)
    solver = CGSEM(polydeg = 3,
                   volume_integral = VolumeIntegralFluxDifferencing(flux_central))
    boundary_conditions = boundary_condition_periodic
    @test_throws ArgumentError SemidiscretizationHyperbolic(mesh, equations,
                                                            initial_condition_constant,
                                                            solver;
                                                            boundary_conditions)

    # Functionality that is not forwarded to the DGSEM solver, e.g., AMR, fails with
    # a `MethodError` instead of silently running the DGSEM code
    @test_throws MethodError Trixi.AdaptorAMR(mesh, solver)

    @test occursin("CGSEM{Float64}", repr(solver))
    @test occursin("CGSEM(polydeg=3)", summary(solver))
    @test occursin("volume integral", repr(MIME"text/plain"(), solver))
end
