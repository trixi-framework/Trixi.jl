@testsnippet P4estMesh2DCGSEM begin
    using Trixi: AssemblyCGSEM, assemble_and_apply_jacobian!, l2_projection!,
                 nnodes_unique, local_nodes_of, eachinterface, index_to_start_step_2d

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

    # Curved mapping of [-1, 1]^2 onto itself that is the identity on the boundary and
    # hence compatible with periodic boundaries, cf.
    # `examples/p4est_2d_dgsem/elixir_advection_diffusion_periodic_curved.jl`.
    function mapping_periodic_curved(xi, eta)
        x = xi + 0.1 * sin(pi * xi) * sin(pi * eta)
        y = eta + 0.1 * sin(pi * xi) * sin(pi * eta)
        return SVector(x, y)
    end

    # Discretization of the compressible Euler equations on a periodic curved mesh
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
        assembly = AssemblyCGSEM(mesh, equations, solver, semi.cache)
        return semi, assembly
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
        assembly = AssemblyCGSEM(mesh, equations, solver, cache)

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
    semi, assembly = setup_euler_periodic_curved(initial_condition_weak_blast_wave)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    u_ode = compute_coefficients(0.0, semi)
    u = Trixi.wrap_array(u_ode, semi)
    for element in eachelement(solver, cache), j in eachnode(solver),
        i in eachnode(solver), v in eachvariable(equations)

        u[v, i, j, element] += 0.1 * sin(element)
    end
    @test max_interface_jump(u, mesh, equations, solver, cache) > 0.01
    integrals = Trixi.integrate(cons2cons, u_ode, semi)

    l2_projection!(u, assembly, mesh, equations, solver, cache)
    @test max_interface_jump(u, mesh, equations, solver, cache) == 0.0

    # The projection preserves the integral of every variable
    @test isapprox(Trixi.integrate(cons2cons, u_ode, semi), integrals, rtol = 1e-14)

    # The projection is idempotent up to roundoff errors
    u_ode_projected = copy(u_ode)
    l2_projection!(u, assembly, mesh, equations, solver, cache)
    @test maximum(abs, u_ode - u_ode_projected) <= 10 * eps() * maximum(abs, u_ode)

    # The projection changes smooth data only by roundoff
    u_ode_smooth = compute_coefficients(initial_condition_convergence_test, 0.0, semi)
    u_ode_smooth_before = copy(u_ode_smooth)
    l2_projection!(Trixi.wrap_array(u_ode_smooth, semi), assembly, mesh, equations,
                   solver, cache)
    @test maximum(abs, u_ode_smooth - u_ode_smooth_before) <=
          100 * eps() * maximum(abs, u_ode_smooth_before)

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities); the threading macro may allocate a few bytes
    @test (@allocated l2_projection!(u, assembly, mesh, equations, solver, cache)) < 1000
end

@testitem "CGSEM 2D (P4estMesh): assembly" setup=[
    Setup,
    P4estMesh2DCGSEM
] tags=[:p4est_part1] begin
    semi, assembly = setup_euler_periodic_curved(initial_condition_weak_blast_wave)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    (; weights) = solver.basis

    # Deterministic pseudo-random element-local residuals
    du_ode = compute_coefficients(0.0, semi)
    du_ode .= sin.(eachindex(du_ode))
    du_ode_raw = copy(du_ode)
    du = Trixi.wrap_array(du_ode, semi)
    du_raw = Trixi.wrap_array(du_ode_raw, semi)

    assemble_and_apply_jacobian!(du, assembly, mesh, equations, solver, cache)
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
    @test (@allocated assemble_and_apply_jacobian!(du, assembly, mesh, equations, solver,
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

    # Gauss-Legendre nodes do not lie on element faces
    solver_gauss = DGSEM(polydeg = 3, basis_type = GaussLegendreBasis)
    cache_gauss = Trixi.create_cache_hyperbolic(mesh, equations, solver_gauss, Float64,
                                                Float64)
    @test_throws ArgumentError AssemblyCGSEM(mesh, equations, solver_gauss, cache_gauss)

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
    @test_throws ArgumentError AssemblyCGSEM(mesh, equations, solver, cache)
end
