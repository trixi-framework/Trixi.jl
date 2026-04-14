# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    SemidiscretizationCoupledP4est

Specialized semidiscretization routines for coupled problems using P4est mesh views.
This is analogous to the implementation for structured meshes.
[`semidiscretize`](@ref) will return an `ODEProblem` that synchronizes time steps between the semidiscretizations.
Each call of `rhs!` will call `rhs!` for each semidiscretization individually.
The semidiscretizations can be coupled by glueing meshes together using [`BoundaryConditionCoupledP4est`](@ref).

See also: [`SemidiscretizationCoupled`](@ref)

!!! warning "Experimental code"
    This is an experimental feature and can change any time.
"""
mutable struct SemidiscretizationCoupledP4est{Semis, Indices, CF} <:
               AbstractSemidiscretization
    semis::Semis
    u_indices::Indices # u_ode[u_indices[i]] is the part of u_ode corresponding to semis[i]
    performance_counter::PerformanceCounter
    view_cell_ids::Vector{Int}
    mesh_ids::Vector{Int}
    coupling_functions::CF # [i, j] converts system j variables to system i variable space
    element_offset::Vector{Int} # 1-based offset into u_global for each system's data block
    # Precomputed lookup: boundary_parent_lookup[i][boundary_index] → parent cell ID
    # for each semidiscretization i. Avoids per-node linear scans at runtime.
    boundary_parent_lookup::Vector{Vector{Int}}
end

"""
    SemidiscretizationCoupledP4est(semis...; coupling_functions = nothing)

Create a coupled semidiscretization that consists of the semidiscretizations passed as arguments.
`coupling_functions[i, j]` is called as `f(x, u, equations_j, equations_i)` and should return
the state vector of system `i` given a state vector `u` of system `j`.
"""
function SemidiscretizationCoupledP4est(semis...; coupling_functions = nothing)
    @assert all(semi -> ndims(semi) == ndims(semis[1]), semis) "All semidiscretizations must have the same dimension!"

    # Number of coefficients for each semidiscretization
    n_coefficients = zeros(Int, length(semis))
    for i in 1:length(semis)
        _, equations, _, _ = mesh_equations_solver_cache(semis[i])
        n_coefficients[i] = ndofs(semis[i]) * nvariables(equations)
    end

    # Compute range of coefficients associated with each semidiscretization
    u_indices = Vector{UnitRange{Int}}(undef, length(semis))
    for i in 1:length(semis)
        offset = sum(n_coefficients[1:(i - 1)]) + 1
        u_indices[i] = range(offset, length = n_coefficients[i])
    end

    # Create correspondence between parent mesh cell IDs and view cell IDs.
    # Use ncells to get the actual number of (possibly AMR-refined) elements.
    n_parent_cells = ncells(semis[1].mesh.parent)
    view_cell_ids = zeros(Int, n_parent_cells)
    mesh_ids = zeros(Int, n_parent_cells)
    for i in eachindex(semis)
        view_cell_ids[semis[i].mesh.cell_ids] = parent_cell_id_to_view(semis[i].mesh.cell_ids,
                                                                       semis[i].mesh)
        mesh_ids[semis[i].mesh.cell_ids] .= i
    end

    performance_counter = PerformanceCounter()

    # Precompute element offsets (1-based) into u_global for each system.
    n_nodes = length(semis[1].mesh.parent.nodes)
    element_offset = zeros(Int, length(semis))
    element_offset[1] = 1
    for i in 2:length(semis)
        element_offset[i] = element_offset[i - 1] +
                            n_nodes^2 * nvariables(semis[i - 1].equations) *
                            length(semis[i - 1].mesh.cell_ids)
    end

    # Precompute boundary → parent cell ID lookup for each semidiscretization.
    # boundary_parent_lookup[i] is a vector indexed by boundary index.
    boundary_parent_lookup = Vector{Vector{Int}}(undef, length(semis))
    for i in eachindex(semis)
        boundary_parent_lookup[i] = semis[i].cache.neighbor_ids_parent
    end

    SemidiscretizationCoupledP4est{typeof(semis), typeof(u_indices),
                                   typeof(coupling_functions)}(semis, u_indices,
                                                               performance_counter,
                                                               view_cell_ids,
                                                               mesh_ids,
                                                               coupling_functions,
                                                               element_offset,
                                                               boundary_parent_lookup)
end

function Base.show(io::IO, semi::SemidiscretizationCoupledP4est)
    @nospecialize semi # reduce precompilation time

    print(io, "SemidiscretizationCoupledP4est with $(nsystems(semi)) systems")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", semi::SemidiscretizationCoupledP4est)
    @nospecialize semi # reduce precompilation time

    if get(io, :compact, false)
        show(io, semi)
    else
        summary_header(io, "SemidiscretizationCoupledP4est")
        summary_line(io, "#spatial dimensions", ndims(semi.semis[1]))
        summary_line(io, "#systems", nsystems(semi))
        for i in eachsystem(semi)
            summary_line(io, "system", i)
            mesh, equations, solver, _ = mesh_equations_solver_cache(semi.semis[i])
            summary_line(increment_indent(io), "mesh", mesh |> typeof |> nameof)
            summary_line(increment_indent(io), "equations",
                         equations |> typeof |> nameof)
            summary_line(increment_indent(io), "initial condition",
                         semi.semis[i].initial_condition)
            # no boundary conditions since that could be too much
            summary_line(increment_indent(io), "source terms",
                         semi.semis[i].source_terms)
            summary_line(increment_indent(io), "solver", solver |> typeof |> nameof)
        end
        summary_line(io, "total #DOFs per field", ndofsglobal(semi))
        summary_footer(io)
    end
end

function print_summary_semidiscretization(io::IO, semi::SemidiscretizationCoupledP4est)
    show(io, MIME"text/plain"(), semi)
    println(io, "\n")
    for i in eachsystem(semi)
        mesh, equations, solver, _ = mesh_equations_solver_cache(semi.semis[i])
        summary_header(io, "System #$i")

        summary_line(io, "mesh", mesh |> typeof |> nameof)
        show(increment_indent(io), MIME"text/plain"(), mesh)

        summary_line(io, "equations", equations |> typeof |> nameof)
        show(increment_indent(io), MIME"text/plain"(), equations)

        summary_line(io, "solver", solver |> typeof |> nameof)
        show(increment_indent(io), MIME"text/plain"(), solver)

        summary_footer(io)
        println(io, "\n")
    end
end

@inline nsystems(semi::SemidiscretizationCoupledP4est) = length(semi.semis)

@inline eachsystem(semi::SemidiscretizationCoupledP4est) = Base.OneTo(nsystems(semi))

@inline Base.real(semi::SemidiscretizationCoupledP4est) = promote_type(real.(semi.semis)...)

@inline function ndofs(semi::SemidiscretizationCoupledP4est)
    return sum(ndofs, semi.semis)
end

"""
    ndofsglobal(semi::SemidiscretizationCoupledP4est)

Return the global number of degrees of freedom associated with each scalar variable across all MPI ranks, and summed up over all coupled systems.
This is the same as [`ndofs`](@ref) for simulations running in serial or
parallelized via threads. It will in general be different for simulations
running in parallel with MPI.
"""
@inline function ndofsglobal(semi::SemidiscretizationCoupledP4est)
    return sum(ndofsglobal, semi.semis)
end

function compute_coefficients(t, semi::SemidiscretizationCoupledP4est)
    @unpack u_indices = semi

    u_ode = Vector{real(semi)}(undef, u_indices[end][end])

    # Distribute the partial solution vectors onto the global one.
    @threaded for i in eachsystem(semi)
        # Call `compute_coefficients` in `src/semidiscretization/semidiscretization.jl`
        u_ode[u_indices[i]] .= compute_coefficients(t, semi.semis[i])
    end

    return u_ode
end

@inline function get_system_u_ode(u_ode, index, semi::SemidiscretizationCoupledP4est)
    return @view u_ode[semi.u_indices[index]]
end

# RHS call for the coupled system.
function rhs!(du_ode, u_ode, semi::SemidiscretizationCoupledP4est, t)
    time_start = time_ns()

    n_nodes = length(semi.semis[1].mesh.parent.nodes)

    # Update element_offset for the current AMR state (cell counts may have changed).
    semi.element_offset[1] = 1
    for i in 2:nsystems(semi)
        semi.element_offset[i] = semi.element_offset[i - 1] +
                                 n_nodes^2 * nvariables(semi.semis[i - 1].equations) *
                                 length(semi.semis[i - 1].mesh.cell_ids)
    end

    # Build the global solution vector for coupled mortar flux computation.
    ndofs_nvars_global = sum(nvariables(semi_.equations) * length(semi_.mesh.cell_ids)
                             for semi_ in semi.semis)
    u_global = Vector{real(semi)}(undef, n_nodes^2 * ndofs_nvars_global)

    # Extract the global solution vector from the local solutions.
    foreach_enumerate(semi.semis) do (i, semi_)
        u_loc = get_system_u_ode(u_ode, i, semi)
        n_vars = nvariables(semi_.equations)
        n_cells = length(semi_.mesh.cell_ids)
        u_loc_reshape = reshape(u_loc, (n_vars, n_nodes, n_nodes, n_cells))
        for (local_element, _) in enumerate(semi_.mesh.cell_ids)
            for j_node in 1:n_nodes, i_node in 1:n_nodes, var in 1:n_vars
                u_global[semi.element_offset[i] +
                (var - 1) +
                n_vars * (i_node - 1) +
                n_vars * n_nodes * (j_node - 1) +
                n_vars * n_nodes^2 * (local_element - 1)] = u_loc_reshape[var,
                                                                          i_node,
                                                                          j_node,
                                                                          local_element]
            end
        end
    end

    # Zero stale coupled mortar flux values in surface_flux_values BEFORE the
    # regular rhs! calls. This prevents calc_surface_integral! (inside rhs!)
    # from picking up stale values from the previous time step.
    foreach_enumerate(semi.semis) do (i, semi_)
        mesh, equations, solver, cache = mesh_equations_solver_cache(semi_)
        if isdefined(cache, :coupled_mortars) &&
           ncoupledmortars(cache.coupled_mortars) > 0
            zero_coupled_mortar_surface_flux!(cache.elements.surface_flux_values,
                                              mesh, equations, solver, cache)
        end
    end

    # Prime BCs and call rhs! for each semi in the same loop iteration.
    # Priming and rhs! MUST be interleaved: if multiple semis share the same BC
    # objects, a separate priming loop followed by a separate rhs! loop would
    # leave all BCs with self_index from the last semi when the first rhs! runs.
    foreach_enumerate(semi.semis) do (i, semi_)
        for bc in semi_.boundary_conditions.boundary_condition_types
            if bc isa BoundaryConditionCoupledP4est
                bc.semi_coupled = semi
                bc.u_ode = u_ode
                bc.self_index = i
                bc.t = t
            end
        end
        u_loc = get_system_u_ode(u_ode, i, semi)
        du_loc = get_system_u_ode(du_ode, i, semi)
        rhs!(du_loc, u_loc, semi_, t)
    end

    # Handle coupled mortars (hanging nodes at mesh view boundaries)
    foreach_enumerate(semi.semis) do (i, semi_)
        mesh, equations, solver, cache = mesh_equations_solver_cache(semi_)

        # Check if this mesh view has coupled mortars
        if isdefined(cache, :coupled_mortars) &&
           ncoupledmortars(cache.coupled_mortars) > 0
            u_loc = get_system_u_ode(u_ode, i, semi)
            du_loc = get_system_u_ode(du_ode, i, semi)

            # Wrap to get correct array structure
            u = wrap_array(u_loc, mesh, equations, solver, cache)
            du = wrap_array(du_loc, mesh, equations, solver, cache)

            # Prolong local elements to coupled mortars
            @trixi_timeit timer() "prolong2coupledmortars" prolong2coupledmortars!(cache,
                                                                                   u,
                                                                                   mesh,
                                                                                   equations,
                                                                                   solver.mortar,
                                                                                   solver)

            # Compute and apply coupled mortar fluxes
            @trixi_timeit timer() "coupled mortar flux" begin
                calc_coupled_mortar_flux!(cache.elements.surface_flux_values,
                                          mesh,
                                          have_nonconservative_terms(equations),
                                          equations,
                                          solver.mortar,
                                          solver.surface_integral,
                                          solver,
                                          cache,
                                          u_global,
                                          semi)
            end

            # Apply surface integral for coupled mortar contributions
            # (the regular surface integral was already computed before coupled mortar fluxes)
            @trixi_timeit timer() "coupled mortar surface integral" begin
                calc_coupled_mortar_surface_integral!(du, u, mesh, equations,
                                                      solver.surface_integral,
                                                      solver, cache)
            end
        end
    end

    runtime = time_ns() - time_start
    put!(semi.performance_counter, runtime)

    return nothing
end

# RHS call for the local system.
# Here we require the data from u_parent for each semidiscretization in order
# to exchange the correct boundary values.
function rhs!(du_ode, u_ode, u_parent, semis,
              semi::SemidiscretizationHyperbolic, t)
    @unpack mesh, equations, boundary_conditions, source_terms, solver, cache = semi

    u = wrap_array(u_ode, mesh, equations, solver, cache)
    du = wrap_array(du_ode, mesh, equations, solver, cache)

    time_start = time_ns()
    @trixi_timeit timer() "rhs!" rhs!(du, u, t, u_parent, semis, mesh, equations,
                                      boundary_conditions, source_terms, solver, cache)
    runtime = time_ns() - time_start
    put!(semi.performance_counter, runtime)

    return nothing
end

################################################################################
### AnalysisCallback
################################################################################

"""
    AnalysisCallbackCoupledP4est(semi, callbacks...)

Combine multiple analysis callbacks for coupled simulations with a
[`SemidiscretizationCoupledP4est`](@ref). For each coupled system, an individual
[`AnalysisCallback`](@ref) **must** be created and passed to the `AnalysisCallbackCoupledP4est` **in
order**, i.e., in the same sequence as the individual semidiscretizations are stored in the
`SemidiscretizationCoupledP4est`.

!!! warning "Experimental code"
    This is an experimental feature and can change any time.
"""
struct AnalysisCallbackCoupledP4est{CB}
    callbacks::CB
end

# Convenience constructor for the coupled callback that gets called directly from the elixirs
function AnalysisCallbackCoupledP4est(semi_coupled, callbacks...)
    if length(callbacks) != nsystems(semi_coupled)
        error("an AnalysisCallbackCoupledP4est requires one AnalysisCallback for each semidiscretization")
    end

    analysis_callback_coupled = AnalysisCallbackCoupledP4est{typeof(callbacks)}(callbacks)

    # This callback is triggered if any of its subsidiary callbacks' condition is triggered
    condition = (u, t, integrator) -> any(callbacks) do callback
        return callback.condition(u, t, integrator)
    end

    return DiscreteCallback(condition, analysis_callback_coupled,
                            save_positions = (false, false),
                            initialize = initialize!)
end

# This method gets called during initialization from OrdinaryDiffEq's `solve(...)`
function initialize!(cb_coupled::DiscreteCallback{Condition, Affect!}, u_ode_coupled, t,
                     integrator) where {Condition,
                                        Affect! <: AnalysisCallbackCoupledP4est}
    analysis_callback_coupled = cb_coupled.affect!
    semi_coupled = integrator.p
    du_ode_coupled = first(get_tmp_cache(integrator))

    # Prime the coupled boundary conditions with the initial solution so that
    # individual AnalysisCallback calls to rhs! can read neighbor state correctly.
    # Priming and initialize! must be interleaved because multiple semis may share
    # the same BC objects, so self_index must be set immediately before use.
    for i in eachsystem(semi_coupled)
        semi = semi_coupled.semis[i]
        for bc in semi.boundary_conditions.boundary_condition_types
            if bc isa BoundaryConditionCoupledP4est
                bc.semi_coupled = semi_coupled
                bc.u_ode = u_ode_coupled
                bc.self_index = i
                bc.t = t
            end
        end
        cb = analysis_callback_coupled.callbacks[i]
        u_ode = get_system_u_ode(u_ode_coupled, i, semi_coupled)
        du_ode = get_system_u_ode(du_ode_coupled, i, semi_coupled)
        initialize!(cb, u_ode, du_ode, t, integrator, semi)
    end
end

# This method gets called from OrdinaryDiffEq's `solve(...)`
function (analysis_callback_coupled::AnalysisCallbackCoupledP4est)(integrator)
    semi_coupled = integrator.p
    u_ode_coupled = integrator.u
    du_ode_coupled = first(get_tmp_cache(integrator))

    # Loop over coupled systems' callbacks and call them individually.
    # Prime BCs before each call since multiple semis may share BC objects.
    for i in eachsystem(semi_coupled)
        @unpack condition = analysis_callback_coupled.callbacks[i]
        analysis_callback = analysis_callback_coupled.callbacks[i].affect!
        u_ode = get_system_u_ode(u_ode_coupled, i, semi_coupled)

        # Check condition and skip callback if it is not yet its turn
        if !condition(u_ode, integrator.t, integrator)
            continue
        end

        semi = semi_coupled.semis[i]
        for bc in semi.boundary_conditions.boundary_condition_types
            if bc isa BoundaryConditionCoupledP4est
                bc.semi_coupled = semi_coupled
                bc.u_ode = u_ode_coupled
                bc.self_index = i
                bc.t = integrator.t
            end
        end
        du_ode = get_system_u_ode(du_ode_coupled, i, semi_coupled)
        analysis_callback(u_ode, du_ode, integrator, semi)
    end
end

# used for error checks and EOC analysis
function (cb::DiscreteCallback{Condition, Affect!})(sol) where {Condition,
                                                                Affect! <:
                                                                AnalysisCallbackCoupledP4est
                                                                }
    semi_coupled = sol.prob.p
    u_ode_coupled = sol.u[end]
    @unpack callbacks = cb.affect!

    uEltype = real(semi_coupled)
    l2_error_collection = uEltype[]
    linf_error_collection = uEltype[]
    for i in eachsystem(semi_coupled)
        analysis_callback = callbacks[i].affect!
        @unpack analyzer = analysis_callback
        cache_analysis = analysis_callback.cache

        semi = semi_coupled.semis[i]
        u_ode = get_system_u_ode(u_ode_coupled, i, semi_coupled)

        l2_error,
        linf_error = calc_error_norms(u_ode, sol.t[end], analyzer, semi,
                                      cache_analysis)
        append!(l2_error_collection, l2_error)
        append!(linf_error_collection, linf_error)
    end

    return (; l2 = l2_error_collection, linf = linf_error_collection)
end

################################################################################
### SaveSolutionCallback
################################################################################

# Save mesh for a coupled semidiscretization, which contains multiple meshes internally
function save_mesh(semi::SemidiscretizationCoupledP4est, output_directory, timestep = 0)
    for i in eachsystem(semi)
        mesh, _, _, _ = mesh_equations_solver_cache(semi.semis[i])

        if mesh.unsaved_changes
            mesh.current_filename = save_mesh_file(mesh, output_directory;
                                                   system = i,
                                                   timestep = timestep)
            mesh.unsaved_changes = false
        end
    end
    return nothing
end

@inline function save_solution_file(semi::SemidiscretizationCoupledP4est, u_ode,
                                    solution_callback,
                                    integrator)
    @unpack semis = semi

    for i in eachsystem(semi)
        u_ode_slice = get_system_u_ode(u_ode, i, semi)
        save_solution_file(semis[i], u_ode_slice, solution_callback, integrator,
                           system = i)
    end
    return nothing
end

################################################################################
### StepsizeCallback
################################################################################

# In case of coupled system, use minimum timestep over all systems
# Case for constant `cfl_number`.
function calculate_dt(u_ode, t, cfl_hyperbolic, cfl_parabolic,
                      semi::SemidiscretizationCoupledP4est)
    dt = minimum(eachsystem(semi)) do i
        u_ode_slice = get_system_u_ode(u_ode, i, semi)
        calculate_dt(u_ode_slice, t, cfl_hyperbolic, cfl_parabolic, semi.semis[i])
    end

    return dt
end

function update_cleaning_speed!(semi_coupled::SemidiscretizationCoupledP4est,
                                glm_speed_callback, dt, t)
    @unpack glm_scale, cfl, semi_indices = glm_speed_callback

    if length(semi_indices) == 0
        error("Since you have more than one semidiscretization you need to specify the 'semi_indices' for which the GLM speed needs to be calculated.")
    end

    # Check that all MHD semidiscretizations received a GLM cleaning speed update.
    for (semi_index, semi) in enumerate(semi_coupled.semis)
        if (typeof(semi.equations) <: AbstractIdealGlmMhdEquations &&
            !(semi_index in semi_indices))
            error("Equation of semidiscretization $semi_index needs to be included in 'semi_indices' of 'GlmSpeedCallback'.")
        end
    end

    if cfl isa Real # Case for constant CFL
        cfl_number = cfl
    else # Variable CFL
        cfl_number = cfl(t)
    end

    for semi_index in semi_indices
        semi = semi_coupled.semis[semi_index]
        mesh, equations, solver, cache = mesh_equations_solver_cache(semi)

        # compute time step for GLM linear advection equation with c_h=1 (redone due to the possible AMR)
        c_h_deltat = calc_dt_for_cleaning_speed(cfl_number,
                                                mesh, equations, solver, cache)

        # c_h is proportional to its own time step divided by the complete MHD time step
        # We use @reset here since the equations are immutable (to work on GPUs etc.).
        # Thus, we need to modify the equations field of the semidiscretization.
        @reset equations.c_h = glm_scale * c_h_deltat / dt
        semi.equations = equations
    end

    return semi_coupled
end

################################################################################
### Boundary conditions
################################################################################

"""
    BoundaryConditionCoupledP4est(coupling_converter)

Boundary condition struct where the user can specify the coupling converter function.

# Arguments
- `coupling_converter::CouplingConverter`: function to call for converting the solution
                                           state of one system to the other system
"""
mutable struct BoundaryConditionCoupledP4est{CouplingConverter, FallbackBC} <:
               AbstractCoupledP4estBC
    const coupling_converter::CouplingConverter
    # Set before each rhs! call by SemidiscretizationCoupledP4est.rhs!
    semi_coupled::Union{Nothing, AbstractSemidiscretization}
    u_ode::Union{Nothing, AbstractVector}
    self_index::Int # index of the system this BC belongs to
    t::Float64     # current time, set before each rhs!
    # Optional fallback BC (e.g. BoundaryConditionDirichlet) called when
    # neighbor_ids_parent == 0, i.e. for physical-domain boundaries in
    # mixed-geometry splits where a face name appears at both the view
    # interface and the physical domain edge.
    const fallback_bc::FallbackBC # Nothing or a callable BC

    function BoundaryConditionCoupledP4est(coupling_converter; fallback_bc = nothing)
        new{typeof(coupling_converter), typeof(fallback_bc)}(coupling_converter,
                                                             nothing, nothing,
                                                             0, 0.0, fallback_bc)
    end
end

"""
Extract the boundary values from the neighboring element.
This requires values from other mesh views.
This currently only works for Cartesian meshes.
"""
function (boundary_condition::BoundaryConditionCoupledP4est)(u_inner, mesh, equations,
                                                             cache,
                                                             i_index, j_index,
                                                             element_index,
                                                             normal_direction,
                                                             surface_flux_function,
                                                             direction,
                                                             boundary_index)
    # Use a function barrier to ensure type stability: the mutable fields
    # semi_coupled and u_ode are abstractly typed (set at runtime), so we
    # extract them once here and pass them into a fully-typed inner function.
    semi_coupled = boundary_condition.semi_coupled
    u_ode = boundary_condition.u_ode
    _boundary_condition_coupled(boundary_condition, semi_coupled, u_ode,
                                u_inner, mesh, equations, cache,
                                i_index, j_index, element_index,
                                normal_direction, surface_flux_function,
                                boundary_index)
end

@inline function _boundary_condition_coupled(boundary_condition, semi_coupled, u_ode,
                                             u_inner, mesh, equations, cache,
                                             i_index, j_index, element_index,
                                             normal_direction, surface_flux_function,
                                             boundary_index)
    n_nodes = length(mesh.parent.nodes)
    lookup = semi_coupled.boundary_parent_lookup[boundary_condition.self_index]

    # Look up the parent cell ID directly by boundary index.
    cell_index_parent = lookup[boundary_index]
    if cell_index_parent == 0
        fallback = boundary_condition.fallback_bc
        if fallback === nothing
            error("BoundaryConditionCoupledP4est: no neighbor found for boundary_index=$boundary_index " *
                  "(semi $(boundary_condition.self_index)). " *
                  "Check that the coupling interface boundary name matches the view_interface_names " *
                  "used in build_view_bcs, and that extract_neighbor_ids_parent set the lookup correctly. " *
                  "For mixed-geometry splits where the same face name appears at both view interfaces " *
                  "and physical boundaries, pass fallback_bc = BoundaryConditionDirichlet(...) " *
                  "to BoundaryConditionCoupledP4est.")
        end
        # Physical-domain boundary in a mixed-geometry split: delegate to the fallback BC.
        x = SVector(cache.elements.node_coordinates[1, i_index, j_index, element_index],
                    cache.elements.node_coordinates[2, i_index, j_index, element_index])
        return fallback(u_inner, normal_direction, x, boundary_condition.t,
                        surface_flux_function, equations)
    end

    # Determine which direction the boundary faces to compute the neighbor node indices.
    if abs(normal_direction[1]) > abs(normal_direction[2])
        i_index_g = i_index
        # Make sure we do not leave the domain.
        if i_index == n_nodes
            i_index_g = 1
        elseif i_index == 1
            i_index_g = n_nodes
        end
        j_index_g = j_index
    else
        j_index_g = j_index
        # Make sure we do not leave the domain.
        if j_index == n_nodes
            j_index_g = 1
        elseif j_index == 1
            j_index_g = n_nodes
        end
        i_index_g = i_index
    end
    # Look up the neighbor element's state from the stored coupled solution.
    idx_other = semi_coupled.mesh_ids[cell_index_parent]
    local_elem = semi_coupled.view_cell_ids[cell_index_parent]
    semi_other = semi_coupled.semis[idx_other]

    # Read the neighbor node variables directly from the flat u_ode vector
    # to avoid per-node SubArray + wrap_array allocations.
    u_boundary_raw = _get_node_vars_coupled(u_ode, semi_coupled, idx_other,
                                            semi_other, i_index_g, j_index_g,
                                            local_elem)
    _compute_boundary_flux(semi_other, u_boundary_raw, boundary_condition,
                           u_inner, mesh, have_nonconservative_terms(equations),
                           equations, cache,
                           i_index, j_index,
                           element_index, normal_direction, surface_flux_function,
                           idx_other)
end

# Read node variables directly from the flat u_ode vector using a computed
# linear index, avoiding SubArray and wrap_array allocations.
@inline function _get_node_vars_coupled(u_ode, semi_coupled, idx_other,
                                        semi_other, i, j, elem)
    offset = first(semi_coupled.u_indices[idx_other]) - 1
    nvars = nvariables(semi_other.equations)
    nn = nnodes(semi_other.solver)
    SVector(ntuple(@inline(v->u_ode[offset + v + nvars * ((i - 1) + nn * ((j - 1) + nn * (elem - 1)))]),
                   Val(nvars)))
end

@inline function _compute_boundary_flux(semi_other, u_boundary_raw, boundary_condition,
                                        u_inner, mesh, nonconservative_terms::False,
                                        equations, cache,
                                        i_index, j_index,
                                        element_index, normal_direction,
                                        surface_flux_function, idx_other)
    u_boundary = _convert_boundary_state(boundary_condition, semi_other,
                                         u_boundary_raw, equations, cache,
                                         i_index, j_index, element_index, idx_other)

    flux = surface_flux_function(u_inner, u_boundary, normal_direction, equations)

    return flux
end

@inline function _compute_boundary_flux(semi_other, u_boundary_raw, boundary_condition,
                                        u_inner, mesh, nonconservative_terms::True,
                                        equations, cache,
                                        i_index, j_index,
                                        element_index, normal_direction,
                                        surface_flux_function, idx_other)
    u_boundary = _convert_boundary_state(boundary_condition, semi_other,
                                         u_boundary_raw, equations, cache,
                                         i_index, j_index, element_index, idx_other)

    flux = (surface_flux_function[1](u_inner, u_boundary, normal_direction, equations) +
            0.5f0 *
            surface_flux_function[2](u_inner, u_boundary, normal_direction, equations))

    return flux
end

# Apply coupling converter to transform from neighbor's equations to ours.
@inline function _convert_boundary_state(boundary_condition, semi_other,
                                         u_boundary_raw, equations, cache,
                                         i_index, j_index, element_index, idx_other)
    x = SVector(ntuple(@inline(idx->cache.elements.node_coordinates[idx, i_index,
                                                                    j_index,
                                                                    element_index]),
                       Val(ndims(equations))))
    converter = boundary_condition.coupling_converter
    if converter isa AbstractMatrix
        converter = converter[boundary_condition.self_index, idx_other]
    end
    return converter(x, u_boundary_raw, semi_other.equations, equations)
end

function calc_boundary_flux!(cache, t, boundary_condition::BC, boundary_indexing,
                             mesh::P4estMeshView{2},
                             equations, surface_integral, dg::DG) where {BC}
    @unpack boundaries = cache
    @unpack surface_flux_values = cache.elements
    index_range = eachnode(dg)

    @threaded for local_index in eachindex(boundary_indexing)
        # Use the local index to get the global boundary index from the pre-sorted list
        boundary = boundary_indexing[local_index]

        # Get information on the adjacent element, compute the surface fluxes,
        # and store them
        element = boundaries.neighbor_ids[boundary]
        node_indices = boundaries.node_indices[boundary]
        direction = indices2direction(node_indices)

        i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
        j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

        i_node = i_node_start
        j_node = j_node_start
        for node in eachnode(dg)
            calc_boundary_flux!(surface_flux_values, t, boundary_condition,
                                mesh, have_nonconservative_terms(equations),
                                equations, surface_integral, dg, cache,
                                i_node, j_node,
                                node, direction, element, boundary)

            i_node += i_node_step
            j_node += j_node_step
        end
    end
    return nothing
end

# Iterate over tuples of boundary condition types and associated indices
# in a type-stable way using "lispy tuple programming".
function calc_boundary_flux_by_type!(cache, t, BCs::NTuple{N, Any},
                                     BC_indices::NTuple{N, Vector{Int}},
                                     mesh::P4estMeshView,
                                     equations, surface_integral, dg::DG) where {N}
    # Extract the boundary condition type and index vector
    boundary_condition = first(BCs)
    boundary_condition_indices = first(BC_indices)
    # Extract the remaining types and indices to be processed later
    remaining_boundary_conditions = Base.tail(BCs)
    remaining_boundary_condition_indices = Base.tail(BC_indices)

    # process the first boundary condition type
    calc_boundary_flux!(cache, t, boundary_condition, boundary_condition_indices,
                        mesh, equations, surface_integral, dg)

    # recursively call this method with the unprocessed boundary types
    calc_boundary_flux_by_type!(cache, t, remaining_boundary_conditions,
                                remaining_boundary_condition_indices,
                                mesh, equations, surface_integral, dg)

    return nothing
end

# terminate the type-stable iteration over tuples
function calc_boundary_flux_by_type!(cache, t, BCs::Tuple{}, BC_indices::Tuple{},
                                     mesh::P4estMeshView,
                                     equations, surface_integral, dg::DG)
    return nothing
end
end # @muladd
