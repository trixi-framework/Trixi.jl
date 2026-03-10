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
The semidiscretizations can be coupled by glueing meshes together using [`BoundaryConditionCoupled`](@ref).

See also: [`SemidiscretizationCoupled`](@ref)

!!! warning "Experimental code"
    This is an experimental feature and can change any time.
"""
mutable struct SemidiscretizationCoupledP4est{Semis, Indices, EquationList} <:
               AbstractSemidiscretization
    semis::Semis
    u_indices::Indices # u_ode[u_indices[i]] is the part of u_ode corresponding to semis[i]
    performance_counter::PerformanceCounter
    parent_cell_ids::Vector{Int}
    view_cell_ids::Vector{Int}
    mesh_ids::Vector{Int}
end

"""
    SemidiscretizationCoupledP4est(semis...)

Create a coupled semidiscretization that consists of the semidiscretizations passed as arguments.
"""
function SemidiscretizationCoupledP4est(semis...)
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
    parent_cell_ids = 1:size(semis[1].mesh.parent.tree_node_coordinates)[end]
    view_cell_ids = zeros(Int, length(parent_cell_ids))
    mesh_ids = zeros(Int, length(parent_cell_ids))
    for i in eachindex(semis)
        view_cell_ids[semis[i].mesh.cell_ids] = parent_cell_id_to_view(parent_cell_ids[semis[i].mesh.cell_ids],
                                                                       semis[i].mesh)
        mesh_ids[semis[i].mesh.cell_ids] .= i
    end

    performance_counter = PerformanceCounter()

    SemidiscretizationCoupledP4est{typeof(semis), typeof(u_indices),
                                   typeof(performance_counter)}(semis, u_indices,
                                                                performance_counter,
                                                                parent_cell_ids,
                                                                view_cell_ids,
                                                                mesh_ids)
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

    # Update all BoundaryConditionCoupledP4est instances with the current solution
    # and semidiscretization reference so they can look up neighbor states.
    foreach_enumerate(semi.semis) do (i, semi_)
        for bc in semi_.boundary_conditions.boundary_condition_types
            if bc isa BoundaryConditionCoupledP4est
                bc.semi_coupled = semi
                bc.u_ode = u_ode
            end
        end
    end

    # Call rhs! for each semidiscretization.
    # u_ode is passed as u_parent but the coupled BCs read from their stored fields.
    foreach_enumerate(semi.semis) do (i, semi_)
        u_loc = get_system_u_ode(u_ode, i, semi)
        du_loc = get_system_u_ode(du_ode, i, semi)
        rhs!(du_loc, u_loc, u_ode, semi, semi_, t)
    end

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
[`SemidiscretizationCoupled`](@ref). For each coupled system, an indididual
[`AnalysisCallback`](@ref) **must** be created and passed to the `AnalysisCallbackCoupledP4est` **in
order**, i.e., in the same sequence as the indidvidual semidiscretizations are stored in the
`SemidiscretizationCoupled`.

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
        callback.condition(u, t, integrator)
    end

    DiscreteCallback(condition, analysis_callback_coupled,
                     save_positions = (false, false),
                     initialize = initialize!)
end

# This method gets called during initialization from OrdinaryDiffEq's `solve(...)`
function initialize!(cb_coupled::DiscreteCallback{Condition, Affect!}, u_ode_coupled, t,
                     integrator) where {Condition, Affect! <: AnalysisCallbackCoupledP4est}
    analysis_callback_coupled = cb_coupled.affect!
    semi_coupled = integrator.p
    du_ode_coupled = first(get_tmp_cache(integrator))

    # Prime the coupled boundary conditions with the initial solution so that
    # individual AnalysisCallback calls to rhs! can read neighbor state correctly.
    foreach_enumerate(semi_coupled.semis) do (i, semi_)
        for bc in semi_.boundary_conditions.boundary_condition_types
            if bc isa BoundaryConditionCoupledP4est
                bc.semi_coupled = semi_coupled
                bc.u_ode = u_ode_coupled
            end
        end
    end

    # Loop over coupled systems' callbacks and initialize them individually
    for i in eachsystem(semi_coupled)
        cb = analysis_callback_coupled.callbacks[i]
        semi = semi_coupled.semis[i]
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

    # Loop over coupled systems' callbacks and call them individually
    for i in eachsystem(semi_coupled)
        @unpack condition = analysis_callback_coupled.callbacks[i]
        analysis_callback = analysis_callback_coupled.callbacks[i].affect!
        u_ode = get_system_u_ode(u_ode_coupled, i, semi_coupled)

        # Check condition and skip callback if it is not yet its turn
        if !condition(u_ode, integrator.t, integrator)
            continue
        end

        semi = semi_coupled.semis[i]
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
    n_vars_upto_semi = cumsum(nvariables(semi_coupled.semis[i].equations)
                              for i in eachindex(semi_coupled.semis))[begin:end]
    error_indices = Array([1, 1 .+ n_vars_upto_semi...])
    length_error_array = sum(nvariables(semi_coupled.semis[i].equations)
                             for i in eachindex(semi_coupled.semis))
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
                                                   system = string(i),
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
function calculate_dt(u_ode, t, cfl_advective, cfl_diffusive,
                      semi::SemidiscretizationCoupledP4est)
    dt = minimum(eachsystem(semi)) do i
        u_ode_slice = get_system_u_ode(u_ode, i, semi)
        calculate_dt(u_ode_slice, t, cfl_advective, cfl_diffusive, semi.semis[i])
    end

    return dt
end

function update_cleaning_speed!(semi_coupled::SemidiscretizationCoupledP4est,
                                glm_speed_callback, dt, t)
    @unpack glm_scale, cfl, semi_indices = glm_speed_callback

    if length(semi_indices) == 0
        throw("Since you have more than one semidiscretization you need to specify the 'semi_indices' for which the GLM speed needs to be calculated.")
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
### Equations
################################################################################

"""
    BoundaryConditionCoupled(other_semi_index, indices, uEltype, coupling_converter)

Boundary condition to glue two meshes together. Solution values at the boundary
of another mesh will be used as boundary values. This requires the use
of [`SemidiscretizationCoupled`](@ref). The other mesh is specified by `other_semi_index`,
which is the index of the mesh in the tuple of semidiscretizations.

Note that the elements and nodes of the two meshes at the coupled boundary must coincide.
This is currently only implemented for [`StructuredMesh`](@ref).

# Arguments
- `other_semi_index`: the index in `SemidiscretizationCoupled` of the semidiscretization
                      from which the values are copied
- `indices::Tuple`: node/cell indices at the boundary of the mesh in the other
                    semidiscretization. See examples below.
- `uEltype::Type`: element type of solution
- `coupling_converter::CouplingConverter`: function to call for converting the solution
                                           state of one system to the other system

# Examples
```julia
# Connect the left boundary of mesh 2 to our boundary such that our positive
# boundary direction will match the positive y direction of the other boundary
BoundaryConditionCoupled(2, (:begin, :i), Float64, fun)

# Connect the same two boundaries oppositely oriented
BoundaryConditionCoupled(2, (:begin, :i_backwards), Float64, fun)

# Using this as y_neg boundary will connect `our_cells[i, 1, j]` to `other_cells[j, end-i, end]`
BoundaryConditionCoupled(2, (:j, :i_backwards, :end), Float64, fun)
```

!!! warning "Experimental code"
    This is an experimental feature and can change any time.
"""
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
mutable struct BoundaryConditionCoupledP4est{CouplingConverter}
    coupling_converter::CouplingConverter
    # Set before each rhs! call by SemidiscretizationCoupledP4est.rhs!
    semi_coupled::Any
    u_ode::Any

    function BoundaryConditionCoupledP4est(coupling_converter)
        new{typeof(coupling_converter)}(coupling_converter, nothing, nothing)
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
                                                             u_ode_coupled)
    n_nodes = length(mesh.parent.nodes)
    # Using a projection onto e_x, -e_x, e_y, -e_y to determine which way our boundary interfaces points to.
    # Knowing this, we then find the cell index in the global (parent) space of the neighboring cell.
    if abs(sum(normal_direction .* (1.0, 0.0))) >
       abs(sum(normal_direction .* (0.0, 1.0)))
        if sum(normal_direction .* (1.0, 0.0)) >
           sum(normal_direction .* (-1.0, 0.0))
            cell_index_parent = cache.neighbor_ids_parent[findfirst((cache.boundaries.name .==
                                                                     :x_pos) .*
                                                                    (cache.boundaries.neighbor_ids .==
                                                                     element_index))]
        else
            cell_index_parent = cache.neighbor_ids_parent[findfirst((cache.boundaries.name .==
                                                                     :x_neg) .*
                                                                    (cache.boundaries.neighbor_ids .==
                                                                     element_index))]
        end
        i_index_g = i_index
        # Make sure we do not leave the domain.
        if i_index == n_nodes
            i_index_g = 1
        elseif i_index == 1
            i_index_g = n_nodes
        end
        j_index_g = j_index
    else
        if sum(normal_direction .* (0.0, 1.0)) > sum(normal_direction .* (0.0, -1.0))
            cell_index_parent = cache.neighbor_ids_parent[findfirst((cache.boundaries.name .==
                                                                     :y_pos) .*
                                                                    (cache.boundaries.neighbor_ids .==
                                                                     element_index))]
        else
            cell_index_parent = cache.neighbor_ids_parent[findfirst((cache.boundaries.name .==
                                                                     :y_neg) .*
                                                                    (cache.boundaries.neighbor_ids .==
                                                                     element_index))]
        end
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
    semi_coupled = boundary_condition.semi_coupled
    u_ode = boundary_condition.u_ode
    idx_other = semi_coupled.mesh_ids[cell_index_parent]
    semi_other = semi_coupled.semis[idx_other]
    local_elem = semi_coupled.view_cell_ids[cell_index_parent]

    u_loc_other = get_system_u_ode(u_ode, idx_other, semi_coupled)
    u_other = wrap_array(u_loc_other, semi_other.mesh, semi_other.equations,
                         semi_other.solver, semi_other.cache)
    u_boundary_raw = get_node_vars(u_other, semi_other.equations, semi_other.solver,
                                   i_index_g, j_index_g, local_elem)

    # Apply coupling converter to transform from neighbor's equations to ours.
    x = cache.elements.node_coordinates[:, i_index, j_index, element_index]
    u_boundary = boundary_condition.coupling_converter(x, u_boundary_raw,
                                                       semi_other.equations, equations)

    orientation = normal_direction

    # Calculate boundary flux
    if have_nonconservative_terms(equations) == true
        flux = (surface_flux_function[1](u_inner, u_boundary, orientation, equations) +
                0.5f0 *
                surface_flux_function[2](u_inner, u_boundary, orientation, equations))
    else
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    end

    return flux
end

function calc_boundary_flux!(cache, t, boundary_condition::BC, boundary_indexing,
                             mesh::P4estMeshView{2},
                             equations, surface_integral, dg::DG, u_parent) where {BC}
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
                                node, direction, element, boundary,
                                u_parent)

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
                                     equations, surface_integral, dg::DG,
                                     u_parent) where {N}
    # Extract the boundary condition type and index vector
    boundary_condition = first(BCs)
    boundary_condition_indices = first(BC_indices)
    # Extract the remaining types and indices to be processed later
    remaining_boundary_conditions = Base.tail(BCs)
    remaining_boundary_condition_indices = Base.tail(BC_indices)

    # process the first boundary condition type
    calc_boundary_flux!(cache, t, boundary_condition, boundary_condition_indices,
                        mesh, equations, surface_integral, dg, u_parent)

    # recursively call this method with the unprocessed boundary types
    calc_boundary_flux_by_type!(cache, t, remaining_boundary_conditions,
                                remaining_boundary_condition_indices,
                                mesh, equations, surface_integral, dg, u_parent)

    return nothing
end

# terminate the type-stable iteration over tuples
function calc_boundary_flux_by_type!(cache, t, BCs::Tuple{}, BC_indices::Tuple{},
                                     mesh::P4estMeshView,
                                     equations, surface_integral, dg::DG, u_parent)
    return nothing
end
end # @muladd
