# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    SemidiscretizationCoupledP4est

Specialized semidiscretization routines for coupled problems using P4est meshes.
This is analogous to the implimantation for structured meshes.
[`semidiscretize`](@ref) will return an `ODEProblem` that synchronizes time steps between the semidiscretizations.
Each call of `rhs!` will call `rhs!` for each semidiscretization individually.
The semidiscretizations can be coupled by gluing meshes together using [`BoundaryConditionCoupled`](@ref).

!!! warning "Experimental code"
    This is an experimental feature and can change any time.
"""
mutable struct SemidiscretizationCoupledP4est{S, Indices, EquationList} <:
               AbstractSemidiscretization
    semis::S
    u_indices::Indices # u_ode[u_indices[i]] is the part of u_ode corresponding to semis[i]
    performance_counter::PerformanceCounter
    global_element_ids::Vector{Int}
    local_element_ids::Vector{Int}
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

    # Create correspondence between global cell IDs and local cell IDs.
    global_element_ids = 1:size(semis[1].mesh.parent.tree_node_coordinates)[end]
    local_element_ids = zeros(Int, size(global_element_ids))
    mesh_ids = zeros(Int, size(global_element_ids))
    for i in 1:length(semis)
        local_element_ids[semis[i].mesh.cell_ids] = global_element_id_to_local(global_element_ids[semis[i].mesh.cell_ids],
                                                                               semis[i].mesh)
        mesh_ids[semis[i].mesh.cell_ids] .= i
    end

    performance_counter = PerformanceCounter()

    SemidiscretizationCoupledP4est{typeof(semis), typeof(u_indices),
                                   typeof(performance_counter)}(semis, u_indices,
                                                                performance_counter,
                                                                global_element_ids,
                                                                local_element_ids,
                                                                mesh_ids)
end

function Base.show(io::IO, semi::SemidiscretizationCoupledP4est)
    @nospecialize semi # reduce precompilation time

    print(io, "SemidiscretizationCoupledP4est($(semi.semis))")
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

@inline Base.ndims(semi::SemidiscretizationCoupledP4est) = ndims(semi.semis[1])

@inline nsystems(semi::SemidiscretizationCoupledP4est) = length(semi.semis)

@inline eachsystem(semi::SemidiscretizationCoupledP4est) = Base.OneTo(nsystems(semi))

@inline Base.real(semi::SemidiscretizationCoupledP4est) = promote_type(real.(semi.semis)...)

@inline function Base.eltype(semi::SemidiscretizationCoupledP4est)
    promote_type(eltype.(semi.semis)...)
end

@inline function ndofs(semi::SemidiscretizationCoupledP4est)
    sum(ndofs, semi.semis)
end

"""
    ndofsglobal(semi::SemidiscretizationCoupledP4est)

Return the global number of degrees of freedom associated with each scalar variable across all MPI ranks, and summed up over all coupled systems.
This is the same as [`ndofs`](@ref) for simulations running in serial or
parallelized via threads. It will in general be different for simulations
running in parallel with MPI.
"""
@inline function ndofsglobal(semi::SemidiscretizationCoupledP4est)
    sum(ndofsglobal, semi.semis)
end

function compute_coefficients(t, semi::SemidiscretizationCoupledP4est)
    @unpack u_indices = semi

    u_ode = Vector{real(semi)}(undef, u_indices[end][end])

    for i in eachsystem(semi)
        # Call `compute_coefficients` in `src/semidiscretization/semidiscretization.jl`
        u_ode[u_indices[i]] .= compute_coefficients(t, semi.semis[i])
    end

    return u_ode
end

@inline function get_system_u_ode(u_ode, index, semi::SemidiscretizationCoupledP4est)
    @view u_ode[semi.u_indices[index]]
end

# Same as `foreach(enumerate(something))`, but without allocations.
#
# Note that compile times may increase if this is used with big tuples.
@inline foreach_enumerate(func, collection) = foreach_enumerate(func, collection, 1)
@inline foreach_enumerate(func, collection::Tuple{}, index) = nothing

@inline function foreach_enumerate(func, collection, index)
    element = first(collection)
    remaining_collection = Base.tail(collection)

    func((index, element))

    # Process remaining collection
    foreach_enumerate(func, remaining_collection, index + 1)
end

function rhs!(du_ode, u_ode, semi::SemidiscretizationCoupledP4est, t)
    time_start = time_ns()

    # @trixi_timeit timer() "copy to coupled boundaries" begin
    #     foreach(semi.semis) do semi_
    #         copy_to_coupled_boundary!(semi_.boundary_conditions, u_ode, semi, semi_)
    #     end
    # end

    u_ode_reformatted = Vector{real(semi)}(undef, ndofs(semi))
    u_ode_reformatted_reshape = reshape(u_ode_reformatted, (4, 4, 4*4))
    foreach_enumerate(semi.semis) do (i, semi_)
        system_ode = get_system_u_ode(u_ode, i, semi)
        system_ode_reshape = reshape(system_ode, (4, 4, Int(length(system_ode)/16)))
        u_ode_reformatted_reshape[:, :, semi.mesh_ids .== i] .= system_ode_reshape
    end

    # Call rhs! for each semidiscretization
    foreach_enumerate(semi.semis) do (i, semi_)
        u_loc = get_system_u_ode(u_ode, i, semi)
        du_loc = get_system_u_ode(du_ode, i, semi)
        rhs!(du_loc, u_loc, semi, semi_, t, u_ode_reformatted)
    end

    runtime = time_ns() - time_start
    put!(semi.performance_counter, runtime)

    return nothing
end

################################################################################
### AnalysisCallback
################################################################################

"""
    AnalysisCallbackCoupled(semi, callbacks...)

Combine multiple analysis callbacks for coupled simulations with a
[`SemidiscretizationCoupled`](@ref). For each coupled system, an indididual
[`AnalysisCallback`](@ref) **must** be created and passed to the `AnalysisCallbackCoupled` **in
order**, i.e., in the same sequence as the indidvidual semidiscretizations are stored in the
`SemidiscretizationCoupled`.

!!! warning "Experimental code"
    This is an experimental feature and can change any time.
"""
struct AnalysisCallbackCoupled{CB}
    callbacks::CB
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb_coupled::DiscreteCallback{<:Any, <:AnalysisCallbackCoupled})
    @nospecialize cb_coupled # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb_coupled)
    else
        analysis_callback_coupled = cb_coupled.affect!

        summary_header(io, "AnalysisCallbackCoupled")
        for (i, cb) in enumerate(analysis_callback_coupled.callbacks)
            summary_line(io, "Callback #$i", "")
            show(increment_indent(io), MIME"text/plain"(), cb)
        end
        summary_footer(io)
    end
end

# Convenience constructor for the coupled callback that gets called directly from the elixirs
function AnalysisCallbackCoupled(semi_coupled, callbacks...)
    if length(callbacks) != nsystems(semi_coupled)
        error("an AnalysisCallbackCoupled requires one AnalysisCallback for each semidiscretization")
    end

    analysis_callback_coupled = AnalysisCallbackCoupled{typeof(callbacks)}(callbacks)

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
                     integrator) where {Condition, Affect! <: AnalysisCallbackCoupled}
    analysis_callback_coupled = cb_coupled.affect!
    semi_coupled = integrator.p
    du_ode_coupled = first(get_tmp_cache(integrator))

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
function (analysis_callback_coupled::AnalysisCallbackCoupled)(integrator)
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
                                                                AnalysisCallbackCoupled}
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

        l2_error, linf_error = calc_error_norms(u_ode, sol.t[end], analyzer, semi,
                                                cache_analysis)
        append!(l2_error_collection, l2_error)
        append!(linf_error_collection, linf_error)
    end

    (; l2 = l2_error_collection, linf = linf_error_collection)
end

################################################################################
### SaveSolutionCallback
################################################################################

# Save mesh for a coupled semidiscretization, which contains multiple meshes internally
function save_mesh(semi::SemidiscretizationCoupledP4est, output_directory, timestep = 0)
    for i in eachsystem(semi)
        mesh, _, _, _ = mesh_equations_solver_cache(semi.semis[i])

        if mesh.unsaved_changes
            mesh.current_filename = save_mesh_file(mesh, output_directory; system = i,
                                                   timestep = timestep)
            mesh.unsaved_changes = false
        end
    end
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
end

################################################################################
### StepsizeCallback
################################################################################

# In case of coupled system, use minimum timestep over all systems
# Case for constant `cfl_number`.
function calculate_dt(u_ode, t, cfl_number::Real, semi::SemidiscretizationCoupledP4est)
    dt = minimum(eachsystem(semi)) do i
        u_ode_slice = get_system_u_ode(u_ode, i, semi)
        calculate_dt(u_ode_slice, t, cfl_number, semi.semis[i])
    end

    return dt
end
# Case for `cfl_number` as a function of time `t`.
function calculate_dt(u_ode, t, cfl_number, semi::SemidiscretizationCoupledP4est)
    cfl_number_ = cfl_number(t)
    dt = minimum(eachsystem(semi)) do i
        u_ode_slice = get_system_u_ode(u_ode, i, semi)
        calculate_dt(u_ode_slice, t, cfl_number_, semi.semis[i])
    end
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
mutable struct BoundaryConditionCoupledP4est{CouplingConverter}
    coupling_converter::CouplingConverter

    function BoundaryConditionCoupledP4est(coupling_converter)
        new{typeof(coupling_converter)}(coupling_converter)
    end
end

function Base.eltype(boundary_condition::BoundaryConditionCoupledP4est)
    eltype(boundary_condition.u_boundary)
end

function (boundary_condition::BoundaryConditionCoupledP4est)(u_inner, mesh, equations, cache,
                                                             i_index, j_index, element_index,
                                                             normal_direction, surface_flux_function, direction,
                                                             u_global)
    # get_node_vars(boundary_condition.u_boundary, equations, solver, surface_node_indices..., cell_indices...),
    # but we don't have a solver here
    @autoinfiltrate
    element_index_y = cld(mesh.cell_ids[element_index], 4)
    element_index_x = mesh.cell_ids[element_index] - (element_index_y - 1) * 4
    if abs(sum(normal_direction .* (1.0, 0.0))) > abs(sum(normal_direction .* (0.0, 1.0)))
        element_index_x += Int(sign(sum(normal_direction .* (1.0, 0.0))))
        if i_index == 4
            i_index_g = 1
        elseif i_index == 1
            i_index_g = 4
        end
        j_index_g = j_index
    else
        element_index_y += Int(sign(sum(normal_direction .* (0.0, 1.0))))
        if j_index == 4
            j_index_g = 1
        elseif j_index == 1
            j_index_g = 4
        end
        i_index_g = i_index
    end
    # Make things periodic across physical boundaries.
    if element_index_x == 0
        element_index_x = 4
    elseif element_index_x == 5
        element_index_x = 1
    end
    if element_index_y == 0
        element_index_y = 4
    elseif element_index_y == 5
        element_index_y = 1
    end
    u_global_reshape = reshape(u_global, (4, 4, 4, 4))
    u_boundary = SVector(u_global_reshape[i_index_g, j_index_g, element_index_x, element_index_y])

    # u_boundary = u_inner
    orientation = normal_direction

    # Calculate boundary flux
    if surface_flux_function isa Tuple
        # In case of conservative (index 1) and non-conservative (index 2) fluxes,
        # add the non-conservative one with a factor of 1/2.
        # if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
            flux = (surface_flux_function[1](u_inner, u_boundary, orientation,
                                             equations) +
                    0.5f0 *
                    surface_flux_function[2](u_inner, u_boundary, orientation,
                                             equations))
        # else # u_boundary is "left" of boundary, u_inner is "right" of boundary
        #     flux = (surface_flux_function[1](u_boundary, u_inner, orientation,
        #                                      equations) +
        #             0.5f0 *
        #             surface_flux_function[2](u_boundary, u_inner, orientation,
        #                                      equations))
        # end
    else
        # if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
            flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
        # else # u_boundary is "left" of boundary, u_inner is "right" of boundary
        #     flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
        # end
    end

    return flux
end

# Don't do anything for other BCs than BoundaryConditionCoupled
function allocate_coupled_boundary_condition(boundary_condition, direction, mesh,
                                             equations,
                                             solver)
    return nothing
end

# # Don't do anything for other BCs than BoundaryConditionCoupled
# function copy_to_coupled_boundary!(boundary_condition, u_ode, semi_coupled, semi)
#     return nothing
# end

# function copy_to_coupled_boundary!(u_ode, semi_coupled, semi, i, n_boundaries,
#                                    boundary_condition, boundary_conditions...)
#     copy_to_coupled_boundary!(boundary_condition, u_ode, semi_coupled, semi)
#     if i < n_boundaries
#         copy_to_coupled_boundary!(u_ode, semi_coupled, semi, i + 1, n_boundaries,
#                                   boundary_conditions...)
#     end
# end

# function copy_to_coupled_boundary!(boundary_conditions::Union{Tuple, NamedTuple}, u_ode,
#                                    semi_coupled, semi)
#     copy_to_coupled_boundary!(u_ode, semi_coupled, semi, 1, length(boundary_conditions),
#                               boundary_conditions...)
# end

# # In 2D
# function copy_to_coupled_boundary!(boundary_condition::BoundaryConditionCoupled{2,
#                                                                                 other_semi_index},
#                                    u_ode, semi_coupled, semi) where {other_semi_index}
#     @unpack u_indices = semi_coupled
#     @unpack other_orientation, indices = boundary_condition
#     @unpack coupling_converter, u_boundary = boundary_condition

#     mesh_own, equations_own, solver_own, cache_own = mesh_equations_solver_cache(semi)
#     other_semi = semi_coupled.semis[other_semi_index]
#     mesh_other, equations_other, solver_other, cache_other = mesh_equations_solver_cache(other_semi)

#     node_coordinates_other = cache_other.elements.node_coordinates
#     u_ode_other = get_system_u_ode(u_ode, other_semi_index, semi_coupled)
#     u_other = wrap_array(u_ode_other, mesh_other, equations_other, solver_other,
#                          cache_other)

#     linear_indices = LinearIndices(size(mesh_other))

#     if other_orientation == 1
#         cells = axes(mesh_other, 2)
#     else # other_orientation == 2
#         cells = axes(mesh_other, 1)
#     end

#     # Copy solution data to the coupled boundary using "delayed indexing" with
#     # a start value and a step size to get the correct face and orientation.
#     node_index_range = eachnode(solver_other)
#     i_node_start, i_node_step = index_to_start_step_2d(indices[1], node_index_range)
#     j_node_start, j_node_step = index_to_start_step_2d(indices[2], node_index_range)

#     i_cell_start, i_cell_step = index_to_start_step_2d(indices[1], axes(mesh_other, 1))
#     j_cell_start, j_cell_step = index_to_start_step_2d(indices[2], axes(mesh_other, 2))

#     # We need indices starting at 1 for the handling of `i_cell` etc.
#     Base.require_one_based_indexing(cells)

#     @threaded for i in eachindex(cells)
#         cell = cells[i]
#         i_cell = i_cell_start + (i - 1) * i_cell_step
#         j_cell = j_cell_start + (i - 1) * j_cell_step

#         i_node = i_node_start
#         j_node = j_node_start
#         element_id = linear_indices[i_cell, j_cell]

#         for element_id in eachnode(solver_other)
#             x_other = get_node_coords(node_coordinates_other, equations_other,
#                                       solver_other,
#                                       i_node, j_node, linear_indices[i_cell, j_cell])
#             u_node_other = get_node_vars(u_other, equations_other, solver_other, i_node,
#                                          j_node, linear_indices[i_cell, j_cell])
#             u_node_converted = coupling_converter(x_other, u_node_other,
#                                                   equations_other,
#                                                   equations_own)

#             for i in eachindex(u_node_converted)
#                 u_boundary[i, element_id, cell] = u_node_converted[i]
#             end

#             i_node += i_node_step
#             j_node += j_node_step
#         end
#     end
# end

################################################################################
### DGSEM/structured
################################################################################

@inline function calc_boundary_flux_by_direction!(surface_flux_values, u, t,
                                                  orientation,
                                                  boundary_condition::BoundaryConditionCoupledP4est,
                                                  mesh::Union{StructuredMesh,
                                                              StructuredMeshView},
                                                  equations,
                                                  surface_integral, dg::DG, cache,
                                                  direction, node_indices,
                                                  surface_node_indices, element)
    @unpack node_coordinates, contravariant_vectors, inverse_jacobian = cache.elements
    @unpack surface_flux = surface_integral

    cell_indices = get_boundary_indices(element, orientation, mesh)

    u_inner = get_node_vars(u, equations, dg, node_indices..., element)

    # If the mapping is orientation-reversing, the contravariant vectors' orientation
    # is reversed as well. The normal vector must be oriented in the direction
    # from `left_element` to `right_element`, or the numerical flux will be computed
    # incorrectly (downwind direction).
    sign_jacobian = sign(inverse_jacobian[node_indices..., element])

    # Contravariant vector Ja^i is the normal vector
    normal = sign_jacobian *
             get_contravariant_vector(orientation, contravariant_vectors,
                                      node_indices..., element)

    # If the mapping is orientation-reversing, the normal vector will be reversed (see above).
    # However, the flux now has the wrong sign, since we need the physical flux in normal direction.
    flux = sign_jacobian * boundary_condition(u_inner, normal, direction, cell_indices,
                              surface_node_indices, surface_flux, equations)

    for v in eachvariable(equations)
        surface_flux_values[v, surface_node_indices..., direction, element] = flux[v]
    end
end

function get_boundary_indices(element, orientation,
                              mesh::Union{StructuredMesh{2}, StructuredMeshView{2}})
    cartesian_indices = CartesianIndices(size(mesh))
    if orientation == 1
        # Get index of element in y-direction
        cell_indices = (cartesian_indices[element][2],)
    else # orientation == 2
        # Get index of element in x-direction
        cell_indices = (cartesian_indices[element][1],)
    end

    return cell_indices
end
end # @muladd
