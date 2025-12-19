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
The semidiscretizations can be coupled by gluing meshes together using [`BoundaryConditionCoupled`](@ref).

!!! warning "Experimental code"
    This is an experimental feature and can change any time.
"""
mutable struct SemidiscretizationCoupledP4est{Semis, Indices, EquationList} <:
               AbstractSemidiscretization
    semis::Semis
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

    # Create correspondence between global (to the parent mesh) cell IDs and local (to the mesh view) cell IDs.
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

function rhs!(du_ode, u_ode, semi::SemidiscretizationCoupledP4est, t)
    time_start = time_ns()

    n_nodes = length(semi.semis[1].mesh.parent.nodes)
    # Reformat the global solutions vector.
    u_ode_reformatted = Vector{real(semi)}(undef, ndofs(semi))
    u_ode_reformatted_reshape = reshape(u_ode_reformatted,
                                        (n_nodes,
                                         n_nodes,
                                         length(semi.mesh_ids)))
    # Extract the global solution vector from the local solutions.
    foreach_enumerate(semi.semis) do (i, semi_)
        system_ode = get_system_u_ode(u_ode, i, semi)
        system_ode_reshape = reshape(system_ode,
                                     (n_nodes, n_nodes,
                                      Int(length(system_ode) /
                                          n_nodes^ndims(semi_.mesh))))
        u_ode_reformatted_reshape[:, :, semi.mesh_ids .== i] .= system_ode_reshape
    end

    # Call rhs! for each semidiscretization
    foreach_enumerate(semi.semis) do (i, semi_)
        u_loc = get_system_u_ode(u_ode, i, semi)
        du_loc = get_system_u_ode(du_ode, i, semi)
        rhs!(du_loc, u_loc, u_ode_reformatted, semi, semi_, t)
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
    if abs(sum(normal_direction .* (1.0, 0.0))) >
       abs(sum(normal_direction .* (0.0, 1.0)))
        if sum(normal_direction .* (1.0, 0.0)) >
           sum(normal_direction .* (-1.0, 0.0))
            element_index_global = cache.neighbor_ids_global[findfirst((cache.boundaries.name .==
                                                                        :x_pos) .*
                                                                       (cache.boundaries.neighbor_ids .==
                                                                        element_index))]
        else
            element_index_global = cache.neighbor_ids_global[findfirst((cache.boundaries.name .==
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
            element_index_global = cache.neighbor_ids_global[findfirst((cache.boundaries.name .==
                                                                        :y_pos) .*
                                                                       (cache.boundaries.neighbor_ids .==
                                                                        element_index))]
        else
            element_index_global = cache.neighbor_ids_global[findfirst((cache.boundaries.name .==
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
    # Perform integer division to get the right shape of the array.
    u_global_reshape = reshape(u_ode_coupled,
                               (n_nodes, n_nodes, length(u_ode_coupled) ÷ n_nodes^2))
    u_boundary = SVector(u_global_reshape[i_index_g, j_index_g, element_index_global])

    # u_boundary = u_inner
    orientation = normal_direction

    # Calculate boundary flux
    if have_nonconservative_terms(equations) == true
        # In case of conservative (index 1) and non-conservative (index 2) fluxes,
        # add the non-conservative one with a factor of 1/2.
        flux = (surface_flux_function[1](u_inner, u_boundary, orientation,
                                         equations) +
                0.5f0 *
                surface_flux_function[2](u_inner, u_boundary, orientation,
                                         equations))
    else
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    end

    return flux
end
end # @muladd
