"""
    SemidiscretizationCoupledDynamic

A struct used to bundle multiple semidiscretizations.
[`semidiscretize`](@ref) will return an `ODEProblem` that synchronizes time steps between the semidiscretizations.
Each call of `rhs!` will call `rhs!` for each semidiscretization individually.
The semidiscretizations can be coupled by gluing meshes together using [`BoundaryConditionCoupled`](@ref).

!!! warning "Experimental code"
    This is an experimental feature and can change any time.
"""
struct SemidiscretizationCoupledDynamic{S, Indices, EquationList} <: AbstractSemidiscretization
    semis::S
    u_indices::Indices # u_ode[u_indices[i]] is the part of u_ode corresponding to semis[i]
    domain_marker::Array
    performance_counter::PerformanceCounter
end

"""
    SemidiscretizationCoupledDynamic(semis...)

Create a coupled semidiscretization that consists of the semidiscretizations passed as arguments.
"""
function SemidiscretizationCoupledDynamic(semis...)
    @assert all(semi -> ndims(semi) == ndims(semis[1]), semis) "All semidiscretizations must have the same dimension!"

    # Number of coefficients for each semidiscretization
    n_coefficients = zeros(Int, length(semis))
    domain_marker = 0
    for i in 1:length(semis)
        _, equations, _, cache = mesh_equations_solver_cache(semis[i])
        n_coefficients[i] = ndofs(semis[i]) * nvariables(equations)
        if i == 1
            # Allocate memory for the domain markers.
            domain_marker = zeros(Int8, size(cache.elements.node_coordinates[1, :, :, :]))
        end
        # TODO: do not hard code this.
        domain_marker[cache.elements.node_coordinates[1, :, :, :] .<= 0] .= 1
        domain_marker[cache.elements.node_coordinates[1, :, :, :] .> 0] .= 2
    end


    # Compute range of coefficients associated with each semidiscretization.
    u_indices = Vector{UnitRange{Int}}(undef, length(semis))
    for i in 1:length(semis)
        offset = sum(n_coefficients[1:(i - 1)]) + 1
        u_indices[i] = range(offset, length = n_coefficients[i])
    end

    performance_counter = PerformanceCounter()

    SemidiscretizationCoupledDynamic{typeof(semis), typeof(u_indices), typeof(performance_counter)
                                     }(semis, u_indices, domain_marker, performance_counter)
end

function Base.show(io::IO, semi::SemidiscretizationCoupledDynamic)
    @nospecialize semi # reduce precompilation time

    print(io, "SemidiscretizationCoupledDynamic($(semi.semis))")
end

function Base.show(io::IO, ::MIME"text/plain", semi::SemidiscretizationCoupledDynamic)
    @nospecialize semi # reduce precompilation time

    if get(io, :compact, false)
        show(io, semi)
    else
        summary_header(io, "SemidiscretizationCoupledDynamic")
        summary_line(io, "#spatial dimensions", ndims(semi.semis[1]))
        summary_line(io, "#systems", nsystems(semi))
        for i in eachsystem(semi)
            summary_line(io, "system", i)
            mesh, equations, solver, _ = mesh_equations_solver_cache(semi.semis[i])
            summary_line(increment_indent(io), "mesh", mesh |> typeof |> nameof)
            summary_line(increment_indent(io), "equations", equations |> typeof |> nameof)
            summary_line(increment_indent(io), "initial condition",
                         semi.semis[i].initial_condition)
            # no boundary conditions since that could be too much
            summary_line(increment_indent(io), "source terms", semi.semis[i].source_terms)
            summary_line(increment_indent(io), "solver", solver |> typeof |> nameof)
        end
        # TODO: Add coupling functions.
        summary_line(io, "total #DOFs", ndofs(semi))
        summary_footer(io)
    end
end

function print_summary_semidiscretization(io::IO, semi::SemidiscretizationCoupledDynamic)
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

@inline Base.ndims(semi::SemidiscretizationCoupledDynamic) = ndims(semi.semis[1])

@inline nsystems(semi::SemidiscretizationCoupledDynamic) = length(semi.semis)

@inline eachsystem(semi::SemidiscretizationCoupledDynamic) = Base.OneTo(nsystems(semi))

@inline Base.real(semi::SemidiscretizationCoupledDynamic) = promote_type(real.(semi.semis)...)

@inline Base.eltype(semi::SemidiscretizationCoupledDynamic) = promote_type(eltype.(semi.semis)...)

@inline function ndofs(semi::SemidiscretizationCoupledDynamic)
    sum(ndofs, semi.semis)
end

@inline function nelements(semi::SemidiscretizationCoupledDynamic)
    return sum(semi.semis) do semi_
        mesh, equations, solver, cache = mesh_equations_solver_cache(semi_)

        nelements(mesh, solver, cache)
    end
end

# TODO: Check if we still need this. Perhaps ad a @autoinfiltrate.
function compute_coefficients(t, semi::SemidiscretizationCoupledDynamic)
    @unpack u_indices = semi

    u_ode = Vector{real(semi)}(undef, u_indices[end][end])

    for i in eachsystem(semi)
        # Call `compute_coefficients` in `src/semidiscretization/semidiscretization.jl`
        u_ode[u_indices[i]] .= compute_coefficients(t, semi.semis[i])
    end

    return u_ode
end

@inline function get_system_u_ode(u_ode, index, semi::SemidiscretizationCoupledDynamic)
    @view u_ode[semi.u_indices[index]]
end

function rhs!(du_ode, u_ode, semi::SemidiscretizationCoupledDynamic, t)
    @unpack u_indices = semi

    time_start = time_ns()

    # Call rhs! for each semidiscretization
    for i in eachsystem(semi)
        u_loc = get_system_u_ode(u_ode, i, semi)
        du_loc = get_system_u_ode(du_ode, i, semi)

        @trixi_timeit timer() "system #$i" rhs!(du_loc, u_loc, semi.semis[i], t)
        copy_to_coupled_boundary!(u_loc, u_ode, i, semi)
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
[`SemidiscretizationCoupledDynamic`](@ref). For each coupled system, an indididual
[`AnalysisCallback`](@ref) **must** be created and passed to the `AnalysisCallbackCoupled` **in
order**, i.e., in the same sequence as the indidvidual semidiscretizations are stored in the
`SemidiscretizationCoupledDynamic`.

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
function save_mesh(semi::SemidiscretizationCoupledDynamic, output_directory, timestep = 0)
    for i in eachsystem(semi)
        mesh, _, _, _ = mesh_equations_solver_cache(semi.semis[i])

        if mesh.unsaved_changes
            mesh.current_filename = save_mesh_file(mesh, output_directory, system = i)
            mesh.unsaved_changes = false
        end
    end
end

@inline function save_solution_file(semi::SemidiscretizationCoupledDynamic, u_ode,
                                    solution_callback,
                                    integrator)
    @unpack semis = semi
    @unpack t, dt = integrator

    mesh, equations, solver, cache = mesh_equations_solver_cache(semi.semis[1])

    # Save the solution.
    for i in eachsystem(semi)
        u_ode_slice = get_system_u_ode(u_ode, i, semi)
        save_solution_file(semis[i], u_ode_slice, solution_callback, integrator, system = i)
    end

    # Save the domain boundary.
    @unpack output_directory, solution_variables = solution_callback
    timestep = integrator.stats.naccept
    filename = joinpath(output_directory, @sprintf("domain_indices_%06d.h5", timestep))
    # Open file (clobber existing content)
    h5open(filename, "w") do file
        # Add context information as attributes
        attributes(file)["ndims"] = ndims(mesh)
        # attributes(file)["equations"] = get_name(equations)
        attributes(file)["polydeg"] = polydeg(solver)
        attributes(file)["n_vars"] = 1
        attributes(file)["n_elements"] = nelements(solver, cache)
        attributes(file)["mesh_type"] = get_name(mesh)
        attributes(file)["mesh_file"] = splitdir(mesh.current_filename)[2]
        attributes(file)["time"] = convert(Float64, t) # Ensure that `time` is written as a double precision scalar
        attributes(file)["dt"] = convert(Float64, dt) # Ensure that `dt` is written as a double precision scalar
        attributes(file)["timestep"] = timestep

        # Store each variable of the solution data
        file["variables_1"] = vec(semi.domain_marker)

        # Add variable name as attribute
        var = file["variables_1"]
        attributes(var)["name"] = "domain_marker"
    end
end

################################################################################
### StepsizeCallback
################################################################################

# In case of coupled system, use minimum timestep over all systems
function calculate_dt(u_ode, t, cfl_number, semi::SemidiscretizationCoupledDynamic)
    dt = minimum(eachsystem(semi)) do i
        u_ode_slice = get_system_u_ode(u_ode, i, semi)
        calculate_dt(u_ode_slice, t, cfl_number, semi.semis[i])
    end

    return dt
end

################################################################################
### Equations
################################################################################

# """
#     BoundaryConditionCoupled(other_semi_index, indices, uEltype)

# Boundary condition to glue two meshes together. Solution values at the boundary
# of another mesh will be used as boundary values. This requires the use
# of [`SemidiscretizationCoupledDynamic`](@ref). The other mesh is specified by `other_semi_index`,
# which is the index of the mesh in the tuple of semidiscretizations.

# Note that the elements and nodes of the two meshes at the coupled boundary must coincide.
# This is currently only implemented for [`StructuredMesh`](@ref).

# # Arguments
# - `other_semi_index`: the index in `SemidiscretizationCoupledDynamic` of the semidiscretization
#                       from which the values are copied
# - `indices::Tuple`: node/cell indices at the boundary of the mesh in the other
#                     semidiscretization. See examples below.
# - `uEltype::Type`: element type of solution

# # Examples
# ```julia
# # Connect the left boundary of mesh 2 to our boundary such that our positive
# # boundary direction will match the positive y direction of the other boundary
# BoundaryConditionCoupled(2, (:begin, :i), Float64)

# # Connect the same two boundaries oppositely oriented
# BoundaryConditionCoupled(2, (:begin, :i_backwards), Float64)

# # Using this as y_neg boundary will connect `our_cells[i, 1, j]` to `other_cells[j, end-i, end]`
# BoundaryConditionCoupled(2, (:j, :i_backwards, :end), Float64)
# ```

# !!! warning "Experimental code"
#     This is an experimental feature and can change any time.
# """
# mutable struct BoundaryConditionCoupled{NDIMS, NDIMST2M1, uEltype <: Real, Indices}
#     # NDIMST2M1 == NDIMS * 2 - 1
#     # Buffer for boundary values: [variable, nodes_i, nodes_j, cell_i, cell_j]
#     u_boundary        :: Array{uEltype, NDIMST2M1} # NDIMS * 2 - 1
#     other_semi_index  :: Int
#     other_orientation :: Int
#     indices           :: Indices
#     coupling_converter :: Function

#     function BoundaryConditionCoupled(other_semi_index, indices, uEltype, coupling_converter)
#         NDIMS = length(indices)
#         u_boundary = Array{uEltype, NDIMS * 2 - 1}(undef, ntuple(_ -> 0, NDIMS * 2 - 1))

#         if indices[1] in (:begin, :end)
#             other_orientation = 1
#         elseif indices[2] in (:begin, :end)
#             other_orientation = 2
#         else # indices[3] in (:begin, :end)
#             other_orientation = 3
#         end

#         new{NDIMS, NDIMS * 2 - 1, uEltype, typeof(indices)}(u_boundary, other_semi_index,
#                                                             other_orientation, indices, coupling_converter)
#     end
# end

# function Base.eltype(boundary_condition::BoundaryConditionCoupled)
#     eltype(boundary_condition.u_boundary)
# end


# In 2D
function copy_to_coupled_boundary!(u_loc, u_ode, i, semi_coupled)
    @unpack u_indices = semi_coupled

    mesh, equations, solver, cache = mesh_equations_solver_cache(semi_coupled.semis[i])
    u_loc = wrap_array(u_loc, mesh, equations, solver, cache)

    j = 1
    if i == 1
        j = 2
    end
    u_other = get_system_u_ode(u_ode, j, semi_coupled)
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi_coupled.semis[j])
    u_other = wrap_array(u_other, mesh, equations, solver, cache)

    a = @view u_loc[1, :, :, :]
    b = @view u_other[1, :, :, :]
    a[semi_coupled.domain_marker .!= i] = deepcopy(b[semi_coupled.domain_marker .!= i]|)
    a = @view u_loc[2, :, :, :]
    b = @view u_other[2, :, :, :]
    a[semi_coupled.domain_marker .!= i] = deepcopy(b[semi_coupled.domain_marker .!= i])
    a = @view u_loc[3, :, :, :]
    b = @view u_other[3, :, :, :]
    a[semi_coupled.domain_marker .!= i] = deepcopy(b[semi_coupled.domain_marker .!= i])
    a = @view u_loc[4, :, :, :]
    b = @view u_other[4, :, :, :]
    a[semi_coupled.domain_marker .!= i] = deepcopy(b[semi_coupled.domain_marker .!= i])
    # u_loc[1, :, :, :][semi_coupled.domain_marker .!= i] .= 1.0
    # u_loc[2, :, :, :][semi_coupled.domain_marker .!= i] .= 0.0
    # u_loc[3, :, :, :][semi_coupled.domain_marker .!= i] .= 0.0
    # u_loc[4, :, :, :][semi_coupled.domain_marker .!= i] .= 1.0

    # Copy the values of the other system the the boundary values.

    # linear_indices = LinearIndices(size(mesh))

    # i_node_start, i_node_step = index_to_start_step_2d(indices[1], node_index_range)
    # j_node_start, j_node_step = index_to_start_step_2d(indices[2], node_index_range)

    # i_cell_start, i_cell_step = index_to_start_step_2d(indices[1], axes(mesh, 1))
    # j_cell_start, j_cell_step = index_to_start_step_2d(indices[2], axes(mesh, 2))

    # i_cell = i_cell_start
    # j_cell = j_cell_start

    # for cell in cells
    #     i_node = 1
    #     j_node = 1

    #         for v in 1:size(u, 1)
    #             x = cache.elements.node_coordinates[:, i_node, j_node, linear_indices[i_cell, j_cell]]
    #             converted_u = boundary_condition.coupling_converter(x, u[:, i_node, j_node, linear_indices[i_cell, j_cell]])
    #             boundary_condition.u_boundary[v, i, cell] = converted_u[v]

    #             # boundary_condition.u_boundary[v, i, cell] = u[v, i_node, j_node,
    #             #                                               linear_indices[i_cell,
    #             #                                                              j_cell]]
    #         end
    #         i_node += i_node_step
    #         j_node += j_node_step
    #     i_cell += 1
    #     j_cell += 1
    # end
end


# function get_boundary_indices(element, orientation, mesh::StructuredMesh{2})
#     cartesian_indices = CartesianIndices(size(mesh))
#     if orientation == 1
#         # Get index of element in y-direction
#         cell_indices = (cartesian_indices[element][2],)
#     else # orientation == 2
#         # Get index of element in x-direction
#         cell_indices = (cartesian_indices[element][1],)
#     end

#     return cell_indices
# end

################################################################################
### Special elixirs
################################################################################

# # Analyze convergence for SemidiscretizationCoupledDynamic
# function analyze_convergence(errors_coupled, iterations,
#                              semi_coupled::SemidiscretizationCoupledDynamic)
#     # Extract errors: the errors are currently stored as
#     # | iter 1 sys 1 var 1...n | iter 1 sys 2 var 1...n | ... | iter 2 sys 1 var 1...n | ...
#     # but for calling `analyze_convergence` below, we need the following layout
#     # sys n: | iter 1 var 1...n | iter 1 var 1...n | ... | iter 2 var 1...n | ...
#     # That is, we need to extract and join the data for a single system
#     errors = Dict{Symbol, Vector{Float64}}[]
#     for i in eachsystem(semi_coupled)
#         push!(errors, Dict(:l2 => Float64[], :linf => Float64[]))
#     end
#     offset = 0
#     for iter in 1:iterations, i in eachsystem(semi_coupled)
#         # Extract information on current semi
#         semi = semi_coupled.semis[i]
#         _, equations, _, _ = mesh_equations_solver_cache(semi)
#         variablenames = varnames(cons2cons, equations)

#         # Compute offset
#         first = offset + 1
#         last = offset + length(variablenames)
#         offset += length(variablenames)

#         # Append errors to appropriate storage
#         append!(errors[i][:l2], errors_coupled[:l2][first:last])
#         append!(errors[i][:linf], errors_coupled[:linf][first:last])
#     end

#     eoc_mean_values = Vector{Dict{Symbol, Any}}(undef, nsystems(semi_coupled))
#     for i in eachsystem(semi_coupled)
#         # Use visual cues to separate output from multiple systems
#         println()
#         println("="^100)
#         println("# System $i")
#         println("="^100)

#         # Extract information on current semi
#         semi = semi_coupled.semis[i]
#         _, equations, _, _ = mesh_equations_solver_cache(semi)
#         variablenames = varnames(cons2cons, equations)

#         eoc_mean_values[i] = analyze_convergence(errors[i], iterations, variablenames)
#     end

#     return eoc_mean_values
# end
