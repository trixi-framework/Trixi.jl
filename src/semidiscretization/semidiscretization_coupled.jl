"""
    SemidiscretizationCoupled

A struct used to bundle multiple semidiscretizations.
[`semidiscretize`](@ref) will return an `ODEProblem` that synchronizes time steps between the semidiscretizations.
Each call of `rhs!` will call `rhs!` for each semidiscretization individually.
The semidiscretizations can be coupled by gluing meshes together using [`BoundaryConditionCoupled`](@ref).

!!! warning "Experimental code"
    This is an experimental feature and can change any time.
"""
struct SemidiscretizationCoupled{S, Indices, EquationList} <: AbstractSemidiscretization
  semis::S
  u_indices::Indices # u_ode[u_indices[i]] is the part of u_ode corresponding to semis[i]
  performance_counter::PerformanceCounter
end


"""
    SemidiscretizationCoupled(semis...)

Create a coupled semidiscretization that consists of the semidiscretizations passed as arguments.
"""
function SemidiscretizationCoupled(semis...)
  @assert all(semi -> ndims(semi) == ndims(semis[1]), semis) "All semidiscretizations must have the same dimension!"

  # Number of coefficients for each semidiscretization
  n_coefficients = zeros(Int, length(semis))
  for i in 1:length(semis)
    _, equations, _, _ = mesh_equations_solver_cache(semis[i])
    n_coefficients[i] = ndofs(semis[i]) * nvariables(equations)
  end

  # Compute range of coefficients associated with each semidiscretization and allocate coupled BCs
  u_indices = Vector{UnitRange{Int}}(undef, length(semis))
  for i in 1:length(semis)
    offset = sum(n_coefficients[1:(i-1)]) + 1
    u_indices[i] = range(offset, length=n_coefficients[i])

    allocate_coupled_boundary_conditions(semis[i])
  end

  performance_counter = PerformanceCounter()

  SemidiscretizationCoupled{typeof(semis), typeof(u_indices), typeof(performance_counter)}(semis, u_indices, performance_counter)
end


function Base.show(io::IO, semi::SemidiscretizationCoupled)
  @nospecialize semi # reduce precompilation time

  print(io, "SemidiscretizationCoupled($(semi.semis))")
end

function Base.show(io::IO, ::MIME"text/plain", semi::SemidiscretizationCoupled)
  @nospecialize semi # reduce precompilation time

  if get(io, :compact, false)
    show(io, semi)
  else
    summary_header(io, "SemidiscretizationCoupled")
    summary_line(io, "#spatial dimensions", ndims(semi.semis[1]))
    summary_line(io, "#systems", nsystems(semi))
    for i in eachsystem(semi)
      summary_line(io, "system", i)
      mesh, equations, solver, _ = mesh_equations_solver_cache(semi.semis[i])
      summary_line(increment_indent(io), "mesh", mesh |> typeof |> nameof)
      summary_line(increment_indent(io), "equations", equations |> typeof |> nameof)
      summary_line(increment_indent(io), "initial condition", semi.semis[i].initial_condition)
      # TODO boundary conditions? That will be 36 BCs for a cubed sphere
      summary_line(increment_indent(io), "source terms", semi.semis[i].source_terms)
      summary_line(increment_indent(io), "solver", solver |> typeof |> nameof)
    end
    summary_line(io, "total #DOFs", ndofs(semi))
    summary_footer(io)
  end
end


function print_summary_semidiscretization(io::IO, semi::SemidiscretizationCoupled)
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


@inline Base.ndims(semi::SemidiscretizationCoupled) = ndims(semi.semis[1])

@inline nsystems(semi::SemidiscretizationCoupled) = length(semi.semis)

@inline eachsystem(semi::SemidiscretizationCoupled) = Base.OneTo(nsystems(semi))

@inline Base.real(semi::SemidiscretizationCoupled) = promote_type(real.(semi.semis)...)

@inline Base.eltype(semi::SemidiscretizationCoupled) = promote_type(eltype.(semi.semis)...)

@inline function ndofs(semi::SemidiscretizationCoupled)
  sum(ndofs, semi.semis)
end

@inline function nelements(semi::SemidiscretizationCoupled)
  return sum(semi.semis) do semi_
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi_)

    nelements(mesh, solver, cache)
  end
end

@inline function total_volume(semi::SemidiscretizationCoupled)
  sum(semi.semis) do semi_
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi_)
    total_volume(mesh, solver, cache)
  end
end


function compute_coefficients(t, semi::SemidiscretizationCoupled)
  @unpack u_indices = semi

  u_ode = Vector{real(semi)}(undef, u_indices[end][end])

  for i in eachsystem(semi)
    # Call `compute_coefficients` in `src/semidiscretization/semidiscretization.jl`
    u_ode[u_indices[i]] .= compute_coefficients(t, semi.semis[i])
  end

  return u_ode
end


function rhs!(du_ode, u_ode, semi::SemidiscretizationCoupled, t)
  @unpack u_indices = semi

  time_start = time_ns()

  @trixi_timeit timer() "copy to coupled boundaries" begin
    for semi_ in semi.semis
      copy_to_coupled_boundary!(semi_.boundary_conditions, u_ode, semi)
    end
  end

  # Call rhs! for each semidiscretization
  for i in eachsystem(semi)
    u_loc  = @view u_ode[u_indices[i]]
    du_loc = @view du_ode[u_indices[i]]

    rhs!(du_loc, u_loc, semi.semis[i], t)
  end

  runtime = time_ns() - time_start
  put!(semi.performance_counter, runtime)

  return nothing
end


################################################################################
### SaveSolutionCallback
################################################################################

# Save mesh for a coupled semidiscretization, which contains multiple meshes internally
function save_mesh(semi::SemidiscretizationCoupled, output_directory, timestep=0)
  for i in eachsystem(semi)
    mesh, _, _, _ = mesh_equations_solver_cache(semi.semis[i])

    if mesh.unsaved_changes
      mesh.current_filename = save_mesh_file(mesh, output_directory, system=i)
      mesh.unsaved_changes = false
    end
  end
end


@inline function save_solution_file(semi::SemidiscretizationCoupled, u_ode, solution_callback,
                                    integrator)
  @unpack semis, u_indices = semi

  for i in eachsystem(semi)
    u_ode_slice = @view u_ode[u_indices[i]]
    save_solution_file(semis[i], u_ode_slice, solution_callback, integrator, system=i)
  end
end


################################################################################
### StepsizeCallback
################################################################################

# In case of coupled system, use minimum timestep over all systems
function calculate_dt(u_ode, t, cfl_number, semi::SemidiscretizationCoupled)
  @unpack u_indices = semi

  dt = minimum(eachsystem(semi)) do i
    u_ode_slice = @view u_ode[u_indices[i]]
    calculate_dt(u_ode_slice, t, cfl_number, semi.semis[i])
  end

  return dt
end


################################################################################
### Equations
################################################################################

"""
    BoundaryConditionCoupled(other_semi_index, indices, uEltype)

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

# Examples
```julia
# Connect the left boundary of mesh 2 to our boundary such that our positive
# boundary direction will match the positive y direction of the other boundary
BoundaryConditionCoupled(2, (:begin, :i), Float64)

# Connect the same two boundaries oppositely oriented
BoundaryConditionCoupled(2, (:begin, :i_backwards), Float64)

# Using this as y_neg boundary will connect `our_cells[i, 1, j]` to `other_cells[j, end-i, end]`
BoundaryConditionCoupled(2, (:j, :i_backwards, :end), Float64)
```

!!! warning "Experimental code"
    This is an experimental feature and can change any time.
"""
mutable struct BoundaryConditionCoupled{NDIMS, NDIMST2M1, uEltype<:Real, Indices}
  # NDIMST2M1 == NDIMS * 2 - 1
  # Buffer for boundary values: [variable, nodes_i, nodes_j, cell_i, cell_j]
  u_boundary       ::Array{uEltype, NDIMST2M1} # NDIMS * 2 - 1
  other_semi_index ::Int
  other_orientation::Int
  indices          ::Indices

  function BoundaryConditionCoupled(other_semi_index, indices, uEltype)
    NDIMS = length(indices)
    u_boundary = Array{uEltype, NDIMS*2-1}(undef, ntuple(_ -> 0, NDIMS*2-1))

    if indices[1] in (:begin, :end)
      other_orientation = 1
    elseif indices[2] in (:begin, :end)
      other_orientation = 2
    else # indices[3] in (:begin, :end)
      other_orientation = 3
    end

    new{NDIMS, NDIMS*2-1, uEltype, typeof(indices)}(
      u_boundary, other_semi_index, other_orientation, indices)
  end
end

Base.eltype(boundary_condition::BoundaryConditionCoupled) = eltype(boundary_condition.u_boundary)

function (boundary_condition::BoundaryConditionCoupled)(u_inner, orientation, direction,
                                                        cell_indices, surface_node_indices,
                                                        surface_flux_function, equations)
  # get_node_vars(boundary_condition.u_boundary, equations, solver, surface_node_indices..., cell_indices...),
  # but we don't have a solver here
  u_boundary = SVector(ntuple(v -> boundary_condition.u_boundary[v, surface_node_indices..., cell_indices...],
                              Val(nvariables(equations))))

  # Calculate boundary flux
  if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
  end

  return flux
end


function allocate_coupled_boundary_conditions(semi::AbstractSemidiscretization)
  n_boundaries = 2 * ndims(semi)
  mesh, equations, solver, _ = mesh_equations_solver_cache(semi)

  for direction in 1:n_boundaries
    boundary_condition = semi.boundary_conditions[direction]

    allocate_coupled_boundary_condition(boundary_condition, direction, mesh, equations, solver)
  end
end


# Don't do anything for other BCs than BoundaryConditionCoupled
function allocate_coupled_boundary_condition(boundary_condition, direction, mesh, equations, solver)
  return nothing
end

# In 2D
function allocate_coupled_boundary_condition(boundary_condition::BoundaryConditionCoupled{2},
                                             direction, mesh, equations, dg::DGSEM)
  if direction in (1, 2)
    cell_size = size(mesh, 2)
  else
    cell_size = size(mesh, 1)
  end

  uEltype = eltype(boundary_condition)
  boundary_condition.u_boundary = Array{uEltype, 3}(undef, nvariables(equations), nnodes(dg),
                                                    cell_size)
end

# Don't do anything for other BCs than BoundaryConditionCoupled
function copy_to_coupled_boundary!(boundary_condition, u_ode, semi)
  return nothing
end

function copy_to_coupled_boundary!(boundary_conditions::Union{Tuple, NamedTuple}, u_ode, semi)
  for boundary_condition in boundary_conditions
    copy_to_coupled_boundary!(boundary_condition, u_ode, semi)
  end
end

# In 2D
function copy_to_coupled_boundary!(boundary_condition::BoundaryConditionCoupled{2}, u_ode, semi)
  @unpack u_indices = semi
  @unpack other_semi_index, other_orientation, indices = boundary_condition

  mesh, equations, solver, cache = mesh_equations_solver_cache(semi.semis[other_semi_index])
  @views u = wrap_array(u_ode[u_indices[other_semi_index]], mesh, equations, solver, cache)

  linear_indices = LinearIndices(size(mesh))

  if other_orientation == 1
    cells = axes(mesh, 2)
  else # other_orientation == 2
    cells = axes(mesh, 1)
  end

  # Copy solution data to the coupled boundary using "delayed indexing" with
  # a start value and a step size to get the correct face and orientation.
  node_index_range = eachnode(solver)
  i_node_start, i_node_step = index_to_start_step_2d(indices[1], node_index_range)
  j_node_start, j_node_step = index_to_start_step_2d(indices[2], node_index_range)

  i_cell_start, i_cell_step = index_to_start_step_2d(indices[1], axes(mesh, 1))
  j_cell_start, j_cell_step = index_to_start_step_2d(indices[2], axes(mesh, 2))

  i_cell = i_cell_start
  j_cell = j_cell_start

  for cell in cells
    i_node = i_node_start
    j_node = j_node_start

    for i in eachnode(solver)
      for v in 1:size(u, 1)
        boundary_condition.u_boundary[v, i, cell] = u[v, i_node, j_node, 
                                                      linear_indices[i_cell, j_cell]]
      end
      i_node += i_node_step
      j_node += j_node_step
    end
    i_cell += i_cell_step
    j_cell += j_cell_step
  end
end


################################################################################
### DGSEM/structured
################################################################################

@inline function calc_boundary_flux_by_direction!(surface_flux_values, u, t, orientation,
                                                  boundary_condition::BoundaryConditionCoupled,
                                                  mesh::StructuredMesh, equations,
                                                  surface_integral, dg::DG, cache,
                                                  direction, node_indices, surface_node_indices, element)
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
  normal = sign_jacobian * get_contravariant_vector(orientation, contravariant_vectors,
                                                    node_indices..., element)

  # If the mapping is orientation-reversing, the normal vector will be reversed (see above).
  # However, the flux now has the wrong sign, since we need the physical flux in normal direction.
  flux = sign_jacobian * boundary_condition(u_inner, normal, direction, cell_indices,
                                            surface_node_indices, surface_flux, equations)

  for v in eachvariable(equations)
    surface_flux_values[v, surface_node_indices..., direction, element] = flux[v]
  end
end

function get_boundary_indices(element, orientation, mesh::StructuredMesh{2})
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
