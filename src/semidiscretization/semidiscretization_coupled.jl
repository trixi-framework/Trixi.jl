"""
    SemidiscretizationCoupled

A struct used to bundle multiple semidiscretizations.
`semidiscretize` will return an `ODEproblem` that synchronizes time steps between the semidiscretizations.
Each call of `rhs!` will call `rhs!` for each semidiscretization individually.
The semidiscretizations can be coupled by gluing meshes together using `BoundaryConditionCoupled`.
"""
struct SemidiscretizationCoupled{S, I} <: AbstractSemidiscretization
  semis::S
  u_indices::I # u_ode[u_indices[i]] is the part of u_ode corresponding to semis[i]
  performance_counter::PerformanceCounter
end


"""
    SemidiscretizationCoupled(semis)

Create a coupled semidiscretization that consists of the semidiscretizations contained in the tuple `semis`.
"""
function SemidiscretizationCoupled(semis)
  @assert all(semi -> ndims(semi) == ndims(semis[1]), semis) "All semidiscretizations must have the same dimension!"

  # Number of coefficients as Vector
  n_coeffs = semis .|> mesh_equations_solver_cache .|> (x -> n_coefficients(x...)) |> collect
  u_indices = Vector{UnitRange{Int}}(undef, length(semis))

  for i in 1:length(semis)
    offset = sum(n_coeffs[1:i-1]) + 1
    u_indices[i] = range(offset, length=n_coeffs[i])

    allocate_coupled_boundary_conditions(semis[i].boundary_conditions, semis[i])
  end

  performance_counter = PerformanceCounter()

  SemidiscretizationCoupled{typeof(semis), typeof(u_indices)}(semis, u_indices, performance_counter)
end


function Base.show(io::IO, semi::SemidiscretizationCoupled)
  @nospecialize semi # reduce precompilation time

  mesh, equations, solver, _ = mesh_equations_solver_cache(semi.semis[1])

  print(io, "SemidiscretizationCoupled(")
  print(io, ", ", equations)
  print(io, ", ", semi.semis[1].initial_condition)
  # print(io, ", ", semi.boundary_conditions) TODO
  print(io, ", ", semi.semis[1].source_terms)
  print(io, ", ", solver)
  print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", semi::SemidiscretizationCoupled)
  @nospecialize semi # reduce precompilation time

  if get(io, :compact, false)
    show(io, semi)
  else
    summary_header(io, "SemidiscretizationCoupled")
    summary_line(io, "#spatial dimensions", ndims(semi.semis[1]))
    summary_line(io, "#meshes", nmeshes(semi))
    summary_line(io, "equations", mesh_equations_solver_cache(semi.semis[1])[2] |> typeof |> nameof)
    summary_line(io, "initial condition", semi.semis[1].initial_condition)
    # TODO boundary conditions?
    summary_line(io, "source terms", semi.semis[1].source_terms)
    summary_line(io, "solver", mesh_equations_solver_cache(semi.semis[1])[3] |> typeof |> nameof)
    summary_line(io, "total #DOFs", ndofs(semi))
    summary_footer(io)
  end
end


function summary_semidiscretization(semi::SemidiscretizationCoupled, io, io_context)
  show(io_context, MIME"text/plain"(), semi)
  println(io, "\n")
  mesh, equations, solver, _ = mesh_equations_solver_cache(semi.semis[1])
  show(io_context, MIME"text/plain"(), mesh)
  println(io, "\n")
  show(io_context, MIME"text/plain"(), equations)
  println(io, "\n")
  show(io_context, MIME"text/plain"(), solver)
  println(io, "\n")
end

function summary_solver(semi::SemidiscretizationCoupled)
  _, _, solver, _ = mesh_equations_solver_cache(semi.semis[1])
  summary(solver)
end


@inline Base.ndims(semi::SemidiscretizationCoupled) = ndims(semi.semis[1])

@inline nmeshes(semi::SemidiscretizationCoupled) = length(semi.semis)

@inline Base.real(semi::SemidiscretizationCoupled) = promote_type(real.(semi.semis)...)

@inline Base.eltype(semi::SemidiscretizationCoupled) = promote_type(eltype.(semi.semis)...)

@inline function ndofs(semi::SemidiscretizationCoupled)
  sum(ndofs, semi.semis)
end

@inline function polydeg(semi::SemidiscretizationCoupled)
  solver(semi_) = mesh_equations_solver_cache(semi_)[3]

  semi.semis .|> solver .|> polydeg
end

@inline function nelements(semi::SemidiscretizationCoupled)
  solver_cache(semi_) = mesh_equations_solver_cache(semi_)[3:4]
  n_elements(solver_cache) = nelements(solver_cache...)

  semi.semis .|> solver_cache .|> n_elements |> sum
end


@inline function mesh_equations_solver_cache(semi::SemidiscretizationCoupled)
  _, equations, _, _ = mesh_equations_solver_cache(semi.semis[1])
  return nothing, equations, nothing, nothing
end


function calc_error_norms(func, u_ode::AbstractVector, t, analyzers,
                          semi::SemidiscretizationCoupled, caches_analysis;
                          normalize=true)
  @unpack semis, u_indices = semi

  # Sum up L2 integrals, use max on Linf error
  op(x, y) = (x[1] + y[1], max(x[2], y[2]))

  l2_error, linf_error = mapreduce(op, 1:nmeshes(semi)) do i
    calc_error_norms(func, u_ode[u_indices[i]], t, analyzers[i],
                     semis[i], caches_analysis[i]; normalize=false)
  end

  if normalize
    # For L2 error, divide by total volume
    total_volume_ = total_volume(semi)
    l2_error = @. sqrt(l2_error / total_volume_)
  end

  return l2_error, linf_error
end


function integrate(func::Func, u_ode::AbstractVector, semi::SemidiscretizationCoupled; normalize=true) where {Func}
  @unpack semis, u_indices = semi

  integral = sum(1:nmeshes(semi)) do i
    mesh, equations, solver, cache = mesh_equations_solver_cache(semis[i])
    u = wrap_array(u_ode[u_indices[i]], mesh, equations, solver, cache)

    integrate(func, u, mesh, equations, solver, cache, normalize=false)
  end

  # Normalize with total volume
  if normalize
    total_volume_ = total_volume(semi)
    integral = integral / total_volume_
  end

  return integral
end


function total_volume(semi::SemidiscretizationCoupled)
  sum(semi.semis) do semi_
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi_)
    total_volume(mesh, solver, cache)
  end
end


function compute_coefficients(t, semi::SemidiscretizationCoupled)
  @unpack u_indices = semi

  u_ode = Vector{real(semi)}(undef, u_indices[end][end])

  for i in 1:nmeshes(semi)
    # Call `compute_coefficients` in `src/semidiscretization/semidiscretization.jl`
    u_ode[u_indices[i]] .= compute_coefficients(t, semi.semis[i])
  end

  return u_ode
end


function allocate_coupled_boundary_conditions(boundary_conditions, semi) end

function allocate_coupled_boundary_conditions(boundary_conditions::Union{Tuple, NamedTuple}, semi)
  n_boundaries = 2 * ndims(semi)
  mesh, equations, solver, _ = mesh_equations_solver_cache(semi)

  for direction in 1:n_boundaries
    boundary_condition = semi.boundary_conditions[direction]

    allocate_coupled_boundary_condition(boundary_condition, direction, mesh, equations, solver)
  end
end

function allocate_coupled_boundary_condition(boundary_condition, direction, mesh, equations, solver) end

function allocate_coupled_boundary_condition(boundary_condition::BoundaryConditionCoupled{3}, direction, mesh, equations, dg::DG)
  if direction in (1, 2)
    cell_size = size(mesh, 2)
  else
    cell_size = size(mesh, 1)
  end
  boundary_condition.u_boundary = Array{Float64, 3}(undef, nvariables(equations), nnodes(dg), cell_size)
end

function allocate_coupled_boundary_condition(boundary_condition::BoundaryConditionCoupled{5}, direction, mesh, equations, dg::DG)
  if direction in (1, 2)
    cell_size = (size(mesh, 2), size(mesh, 3))
  elseif direction in (3, 4)
    cell_size = (size(mesh, 1), size(mesh, 3))
  else # direction in (5, 6)
    (size(mesh, 1), size(mesh, 2))
  end
  boundary_condition.u_boundary = Array{Float64, 5}(undef, nvariables(equations), nnodes(dg), nnodes(dg), cell_size...)
end


function rhs!(du_ode, u_ode, semi::SemidiscretizationCoupled, t)
  @unpack u_indices = semi

  time_start = time_ns()

  @trixi_timeit timer() "copy to coupled boundaries" begin
    for semi_ in semi.semis
      copy_to_coupled_boundary(semi_.boundary_conditions, u_ode, semi)
    end
  end

  # Call rhs! for each semidiscretization
  for i in 1:nmeshes(semi)
    u_loc  = @view u_ode[u_indices[i]]
    du_loc = @view du_ode[u_indices[i]]

    rhs!(du_loc, u_loc, semi.semis[i], t)
  end

  runtime = time_ns() - time_start
  put!(semi.performance_counter, runtime)

  return nothing
end


function copy_to_coupled_boundary(boundary_condition, u_ode, semi) end

function copy_to_coupled_boundary(boundary_conditions::Union{Tuple, NamedTuple}, u_ode, semi)
  for boundary_condition in boundary_conditions
    copy_to_coupled_boundary(boundary_condition, u_ode, semi)
  end
end

# In 2D
function copy_to_coupled_boundary(boundary_condition::BoundaryConditionCoupled{3}, u_ode, semi)
  @unpack u_indices = semi
  @unpack other_semi_index, other_orientation, indices = boundary_condition

  mesh, equations, solver, cache = mesh_equations_solver_cache(semi.semis[other_semi_index])
  @views u = wrap_array(u_ode[u_indices[other_semi_index]], mesh, equations, solver, cache)

  size_ = (nnodes(solver), nnodes(solver))
  linear_indices = LinearIndices(size(mesh))

  if other_orientation == 1
    cells = axes(mesh, 2)
  else # other_orientation == 2
    cells = axes(mesh, 1)
  end

  for cell in cells, i in eachnode(solver), v in 1:size(u, 1)
    boundary_condition.u_boundary[v, i, cell] = u[v, indexfunction(indices, size_, 1, i),
                                                     indexfunction(indices, size_, 2, i),
                                                     linear_indices[indexfunction(indices, size(mesh), 1, cell),
                                                                    indexfunction(indices, size(mesh), 2, cell)]]
  end
end


# In 3D
function copy_to_coupled_boundary(boundary_condition::BoundaryConditionCoupled{5}, u_ode, semi)
  @unpack u_indices = semi
  @unpack other_semi_index, other_orientation, indices = boundary_condition

  mesh, equations, solver, cache = mesh_equations_solver_cache(semi.semis[other_semi_index])
  @views u = wrap_array(u_ode[u_indices[other_semi_index]], mesh, equations, solver, cache)

  size_ = (nnodes(solver), nnodes(solver), nnodes(solver))
  linear_indices = LinearIndices(size(mesh))

  if other_orientation == 1
    cells = (axes(mesh, 2), axes(mesh, 3))
  elseif other_orientation == 2
    cells = (axes(mesh, 1), axes(mesh, 3))
  else # other_orientation == 3
    cells = (axes(mesh, 1), axes(mesh, 2))
  end

  for cell_j in cells[2], cell_i in cells[1], j in eachnode(solver), i in eachnode(solver), v in 1:size(u, 1)
    boundary_condition.u_boundary[v, i, j, cell_i, cell_j] = u[v, indexfunction(indices, size_, 1, i, j),
                                                                  indexfunction(indices, size_, 2, i, j),
                                                                  indexfunction(indices, size_, 3, i, j),
                                                                  linear_indices[indexfunction(indices, size(mesh), 1, cell_i, cell_j),
                                                                                 indexfunction(indices, size(mesh), 2, cell_i, cell_j),
                                                                                 indexfunction(indices, size(mesh), 3, cell_i, cell_j)]]
  end
end


function indexfunction(indices, size, dim, i, j=0)
  if indices[dim] === :i
    return i
  elseif indices[dim] === :mi
    return size[dim] - i + 1
  elseif indices[dim] === :j
    return j
  elseif indices[dim] === :mj
    return size[dim] - j + 1
  elseif indices[dim] === :one
    return 1
  elseif indices[dim] === :end
    return size[dim]
  end

  error("Invalid identifier: Only 1, :end, :i, :j, :mi, :mj are valid index identifiers")
end
