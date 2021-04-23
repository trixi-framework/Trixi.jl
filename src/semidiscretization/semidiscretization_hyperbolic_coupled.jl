struct SemidiscretizationHyperbolicCoupled{S, I} <: AbstractSemidiscretization
  semis::S
  u_indices::I
  performance_counter::PerformanceCounter
end

function SemidiscretizationHyperbolicCoupled(semis)
  # TODO check that equations are all equal
  n_coeffs = semis .|> mesh_equations_solver_cache .|> (x -> n_coefficients(x...)) |> collect
  u_indices = Vector{UnitRange{Int}}(undef, length(semis))

  for i in 1:length(semis)
    offset = sum(n_coeffs[1:i-1]) + 1
    u_indices[i] = range(offset, length=n_coeffs[i])
  end

  allocate_coupled_boundary_conditions(semis)

  performance_counter = PerformanceCounter()

  SemidiscretizationHyperbolicCoupled{typeof(semis), typeof(u_indices)}(semis, u_indices, performance_counter)
end


function allocate_coupled_boundary_conditions(semis)
  for semi in semis
    mesh, equations, solver, _ = mesh_equations_solver_cache(semi)

    for direction in 1:4 # TODO
      boundary_condition = semi.boundary_conditions[direction]

      allocate_coupled_boundary_condition(boundary_condition, direction, mesh, equations, solver)
    end
  end
end

function allocate_coupled_boundary_condition(boundary_condition, direction, mesh, equations, dg) end

function allocate_coupled_boundary_condition(boundary_condition::BoundaryConditionCoupled, direction, mesh, equations, dg)
  if direction in (1, 2)
    cell_size = size(mesh, 2)
  else
    cell_size = size(mesh, 1)
  end
  boundary_condition.u_boundary = Array{Float64, 3}(undef, nvariables(equations), nnodes(dg), cell_size)
end


function Base.show(io::IO, semi::SemidiscretizationHyperbolicCoupled)
  @nospecialize semi # reduce precompilation time

  print(io, "SemidiscretizationHyperbolicCoupled(")
  print(io,       semis)
  # print(io, ", ", semi.equations)
  # print(io, ", ", semi.initial_condition)
  # print(io, ", ", semi.boundary_conditions)
  # print(io, ", ", semi.source_terms)
  # print(io, ", ", semi.solver)
  # print(io, ", cache(")
  # for (idx,key) in enumerate(keys(semi.cache))
  #   idx > 1 && print(io, " ")
  #   print(io, key)
  # end
  # print(io, "))")
end

function Base.show(io::IO, ::MIME"text/plain", semi::SemidiscretizationHyperbolicCoupled)
  @nospecialize semi # reduce precompilation time

  if get(io, :compact, false)
    show(io, semi)
  else
    summary_header(io, "SemidiscretizationHyperbolicCoupled")
    summary_line(io, "#spatial dimensions", ndims(semi.semis[1].equations))
    summary_line(io, "#meshes", nmeshes(semi))
    summary_line(io, "equations", semi.semis[1].equations |> typeof |> nameof)
    summary_line(io, "initial condition", semi.semis[1].initial_condition)
    # summary_line(io, "boundary conditions", 2*ndims(semi))
    # if (semi.boundary_conditions isa Tuple ||
    #     semi.boundary_conditions isa NamedTuple ||
    #     semi.boundary_conditions isa AbstractArray)
    #   bcs = semi.boundary_conditions
    # else
    #   bcs = collect(semi.boundary_conditions for _ in 1:(2*ndims(semi)))
    # end
    # summary_line(increment_indent(io), "negative x", bcs[1])
    # summary_line(increment_indent(io), "positive x", bcs[2])
    # if ndims(semi) > 1
    #   summary_line(increment_indent(io), "negative y", bcs[3])
    #   summary_line(increment_indent(io), "positive y", bcs[4])
    # end
    # if ndims(semi) > 2
    #   summary_line(increment_indent(io), "negative z", bcs[5])
    #   summary_line(increment_indent(io), "positive z", bcs[6])
    # end
    summary_line(io, "source terms", semi.semis[1].source_terms)
    summary_line(io, "solver", semi.semis[1].solver |> typeof |> nameof)
    summary_line(io, "total #DOFs", ndofs(semi))
    summary_footer(io)
  end
end


@inline Base.ndims(semi::SemidiscretizationHyperbolicCoupled) = ndims(semi.semis[1].mesh)
@inline nmeshes(semi::SemidiscretizationHyperbolicCoupled) = length(semi.semis)
@inline Base.real(semi::SemidiscretizationHyperbolicCoupled) = real(semi.semis[1])

@inline function ndofs(semi::SemidiscretizationHyperbolicCoupled)
  sum(ndofs, semi.semis)
end

@inline function polydeg(semi::SemidiscretizationHyperbolicCoupled)
  solver(semi_) = mesh_equations_solver_cache(semi_)[3]

  semi.semis .|> solver .|> polydeg
end

@inline function nelements(semi::SemidiscretizationHyperbolicCoupled)
  solver_cache(semi_) = mesh_equations_solver_cache(semi_)[3:4]
  n_elements(solver_cache) = nelements(solver_cache...)

  semi.semis .|> solver_cache .|> n_elements
end


@inline function mesh_equations_solver_cache(semi::SemidiscretizationHyperbolicCoupled)
  @unpack equations = semi.semis[1]
  return nothing, equations, nothing, nothing
end


function compute_coefficients(t, semi::SemidiscretizationHyperbolicCoupled)
  @unpack u_indices = semi

  u_ode = Vector{real(semi)}(undef, u_indices[end][end])

  for i in 1:nmeshes(semi)
    # Call `compute_coefficients` in `src/semidiscretization/semidiscretization.jl`
    u_ode[u_indices[i]] .= compute_coefficients(semi.semis[i].initial_condition, t, semi.semis[i])
  end
  
  return u_ode
end


function calc_error_norms(func, u_ode::AbstractVector, t, analyzers, semi::SemidiscretizationHyperbolicCoupled, caches_analysis)
  @unpack semis, u_indices = semi

  # TODO This is horrible
  u_temp = wrap_array(u_ode[u_indices[1]], semis[1].mesh, semis[1].equations, semis[1].solver, semis[1].cache)
  l2_integral = zero(func(get_node_vars(u_temp, semis[1].equations, semis[1].solver, 1, 1, 1), semis[1].equations))
  linf_error = copy(l2_integral)
  total_volume = zero(real(semi))

  for i in 1:nmeshes(semi)
    @unpack mesh, equations, initial_condition, solver, cache = semis[i]
    u = wrap_array(u_ode[u_indices[i]], mesh, equations, solver, cache)

    l2_integral_, total_volume_, linf_error_ = calc_l2_integral_and_linf(
      func, u, t, analyzers[i], mesh, equations, initial_condition, solver, cache, caches_analysis[i])

    l2_integral += l2_integral_
    total_volume += total_volume_
    linf_error = max(linf_error, linf_error_)
  end

  # For L2 error, divide by total volume
  l2_error = @. sqrt(l2_integral / total_volume)

  return l2_error, linf_error
end


function integrate(func::Func, u_ode::AbstractVector, semi::SemidiscretizationHyperbolicCoupled; normalize=true) where {Func}
  @unpack semis, u_indices = semi
  # TODO This is horrible
  u_temp = wrap_array(u_ode[u_indices[1]], semis[1].mesh, semis[1].equations, semis[1].solver, semis[1].cache)
  integral = zero(func(get_node_vars(u_temp, semis[1].equations, semis[1].solver, 1, 1, 1), semis[1].equations))

  for i in 1:nmeshes(semi)
    mesh, equations, solver, cache = mesh_equations_solver_cache(semis[i])

    u = wrap_array(u_ode[u_indices[i]], mesh, equations, solver, cache)
    integral += integrate(func, u, mesh, equations, solver, cache, normalize=normalize)
  end

  return integral
end


function rhs!(du_ode, u_ode, semi::SemidiscretizationHyperbolicCoupled, t)
  @unpack u_indices = semi

  time_start = time_ns()

  @timeit_debug timer() "prolong to coupled boundaries" prolong_to_coupled_boundaries(u_ode, semi)

  for i in 1:nmeshes(semi)
    u_loc  = @view u_ode[u_indices[i]]
    du_loc = @view du_ode[u_indices[i]]

    rhs!(du_loc, u_loc, semi.semis[i], t)
  end

  runtime = time_ns() - time_start
  put!(semi.performance_counter, runtime)

  return nothing
end


function prolong_to_coupled_boundaries(u_ode, semi)
  for semi_ in semi.semis
    for boundary_condition in semi_.boundary_conditions
      prolong2boundary(boundary_condition, u_ode, semi)
    end
  end
end


function prolong2boundary(boundary_condition, u_ode, semi) end

function indexfunction(indices, i, size, dim)
  if indices[dim] === :i
    return i
  elseif indices[dim] === :mi
    return size[dim] - i + 1
  elseif indices[dim] == 1
    return 1
  elseif indices[dim] == :end
    return size[dim]
  end
end

function prolong2boundary(boundary_condition::BoundaryConditionCoupled, u_ode, semi)
  @unpack u_indices = semi
  @unpack other_mesh_id, other_mesh_orientation, indices = boundary_condition

  mesh, equations, solver, cache = mesh_equations_solver_cache(semi.semis[other_mesh_id])
  @views u = wrap_array(u_ode[u_indices[other_mesh_id]], mesh, equations, solver, cache)

  linear_indices = LinearIndices(size(mesh))
  size_ = (nnodes(solver), nnodes(solver))

  if other_mesh_orientation == 1
    for cell_y in axes(mesh, 2), i in 1:size(u, 3), v in 1:size(u, 1)
      boundary_condition.u_boundary[v, i, cell_y] = u[v, indexfunction(indices, i, size_, 1), 
                                                         indexfunction(indices, i, size_, 2), 
                                                         linear_indices[indexfunction(indices, cell_y, size(mesh), 1), 
                                                                        indexfunction(indices, cell_y, size(mesh), 2)]]
    end
  elseif other_mesh_orientation == 2
    for cell_x in axes(mesh, 1), j in 1:size(u, 3), v in 1:size(u, 1)
      boundary_condition.u_boundary[v, j, cell_x] = u[v, indexfunction(indices, j, size_, 1), 
                                                         indexfunction(indices, j, size_, 2), 
                                                         linear_indices[indexfunction(indices, cell_x, size(mesh), 1), 
                                                                        indexfunction(indices, cell_x, size(mesh), 2)]]
    end
  else
    error("Something went horribly wrong!")
  end
end
