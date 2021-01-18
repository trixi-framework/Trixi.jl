
# TODO: Taal refactor, Mesh<:AbstractMesh{NDIMS}, Equations<:AbstractEquations{NDIMS} ?
"""
    SemidiscretizationParabolicAuxVars

A struct containing everything needed to describe a spatial semidiscretization
of a parabolic conservation law.
"""
struct SemidiscretizationParabolicAuxVars{Mesh, Equations, InitialCondition, BoundaryConditions,
                                          SourceTerms, Solver, Cache} <: AbstractSemidiscretization

  mesh::Mesh
  equations::Equations

  # This guy is a bit messy since we abuse it as some kind of "exact solution"
  # although this doesn't really exist...
  initial_condition::InitialCondition

  boundary_conditions::BoundaryConditions
  source_terms::SourceTerms
  solver::Solver
  cache::Cache
  performance_counter::PerformanceCounter

  function SemidiscretizationParabolicAuxVars{Mesh, Equations, InitialCondition, BoundaryConditions,
                                              SourceTerms, Solver, Cache}(
      mesh::Mesh, equations::Equations,
      initial_condition::InitialCondition, boundary_conditions::BoundaryConditions,
      source_terms::SourceTerms,
      solver::Solver, cache::Cache) where {Mesh, Equations, InitialCondition, BoundaryConditions, SourceTerms, Solver, Cache}
    @assert ndims(mesh) == ndims(equations)

    performance_counter = PerformanceCounter()

    new(mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache, performance_counter)
  end
end

"""
    SemidiscretizationParabolicAuxVars(mesh, equations, initial_condition, solver;
                                 source_terms=nothing,
                                 boundary_conditions=boundary_condition_periodic)

Construct a semidiscretization of a parabolic PDE.
"""
function SemidiscretizationParabolicAuxVars(mesh, equations, initial_condition, solver;
                                            source_terms=nothing,
                                            boundary_conditions=boundary_condition_periodic,
                                            RealT=real(solver))

  cache = create_cache(mesh, equations, solver, RealT)
  _boundary_conditions = digest_boundary_conditions(boundary_conditions)

  solver_auxvars = create_solver_auxvars(solver)

  equations_grad_x = GradientEquations2D(nvariables(equations), 1)
  semi_grad_x = SemidiscretizationHyperbolic(mesh, equations_grad_x,
                                             initial_condition_constant, solver_auxvars)

  equations_grad_y = GradientEquations2D(nvariables(equations), 2)
  semi_grad_y = SemidiscretizationHyperbolic(mesh, equations_grad_y,
                                             initial_condition_constant, solver_auxvars)

  u_ode_grad_x = compute_coefficients(nothing, semi_grad_x)
  u_ode_grad_y = compute_coefficients(nothing, semi_grad_y)
  u_ode_grad_xx = compute_coefficients(nothing, semi_grad_x)
  u_ode_grad_yy = compute_coefficients(nothing, semi_grad_y)

  cache = (; cache..., semi_grad_x, u_ode_grad_x, semi_grad_y, u_ode_grad_y, u_ode_grad_xx, u_ode_grad_yy)

  SemidiscretizationParabolicAuxVars{typeof(mesh), typeof(equations), typeof(initial_condition), typeof(_boundary_conditions), typeof(source_terms), typeof(solver), typeof(cache)}(
    mesh, equations, initial_condition, _boundary_conditions, source_terms, solver, cache)
end


function Base.show(io::IO, semi::SemidiscretizationParabolicAuxVars)
  print(io, "SemidiscretizationParabolicAuxVars(")
  print(io,       semi.mesh)
  print(io, ", ", semi.equations)
  print(io, ", ", semi.initial_condition)
  print(io, ", ", semi.boundary_conditions)
  print(io, ", ", semi.source_terms)
  print(io, ", ", semi.solver)
  print(io, ", cache(")
  for (idx,key) in enumerate(keys(semi.cache))
    idx > 1 && print(io, " ")
    print(io, key)
  end
  print(io, "))")
end

function Base.show(io::IO, ::MIME"text/plain", semi::SemidiscretizationParabolicAuxVars)
  if get(io, :compact, false)
    show(io, semi)
  else
    summary_header(io, "SemidiscretizationParabolicAuxVars")
    summary_line(io, "#spatial dimensions", ndims(semi.equations))
    summary_line(io, "mesh", semi.mesh)
    summary_line(io, "equations", typeof(semi.equations).name)
    summary_line(io, "initial condition", semi.initial_condition)
    summary_line(io, "boundary conditions", 2*ndims(semi))
    if (semi.boundary_conditions isa Tuple ||
        semi.boundary_conditions isa NamedTuple ||
        semi.boundary_conditions isa AbstractArray)
      bcs = semi.boundary_conditions
    else
      bcs = collect(semi.boundary_conditions for _ in 1:(2*ndims(semi)))
    end
    summary_line(increment_indent(io), "negative x", bcs[1])
    summary_line(increment_indent(io), "positive x", bcs[2])
    if ndims(semi) > 1
      summary_line(increment_indent(io), "negative y", bcs[3])
      summary_line(increment_indent(io), "positive y", bcs[4])
    end
    if ndims(semi) > 2
      summary_line(increment_indent(io), "negative z", bcs[5])
      summary_line(increment_indent(io), "positive z", bcs[6])
    end
    summary_line(io, "source terms", semi.source_terms)
    summary_line(io, "solver", typeof(semi.solver).name)
    summary_line(io, "total #DOFs", ndofs(semi))
    summary_footer(io)
  end
end


@inline Base.ndims(semi::SemidiscretizationParabolicAuxVars) = ndims(semi.mesh)

@inline nvariables(semi::SemidiscretizationParabolicAuxVars) = nvariables(semi.equations)

@inline Base.real(semi::SemidiscretizationParabolicAuxVars) = real(semi.solver)


@inline function mesh_equations_solver_cache(semi::SemidiscretizationParabolicAuxVars)
  @unpack mesh, equations, solver, cache = semi
  return mesh, equations, solver, cache
end


function calc_error_norms(func, u_ode::AbstractVector, t, analyzer, semi::SemidiscretizationParabolicAuxVars, cache_analysis)
  @unpack mesh, equations, initial_condition, solver, cache = semi
  u = wrap_array(u_ode, mesh, equations, solver, cache)

  calc_error_norms(func, u, t, analyzer, mesh, equations, initial_condition, solver, cache, cache_analysis)
end

function calc_error_norms(func, u, t, analyzer, semi::SemidiscretizationParabolicAuxVars, cache_analysis)
  @unpack mesh, equations, initial_condition, solver, cache = semi

  calc_error_norms(func, u, t, analyzer, mesh, equations, initial_condition, solver, cache, cache_analysis)
end


function compute_coefficients(t, semi::SemidiscretizationParabolicAuxVars)
  compute_coefficients(semi.initial_condition, t, semi)
end

function compute_coefficients!(u_ode::AbstractVector, t, semi::SemidiscretizationParabolicAuxVars)
  compute_coefficients!(u_ode, semi.initial_condition, t, semi)
end


function rhs!(du_ode, u_ode, semi::SemidiscretizationParabolicAuxVars, t)
  @unpack mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache = semi

  u  = wrap_array(u_ode,  mesh, equations, solver, cache)
  du = wrap_array(du_ode, mesh, equations, solver, cache)

  # Compute gradients
  @unpack semi_grad_x, u_ode_grad_x, semi_grad_y, u_ode_grad_y = cache
  rhs!(u_ode_grad_x, u_ode, semi_grad_x, t)
  rhs!(u_ode_grad_y, u_ode, semi_grad_y, t)
  gradients = (wrap_array(u_ode_grad_x, mesh, equations, solver, cache),
               wrap_array(u_ode_grad_y, mesh, equations, solver, cache))
  cache_gradients = (semi_grad_x.cache, semi_grad_y.cache)

  # TODO: Taal decide, do we need to pass the mesh?
  time_start = time_ns()
  @timeit_debug timer() "rhs!" rhs!(du, u, gradients, t, mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache, cache_gradients)
  runtime = time_ns() - time_start
  put!(semi.performance_counter, runtime)

  return nothing
end
