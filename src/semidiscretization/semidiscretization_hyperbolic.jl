
# TODO: Taal refactor, Mesh<:AbstractMesh{NDIMS}, Equations<:AbstractEquations{NDIMS} ?
"""
    SemidiscretizationHyperbolic

A struct containing everything needed to describe a spatial semidiscretization
of a hyperbolic conservation law.
"""
struct SemidiscretizationHyperbolic{Mesh, Equations, InitialCondition, BoundaryConditions,
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

  function SemidiscretizationHyperbolic{Mesh, Equations, InitialCondition, BoundaryConditions, SourceTerms, Solver, Cache}(
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
    SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                 source_terms=nothing,
                                 boundary_conditions=boundary_condition_periodic)

Construct a semidiscretization of a hyperbolic PDE.
"""
function SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                      source_terms=nothing,
                                      boundary_conditions=boundary_condition_periodic, RealT=real(solver))

  cache = create_cache(mesh, equations, solver, RealT)
  _boundary_conditions = digest_boundary_conditions(boundary_conditions)

  SemidiscretizationHyperbolic{typeof(mesh), typeof(equations), typeof(initial_condition), typeof(_boundary_conditions), typeof(source_terms), typeof(solver), typeof(cache)}(
    mesh, equations, initial_condition, _boundary_conditions, source_terms, solver, cache)
end


# allow passing named tuples of BCs constructed in an arbitrary order
digest_boundary_conditions(boundary_conditions) = boundary_conditions

function digest_boundary_conditions(boundary_conditions::NamedTuple{Keys,ValueTypes}) where {Keys, ValueTypes<:NTuple{2,Any}} # 1D
  @unpack x_neg, x_pos = boundary_conditions
  (; x_neg, x_pos)
end
function digest_boundary_conditions(boundary_conditions::NamedTuple{Keys,ValueTypes}) where {Keys, ValueTypes<:NTuple{4,Any}} # 2D
  @unpack x_neg, x_pos, y_neg, y_pos = boundary_conditions
  (; x_neg, x_pos, y_neg, y_pos)
end
function digest_boundary_conditions(boundary_conditions::NamedTuple{Keys,ValueTypes}) where {Keys, ValueTypes<:NTuple{6,Any}} # 3D
  @unpack x_neg, x_pos, y_neg, y_pos, z_neg, z_pos = boundary_conditions
  (; x_neg, x_pos, y_neg, y_pos, z_neg, z_pos)
end

function digest_boundary_conditions(boundary_conditions::AbstractArray)
  throw(ArgumentError("Please use a (named) tuple instead of an (abstract) array to supply multiple boundary conditions (to improve performance)."))
end



function Base.show(io::IO, semi::SemidiscretizationHyperbolic)
  print(io, "SemidiscretizationHyperbolic(")
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

function Base.show(io::IO, ::MIME"text/plain", semi::SemidiscretizationHyperbolic)
  if get(io, :compact, false)
    show(io, semi)
  else
    summary_header(io, "SemidiscretizationHyperbolic")
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


@inline Base.ndims(semi::SemidiscretizationHyperbolic) = ndims(semi.mesh)

@inline nvariables(semi::SemidiscretizationHyperbolic) = nvariables(semi.equations)

@inline Base.real(semi::SemidiscretizationHyperbolic) = real(semi.solver)


@inline function mesh_equations_solver_cache(semi::SemidiscretizationHyperbolic)
  @unpack mesh, equations, solver, cache = semi
  return mesh, equations, solver, cache
end


function calc_error_norms(func, u_ode::AbstractVector, t, analyzer, semi::SemidiscretizationHyperbolic, cache_analysis)
  @unpack mesh, equations, initial_condition, solver, cache = semi
  u = wrap_array(u_ode, mesh, equations, solver, cache)

  calc_error_norms(func, u, t, analyzer, mesh, equations, initial_condition, solver, cache, cache_analysis)
end

function calc_error_norms(func, u, t, analyzer, semi::SemidiscretizationHyperbolic, cache_analysis)
  @unpack mesh, equations, initial_condition, solver, cache = semi

  calc_error_norms(func, u, t, analyzer, mesh, equations, initial_condition, solver, cache, cache_analysis)
end


function compute_coefficients(t, semi::SemidiscretizationHyperbolic)
  compute_coefficients(semi.initial_condition, t, semi)
end

function compute_coefficients!(u_ode::AbstractVector, t, semi::SemidiscretizationHyperbolic)
  compute_coefficients!(u_ode, semi.initial_condition, t, semi)
end


function rhs!(du_ode, u_ode, semi::SemidiscretizationHyperbolic, t)
  @unpack mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache = semi

  u  = wrap_array(u_ode,  mesh, equations, solver, cache)
  du = wrap_array(du_ode, mesh, equations, solver, cache)

  # TODO: Taal decide, do we need to pass the mesh?
  time_start = time_ns()
  @timeit_debug timer() "rhs!" rhs!(du, u, t, mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache)
  runtime = time_ns() - time_start
  put!(semi.performance_counter, runtime)

  return nothing
end
