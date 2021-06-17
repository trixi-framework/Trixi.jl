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
                                 boundary_conditions=boundary_condition_periodic,
                                 RealT=real(solver),
                                 uEltype=RealT,
                                 initial_cache=NamedTuple())

Construct a semidiscretization of a hyperbolic PDE.
"""
function SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                      source_terms=nothing,
                                      boundary_conditions=boundary_condition_periodic,
                                      # `RealT` is used as real type for node locations etc.
                                      # while `uEltype` is used as element type of solutions etc.
                                      RealT=real(solver), uEltype=RealT,
                                      initial_cache=NamedTuple())

  cache = (; create_cache(mesh, equations, solver, RealT, uEltype)..., initial_cache...)
  _boundary_conditions = digest_boundary_conditions(boundary_conditions, cache)

  SemidiscretizationHyperbolic{typeof(mesh), typeof(equations), typeof(initial_condition), typeof(_boundary_conditions), typeof(source_terms), typeof(solver), typeof(cache)}(
    mesh, equations, initial_condition, _boundary_conditions, source_terms, solver, cache)
end


# Create a new semidiscretization but change some parameters compared to the input.
# `Base.similar` follows a related concept but would require us to `copy` the `mesh`,
# which would impact the performance. Instead, `SciMLBase.remake` has exactly the
# semantics we want to use here. In particular, it allows us to re-use mutable parts,
# e.g. `remake(semi).mesh === semi.mesh`.
function remake(semi::SemidiscretizationHyperbolic; uEltype=real(semi.solver),
                                                    mesh=semi.mesh,
                                                    equations=semi.equations,
                                                    initial_condition=semi.initial_condition,
                                                    solver=semi.solver,
                                                    source_terms=semi.source_terms,
                                                    boundary_conditions=semi.boundary_conditions
                                                    )
  # TODO: Which parts do we want to `remake`? At least the solver needs some
  #       special care if shock-capturing volume integrals are used (because of
  #       the indicators and their own caches...).
  SemidiscretizationHyperbolic(
    mesh,  equations, initial_condition, solver; source_terms, boundary_conditions, uEltype)
end


# allow passing named tuples of BCs constructed in an arbitrary order
digest_boundary_conditions(boundary_conditions, cache) = boundary_conditions

function digest_boundary_conditions(boundary_conditions::NamedTuple{Keys,ValueTypes}, cache) where {Keys, ValueTypes<:NTuple{2,Any}} # 1D
  @unpack x_neg, x_pos = boundary_conditions
  (; x_neg, x_pos)
end
function digest_boundary_conditions(boundary_conditions::NamedTuple{Keys,ValueTypes}, cache) where {Keys, ValueTypes<:NTuple{4,Any}} # 2D
  @unpack x_neg, x_pos, y_neg, y_pos = boundary_conditions
  (; x_neg, x_pos, y_neg, y_pos)
end
function digest_boundary_conditions(boundary_conditions::NamedTuple{Keys,ValueTypes}, cache) where {Keys, ValueTypes<:NTuple{6,Any}} # 3D
  @unpack x_neg, x_pos, y_neg, y_pos, z_neg, z_pos = boundary_conditions
  (; x_neg, x_pos, y_neg, y_pos, z_neg, z_pos)
end

# sort the boundary conditions from a dictionary and into tuples
function digest_boundary_conditions(boundary_conditions::Dict, cache)
  UnstructuredQuadSortedBoundaryTypes(boundary_conditions, cache)
end

function digest_boundary_conditions(boundary_conditions::AbstractArray, cache)
  throw(ArgumentError("Please use a (named) tuple instead of an (abstract) array to supply multiple boundary conditions (to improve performance)."))
end


function Base.show(io::IO, semi::SemidiscretizationHyperbolic)
  @nospecialize semi # reduce precompilation time

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
  @nospecialize semi # reduce precompilation time

  if get(io, :compact, false)
    show(io, semi)
  else
    summary_header(io, "SemidiscretizationHyperbolic")
    summary_line(io, "#spatial dimensions", ndims(semi.equations))
    summary_line(io, "mesh", semi.mesh)
    summary_line(io, "equations", semi.equations |> typeof |> nameof)
    summary_line(io, "initial condition", semi.initial_condition)
    if semi.boundary_conditions isa UnstructuredQuadSortedBoundaryTypes
      @unpack boundary_dictionary = semi.boundary_conditions
      summary_line(io, "boundary conditions", length(boundary_dictionary))
      for (boundary_name, boundary_condition) in boundary_dictionary
        summary_line(increment_indent(io), boundary_name, typeof(boundary_condition))
      end
    else # non dictionary boundary conditions container
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
    end
    summary_line(io, "source terms", semi.source_terms)
    summary_line(io, "solver", semi.solver |> typeof |> nameof)
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


function compute_coefficients(t, semi::SemidiscretizationHyperbolic)
  # Call `compute_coefficients` in `src/semidiscretization/semidiscretization.jl`
  compute_coefficients(semi.initial_condition, t, semi)
end

function compute_coefficients!(u_ode, t, semi::SemidiscretizationHyperbolic)
  compute_coefficients!(u_ode, semi.initial_condition, t, semi)
end


function rhs!(du_ode, u_ode, semi::SemidiscretizationHyperbolic, t)
  @unpack mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache = semi

  u  = wrap_array(u_ode,  mesh, equations, solver, cache)
  du = wrap_array(du_ode, mesh, equations, solver, cache)

  # TODO: Taal decide, do we need to pass the mesh?
  time_start = time_ns()
  @trixi_timeit timer() "rhs!" rhs!(du, u, t, mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache)
  runtime = time_ns() - time_start
  put!(semi.performance_counter, runtime)

  return nothing
end
