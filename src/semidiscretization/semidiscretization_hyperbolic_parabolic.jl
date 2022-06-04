# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


"""
    SemidiscretizationHyperbolicParabolic

A struct containing everything needed to describe a spatial semidiscretization
of a mixed hyperbolic-parabolic conservation law.
"""
struct SemidiscretizationHyperbolicParabolic{Mesh, Equations <: EquationsHyperbolicParabolic, InitialCondition,
                                             BoundaryConditions, BoundaryConditionsParabolic,
                                             SourceTerms, Solver, SolverParabolic, Cache, CacheParabolic} <: AbstractSemidiscretization

  mesh::Mesh

  equations::Equations

  # This guy is a bit messy since we abuse it as some kind of "exact solution"
  # although this doesn't really exist...
  initial_condition::InitialCondition

  # TODO: rename boundary_conditions to boundary_conditions_hyperbolic?
  boundary_conditions::BoundaryConditions
  boundary_conditions_parabolic::BoundaryConditionsParabolic

  source_terms::SourceTerms

  solver::Solver
  solver_parabolic::SolverParabolic

  # TODO: rename cache -> cache_hyperbolic?
  cache::Cache
  cache_parabolic::CacheParabolic

  performance_counter::PerformanceCounter

  function SemidiscretizationHyperbolicParabolic{Mesh, Equations, InitialCondition, BoundaryConditions, BoundaryConditionsParabolic, SourceTerms, Solver, SolverParabolic, Cache, CacheParabolic}(
      mesh::Mesh, equations::Equations, initial_condition::InitialCondition,
      boundary_conditions::BoundaryConditions, boundary_conditions_parabolic::BoundaryConditionsParabolic,
      source_terms::SourceTerms, solver::Solver, solver_parabolic::SolverParabolic, cache::Cache, cache_parabolic::CacheParabolic) where {Mesh, Equations, InitialCondition, BoundaryConditions, BoundaryConditionsParabolic, SourceTerms, Solver, SolverParabolic, Cache, CacheParabolic}
    @assert ndims(mesh) == ndims(equations)

    # Todo: assert nvariables(equations)==nvariables(equations_parabolic)

    performance_counter = PerformanceCounter()

    new(mesh, equations, initial_condition,
        boundary_conditions, boundary_conditions_parabolic,
        source_terms, solver, solver_parabolic, cache, cache_parabolic, performance_counter)
  end
end

"""
    SemidiscretizationHyperbolicParabolic(mesh, both_equations, initial_condition, solver;
                                          solver_parabolic=default_parabolic_solver(),
                                          source_terms=nothing,
                                          both_boundary_conditions=(boundary_condition_periodic, boundary_condition_periodic),
                                          RealT=real(solver),
                                          uEltype=RealT,
                                          both_initial_caches=(NamedTuple(), NamedTuple()))

Construct a semidiscretization of a hyperbolic-parabolic PDE.
"""

function SemidiscretizationHyperbolicParabolic(mesh, equations,
                                               initial_condition, solver;
                                               solver_parabolic=default_parabolic_solver(),
                                               source_terms=nothing,
                                               boundary_conditions=(boundary_condition_periodic, boundary_condition_periodic),
                                               # `RealT` is used as real type for node locations etc.
                                               # while `uEltype` is used as element type of solutions etc.
                                               RealT=real(solver), uEltype=RealT,
                                               initial_caches=(NamedTuple(), NamedTuple()))

  equations_hyperbolic = equations.equations_hyperbolic
  equations_parabolic = equations.equations_parabolic
  boundary_conditions_hyperbolic, boundary_conditions_parabolic = boundary_conditions

  initial_cache_hyperbolic, initial_cache_parabolic = initial_caches
  cache_hyperbolic = (; create_cache(mesh, equations_hyperbolic, solver, RealT, uEltype)..., initial_cache_hyperbolic...)
  _boundary_conditions_hyperbolic = digest_boundary_conditions(boundary_conditions_hyperbolic, mesh, solver, cache_hyperbolic)
  _boundary_conditions_parabolic = digest_boundary_conditions(boundary_conditions_parabolic, mesh, solver, cache_hyperbolic)

  cache_parabolic = (; create_cache_parabolic(mesh, equations_parabolic, solver, solver_parabolic, RealT, uEltype)...,
                                              initial_cache_parabolic...)

  SemidiscretizationHyperbolicParabolic{typeof(mesh), typeof(equations),
                                        typeof(initial_condition), typeof(_boundary_conditions_hyperbolic), typeof(_boundary_conditions_parabolic),
                                        typeof(source_terms), typeof(solver), typeof(solver_parabolic), typeof(cache_hyperbolic), typeof(cache_parabolic)}(
    mesh, equations, initial_condition,
    _boundary_conditions_hyperbolic, _boundary_conditions_parabolic, source_terms,
    solver, solver_parabolic, cache_hyperbolic, cache_parabolic)
end

# Create a new semidiscretization but change some parameters compared to the input.
# `Base.similar` follows a related concept but would require us to `copy` the `mesh`,
# which would impact the performance. Instead, `SciMLBase.remake` has exactly the
# semantics we want to use here. In particular, it allows us to re-use mutable parts,
# e.g. `remake(semi).mesh === semi.mesh`.
function remake(semi::SemidiscretizationHyperbolicParabolic; uEltype=real(semi.solver),
                                                             mesh=semi.mesh,
                                                             equations=semi.equations,
                                                             initial_condition=semi.initial_condition,
                                                             solver=semi.solver,
                                                             solver_parabolic=semi.solver_parabolic,
                                                             source_terms=semi.source_terms,
                                                             boundary_conditions=semi.boundary_conditions,
                                                             boundary_conditions_parabolic=semi.boundary_conditions_parabolic
                                                             )
  # TODO: Which parts do we want to `remake`? At least the solver needs some
  #       special care if shock-capturing volume integrals are used (because of
  #       the indicators and their own caches...).
  SemidiscretizationHyperbolicParabolic(
    mesh, equations, initial_condition, solver; solver_parabolic, source_terms, boundary_conditions=(boundary_conditions, boundary_conditions_parabolic), uEltype)
end

function Base.show(io::IO, semi::SemidiscretizationHyperbolicParabolic)
  @nospecialize semi # reduce precompilation time

  print(io, "SemidiscretizationHyperbolicParabolic(")
  print(io,       semi.mesh)
  print(io, ", ", semi.equations.equations_hyperbolic)
  print(io, ", ", semi.equations.equations_parabolic)
  print(io, ", ", semi.initial_condition)
  print(io, ", ", semi.boundary_conditions)
  print(io, ", ", semi.boundary_conditions_parabolic)
  print(io, ", ", semi.source_terms)
  print(io, ", ", semi.solver)
  print(io, ", ", semi.solver_parabolic)
  print(io, ", cache(")
  for (idx,key) in enumerate(keys(semi.cache))
    idx > 1 && print(io, " ")
    print(io, key)
  end
  print(io, "))")
end

function Base.show(io::IO, ::MIME"text/plain", semi::SemidiscretizationHyperbolicParabolic)
  @nospecialize semi # reduce precompilation time

  if get(io, :compact, false)
    show(io, semi)
  else
    summary_header(io, "SemidiscretizationHyperbolicParabolic")
    summary_line(io, "#spatial dimensions", ndims(semi.equations))
    summary_line(io, "mesh", semi.mesh)
    summary_line(io, "hyperbolic equations", semi.equations.equations_hyperbolic |> typeof |> nameof)
    summary_line(io, "parabolic equations", semi.equations.equations_parabolic |> typeof |> nameof)
    summary_line(io, "initial condition", semi.initial_condition)

    # print_boundary_conditions(io, semi)

    summary_line(io, "source terms", semi.source_terms)
    summary_line(io, "solver", semi.solver |> typeof |> nameof)
    summary_line(io, "parabolic solver", semi.solver_parabolic |> typeof |> nameof)
    summary_line(io, "total #DOFs", ndofs(semi))
    summary_footer(io)
  end
end

@inline Base.ndims(semi::SemidiscretizationHyperbolicParabolic) = ndims(semi.mesh)

@inline nvariables(semi::SemidiscretizationHyperbolicParabolic) = nvariables(semi.equations.equations_hyperbolic)

@inline Base.real(semi::SemidiscretizationHyperbolicParabolic) = real(semi.solver)

# TODO: functions which depend on `equations` should dispatch on equations.equations_hyperbolic
# retain dispatch on hyperbolic equations only
@inline function mesh_equations_solver_cache(semi::SemidiscretizationHyperbolicParabolic)
  @unpack mesh, equations, solver, cache = semi
  return mesh, equations.equations_hyperbolic, solver, cache
end

function calc_error_norms(func, u_ode, t, analyzer, semi::SemidiscretizationHyperbolicParabolic, cache_analysis)
  @unpack mesh, equations, initial_condition, solver, cache = semi
  u = wrap_array(u_ode, mesh, equations.equations_hyperbolic, solver, cache)

  # pass EquationsHyperbolicParabolic through `calc_error_norms` to `initial_condition`
  calc_error_norms(func, u, t, analyzer, mesh, equations, initial_condition, solver, cache, cache_analysis)
end

function compute_coefficients(t, semi::SemidiscretizationHyperbolicParabolic)
  # Call `compute_coefficients` in `src/semidiscretization/semidiscretization.jl`
  compute_coefficients(semi.initial_condition, t, semi)
end

function compute_coefficients!(u_ode, t, semi::SemidiscretizationHyperbolicParabolic)
  compute_coefficients!(u_ode, semi.initial_condition, t, semi)
end

# Specialize: does not dispatch on semi.equations.equations_hyperbolic so that
# EquationsHyperbolicParabolic is available to the initial condition
function compute_coefficients!(u_ode, func, t, semi::SemidiscretizationHyperbolicParabolic)
  u = wrap_array(u_ode, semi)
  # Call `compute_coefficients` defined by the solver
  compute_coefficients!(u, func, t, semi.mesh, semi.equations, semi.solver, semi.cache)
end


"""
    semidiscretize(semi::SemidiscretizationHyperbolicParabolic, tspan)

Wrap the semidiscretization `semi` as a Split ODE problem in the time interval `tspan`
that can be passed to `solve` from the [SciML ecosystem](https://diffeq.sciml.ai/latest/).
"""
function semidiscretize(semi::SemidiscretizationHyperbolicParabolic, tspan)
  u0_ode = compute_coefficients(first(tspan), semi)
  # TODO: MPI, do we want to synchonize loading and print debug statements, e.g. using
  #       mpi_isparallel() && MPI.Barrier(mpi_comm())
  #       See https://github.com/trixi-framework/Trixi.jl/issues/328
  return SplitODEProblem(rhs!, rhs_parabolic!, u0_ode, tspan, semi)
end

function rhs!(du_ode, u_ode, semi::SemidiscretizationHyperbolicParabolic, t)
  @unpack mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache = semi

  u  = wrap_array(u_ode,  mesh, equations, solver, cache)
  du = wrap_array(du_ode, mesh, equations, solver, cache)

  # TODO: Taal decide, do we need to pass the mesh?
  time_start = time_ns()
  @trixi_timeit timer() "rhs!" rhs!(du, u, t, mesh, equations.equations_hyperbolic, initial_condition,
                                    boundary_conditions, source_terms, solver, cache)
  runtime = time_ns() - time_start
  put!(semi.performance_counter, runtime)

  return nothing
end

function rhs_parabolic!(du_ode, u_ode, semi::SemidiscretizationHyperbolicParabolic, t)
  @unpack mesh, equations, initial_condition, boundary_conditions_parabolic, source_terms, solver, solver_parabolic, cache, cache_parabolic = semi

  u  = wrap_array(u_ode,  mesh, equations, solver, cache_parabolic)
  du = wrap_array(du_ode, mesh, equations, solver, cache_parabolic)

  # TODO: Taal decide, do we need to pass the mesh?
  time_start = time_ns()
  @trixi_timeit timer() "parabolic rhs!" rhs_parabolic!(du, u, t, mesh, equations.equations_parabolic, initial_condition,
                                                        boundary_conditions_parabolic, source_terms,
                                                        solver, solver_parabolic, cache, cache_parabolic)
  runtime = time_ns() - time_start
  put!(semi.performance_counter, runtime)

  return nothing
end


end # @muladd
