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
struct SemidiscretizationHyperbolicParabolic{Mesh, Equations, EquationsParabolic, InitialCondition,
                                             BoundaryConditions, BoundaryConditionsParabolic,
                                             SourceTerms, Solver, SolverParabolic, Cache, CacheParabolic} <: AbstractSemidiscretization

  mesh::Mesh

  equations::Equations
  equations_parabolic::EquationsParabolic

  # This guy is a bit messy since we abuse it as some kind of "exact solution"
  # although this doesn't really exist...
  initial_condition::InitialCondition

  boundary_conditions::BoundaryConditions
  boundary_conditions_parabolic::BoundaryConditionsParabolic

  source_terms::SourceTerms

  solver::Solver
  solver_parabolic::SolverParabolic

  cache::Cache
  cache_parabolic::CacheParabolic

  performance_counter::PerformanceCounterList{2}

  function SemidiscretizationHyperbolicParabolic{Mesh, Equations, EquationsParabolic, InitialCondition, BoundaryConditions, BoundaryConditionsParabolic, SourceTerms, Solver, SolverParabolic, Cache, CacheParabolic}(
      mesh::Mesh, equations::Equations, equations_parabolic::EquationsParabolic, initial_condition::InitialCondition,
      boundary_conditions::BoundaryConditions, boundary_conditions_parabolic::BoundaryConditionsParabolic,
      source_terms::SourceTerms, solver::Solver, solver_parabolic::SolverParabolic, cache::Cache, cache_parabolic::CacheParabolic) where {Mesh, Equations, EquationsParabolic, InitialCondition, BoundaryConditions, BoundaryConditionsParabolic, SourceTerms, Solver, SolverParabolic, Cache, CacheParabolic}
    @assert ndims(mesh) == ndims(equations)

    # Todo: assert nvariables(equations)==nvariables(equations_parabolic)

    performance_counter = PerformanceCounterList{2}()

    new(mesh, equations, equations_parabolic, initial_condition,
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
function SemidiscretizationHyperbolicParabolic(mesh, equations::Tuple,
                                               initial_condition, solver;
                                               solver_parabolic=default_parabolic_solver(),
                                               source_terms=nothing,
                                               boundary_conditions=(boundary_condition_periodic, boundary_condition_periodic),
                                               # `RealT` is used as real type for node locations etc.
                                               # while `uEltype` is used as element type of solutions etc.
                                               RealT=real(solver), uEltype=RealT,
                                               initial_caches=(NamedTuple(), NamedTuple()))

  equations_hyperbolic, equations_parabolic = equations
  boundary_conditions_hyperbolic, boundary_conditions_parabolic = boundary_conditions
  initial_hyperbolic_cache, initial_cache_parabolic = initial_caches

  return SemidiscretizationHyperbolicParabolic(mesh, equations_hyperbolic, equations_parabolic,
                                               initial_condition, solver; solver_parabolic, source_terms,
                                               boundary_conditions=boundary_conditions_hyperbolic,
                                               boundary_conditions_parabolic=boundary_conditions_parabolic,
                                               RealT, uEltype, initial_cache=initial_hyperbolic_cache,
                                               initial_cache_parabolic=initial_cache_parabolic)
end

function SemidiscretizationHyperbolicParabolic(mesh, equations, equations_parabolic,
                                               initial_condition, solver;
                                               solver_parabolic=default_parabolic_solver(),
                                               source_terms=nothing,
                                               boundary_conditions=boundary_condition_periodic,
                                               boundary_conditions_parabolic=boundary_condition_periodic,
                                               # `RealT` is used as real type for node locations etc.
                                               # while `uEltype` is used as element type of solutions etc.
                                               RealT=real(solver), uEltype=RealT,
                                               initial_cache=NamedTuple(),
                                               initial_cache_parabolic=NamedTuple())

  cache = (; create_cache(mesh, equations, solver, RealT, uEltype)..., initial_cache...)
  _boundary_conditions = digest_boundary_conditions(boundary_conditions, mesh, solver, cache)
  _boundary_conditions_parabolic = digest_boundary_conditions(boundary_conditions_parabolic, mesh, solver, cache)

  cache_parabolic = (; create_cache_parabolic(mesh, equations, equations_parabolic,
                                              solver, solver_parabolic, RealT, uEltype)...,
                                              initial_cache_parabolic...)

  SemidiscretizationHyperbolicParabolic{typeof(mesh), typeof(equations), typeof(equations_parabolic),
                                        typeof(initial_condition), typeof(_boundary_conditions), typeof(_boundary_conditions_parabolic),
                                        typeof(source_terms), typeof(solver), typeof(solver_parabolic), typeof(cache), typeof(cache_parabolic)}(
    mesh, equations, equations_parabolic, initial_condition,
    _boundary_conditions, _boundary_conditions_parabolic, source_terms,
    solver, solver_parabolic, cache, cache_parabolic)
end


# Create a new semidiscretization but change some parameters compared to the input.
# `Base.similar` follows a related concept but would require us to `copy` the `mesh`,
# which would impact the performance. Instead, `SciMLBase.remake` has exactly the
# semantics we want to use here. In particular, it allows us to re-use mutable parts,
# e.g. `remake(semi).mesh === semi.mesh`.
function remake(semi::SemidiscretizationHyperbolicParabolic; uEltype=real(semi.solver),
                                                             mesh=semi.mesh,
                                                             equations=semi.equations,
                                                             equations_parabolic=semi.equations_parabolic,
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
    mesh, equations, equations_parabolic, initial_condition, solver; solver_parabolic, source_terms, boundary_conditions, boundary_conditions_parabolic, uEltype)
end

function Base.show(io::IO, semi::SemidiscretizationHyperbolicParabolic)
  @nospecialize semi # reduce precompilation time

  print(io, "SemidiscretizationHyperbolicParabolic(")
  print(io,       semi.mesh)
  print(io, ", ", semi.equations)
  print(io, ", ", semi.equations_parabolic)
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
    summary_line(io, "hyperbolic equations", semi.equations |> typeof |> nameof)
    summary_line(io, "parabolic equations", semi.equations_parabolic |> typeof |> nameof)
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

@inline nvariables(semi::SemidiscretizationHyperbolicParabolic) = nvariables(semi.equations)

@inline Base.real(semi::SemidiscretizationHyperbolicParabolic) = real(semi.solver)

# retain dispatch on hyperbolic equations only
@inline function mesh_equations_solver_cache(semi::SemidiscretizationHyperbolicParabolic)
  @unpack mesh, equations, solver, cache = semi
  return mesh, equations, solver, cache
end


function calc_error_norms(func, u_ode, t, analyzer, semi::SemidiscretizationHyperbolicParabolic, cache_analysis)
  @unpack mesh, equations, initial_condition, solver, cache = semi
  u = wrap_array(u_ode, mesh, equations, solver, cache)

  calc_error_norms(func, u, t, analyzer, mesh, equations, initial_condition, solver, cache, cache_analysis)
end


function compute_coefficients(t, semi::SemidiscretizationHyperbolicParabolic)
  # Call `compute_coefficients` in `src/semidiscretization/semidiscretization.jl`
  compute_coefficients(semi.initial_condition, t, semi)
end

function compute_coefficients!(u_ode, t, semi::SemidiscretizationHyperbolicParabolic)
  compute_coefficients!(u_ode, semi.initial_condition, t, semi)
end

"""
    semidiscretize(semi::SemidiscretizationHyperbolicParabolic, tspan)

Wrap the semidiscretization `semi` as a split ODE problem in the time interval `tspan`
that can be passed to `solve` from the [SciML ecosystem](https://diffeq.sciml.ai/latest/).
The parabolic right-hand side is the first function of the split ODE problem and
will be used by default by the implicit part of IMEX methods from the
SciML ecosystem.
"""
function semidiscretize(semi::SemidiscretizationHyperbolicParabolic, tspan)
  u0_ode = compute_coefficients(first(tspan), semi)
  # TODO: MPI, do we want to synchonize loading and print debug statements, e.g. using
  #       mpi_isparallel() && MPI.Barrier(mpi_comm())
  #       See https://github.com/trixi-framework/Trixi.jl/issues/328
  iip = true # is-inplace, i.e., we modify a vector when calling rhs_parabolic!, rhs!
  # Note that the IMEX time integration methods of OrdinaryDiffEq.jl treat the
  # first function implicitly and the second one explicitly. Thus, we pass the
  # stiffer parabolic function first.
  return SplitODEProblem{iip}(rhs_parabolic!, rhs!, u0_ode, tspan, semi)
end

function rhs!(du_ode, u_ode, semi::SemidiscretizationHyperbolicParabolic, t)
  @unpack mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache = semi

  u  = wrap_array(u_ode,  mesh, equations, solver, cache)
  du = wrap_array(du_ode, mesh, equations, solver, cache)

  # TODO: Taal decide, do we need to pass the mesh?
  time_start = time_ns()
  @trixi_timeit timer() "rhs!" rhs!(du, u, t, mesh, equations, initial_condition,
                                    boundary_conditions, source_terms, solver, cache)
  runtime = time_ns() - time_start
  put!(semi.performance_counter.counters[1], runtime)

  return nothing
end

function rhs_parabolic!(du_ode, u_ode, semi::SemidiscretizationHyperbolicParabolic, t)
  @unpack mesh, equations_parabolic, initial_condition, boundary_conditions_parabolic, source_terms, solver, solver_parabolic, cache, cache_parabolic = semi

  u  = wrap_array(u_ode,  mesh, equations_parabolic, solver, cache_parabolic)
  du = wrap_array(du_ode, mesh, equations_parabolic, solver, cache_parabolic)

  # TODO: Taal decide, do we need to pass the mesh?
  time_start = time_ns()
  @trixi_timeit timer() "parabolic rhs!" rhs_parabolic!(du, u, t, mesh, equations_parabolic, initial_condition,
                                                        boundary_conditions_parabolic, source_terms,
                                                        solver, solver_parabolic, cache, cache_parabolic)
  runtime = time_ns() - time_start
  put!(semi.performance_counter.counters[2], runtime)

  return nothing
end


end # @muladd
