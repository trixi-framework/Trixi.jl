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

struct SemidiscretizationHyperbolicParabolic{Mesh, Equations, ParabolicEquations, InitialCondition,
                                             BoundaryConditions, ParabolicBoundaryConditions,
                                             SourceTerms, Solver, Cache, ParabolicCache} <: AbstractSemidiscretization

  mesh::Mesh

  equations::Equations
  parabolic_equations::ParabolicEquations

  # This guy is a bit messy since we abuse it as some kind of "exact solution"
  # although this doesn't really exist...
  initial_condition::InitialCondition

  boundary_conditions::BoundaryConditions
  parabolic_boundary_conditions::ParabolicBoundaryConditions

  source_terms::SourceTerms

  solver::Solver
  # TODO: do we want to introduce `parabolic_solver` for future specialization?

  cache::Cache
  parabolic_cache::ParabolicCache

  performance_counter::PerformanceCounter

  function SemidiscretizationHyperbolicParabolic{Mesh, Equations, ParabolicEquations, InitialCondition, BoundaryConditions, ParabolicBoundaryConditions, SourceTerms, Solver, Cache, ParabolicCache}(
      mesh::Mesh, equations::Equations, parabolic_equations::ParabolicEquations, initial_condition::InitialCondition,
      boundary_conditions::BoundaryConditions, parabolic_boundary_conditions::ParabolicBoundaryConditions,
      source_terms::SourceTerms, solver::Solver, cache::Cache, parabolic_cache::ParabolicCache) where {Mesh, Equations, ParabolicEquations, InitialCondition, BoundaryConditions, ParabolicBoundaryConditions, SourceTerms, Solver, Cache, ParabolicCache}
    @assert ndims(mesh) == ndims(equations)

    # Todo: assert nvariables(equations)==nvariables(parabolic_equations)

    performance_counter = PerformanceCounter()

    new(mesh, equations, parabolic_equations, initial_condition,
        boundary_conditions, parabolic_boundary_conditions,
        source_terms, solver, cache, parabolic_cache, performance_counter)
  end
end

"""
    SemidiscretizationHyperbolicParabolic(mesh, equations, initial_condition, solver;
                                  source_terms=nothing,
                                  boundary_conditions=boundary_condition_periodic,
                                  RealT=real(solver),
                                  uEltype=RealT,
                                  initial_cache=NamedTuple())

Construct a semidiscretization of a hyperbolic PDE.
"""
function SemidiscretizationHyperbolicParabolic(mesh, equations, parabolic_equations,
                                               initial_condition, solver;
                                               source_terms=nothing,
                                               boundary_conditions=boundary_condition_periodic,
                                               parabolic_boundary_conditions=boundary_condition_periodic,
                                               # `RealT` is used as real type for node locations etc.
                                               # while `uEltype` is used as element type of solutions etc.
                                               RealT=real(solver), uEltype=RealT,
                                               initial_cache=NamedTuple(),
                                               initial_parabolic_cache=NamedTuple())

  cache = (; create_cache(mesh, equations, solver, RealT, uEltype)..., initial_cache...)
  _boundary_conditions = digest_boundary_conditions(boundary_conditions, mesh, solver, cache)
  _parabolic_boundary_conditions = digest_boundary_conditions(parabolic_boundary_conditions, mesh, solver, cache)

  parabolic_cache = (; create_cache(mesh, parabolic_equations, solver, RealT, uEltype)...,
                       initial_parabolic_cache...)

  SemidiscretizationHyperbolicParabolic{typeof(mesh), typeof(equations), typeof(parabolic_equations),
                                        typeof(initial_condition), typeof(_boundary_conditions), typeof(_parabolic_boundary_conditions),
                                        typeof(source_terms), typeof(solver), typeof(cache), typeof(parabolic_cache)}(
    mesh, equations, parabolic_equations, initial_condition,
    _boundary_conditions, _parabolic_boundary_conditions, source_terms,
    solver, cache, parabolic_cache)
end


# Create a new semidiscretization but change some parameters compared to the input.
# `Base.similar` follows a related concept but would require us to `copy` the `mesh`,
# which would impact the performance. Instead, `SciMLBase.remake` has exactly the
# semantics we want to use here. In particular, it allows us to re-use mutable parts,
# e.g. `remake(semi).mesh === semi.mesh`.
function remake(semi::SemidiscretizationHyperbolicParabolic; uEltype=real(semi.solver),
                                                    mesh=semi.mesh,
                                                    equations=semi.equations,
                                                    parabolic_equations=semi.parabolic_equations,
                                                    initial_condition=semi.initial_condition,
                                                    solver=semi.solver,
                                                    source_terms=semi.source_terms,
                                                    boundary_conditions=semi.boundary_conditions,
                                                    parabolic_boundary_conditions=semi.parabolic_boundary_conditions
                                                    )
  # TODO: Which parts do we want to `remake`? At least the solver needs some
  #       special care if shock-capturing volume integrals are used (because of
  #       the indicators and their own caches...).
  SemidiscretizationHyperbolicParabolic(
    mesh, equations, parabolic_equations, initial_condition, solver; source_terms, boundary_conditions, parabolic_boundary_conditions, uEltype)
end

function Base.show(io::IO, semi::SemidiscretizationHyperbolicParabolic)
  @nospecialize semi # reduce precompilation time

  print(io, "SemidiscretizationHyperbolicParabolic(")
  print(io,       semi.mesh)
  print(io, ", ", semi.equations)
  print(io, ", ", semi.parabolic_equations)
  print(io, ", ", semi.initial_condition)
  print(io, ", ", semi.boundary_conditions)
  print(io, ", ", semi.parabolic_boundary_conditions)
  print(io, ", ", semi.source_terms)
  print(io, ", ", semi.solver)
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
    summary_line(io, "equations", semi.equations |> typeof |> nameof)
    summary_line(io, "parabolic_equations", semi.parabolic_equations |> typeof |> nameof)
    summary_line(io, "initial condition", semi.initial_condition)

    # print_boundary_conditions(io, semi)

    summary_line(io, "source terms", semi.source_terms)
    summary_line(io, "solver", semi.solver |> typeof |> nameof)
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
  @trixi_timeit timer() "rhs!" rhs!(du, u, t, mesh, equations, initial_condition,
                                    boundary_conditions, source_terms, solver, cache)
  runtime = time_ns() - time_start
  put!(semi.performance_counter, runtime)

  return nothing
end

function rhs_parabolic!(du_ode, u_ode, semi::SemidiscretizationHyperbolicParabolic, t)
  @unpack mesh, parabolic_equations, initial_condition, parabolic_boundary_conditions, source_terms, solver, cache, parabolic_cache = semi

  u  = wrap_array(u_ode,  mesh, parabolic_equations, solver, parabolic_cache)
  du = wrap_array(du_ode, mesh, parabolic_equations, solver, parabolic_cache)

  # TODO: Taal decide, do we need to pass the mesh?
  time_start = time_ns()
  @trixi_timeit timer() "parabolic rhs!" rhs!(du, u, mesh, parabolic_equations, initial_condition,
                                              parabolic_boundary_conditions, source_terms,
                                              solver, cache, parabolic_cache)
  runtime = time_ns() - time_start
  put!(semi.performance_counter, runtime)

  return nothing
end


end # @muladd
