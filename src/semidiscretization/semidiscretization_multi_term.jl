# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

"""
    SemidiscretizationMultiTerm

A struct containing everything needed to describe a spatial semidiscretization
of a conservation law with multiple equation terms (e.g., a hyperbolic and parabolic equation).
"""
struct SemidiscretizationMultiTerm{NTerms, Mesh, Equations <: NTuple{NTerms}, InitialCondition, BoundaryConditions,
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

  function SemidiscretizationMultiTerm{NTerms, Mesh, Equations, InitialCondition, BoundaryConditions, SourceTerms, Solver, Cache}(
      mesh::Mesh, equations::Equations,
      initial_condition::InitialCondition, boundary_conditions::BoundaryConditions,
      source_terms::SourceTerms,
      solver::Solver, cache::Cache) where {NTerms, Mesh, Equations<:NTuple{NTerms}, InitialCondition, BoundaryConditions, SourceTerms, Solver, Cache}
    @assert ndims(mesh) == ndims(first(equations))

    # check that all equations have the same number of variables and same dimension
    @assert all(ndims.(equations) .== ndims(first(equations)))
    @assert all(nvariables.(equations) .== nvariables(first(equations)))

    performance_counter = PerformanceCounter()

    new(mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache, performance_counter)
  end
end

"""
    SemidiscretizationMultiTerm(mesh, equations, initial_condition, solver;
                                 source_terms=nothing,
                                 boundary_conditions=boundary_condition_periodic,
                                 RealT=real(solver),
                                 uEltype=RealT,
                                 initial_cache=NamedTuple())

Construct a semidiscretization of a hyperbolic PDE.
"""
function SemidiscretizationMultiTerm(mesh, equations, initial_condition, solver;
                                     source_terms=nothing,
                                     boundary_conditions=boundary_condition_periodic,
                                     # `RealT` is used as real type for node locations etc.
                                     # while `uEltype` is used as element type of solutions etc.
                                     RealT=real(solver), uEltype=RealT,
                                     initial_cache=NamedTuple())

  cache = (; create_cache(mesh, equations, solver, RealT, uEltype)..., initial_cache...)
  _boundary_conditions = digest_boundary_conditions(boundary_conditions, mesh, solver, cache)

  SemidiscretizationMultiTerm{length(equations), typeof(mesh), typeof(equations), typeof(initial_condition), typeof(_boundary_conditions), typeof(source_terms), typeof(solver), typeof(cache)}(
    mesh, equations, initial_condition, _boundary_conditions, source_terms, solver, cache)
end


# Create a new semidiscretization but change some parameters compared to the input.
# `Base.similar` follows a related concept but would require us to `copy` the `mesh`,
# which would impact the performance. Instead, `SciMLBase.remake` has exactly the
# semantics we want to use here. In particular, it allows us to re-use mutable parts,
# e.g. `remake(semi).mesh === semi.mesh`.
function remake(semi::SemidiscretizationMultiTerm; uEltype=real(semi.solver),
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
  SemidiscretizationMultiTerm(
    mesh, equations, initial_condition, solver; source_terms, boundary_conditions, uEltype)
end

function Base.show(io::IO, semi::SemidiscretizationMultiTerm)
  @nospecialize semi # reduce precompilation time

  print(io, "SemidiscretizationMultiTerm(")
  print(io,       semi.mesh)
  for equation in semi.equations
    print(io, ", ", equation)
  end
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

function Base.show(io::IO, ::MIME"text/plain", semi::SemidiscretizationMultiTerm)
  @nospecialize semi # reduce precompilation time

  if get(io, :compact, false)
    show(io, semi)
  else
    summary_header(io, "SemidiscretizationMultiTerm")
    summary_line(io, "#spatial dimensions", ndims(first(semi.equations)))
    summary_line(io, "mesh", semi.mesh)
    for (i, equation) in enumerate(semi.equations)
      summary_line(io, "equation $i", equation |> typeof |> nameof)
    end
    summary_line(io, "initial condition", semi.initial_condition)

    # print_boundary_conditions(io, semi)

    summary_line(io, "source terms", semi.source_terms)
    summary_line(io, "solver", semi.solver |> typeof |> nameof)
    summary_line(io, "total #DOFs", ndofs(semi))
    summary_footer(io)
  end
end

@inline Base.ndims(semi::SemidiscretizationMultiTerm) = ndims(semi.mesh)

@inline nvariables(semi::SemidiscretizationMultiTerm) = nvariables(first(semi.equations))
@inline nvariables(equations::NTuple) = nvariables(first(equations))

@inline Base.real(semi::SemidiscretizationMultiTerm) = real(semi.solver)


# to reuse existing functionality, we dispatch based on the first set of equations
@inline function mesh_equations_solver_cache(semi::SemidiscretizationMultiTerm)
  @unpack mesh, equations, solver, cache = semi
  return mesh, first(equations), solver, cache
end

function calc_error_norms(func, u_ode, t, analyzer, semi::SemidiscretizationMultiTerm, cache_analysis)
  @unpack mesh, equations, initial_condition, solver, cache = semi
  u = wrap_array(u_ode, mesh, first(equations), solver, cache)

  calc_error_norms(func, u, t, analyzer, mesh, first(equations), initial_condition, solver, cache, cache_analysis)
end

function compute_coefficients(t, semi::SemidiscretizationMultiTerm)
  # Call `compute_coefficients` in `src/semidiscretization/semidiscretization.jl`
  compute_coefficients(semi.initial_condition, t, semi)
end

function compute_coefficients!(u_ode, t, semi::SemidiscretizationMultiTerm)
  compute_coefficients!(u_ode, semi.initial_condition, t, semi)
end

function rhs!(du_ode, u_ode, semi::SemidiscretizationMultiTerm, t)
  @unpack mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache = semi

  u  = wrap_array(u_ode,  mesh, first(equations), solver, cache)
  du = wrap_array(du_ode, mesh, first(equations), solver, cache)
  du_single_term = similar(du)

  # compute RHS for each term/equation
  time_start = time_ns()
  @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, solver, cache)
  for (i, equation) in enumerate(equations)
    @trixi_timeit timer() "`rhs!` number $(i)" rhs!(du_single_term, u, t, mesh, equation,
                                                    initial_condition, boundary_conditions, source_terms,
                                                    solver, cache)
    @threaded for i in eachindex(du)
      du[i] = du[i] + du_single_term[i]
    end
  end
  runtime = time_ns() - time_start
  put!(semi.performance_counter, runtime)

  return nothing
end

end # @muladd
