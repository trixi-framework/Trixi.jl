
# TODO: Taal refactor, Mesh<:AbstractMesh{NDIMS}, Equations<:AbstractEquations{NDIMS} ?
"""
    SemidiscretizationHyperbolicParabolic

A struct containing everything needed to describe a spatial semidiscretization
of a hyperbolic-parabolic conservation law.
"""
struct SemidiscretizationHyperbolicParabolic{SemiHyperbolic, SemiParabolic, Cache} <: AbstractSemidiscretization
  semi_hyperbolic::SemiHyperbolic
  semi_parabolic::SemiParabolic
  cache::Cache
  performance_counter::PerformanceCounter

  function SemidiscretizationHyperbolicParabolic{SemiHyperbolic, SemiParabolic, Cache}(
      semi_hyperbolic::SemiHyperbolic, semi_parabolic::SemiParabolic,
      cache::Cache) where {SemiHyperbolic, SemiParabolic, Cache}

    performance_counter = PerformanceCounter()

    new(semi_hyperbolic, semi_parabolic, cache, performance_counter)
  end
end

"""
    SemidiscretizationHyperbolicParabolic(semi_hyperbolic, semi_parabolic)

Construct a semidiscretization of a hyperbolic-parabolic PDE by combining the purely hyperbolic and
purely parabolic semidiscretizations.
"""
function SemidiscretizationHyperbolicParabolic(semi_hyperbolic, semi_parabolic)
  cache = (; du_ode_parabolic=Vector{real(semi_parabolic)}())

  SemidiscretizationHyperbolicParabolic{typeof(semi_hyperbolic), typeof(semi_parabolic), typeof(cache)}(
    semi_hyperbolic, semi_parabolic, cache)
end


function Base.show(io::IO, semi::SemidiscretizationHyperbolicParabolic)
  print(io, "SemidiscretizationHyperbolicParabolic(")
  print(io,       semi.semi_hyperbolic)
  print(io, ", ", semi.semi_parabolic)
  print(io, ", cache(")
  for (idx,key) in enumerate(keys(semi.cache))
    idx > 1 && print(io, " ")
    print(io, key)
  end
  print(io, "))")
end

function Base.show(io::IO, ::MIME"text/plain", semi::SemidiscretizationHyperbolicParabolic)
  if get(io, :compact, false)
    show(io, semi)
  else
    summary_header(io, "SemidiscretizationHyperbolicParabolic")
    summary_line(io, "hyperbolic system", semi.semi_hyperbolic)
    summary_line(io, "parabolic system", semi.semi_parabolic)
    summary_footer(io)
  end
end


@inline Base.ndims(semi::SemidiscretizationHyperbolicParabolic) = ndims(semi.semi_hyperbolic.mesh)

@inline nvariables(semi::SemidiscretizationHyperbolicParabolic) = nvariables(semi.semi_hyperbolic.equations)

@inline Base.real(semi::SemidiscretizationHyperbolicParabolic) = real(semi.semi_hyperbolic.solver)


@inline function mesh_equations_solver_cache(semi::SemidiscretizationHyperbolicParabolic)
  @unpack mesh, equations, solver, cache = semi.semi_hyperbolic
  return mesh, equations, solver, cache
end


function calc_error_norms(func, u_ode::AbstractVector, t, analyzer, semi::SemidiscretizationHyperbolicParabolic, cache_analysis)
  @unpack mesh, equations, initial_condition, solver, cache = semi.semi_hyperbolic
  u = wrap_array(u_ode, mesh, equations, solver, cache)

  calc_error_norms(func, u, t, analyzer, mesh, equations, initial_condition, solver, cache, cache_analysis)
end

function calc_error_norms(func, u, t, analyzer, semi::SemidiscretizationHyperbolicParabolic, cache_analysis)
  @unpack mesh, equations, initial_condition, solver, cache = semi.semi_hyperbolic

  calc_error_norms(func, u, t, analyzer, mesh, equations, initial_condition, solver, cache, cache_analysis)
end


function compute_coefficients(t, semi::SemidiscretizationHyperbolicParabolic)
  compute_coefficients(semi.semi_hyperbolic.initial_condition, t, semi.semi_hyperbolic)
end

function compute_coefficients!(u_ode::AbstractVector, t, semi::SemidiscretizationHyperbolicParabolic)
  compute_coefficients!(u_ode, semi.semi_hyperbolic.initial_condition, t, semi.semi_hyperbolic)
end


function rhs!(du_ode, u_ode, semi::SemidiscretizationHyperbolicParabolic, t)
  @unpack du_ode_parabolic = semi.cache

  # TODO: Taal decide, do we need to pass the mesh?
  time_start = time_ns()

  # Resize the storage for the parabolic time derivative in case it was changed
  resize!(du_ode_parabolic, length(du_ode))

  @timeit_debug timer() "rhs!" begin
    # Compute hyperbolic time derivative
    rhs!(du_ode, u_ode, semi.semi_hyperbolic, t)

    # Compute parabolic time derivative
    rhs!(du_ode_parabolic, u_ode, semi.semi_parabolic, t)

    # Add parabolic update to hyperbolic time derivative
    du_ode .+= du_ode_parabolic
  end

  runtime = time_ns() - time_start
  put!(semi.performance_counter, runtime)

  return nothing
end


"""
    SemidiscretizationHyperbolicParabolicBR1(mesh, equations, initial_condition, solver::DGSEM;
                                             source_terms=nothing,
                                             boundary_conditions=boundary_condition_periodic)

Convenience function to construct a semidiscretization of a hyperbolic-parabolic PDE with the BR1
scheme for a DGSEM solver. The passed arguments must correspond to the hyperbolic setup, including
the `solver`. Internally, a matching solver for the parabolic system is created and both a
hyperbolic and a parabolic semidiscretization are created and passed to a
`SemidiscretizationHyperbolicParabolic`.
"""
function SemidiscretizationHyperbolicParabolicBR1(mesh, equations, initial_condition, solver::DGSEM;
                                                  source_terms=nothing,
                                                  boundary_conditions=boundary_condition_periodic,
                                                  RealT=real(solver))
  semi_hyperbolic = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                                 source_terms=source_terms,
                                                 boundary_conditions=boundary_conditions,
                                                 RealT=RealT)

  solver_parabolic = DGSEM(solver.basis, flux_central, solver.volume_integral, solver.mortar)
  semi_parabolic = SemidiscretizationParabolicAuxVars(mesh, equations, initial_condition,
                                                      solver_parabolic,
                                                      source_terms=source_terms,
                                                      boundary_conditions=boundary_conditions,
                                                      RealT=RealT)

  return SemidiscretizationHyperbolicParabolic(semi_hyperbolic, semi_parabolic)
end
