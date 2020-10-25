
"""
    ndofs(semi::AbstractSemidiscretization)

Return the number of degrees of freedom associated with each scalar variable.
"""
@inline function ndofs(semi::AbstractSemidiscretization)
  mesh, _, solver, cache = mesh_equations_solver_cache(semi)
  ndofs(mesh, solver, cache)
end


"""
    integrate_via_indices(func, u_ode::AbstractVector, semi::AbstractSemidiscretization, args...; normalize=true)

Call `func(u, i..., element, equations, solver, args...)` for all nodal indices `i..., element`
and integrate the result using a quadrature associated with the semidiscretization `semi`.

If `normalize` is true, the result is divided by the total volume of the computational domain.
"""
function integrate_via_indices(func, u_ode::AbstractVector, semi::AbstractSemidiscretization, args...; normalize=true)
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)

  u = wrap_array(u_ode, mesh, equations, solver, cache)
  integrate_via_indices(func, u, mesh, equations, solver, cache, args..., normalize=normalize)
end

"""
    integrate([func=(u_node,equations)->u_node,] u_ode::AbstractVector, semi::AbstractSemidiscretization; normalize=true)

Call `func(u_node, equations)` for each vector of nodal variables `u_node` in `u_ode`
and integrate the result using a quadrature associated with the semidiscretization `semi`.

If `normalize` is true, the result is divided by the total volume of the computational domain.
"""
function integrate(func, u_ode::AbstractVector, semi::AbstractSemidiscretization; normalize=true)
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)

  u = wrap_array(u_ode, mesh, equations, solver, cache)
  integrate(func, u, mesh, equations, solver, cache, normalize=normalize)
end

function integrate(u, semi::AbstractSemidiscretization; normalize=true)
  integrate(cons2cons, u, semi; normalize=normalize)
end


"""
    calc_error_norms([func=(u_node,equations)->u_node,] u_ode, t, analyzer, semi::AbstractSemidiscretization)

Calculate discrete L2 and L∞ error norms of `func` applied to each nodal variable `u_node` in `u_ode`.
If no exact solution is available, "errors" are calculated using some reference state and can be useful
for regression tests.
"""
calc_error_norms(u_ode, t, analyzer, semi::AbstractSemidiscretization) = calc_error_norms(cons2cons, u_ode, t, analyzer, semi)


"""
    semidiscretize(semi::AbstractSemidiscretization, tspan)

Wrap the semidiscretization `semi` as an ODE problem in the time interval `tspan`
that can be passed to `solve` from the [SciML ecosystem](https://diffeq.sciml.ai/latest/).
"""
function semidiscretize(semi::AbstractSemidiscretization, tspan)
  u0_ode = compute_coefficients(first(tspan), semi)
  return ODEProblem(rhs!, u0_ode, tspan, semi)
end


"""
    compute_coefficients(func, t, semi::AbstractSemidiscretization)

Compute the discrete coefficients of the continuous function `func` at time `t`
associated with the semidiscretization `semi`.
For example, the discrete coefficients of `func` for a discontinuous Galerkin
spectral element method ([`DGSEM`](@ref)) are the values of `func` at the
Lobatto-Legendre nodes. Similarly, a classical finite difference method will use
the values of `func` at the nodes of the grid assoociated with the semidiscretization
`semi`.
"""
function compute_coefficients(func, t, semi::AbstractSemidiscretization)
  u_ode = allocate_coefficients(mesh_equations_solver_cache(semi)...)
  compute_coefficients!(u_ode, func, t, semi)
  return u_ode
end

"""
    compute_coefficients!(u_ode, func, t, semi::AbstractSemidiscretization)

Same as [`compute_coefficients`](@ref) but stores the result in `u_ode`.
"""
function compute_coefficients!(u_ode::AbstractVector, func, t, semi::AbstractSemidiscretization)
  u = wrap_array(u_ode, semi)
  compute_coefficients!(u, func, t, mesh_equations_solver_cache(semi)...)
end


# Sometimes, it can be useful to save some (scalar) variables associated with each element,
# e.g. AMR indicators or shock indicators. Since these usually have to be re-computed
# directly before IO and do not necessarily need to be stored in memory before,
#   get_element_variables!(element_variables, ..)
# is used to retrieve such up to date element variables, modifying
# `element_variables::Dict{Symbol,Any}` in place.
function get_element_variables!(element_variables, u_ode::AbstractVector, semi::AbstractSemidiscretization)
  u = wrap_array(u_ode, semi)
  get_element_variables!(element_variables, u, mesh_equations_solver_cache(semi)...)
end


# To implement AMR and use OrdinaryDiffEq.jl etc., we have to be a bit creative.
# Since the caches of the SciML ecosystem are immutable structs, we cannot simply
# change the underlying arrays therein. Hence, to support changing the number of
# DOFs, we need to use `resize!`. In some sense, this will force us to write more
# efficient code, since `resize!` will make use of previously allocated memory
# instead of allocating memory from scratch every time.
#
# However, multidimensional `Array`s don't support `resize!`. One option might be
# to use ElasticArrays.jl. But I don't really like that approach. Needing to use
# ElasticArray doesn't feel completely good to me, since we also want to experiment
# with other array types such as PaddedMatrices.jl, see trixi-framework/Trixi.jl#166.
# Then, we would need to wrap an Array inside something from PaddedMatrices.jl inside
# something from ElasticArrays.jl - or the other way round? Is that possible at all?
# If we go further, this looks like it could easily explode.
#
# Currently, the best option seems to be to let OrdinaryDiffEq.jl use `Vector`s,
# which can be `resize!`ed for AMR. Then, we have to wrap these `Vector`s inside
# Trixi.jl as our favorite multidimensional array type. We need to do this wrapping
# in every method exposed to OrdinaryDiffEq, i.e. in the first levels of things like
# rhs!, AMRCallback, StepsizeCallback, AnalysisCallback, SaveSolutionCallback
#
# This wrapping will also allow us to experiment more easily with additional
# kinds of wrapping, e.g. HybridArrays.jl or PaddedMatrices.jl to inform the
# compiler about the sizes of the first few dimensions in DG methods, i.e.
# nvariables(equations) and nnodes(dg).
#
# In some sense, having plain multidimensional `Array`s not support `resize!`
# isn't necessarily a bug (although it would be nice to add this possibility to
# base Julia) but can turn out to be a feature for us, because it will aloow us
# more specializations.
# Since we can use multiple dispatch, these kinds of specializations can be
# tailored specifically to each combinations of mesh/solver etc.
#
# Under the hood, `wrap_array(u_ode, mesh, equations, solver, cache)` might
# (and probably will) use `unsafe_wrap`. Hence, you have to remember to
# `GC.@preserve` temporaries that are only used indirectly via `wrap_array`
# to avoid stochastic memory errors.
#
# Xref https://github.com/SciML/OrdinaryDiffEq.jl/pull/1275
function wrap_array(u_ode::AbstractVector, semi::AbstractSemidiscretization)
  wrap_array(u_ode, mesh_equations_solver_cache(semi)...)
end



# TODO: Taal refactor, Mesh<:AbstractMesh{NDIMS}, Equations<:AbstractEquations{NDIMS} ?
"""
    SemidiscretizationHyperbolic

A struct containing everything needed to describe a spatial semidiscretization
of a hyperbolic conservation law.
"""
struct SemidiscretizationHyperbolic{Mesh, Equations, InitialConditions, BoundaryConditions,
                                    SourceTerms, Solver, Cache} <: AbstractSemidiscretization

  mesh::Mesh
  equations::Equations

  # This guy is a bit messy since we abuse it as some kind of "exact solution"
  # although this doesn't really exist...
  initial_conditions::InitialConditions

  boundary_conditions::BoundaryConditions
  source_terms::SourceTerms
  solver::Solver
  cache::Cache
  performance_counter::PerformanceCounter

  function SemidiscretizationHyperbolic{Mesh, Equations, InitialConditions, BoundaryConditions, SourceTerms, Solver, Cache}(
      mesh::Mesh, equations::Equations,
      initial_conditions::InitialConditions, boundary_conditions::BoundaryConditions,
      source_terms::SourceTerms,
      solver::Solver, cache::Cache) where {Mesh, Equations, InitialConditions, BoundaryConditions, SourceTerms, Solver, Cache}
    @assert ndims(mesh) == ndims(equations)

    performance_counter = PerformanceCounter()

    new(mesh, equations, initial_conditions, boundary_conditions, source_terms, solver, cache, performance_counter)
  end
end

"""
    SemidiscretizationHyperbolic(mesh, equations, initial_conditions, solver;
                                 source_terms=nothing,
                                 boundary_conditions=nothing)

Construct a semidiscretization of a hyperbolic PDE.
"""
function SemidiscretizationHyperbolic(mesh, equations, initial_conditions, solver;
                                      source_terms=nothing,
                                      boundary_conditions=nothing, RealT=real(solver))

  cache = create_cache(mesh, equations, solver, RealT)

  SemidiscretizationHyperbolic{typeof(mesh), typeof(equations), typeof(initial_conditions), typeof(boundary_conditions), typeof(source_terms), typeof(solver), typeof(cache)}(
    mesh, equations, initial_conditions, boundary_conditions, source_terms, solver, cache)
end


function Base.show(io::IO, semi::SemidiscretizationHyperbolic)
  print(io, "SemidiscretizationHyperbolic(")
  print(io,       semi.mesh)
  print(io, ", ", semi.equations)
  print(io, ", ", semi.initial_conditions)
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
  println(io, "SemidiscretizationHyperbolic using")
  println(io, "- ", semi.mesh)
  println(io, "- ", semi.equations)
  println(io, "- ", semi.initial_conditions)
  println(io, "- ", semi.boundary_conditions)
  println(io, "- ", semi.source_terms)
  println(io, "- ", semi.solver)
  print(io, "- cache with fields:")
  for key in keys(semi.cache)
    print(io, " ", key)
  end
  print(io, "\nTotal number of degrees of freedom: ", ndofs(semi))
end


@inline Base.ndims(semi::SemidiscretizationHyperbolic) = ndims(semi.mesh)

@inline nvariables(semi::SemidiscretizationHyperbolic) = nvariables(semi.equations)

@inline Base.real(semi::SemidiscretizationHyperbolic) = real(semi.solver)


@inline function mesh_equations_solver_cache(semi::SemidiscretizationHyperbolic)
  @unpack mesh, equations, solver, cache = semi
  return mesh, equations, solver, cache
end


function calc_error_norms(func, u_ode::AbstractVector, t, analyzer, semi::SemidiscretizationHyperbolic)
  @unpack mesh, equations, initial_conditions, solver, cache = semi
  u = wrap_array(u_ode, mesh, equations, solver, cache)

  calc_error_norms(func, u, t, analyzer, mesh, equations, initial_conditions, solver, cache)
end

function calc_error_norms(func, u, t, analyzer, semi::SemidiscretizationHyperbolic)
  @unpack mesh, equations, initial_conditions, solver, cache = semi

  calc_error_norms(func, u, t, analyzer, mesh, equations, initial_conditions, solver, cache)
end


function compute_coefficients(t, semi::SemidiscretizationHyperbolic)
  compute_coefficients(semi.initial_conditions, t, semi)
end

function compute_coefficients!(u_ode::AbstractVector, t, semi::SemidiscretizationHyperbolic)
  compute_coefficients!(u_ode, semi.initial_conditions, t, semi)
end


# TODO: Taal better name
function rhs!(du_ode, u_ode, semi::SemidiscretizationHyperbolic, t)
  @unpack mesh, equations, initial_conditions, boundary_conditions, source_terms, solver, cache = semi

  u  = wrap_array(u_ode,  mesh, equations, solver, cache)
  du = wrap_array(du_ode, mesh, equations, solver, cache)

  # TODO: Taal decide, do we need to pass the mesh?
  time_start = time_ns()
  @timeit_debug timer() "rhs!" rhs!(du, u, t, mesh, equations, initial_conditions, boundary_conditions, source_terms, solver, cache)
  runtime = time_ns() - time_start
  put!(semi.performance_counter, runtime)

  return nothing
end


# TODO: Taal interface
# New mesh/solver combinations have to implement
# - ndofs(mesh, solver, cache)
# - ndims(mesh)
# - nnodes(solver)
# - real(solver)
# - create_cache(mesh, equations, solver, RealT)
# - wrap_array(u_ode::AbstractVector, mesh, equations, solver, cache)
# - integrate(func, u, mesh, equations, solver, cache; normalize=true)
# - integrate_via_indices(func, u, mesh, equations, solver, cache, args...; normalize=true)
# - calc_error_norms(func, u, t, analyzer, mesh, equations, initial_conditions, solver, cache)
# - allocate_coefficients(mesh, equations, solver, cache)
# - compute_coefficients!(u, func, mesh, equations, solver, cache)
# - rhs!(du, u, t, mesh, equations, initial_conditions, boundary_conditions, source_terms, solver, cache)
#
