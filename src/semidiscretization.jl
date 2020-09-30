

mutable struct PerformanceCounter
  ncalls_since_readout::Int
  runtime::Float64
end

PerformanceCounter() = PerformanceCounter(0, 0.0)

function Base.take!(counter::PerformanceCounter)
  time_per_call = counter.runtime / counter.ncalls_since_readout
  counter.ncalls_since_readout = 0
  counter.runtime = 0.0
  return time_per_call
end

function Base.put!(counter::PerformanceCounter, runtime::Real)
  counter.ncalls_since_readout += 1
  counter.runtime += runtime
end


abstract type AbstractSemidiscretization end


# TODO: Taal refactor, Mesh<:AbstractMesh{NDIMS}, Equations<:AbstractEquations{NDIMS} ?
"""
    SemidiscretizationHyperbolic

A struct containing everything needed to describe a spatial semidiscretization
of a hyperbolic conservation law.
"""
struct SemidiscretizationHyperbolic{Mesh, Equations, InitialConditions, BoundaryConditions, SourceTerms, Solver, Cache} <: AbstractSemidiscretization
  mesh::Mesh

  equations::Equations

  # This guy is a bit messy since we abuse it as some kind of "exact solution"
  # although this doesn't really exist...
  initial_conditions::InitialConditions

  # TODO: Taal BCs
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

function SemidiscretizationHyperbolic(mesh, equations, initial_conditions, solver;
                                      source_terms=nothing,
                                      boundary_conditions=nothing, RealT=real(solver))

  cache = create_cache(mesh, equations, boundary_conditions, solver, RealT)

  SemidiscretizationHyperbolic{typeof(mesh), typeof(equations), typeof(initial_conditions), typeof(boundary_conditions), typeof(source_terms), typeof(solver), typeof(cache)}(
    mesh, equations, initial_conditions, boundary_conditions, source_terms, solver, cache)
end

# TODO: Taal bikeshedding, implement a method with reduced information and the signature
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

@inline nnodes(semi::SemidiscretizationHyperbolic) = nnodes(semi.solver)

@inline function ndofs(semi::AbstractSemidiscretization)
  mesh, +, solver, cache = mesh_equations_solver_cache(semi)
  ndofs(mesh, solver, cache)
end



@inline function get_node_coords(x, semi::SemidiscretizationHyperbolic, indices...)
  @unpack equations, solver = semi

  get_node_coords(x, equations, solver, indices...)
end

@inline function get_node_vars(u, semi::SemidiscretizationHyperbolic, indices...)
  @unpack equations, solver = semi

  get_node_vars(u, equations, solver, indices...)
end

@inline function get_surface_node_vars(u, semi::SemidiscretizationHyperbolic, indices...)
  @unpack equations, solver = semi

  get_surface_node_vars(u, equations, solver, indices...)
end

@inline function set_node_vars!(u, u_node, semi::SemidiscretizationHyperbolic, indices...)
  @unpack equations, solver = semi

  set_node_vars!(u, u_node, equations, solver, indices...)
  return nothing
end

@inline function add_to_node_vars!(u, u_node, semi::SemidiscretizationHyperbolic, indices...)
  @unpack equations, solver = semi

  add_to_node_vars!(u, u_node, equations, solver, indices...)
  return nothing
end


@inline function mesh_equations_solver_cache(semi::SemidiscretizationHyperbolic)
  @unpack mesh, equations, solver, cache = semi
  return mesh, equations, solver, cache
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


function integrate(func, semi::AbstractSemidiscretization, u_ode::AbstractVector, args...; normalize=true)
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)

  u = wrap_array(u_ode, mesh, equations, solver, cache)
  integrate(func, mesh, equations, solver, cache, u, args..., normalize=normalize)
end

function integrate(func, u_ode::AbstractVector, semi::AbstractSemidiscretization; normalize=true)
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)

  u = wrap_array(u_ode, mesh, equations, solver, cache)
  integrate(func, u, mesh, equations, solver, cache, normalize=normalize)
end

function integrate(u, semi::AbstractSemidiscretization; normalize=true)
  integrate(cons2cons, u, semi; normalize=normalize)
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
calc_error_norms(u, t, analyzer, semi::AbstractSemidiscretization) = calc_error_norms(cons2cons, u, t, analyzer, semi)


function compute_coefficients(t, semi::SemidiscretizationHyperbolic)
  compute_coefficients(semi.initial_conditions, t, semi)
end

function compute_coefficients(func, t, semi::SemidiscretizationHyperbolic)
  @unpack mesh, equations, solver, cache = semi

  u_ode = allocate_coefficients(mesh, equations, solver, cache)
  compute_coefficients!(u_ode, func, t, semi)
  return u_ode
end

function compute_coefficients!(u_ode::AbstractVector, t, semi::SemidiscretizationHyperbolic)
  compute_coefficients!(u_ode, semi.initial_conditions, t, semi)
end

function compute_coefficients!(u_ode::AbstractVector, func, t, semi::SemidiscretizationHyperbolic)
  @unpack mesh, equations, solver, cache = semi
  u = wrap_array(u_ode, mesh, equations, solver, cache)

  compute_coefficients!(u, func, t, mesh, equations, solver, cache)
end


function semidiscretize(semi::AbstractSemidiscretization, tspan)
  u0_ode = compute_coefficients(first(tspan), semi)
  return ODEProblem(rhs!, u0_ode, tspan, semi)
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


function get_element_variables!(element_variables, u_ode::AbstractVector, semi::AbstractSemidiscretization)
  u = wrap_array(u_ode, semi)
  get_element_variables!(element_variables, u, mesh_equations_solver_cache(semi)...)
end


# TODO: Taal interface
# New mesh/solver combinations have to implement
# - ndims(mesh)
# - nnodes(solver)
# - real(solver)
# - ndofs(mesh, solver, cache)
# - create_cache(mesh, equations, boundary_conditions, solver)
# - wrap_array(u_ode::AbstractVector, mesh, equations, solver, cache)
# - integrate(func, mesh, equations, solver, cache, u; normalize=true)
# - integrate(func, u, mesh, equations, solver, cache, args...; normalize=true)
# - calc_error_norms(func, u, t, analyzer, mesh, equations, initial_conditions, solver, cache)
# - allocate_coefficients(mesh, equations, solver, cache)
# - compute_coefficients!(u, func, mesh, equations, solver, cache)
# - rhs!(du, u, t, mesh, equations, initial_conditions, boundary_conditions, source_terms, solver, cache)
#
