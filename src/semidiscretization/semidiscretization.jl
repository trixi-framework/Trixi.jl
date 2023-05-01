# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


"""
    ndofs(semi::AbstractSemidiscretization)

Return the number of degrees of freedom associated with each scalar variable.
"""
@inline function ndofs(semi::AbstractSemidiscretization)
  mesh, _, solver, cache = mesh_equations_solver_cache(semi)
  ndofs(mesh, solver, cache)
end


"""
    integrate_via_indices(func, u_ode, semi::AbstractSemidiscretization, args...; normalize=true)

Call `func(u, i..., element, equations, solver, args...)` for all nodal indices `i..., element`
and integrate the result using a quadrature associated with the semidiscretization `semi`.

If `normalize` is true, the result is divided by the total volume of the computational domain.
"""
function integrate_via_indices(func::Func, u_ode, semi::AbstractSemidiscretization, args...; normalize=true) where {Func}
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)

  u = wrap_array(u_ode, mesh, equations, solver, cache)
  integrate_via_indices(func, u, mesh, equations, solver, cache, args..., normalize=normalize)
end

"""
    integrate([func=(u_node,equations)->u_node,] u_ode, semi::AbstractSemidiscretization; normalize=true)

Call `func(u_node, equations)` for each vector of nodal variables `u_node` in `u_ode`
and integrate the result using a quadrature associated with the semidiscretization `semi`.

If `normalize` is true, the result is divided by the total volume of the computational domain.
"""
function integrate(func::Func, u_ode, semi::AbstractSemidiscretization; normalize=true) where {Func}
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)

  u = wrap_array(u_ode, mesh, equations, solver, cache)
  integrate(func, u, mesh, equations, solver, cache, normalize=normalize)
end

function integrate(u, semi::AbstractSemidiscretization; normalize=true)
  integrate(cons2cons, u, semi; normalize=normalize)
end


"""
    calc_error_norms([func=(u_node,equations)->u_node,] u_ode, t, analyzer, semi::AbstractSemidiscretization, cache_analysis)

Calculate discrete L2 and L∞ error norms of `func` applied to each nodal variable `u_node` in `u_ode`.
If no exact solution is available, "errors" are calculated using some reference state and can be useful
for regression tests.
"""
calc_error_norms(u_ode, t, analyzer, semi::AbstractSemidiscretization, cache_analysis) = calc_error_norms(cons2cons, u_ode, t, analyzer, semi, cache_analysis)


"""
    semidiscretize(semi::AbstractSemidiscretization, tspan)

Wrap the semidiscretization `semi` as an ODE problem in the time interval `tspan`
that can be passed to `solve` from the [SciML ecosystem](https://diffeq.sciml.ai/latest/).
"""
function semidiscretize(semi::AbstractSemidiscretization, tspan)
  u0_ode = compute_coefficients(first(tspan), semi)
  # TODO: MPI, do we want to synchronize loading and print debug statements, e.g. using
  #       mpi_isparallel() && MPI.Barrier(mpi_comm())
  #       See https://github.com/trixi-framework/Trixi.jl/issues/328
  iip = true # is-inplace, i.e., we modify a vector when calling rhs!
  return ODEProblem{iip}(rhs!, u0_ode, tspan, semi)
end


"""
    semidiscretize(semi::AbstractSemidiscretization, tspan, restart_file::AbstractString)

Wrap the semidiscretization `semi` as an ODE problem in the time interval `tspan`
that can be passed to `solve` from the [SciML ecosystem](https://diffeq.sciml.ai/latest/).
The initial condition etc. is taken from the `restart_file`.
"""
function semidiscretize(semi::AbstractSemidiscretization, tspan, restart_file::AbstractString)
  u0_ode = load_restart_file(semi, restart_file)
  # TODO: MPI, do we want to synchronize loading and print debug statements, e.g. using
  #       mpi_isparallel() && MPI.Barrier(mpi_comm())
  #       See https://github.com/trixi-framework/Trixi.jl/issues/328
  iip = true # is-inplace, i.e., we modify a vector when calling rhs!
  return ODEProblem{iip}(rhs!, u0_ode, tspan, semi)
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

For semidiscretizations `semi` associated with an initial condition, `func` can be omitted
to use the given initial condition at time `t`.
"""
function compute_coefficients(func, t, semi::AbstractSemidiscretization)
  u_ode = allocate_coefficients(mesh_equations_solver_cache(semi)...)
  # Call `compute_coefficients` defined below
  compute_coefficients!(u_ode, func, t, semi)
  return u_ode
end

"""
    compute_coefficients!(u_ode, func, t, semi::AbstractSemidiscretization)

Same as [`compute_coefficients`](@ref) but stores the result in `u_ode`.
"""
function compute_coefficients!(u_ode, func, t, semi::AbstractSemidiscretization)
  u = wrap_array(u_ode, semi)
  # Call `compute_coefficients` defined by the solver
  compute_coefficients!(u, func, t, mesh_equations_solver_cache(semi)...)
end


"""
    linear_structure(semi::AbstractSemidiscretization;
                     t0=zero(real(semi)))

Wraps the right-hand side operator of the semidiscretization `semi`
at time `t0` as an affine-linear operator given by a linear operator `A`
and a vector `b`.
"""
function linear_structure(semi::AbstractSemidiscretization;
                          t0=zero(real(semi)))
  # allocate memory
  u_ode = allocate_coefficients(mesh_equations_solver_cache(semi)...)
  du_ode = similar(u_ode)

  # get the right hand side from possible source terms
  u_ode .= zero(eltype(u_ode))
  rhs!(du_ode, u_ode, semi, t0)
  # Create a copy of `b` used internally to extract the linear part of `semi`.
  # This is necessary to get everything correct when the users updates the
  # returned vector `b`.
  b = -du_ode
  b_tmp = copy(b)

  # wrap the linear operator
  A = LinearMap(length(u_ode), ismutating=true) do dest,src
    rhs!(dest, src, semi, t0)
    @. dest += b_tmp
    dest
  end

  return A, b
end


"""
    jacobian_fd(semi::AbstractSemidiscretization;
                t0=zero(real(semi)),
                u0_ode=compute_coefficients(t0, semi))

Uses the right-hand side operator of the semidiscretization `semi`
and simple second order finite difference to compute the Jacobian `J`
of the semidiscretization `semi` at state `u0_ode`.
"""
function jacobian_fd(semi::AbstractSemidiscretization;
                     t0=zero(real(semi)),
                     u0_ode=compute_coefficients(t0, semi))
  # copy the initial state since it will be modified in the following
  u_ode = copy(u0_ode)
  du0_ode = similar(u_ode)
  dup_ode = similar(u_ode)
  dum_ode = similar(u_ode)

  # compute residual of linearization state
  rhs!(du0_ode, u_ode, semi, t0)

  # initialize Jacobian matrix
  J = zeros(eltype(u_ode), length(u_ode), length(u_ode))

  # use second order finite difference to estimate Jacobian matrix
  for idx in eachindex(u0_ode)
    # determine size of fluctuation
    epsilon = sqrt(eps(u0_ode[idx]))

    # plus fluctuation
    u_ode[idx] = u0_ode[idx] + epsilon
    rhs!(dup_ode, u_ode, semi, t0)

    # minus fluctuation
    u_ode[idx] = u0_ode[idx] - epsilon
    rhs!(dum_ode, u_ode, semi, t0)

    # restore linearisation state
    u_ode[idx] = u0_ode[idx]

    # central second order finite difference
    @. J[:, idx] = (dup_ode - dum_ode) / (2 * epsilon)
  end

  return J
end


"""
    jacobian_ad_forward(semi::AbstractSemidiscretization;
                        t0=zero(real(semi)),
                        u0_ode=compute_coefficients(t0, semi))

Uses the right-hand side operator of the semidiscretization `semi`
and forward mode automatic differentiation to compute the Jacobian `J`
of the semidiscretization `semi` at state `u0_ode`.
"""
function jacobian_ad_forward(semi::AbstractSemidiscretization;
                             t0=zero(real(semi)),
                             u0_ode=compute_coefficients(t0, semi))
  jacobian_ad_forward(semi, t0, u0_ode)
end

# The following version is for plain arrays
function jacobian_ad_forward(semi::AbstractSemidiscretization, t0, u0_ode)
  du_ode = similar(u0_ode)
  config = ForwardDiff.JacobianConfig(nothing, du_ode, u0_ode)

  # Use a function barrier since the generation of the `config` we use above
  # is not type-stable
  _jacobian_ad_forward(semi, t0, u0_ode, du_ode, config)
end

function _jacobian_ad_forward(semi, t0, u0_ode, du_ode, config)

  new_semi = remake(semi, uEltype=eltype(config))
  J = ForwardDiff.jacobian(du_ode, u0_ode, config) do du_ode, u_ode
    Trixi.rhs!(du_ode, u_ode, new_semi, t0)
  end

  return J
end

# This version is specialized to `StructArray`s used by some `DGMulti` solvers.
# We need to convert the numerical solution vectors since ForwardDiff cannot
# handle arrays of `SVector`s.
function jacobian_ad_forward(semi::AbstractSemidiscretization, t0, _u0_ode::StructArray)
  u0_ode_plain = similar(_u0_ode, eltype(eltype(_u0_ode)), (size(_u0_ode)..., nvariables(semi)))
  for (v, u_v) in enumerate(StructArrays.components(_u0_ode))
    u0_ode_plain[.., v] = u_v
  end
  du_ode_plain = similar(u0_ode_plain)
  config = ForwardDiff.JacobianConfig(nothing, du_ode_plain, u0_ode_plain)

  # Use a function barrier since the generation of the `config` we use above
  # is not type-stable
  _jacobian_ad_forward_structarrays(semi, t0, u0_ode_plain, du_ode_plain, config)
end

function _jacobian_ad_forward_structarrays(semi, t0, u0_ode_plain, du_ode_plain, config)

  new_semi = remake(semi, uEltype=eltype(config))
  J = ForwardDiff.jacobian(du_ode_plain, u0_ode_plain, config) do du_ode_plain, u_ode_plain
    u_ode  = StructArray{SVector{nvariables(semi), eltype(config)}}(ntuple(v -> view(u_ode_plain,  :, :, v), nvariables(semi)))
    du_ode = StructArray{SVector{nvariables(semi), eltype(config)}}(ntuple(v -> view(du_ode_plain, :, :, v), nvariables(semi)))
    Trixi.rhs!(du_ode, u_ode, new_semi, t0)
  end

  return J
end

# This version is specialized to arrays of `StaticArray`s used by some `DGMulti` solvers.
# We need to convert the numerical solution vectors since ForwardDiff cannot
# handle arrays of `SVector`s.
function jacobian_ad_forward(semi::AbstractSemidiscretization, t0, _u0_ode::AbstractArray{<:SVector})
  u0_ode_plain = reinterpret(eltype(eltype(_u0_ode)), _u0_ode)
  du_ode_plain = similar(u0_ode_plain)
  config = ForwardDiff.JacobianConfig(nothing, du_ode_plain, u0_ode_plain)

  # Use a function barrier since the generation of the `config` we use above
  # is not type-stable
  _jacobian_ad_forward_staticarrays(semi, t0, u0_ode_plain, du_ode_plain, config)
end

function _jacobian_ad_forward_staticarrays(semi, t0, u0_ode_plain, du_ode_plain, config)

  new_semi = remake(semi, uEltype=eltype(config))
  J = ForwardDiff.jacobian(du_ode_plain, u0_ode_plain, config) do du_ode_plain, u_ode_plain
    u_ode  = reinterpret(SVector{nvariables(semi), eltype(config)}, u_ode_plain)
    du_ode = reinterpret(SVector{nvariables(semi), eltype(config)}, du_ode_plain)
    Trixi.rhs!(du_ode, u_ode, new_semi, t0)
  end

  return J
end



# Sometimes, it can be useful to save some (scalar) variables associated with each element,
# e.g. AMR indicators or shock indicators. Since these usually have to be re-computed
# directly before IO and do not necessarily need to be stored in memory before,
#   get_element_variables!(element_variables, ..)
# is used to retrieve such up to date element variables, modifying
# `element_variables::Dict{Symbol,Any}` in place.
function get_element_variables!(element_variables, u_ode, semi::AbstractSemidiscretization)
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
function wrap_array(u_ode, semi::AbstractSemidiscretization)
  wrap_array(u_ode, mesh_equations_solver_cache(semi)...)
end

# Like `wrap_array`, but guarantees to return a plain `Array`, which can be better
# for writing solution files etc.
function wrap_array_native(u_ode, semi::AbstractSemidiscretization)
  wrap_array_native(u_ode, mesh_equations_solver_cache(semi)...)
end



# TODO: Taal, document interface?
# New mesh/solver combinations have to implement
# - ndofs(mesh, solver, cache)
# - ndims(mesh)
# - nnodes(solver)
# - real(solver)
# - create_cache(mesh, equations, solver, RealT)
# - wrap_array(u_ode, mesh, equations, solver, cache)
# - integrate(func, u, mesh, equations, solver, cache; normalize=true)
# - integrate_via_indices(func, u, mesh, equations, solver, cache, args...; normalize=true)
# - calc_error_norms(func, u, t, analyzer, mesh, equations, initial_condition, solver, cache, cache_analysis)
# - allocate_coefficients(mesh, equations, solver, cache)
# - compute_coefficients!(u, func, mesh, equations, solver, cache)
# - rhs!(du, u, t, mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache)
#


end # @muladd
