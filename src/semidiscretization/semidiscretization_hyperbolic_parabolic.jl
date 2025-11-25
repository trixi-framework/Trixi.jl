# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    SemidiscretizationHyperbolicParabolic

A struct containing everything needed to describe a spatial semidiscretization
of a mixed hyperbolic-parabolic conservation law.
"""
struct SemidiscretizationHyperbolicParabolic{Mesh, Equations, EquationsParabolic,
                                             InitialCondition,
                                             BoundaryConditions,
                                             BoundaryConditionsParabolic,
                                             SourceTerms, Solver, SolverParabolic,
                                             Cache, CacheParabolic} <:
       AbstractSemidiscretization
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
end

"""
    SemidiscretizationHyperbolicParabolic(mesh, both_equations, initial_condition, solver;
                                          solver_parabolic=default_parabolic_solver(),
                                          source_terms=nothing,
                                          both_boundary_conditions=(boundary_condition_periodic, boundary_condition_periodic),
                                          RealT=real(solver),
                                          uEltype=RealT)

Construct a semidiscretization of a hyperbolic-parabolic PDE.
"""
function SemidiscretizationHyperbolicParabolic(mesh, equations::Tuple,
                                               initial_condition, solver;
                                               solver_parabolic = default_parabolic_solver(),
                                               source_terms = nothing,
                                               boundary_conditions = (boundary_condition_periodic,
                                                                      boundary_condition_periodic),
                                               # `RealT` is used as real type for node locations etc.
                                               # while `uEltype` is used as element type of solutions etc.
                                               RealT = real(solver), uEltype = RealT)
    equations, equations_parabolic = equations
    boundary_conditions, boundary_conditions_parabolic = boundary_conditions

    @assert ndims(mesh) == ndims(equations)
    @assert ndims(mesh) == ndims(equations_parabolic)

    if !(nvariables(equations) == nvariables(equations_parabolic))
        throw(ArgumentError("Current implementation of viscous terms requires the same number of conservative and gradient variables."))
    end

    cache = create_cache(mesh, equations, solver, RealT, uEltype)
    _boundary_conditions = digest_boundary_conditions(boundary_conditions,
                                                      mesh, solver, cache)
    check_periodicity_mesh_boundary_conditions(mesh, _boundary_conditions)

    cache_parabolic = create_cache_parabolic(mesh, equations, equations_parabolic,
                                             solver, solver_parabolic,
                                             nelements(cache.elements), RealT, uEltype)

    _boundary_conditions_parabolic = digest_boundary_conditions(boundary_conditions_parabolic,
                                                                mesh, solver, cache)
    check_periodicity_mesh_boundary_conditions(mesh, _boundary_conditions_parabolic)

    performance_counter = PerformanceCounterList{2}(false)

    return SemidiscretizationHyperbolicParabolic{typeof(mesh),
                                                 typeof(equations),
                                                 typeof(equations_parabolic),
                                                 typeof(initial_condition),
                                                 typeof(_boundary_conditions),
                                                 typeof(_boundary_conditions_parabolic),
                                                 typeof(source_terms),
                                                 typeof(solver),
                                                 typeof(solver_parabolic),
                                                 typeof(cache),
                                                 typeof(cache_parabolic)}(mesh,
                                                                          equations,
                                                                          equations_parabolic,
                                                                          initial_condition,
                                                                          _boundary_conditions,
                                                                          _boundary_conditions_parabolic,
                                                                          source_terms,
                                                                          solver,
                                                                          solver_parabolic,
                                                                          cache,
                                                                          cache_parabolic,
                                                                          performance_counter)
end

# Create a new semidiscretization but change some parameters compared to the input.
# `Base.similar` follows a related concept but would require us to `copy` the `mesh`,
# which would impact the performance. Instead, `SciMLBase.remake` has exactly the
# semantics we want to use here. In particular, it allows us to re-use mutable parts,
# e.g. `remake(semi).mesh === semi.mesh`.
function remake(semi::SemidiscretizationHyperbolicParabolic;
                uEltype = real(semi.solver),
                mesh = semi.mesh,
                equations = semi.equations,
                equations_parabolic = semi.equations_parabolic,
                initial_condition = semi.initial_condition,
                solver = semi.solver,
                solver_parabolic = semi.solver_parabolic,
                source_terms = semi.source_terms,
                boundary_conditions = semi.boundary_conditions,
                boundary_conditions_parabolic = semi.boundary_conditions_parabolic)
    # TODO: Which parts do we want to `remake`? At least the solver needs some
    #       special care if shock-capturing volume integrals are used (because of
    #       the indicators and their own caches...).
    return SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                                 initial_condition, solver;
                                                 solver_parabolic, source_terms,
                                                 boundary_conditions = (boundary_conditions,
                                                                        boundary_conditions_parabolic),
                                                 uEltype)
end

function Base.show(io::IO, semi::SemidiscretizationHyperbolicParabolic)
    @nospecialize semi # reduce precompilation time

    print(io, "SemidiscretizationHyperbolicParabolic(")
    print(io, semi.mesh)
    print(io, ", ", semi.equations)
    print(io, ", ", semi.equations_parabolic)
    print(io, ", ", semi.initial_condition)
    print(io, ", ", semi.boundary_conditions)
    print(io, ", ", semi.boundary_conditions_parabolic)
    print(io, ", ", semi.source_terms)
    print(io, ", ", semi.solver)
    print(io, ", ", semi.solver_parabolic)
    print(io, ", cache(")
    for (idx, key) in enumerate(keys(semi.cache))
        idx > 1 && print(io, " ")
        print(io, key)
    end
    print(io, "))")
end

function Base.show(io::IO, ::MIME"text/plain",
                   semi::SemidiscretizationHyperbolicParabolic)
    @nospecialize semi # reduce precompilation time

    if get(io, :compact, false)
        show(io, semi)
    else
        summary_header(io, "SemidiscretizationHyperbolicParabolic")
        summary_line(io, "#spatial dimensions", ndims(semi.equations))
        summary_line(io, "mesh", semi.mesh)
        summary_line(io, "hyperbolic equations", semi.equations |> typeof |> nameof)
        summary_line(io, "parabolic equations",
                     semi.equations_parabolic |> typeof |> nameof)
        summary_line(io, "initial condition", semi.initial_condition)

        # print_boundary_conditions(io, semi)

        summary_line(io, "source terms", semi.source_terms)
        summary_line(io, "solver", semi.solver |> typeof |> nameof)
        summary_line(io, "parabolic solver", semi.solver_parabolic |> typeof |> nameof)
        summary_line(io, "total #DOFs per field", ndofsglobal(semi))
        summary_footer(io)
    end
end

@inline Base.ndims(semi::SemidiscretizationHyperbolicParabolic) = ndims(semi.mesh)

@inline function nvariables(semi::SemidiscretizationHyperbolicParabolic)
    nvariables(semi.equations)
end

@inline Base.real(semi::SemidiscretizationHyperbolicParabolic) = real(semi.solver)

# retain dispatch on hyperbolic equations only
@inline function mesh_equations_solver_cache(semi::SemidiscretizationHyperbolicParabolic)
    @unpack mesh, equations, solver, cache = semi
    return mesh, equations, solver, cache
end

function calc_error_norms(func, u_ode, t, analyzer,
                          semi::SemidiscretizationHyperbolicParabolic, cache_analysis)
    @unpack mesh, equations, initial_condition, solver, cache = semi
    u = wrap_array(u_ode, mesh, equations, solver, cache)

    calc_error_norms(func, u, t, analyzer, mesh, equations, initial_condition, solver,
                     cache, cache_analysis)
end

function compute_coefficients(t, semi::SemidiscretizationHyperbolicParabolic)
    # Call `compute_coefficients` in `src/semidiscretization/semidiscretization.jl`
    compute_coefficients(semi.initial_condition, t, semi)
end

function compute_coefficients!(u_ode, t, semi::SemidiscretizationHyperbolicParabolic)
    compute_coefficients!(u_ode, semi.initial_condition, t, semi)
end

# Required for storing `extra_node_variables` in the `SaveSolutionCallback`.
# Not to be confused with `get_node_vars` which returns the variables of the simulated equation.
function get_node_variables!(node_variables, u_ode,
                             semi::SemidiscretizationHyperbolicParabolic)
    get_node_variables!(node_variables, u_ode, mesh_equations_solver_cache(semi)...,
                        semi.equations_parabolic, semi.cache_parabolic)
end

"""
    semidiscretize(semi::SemidiscretizationHyperbolicParabolic, tspan;
                   jac_prototype_parabolic::Union{AbstractMatrix, Nothing} = nothing,
                   colorvec_parabolic::Union{AbstractVector, Nothing} = nothing)

Wrap the semidiscretization `semi` as a split ODE problem in the time interval `tspan`
that can be passed to `solve` from the [SciML ecosystem](https://diffeq.sciml.ai/latest/).
The parabolic right-hand side is the first function of the split ODE problem and
will be used by default by the implicit part of IMEX methods from the
SciML ecosystem.

Optional keyword arguments:
- `jac_prototype_parabolic`: Expected to come from [SparseConnectivityTracer.jl](https://github.com/adrhill/SparseConnectivityTracer.jl).
  Specifies the sparsity structure of the parabolic function's Jacobian to enable e.g. efficient implicit time stepping.
  The [`SplitODEProblem`](https://docs.sciml.ai/DiffEqDocs/stable/types/split_ode_types/#SciMLBase.SplitODEProblem) only expects the Jacobian
  to be defined on the first function it takes in, which is treated implicitly. This corresponds to the parabolic right-hand side in Trixi.jl.
  The hyperbolic right-hand side is expected to be treated explicitly, and therefore its Jacobian is irrelevant.
- `colorvec_parabolic`: Expected to come from [SparseMatrixColorings.jl](https://github.com/gdalle/SparseMatrixColorings.jl).
  Allows for even faster Jacobian computation. Not necessarily required when `jac_prototype_parabolic` is given.
"""
function semidiscretize(semi::SemidiscretizationHyperbolicParabolic, tspan;
                        jac_prototype_parabolic::Union{AbstractMatrix, Nothing} = nothing,
                        colorvec_parabolic::Union{AbstractVector, Nothing} = nothing,
                        reset_threads = true)
    # Optionally reset Polyester.jl threads. See
    # https://github.com/trixi-framework/Trixi.jl/issues/1583
    # https://github.com/JuliaSIMD/Polyester.jl/issues/30
    if reset_threads
        Polyester.reset_threads!()
    end

    u0_ode = compute_coefficients(first(tspan), semi)
    # TODO: MPI, do we want to synchronize loading and print debug statements, e.g. using
    #       mpi_isparallel() && MPI.Barrier(mpi_comm())
    #       See https://github.com/trixi-framework/Trixi.jl/issues/328
    iip = true # is-inplace, i.e., we modify a vector when calling rhs_parabolic!, rhs!

    # Check if Jacobian prototype is provided for sparse Jacobian
    if jac_prototype_parabolic !== nothing
        # Convert `jac_prototype_parabolic` to real type, as seen here:
        # https://docs.sciml.ai/DiffEqDocs/stable/tutorials/advanced_ode_example/#Declaring-a-Sparse-Jacobian-with-Automatic-Sparsity-Detection
        parabolic_ode = SciMLBase.ODEFunction(rhs_parabolic!,
                                              jac_prototype = convert.(eltype(u0_ode),
                                                                       jac_prototype_parabolic),
                                              colorvec = colorvec_parabolic) # coloring vector is optional

        # Note that the IMEX time integration methods of OrdinaryDiffEq.jl treat the
        # first function implicitly and the second one explicitly. Thus, we pass the
        # (potentially) stiffer parabolic function first.
        return SplitODEProblem{iip}(parabolic_ode, rhs!, u0_ode, tspan, semi)
    else
        # We could also construct an `ODEFunction` explicitly without the Jacobian here,
        # but we stick to the lean direct in-place functions `rhs_parabolic!` and
        # let OrdinaryDiffEq.jl handle the rest
        return SplitODEProblem{iip}(rhs_parabolic!, rhs!, u0_ode, tspan, semi)
    end
end

"""
    semidiscretize(semi::SemidiscretizationHyperbolicParabolic, tspan,
                   restart_file::AbstractString;
                   jac_prototype_parabolic::Union{AbstractMatrix, Nothing} = nothing,
                   colorvec_parabolic::Union{AbstractVector, Nothing} = nothing)

Wrap the semidiscretization `semi` as a split ODE problem in the time interval `tspan`
that can be passed to `solve` from the [SciML ecosystem](https://diffeq.sciml.ai/latest/).
The parabolic right-hand side is the first function of the split ODE problem and
will be used by default by the implicit part of IMEX methods from the
SciML ecosystem.

The initial condition etc. is taken from the `restart_file`.

Optional keyword arguments:
- `jac_prototype_parabolic`: Expected to come from [SparseConnectivityTracer.jl](https://github.com/adrhill/SparseConnectivityTracer.jl).
  Specifies the sparsity structure of the parabolic function's Jacobian to enable e.g. efficient implicit time stepping.
  The [`SplitODEProblem`](https://docs.sciml.ai/DiffEqDocs/stable/types/split_ode_types/#SciMLBase.SplitODEProblem) only expects the Jacobian
  to be defined on the first function it takes in, which is treated implicitly. This corresponds to the parabolic right-hand side in Trixi.jl.
  The hyperbolic right-hand side is expected to be treated explicitly, and therefore its Jacobian is irrelevant.
- `colorvec_parabolic`: Expected to come from [SparseMatrixColorings.jl](https://github.com/gdalle/SparseMatrixColorings.jl).
  Allows for even faster Jacobian computation. Not necessarily required when `jac_prototype_parabolic` is given.
"""
function semidiscretize(semi::SemidiscretizationHyperbolicParabolic, tspan,
                        restart_file::AbstractString;
                        jac_prototype_parabolic::Union{AbstractMatrix, Nothing} = nothing,
                        colorvec_parabolic::Union{AbstractVector, Nothing} = nothing,
                        reset_threads = true)
    # Optionally reset Polyester.jl threads. See
    # https://github.com/trixi-framework/Trixi.jl/issues/1583
    # https://github.com/JuliaSIMD/Polyester.jl/issues/30
    if reset_threads
        Polyester.reset_threads!()
    end

    u0_ode = load_restart_file(semi, restart_file)
    # TODO: MPI, do we want to synchronize loading and print debug statements, e.g. using
    #       mpi_isparallel() && MPI.Barrier(mpi_comm())
    #       See https://github.com/trixi-framework/Trixi.jl/issues/328
    iip = true # is-inplace, i.e., we modify a vector when calling rhs_parabolic!, rhs!

    # Check if Jacobian prototype is provided for sparse Jacobian
    if jac_prototype_parabolic !== nothing
        # Convert `jac_prototype_parabolic` to real type, as seen here:
        # https://docs.sciml.ai/DiffEqDocs/stable/tutorials/advanced_ode_example/#Declaring-a-Sparse-Jacobian-with-Automatic-Sparsity-Detection
        parabolic_ode = SciMLBase.ODEFunction(rhs_parabolic!,
                                              jac_prototype = convert.(eltype(u0_ode),
                                                                       jac_prototype_parabolic),
                                              colorvec = colorvec_parabolic) # coloring vector is optional

        # Note that the IMEX time integration methods of OrdinaryDiffEq.jl treat the
        # first function implicitly and the second one explicitly. Thus, we pass the
        # (potentially) stiffer parabolic function first.
        return SplitODEProblem{iip}(parabolic_ode, rhs!, u0_ode, tspan, semi)
    else
        # We could also construct an `ODEFunction` explicitly without the Jacobian here,
        # but we stick to the lean direct in-place function `rhs_parabolic!` and
        # let OrdinaryDiffEq.jl handle the rest
        return SplitODEProblem{iip}(rhs_parabolic!, rhs!, u0_ode, tspan, semi)
    end
end

function rhs!(du_ode, u_ode, semi::SemidiscretizationHyperbolicParabolic, t)
    @unpack mesh, equations, boundary_conditions, source_terms, solver, cache = semi

    u = wrap_array(u_ode, mesh, equations, solver, cache)
    du = wrap_array(du_ode, mesh, equations, solver, cache)

    # TODO: Taal decide, do we need to pass the mesh?
    time_start = time_ns()
    @trixi_timeit timer() "rhs!" rhs!(du, u, t, mesh, equations,
                                      boundary_conditions, source_terms, solver, cache)
    runtime = time_ns() - time_start
    put!(semi.performance_counter.counters[1], runtime)

    return nothing
end

function rhs_parabolic!(du_ode, u_ode, semi::SemidiscretizationHyperbolicParabolic, t)
    @unpack mesh, equations_parabolic, boundary_conditions_parabolic, source_terms, solver, solver_parabolic, cache, cache_parabolic = semi

    u = wrap_array(u_ode, mesh, equations_parabolic, solver, cache)
    du = wrap_array(du_ode, mesh, equations_parabolic, solver, cache)

    # TODO: Taal decide, do we need to pass the mesh?
    time_start = time_ns()
    @trixi_timeit timer() "parabolic rhs!" rhs_parabolic!(du, u, t, mesh,
                                                          equations_parabolic,
                                                          boundary_conditions_parabolic,
                                                          source_terms,
                                                          solver, solver_parabolic,
                                                          cache, cache_parabolic)
    runtime = time_ns() - time_start
    put!(semi.performance_counter.counters[2], runtime)

    return nothing
end

"""
    linear_structure(semi::SemidiscretizationHyperbolicParabolic;
                     t0 = zero(real(semi)))

Wraps the right-hand side operator of the hyperbolic-parabolic semidiscretization `semi`
at time `t0` as an affine-linear operator given by a linear operator `A`
and a vector `b`:
```math
\\partial_t u(t) = A u(t) - b.
```
Works only for linear equations, i.e., equations with `have_constant_speed(equations) == True()`.

This has the benefit of greatly reduced memory consumption compared to constructing
the full system matrix explicitly, as done for instance in
[`jacobian_fd`](@ref) and [`jacobian_ad_forward`](@ref).

The returned linear operator `A` is a matrix-free representation which can be
supplied to iterative solvers from, e.g., [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl).
"""
function linear_structure(semi::SemidiscretizationHyperbolicParabolic;
                          t0 = zero(real(semi)))
    if have_constant_speed(semi.equations) == False()
        throw(ArgumentError("`linear_structure` expects linear equations."))
    end

    # allocate memory
    u_ode = allocate_coefficients(mesh_equations_solver_cache(semi)...)
    du_ode = similar(u_ode)

    # get the right hand side from boundary conditions and optional source terms
    u_ode .= zero(eltype(u_ode))
    rhs!(du_ode, u_ode, semi, t0)
    b = -du_ode

    # Repeat for parabolic part
    rhs_parabolic!(du_ode, u_ode, semi, t0)
    @. b -= du_ode

    # Create a copy of `b` used internally to extract the linear part of `semi`.
    # This is necessary to get everything correct when the user updates the
    # returned vector `b`.
    b_tmp = copy(b)

    # additional storage for parabolic part
    dest_para = similar(du_ode)

    # wrap the linear operator
    A = LinearMap(length(u_ode), ismutating = true) do dest, src
        rhs!(dest, src, semi, t0)
        rhs_parabolic!(dest_para, src, semi, t0)

        @. dest += dest_para + b_tmp
        return dest
    end

    return A, b
end

function _jacobian_ad_forward(semi::SemidiscretizationHyperbolicParabolic, t0, u0_ode,
                              du_ode, config)
    new_semi = remake(semi, uEltype = eltype(config))

    du_ode_hyp = Vector{eltype(config)}(undef, length(du_ode))
    J = ForwardDiff.jacobian(du_ode, u0_ode, config) do du_ode, u_ode
        # Implementation of split ODE problem in OrdinaryDiffEq
        rhs!(du_ode_hyp, u_ode, new_semi, t0)
        rhs_parabolic!(du_ode, u_ode, new_semi, t0)
        du_ode .+= du_ode_hyp
    end

    return J
end

"""
    jacobian_ad_forward_parabolic(semi::SemidiscretizationHyperbolicParabolic;
                                  t0 = zero(real(semi)),
                                  u0_ode = compute_coefficients(t0, semi))

Uses the *parabolic part* of the right-hand side operator of the [`SemidiscretizationHyperbolicParabolic`](@ref) `semi`
and forward mode automatic differentiation to compute the Jacobian `J` of the 
parabolic/diffusive contribution only at time `t0` and state `u0_ode`.

This might be useful for operator-splitting methods, e.g., the construction of optimized 
time integrators which optimize different methods for the hyperbolic and parabolic part separately.
"""
function jacobian_ad_forward_parabolic(semi::SemidiscretizationHyperbolicParabolic;
                                       t0 = zero(real(semi)),
                                       u0_ode = compute_coefficients(t0, semi))
    jacobian_ad_forward_parabolic(semi, t0, u0_ode)
end

# The following version is for plain arrays
function jacobian_ad_forward_parabolic(semi::SemidiscretizationHyperbolicParabolic,
                                       t0, u0_ode)
    du_ode = similar(u0_ode)
    config = ForwardDiff.JacobianConfig(nothing, du_ode, u0_ode)

    # Use a function barrier since the generation of the `config` we use above
    # is not type-stable
    _jacobian_ad_forward_parabolic(semi, t0, u0_ode, du_ode, config)
end

function _jacobian_ad_forward_parabolic(semi, t0, u0_ode, du_ode, config)
    new_semi = remake(semi, uEltype = eltype(config))
    # Create anonymous function passed as first argument to `ForwardDiff.jacobian` to match
    # `ForwardDiff.jacobian(f!, y::AbstractArray, x::AbstractArray, 
    #                       cfg::JacobianConfig = JacobianConfig(f!, y, x), check=Val{true}())`
    J = ForwardDiff.jacobian(du_ode, u0_ode, config) do du_ode, u_ode
        Trixi.rhs_parabolic!(du_ode, u_ode, new_semi, t0)
    end

    return J
end
end # @muladd
