# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    SemidiscretizationParabolic

A struct containing everything needed to describe a spatial semidiscretization
of a purely parabolic conservation law.
"""
mutable struct SemidiscretizationParabolic{Mesh, EquationsParabolic, InitialCondition,
                                           BoundaryConditions, SourceTermsParabolic,
                                           Solver, SolverParabolic,
                                           Cache, CacheParabolic} <:
               AbstractSemidiscretization
    mesh::Mesh
    equations::EquationsParabolic

    # This guy is a bit messy since we abuse it as some kind of "exact solution"
    # although this doesn't really exist...
    const initial_condition::InitialCondition

    const boundary_conditions::BoundaryConditions
    const source_terms::SourceTermsParabolic

    const solver::Solver
    const solver_parabolic::SolverParabolic

    cache::Cache
    cache_parabolic::CacheParabolic

    performance_counter::PerformanceCounter
end
# We assume some properties of the fields of the semidiscretization, e.g.,
# the `equations` and the `mesh` should have the same dimension. We check these
# properties in the outer constructor defined below. While we could ensure
# them even better in an inner constructor, we do not use this approach to
# simplify the integration with Adapt.jl for GPU usage, see
# https://github.com/trixi-framework/Trixi.jl/pull/2677#issuecomment-3591789921

"""
    SemidiscretizationParabolic(mesh, equations, initial_condition, solver;
                                solver_parabolic=default_parabolic_solver(),
                                source_terms=nothing,
                                boundary_conditions,
                                RealT=real(solver),
                                uEltype=RealT)

Construct a semidiscretization of a purely parabolic PDE.

Boundary conditions must be provided explicitly either as a `NamedTuple` or as a
single boundary condition that gets applied to all boundaries.
"""
function SemidiscretizationParabolic(mesh, equations::AbstractEquationsParabolic,
                                     initial_condition, solver;
                                     solver_parabolic = default_parabolic_solver(),
                                     source_terms = nothing,
                                     boundary_conditions,
                                     # `RealT` is used as real type for node locations etc.
                                     # while `uEltype` is used as element type of solutions etc.
                                     RealT = real(solver), uEltype = RealT)
    @assert ndims(mesh) == ndims(equations)

    cache = create_cache(mesh, equations, solver, RealT, uEltype)
    _boundary_conditions = digest_boundary_conditions(boundary_conditions, mesh, solver,
                                                      cache)
    check_periodicity_mesh_boundary_conditions(mesh, _boundary_conditions)

    cache_parabolic = create_cache_parabolic(mesh, equations, solver,
                                             nelements(solver, cache), uEltype)

    performance_counter = PerformanceCounter()

    return SemidiscretizationParabolic{typeof(mesh), typeof(equations),
                                       typeof(initial_condition),
                                       typeof(_boundary_conditions),
                                       typeof(source_terms),
                                       typeof(solver), typeof(solver_parabolic),
                                       typeof(cache), typeof(cache_parabolic)}(mesh,
                                                                               equations,
                                                                               initial_condition,
                                                                               _boundary_conditions,
                                                                               source_terms,
                                                                               solver,
                                                                               solver_parabolic,
                                                                               cache,
                                                                               cache_parabolic,
                                                                               performance_counter)
end

# @eval due to @muladd
@eval Adapt.@adapt_structure(SemidiscretizationParabolic)

# Create a new semidiscretization but change some parameters compared to the input.
# `Base.similar` follows a related concept but would require us to `copy` the `mesh`,
# which would impact the performance. Instead, `SciMLBase.remake` has exactly the
# semantics we want to use here. In particular, it allows us to re-use mutable parts,
# e.g. `remake(semi).mesh === semi.mesh`.
function remake(semi::SemidiscretizationParabolic; uEltype = real(semi.solver),
                mesh = semi.mesh,
                equations = semi.equations,
                initial_condition = semi.initial_condition,
                solver = semi.solver,
                solver_parabolic = semi.solver_parabolic,
                source_terms = semi.source_terms,
                boundary_conditions = semi.boundary_conditions)
    return SemidiscretizationParabolic(mesh, equations, initial_condition, solver;
                                       solver_parabolic = solver_parabolic,
                                       source_terms = source_terms,
                                       boundary_conditions = boundary_conditions,
                                       uEltype = uEltype)
end

function Base.show(io::IO, semi::SemidiscretizationParabolic)
    @nospecialize semi # reduce precompilation time

    print(io, "SemidiscretizationParabolic(")
    print(io, semi.mesh)
    print(io, ", ", semi.equations)
    print(io, ", ", semi.initial_condition)
    print(io, ", ", semi.boundary_conditions)
    print(io, ", ", semi.source_terms)
    print(io, ", ", semi.solver)
    print(io, ", ", semi.solver_parabolic)
    print(io, ", cache(")
    for (idx, key) in enumerate(keys(semi.cache))
        idx > 1 && print(io, " ")
        print(io, key)
    end
    print(io, "))")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", semi::SemidiscretizationParabolic)
    @nospecialize semi # reduce precompilation time

    if get(io, :compact, false)
        show(io, semi)
    else
        summary_header(io, "SemidiscretizationParabolic")
        summary_line(io, "#spatial dimensions", ndims(semi.equations))
        summary_line(io, "mesh", semi.mesh)
        summary_line(io, "parabolic equations", semi.equations |> typeof |> nameof)
        summary_line(io, "initial condition", semi.initial_condition)
        summary_line(io, "source terms parabolic", semi.source_terms)
        summary_line(io, "solver", semi.solver |> typeof |> nameof)
        summary_line(io, "parabolic solver", semi.solver_parabolic |> typeof |> nameof)
        summary_line(io, "total #DOFs per field", ndofsglobal(semi))
        summary_footer(io)
    end
end

@inline Base.ndims(semi::SemidiscretizationParabolic) = ndims(semi.mesh)

@inline function nvariables(semi::SemidiscretizationParabolic)
    return nvariables(semi.equations)
end

@inline Base.real(semi::SemidiscretizationParabolic) = real(semi.solver)

@inline function mesh_equations_solver_cache(semi::SemidiscretizationParabolic)
    @unpack mesh, equations, solver, cache = semi
    return mesh, equations, solver, cache
end

function calc_error_norms(func, u_ode, t, analyzer,
                          semi::SemidiscretizationParabolic, cache_analysis)
    @unpack mesh, equations, initial_condition, solver, cache = semi
    u = wrap_array(u_ode, mesh, equations, solver, cache)

    return calc_error_norms(func, u, t, analyzer, mesh, equations, initial_condition,
                            solver, cache, cache_analysis)
end

function compute_coefficients(t, semi::SemidiscretizationParabolic)
    # Call `compute_coefficients` in `src/semidiscretization/semidiscretization.jl`
    return compute_coefficients(semi.initial_condition, t, semi)
end

function compute_coefficients!(u_ode, t, semi::SemidiscretizationParabolic)
    return compute_coefficients!(u_ode, semi.initial_condition, t, semi)
end

function rhs!(du_ode, u_ode, semi::SemidiscretizationParabolic, t)
    @unpack mesh, equations, boundary_conditions, source_terms, solver, solver_parabolic,
    cache, cache_parabolic = semi

    u = wrap_array(u_ode, mesh, equations, solver, cache)
    du = wrap_array(du_ode, mesh, equations, solver, cache)

    time_start = time_ns()
    @trixi_timeit timer() "rhs!" rhs_parabolic!(du, u, t, mesh, equations,
                                                boundary_conditions, source_terms,
                                                solver, solver_parabolic, cache,
                                                cache_parabolic)
    runtime = time_ns() - time_start
    put!(semi.performance_counter, runtime)

    return nothing
end

end # @muladd
