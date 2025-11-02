# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
	SemidiscretizationHyperbolicSplit

A struct containing everything needed to describe a spatial semidiscretization
of a splitting rhs in time for hyperbolic conservation law.
"""
struct SemidiscretizationHyperbolicSplit{Mesh, Equations1, Equations2,
                                         InitialCondition,
                                         BoundaryConditions1,
                                         BoundaryConditions2,
                                         SourceTerms1, SourceTerms2, Solver1, Solver2,
                                         Cache1, Cache2} <:
       AbstractSemidiscretization
    mesh::Mesh

    equations1::Equations1
    equations2::Equations2

    initial_condition::InitialCondition

    boundary_conditions1::BoundaryConditions1
    boundary_conditions2::BoundaryConditions2

    source_terms1::SourceTerms1
    source_terms2::SourceTerms2

    solver1::Solver1
    solver2::Solver2

    cache1::Cache1
    cache2::Cache2

    performance_counter::PerformanceCounterList{2}

    function SemidiscretizationHyperbolicSplit{Mesh, Equations1, Equations2,
                                               InitialCondition, BoundaryConditions1,
                                               BoundaryConditions2,
                                               SourceTerms1, SourceTerms2, Solver1,
                                               Solver2, Cache1,
                                               Cache2}(mesh::Mesh,
                                                       equations1::Equations1,
                                                       equations2::Equations2,
                                                       initial_condition::InitialCondition,
                                                       boundary_conditions1::BoundaryConditions1,
                                                       boundary_conditions2::BoundaryConditions2,
                                                       source_terms1::SourceTerms1,
                                                       source_terms2::SourceTerms2,
                                                       solver1::Solver1,
                                                       solver2::Solver2,
                                                       cache1::Cache1,
                                                       cache2::Cache2) where {
                                                                              Mesh,
                                                                              Equations1,
                                                                              Equations2,
                                                                              InitialCondition,
                                                                              BoundaryConditions1,
                                                                              BoundaryConditions2,
                                                                              SourceTerms1,
                                                                              SourceTerms2,
                                                                              Solver1,
                                                                              Solver2,
                                                                              Cache1,
                                                                              Cache2}
        @assert ndims(mesh) == ndims(equations1)

        # Todo: assert nvariables(equations)==nvariables(equations_parabolic)

        performance_counter = PerformanceCounterList{2}(false)

        new(mesh, equations1, equations2, initial_condition,
            boundary_conditions1, boundary_conditions2,
            source_terms1, source_terms2, solver1, solver2, cache1, cache2,
            performance_counter)
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
function SemidiscretizationHyperbolicSplit(mesh, equations::Tuple,
                                           initial_condition, solver1, solver2;
                                           source_terms = (nothing, nothing),
                                           boundary_conditions = (boundary_condition_periodic,
                                                                  boundary_condition_periodic),
                                           # `RealT` is used as real type for node locations etc.
                                           # while `uEltype` is used as element type of solutions etc.
                                           RealT = real(solver1), uEltype = RealT,
                                           initial_caches = (NamedTuple(),
                                                             NamedTuple()))
    equations1, equations2 = equations
    boundary_conditions1, boundary_conditions2 = boundary_conditions
    initial_hyperbolic_cache1, initial_hyperbolic_cache2 = initial_caches
    source_terms1, source_terms2 = source_terms
    return SemidiscretizationHyperbolicSplit(mesh, equations1,
                                             equations2,
                                             initial_condition, solver1, solver2;
                                             source_terms1 = source_terms1,
                                             source_terms2 = source_terms2,
                                             boundary_conditions1 = boundary_conditions1,
                                             boundary_conditions2 = boundary_conditions2,
                                             RealT, uEltype,
                                             initial_cache1 = initial_hyperbolic_cache1,
                                             initial_cache2 = initial_hyperbolic_cache2)
end

function SemidiscretizationHyperbolicSplit(mesh, equations1, equations2,
                                           initial_condition, solver1, solver2;
                                           source_terms1 = nothing,
                                           source_terms2 = nothing,
                                           boundary_conditions1 = boundary_condition_periodic,
                                           boundary_conditions2 = boundary_condition_periodic,
                                           # `RealT` is used as real type for node locations etc.
                                           # while `uEltype` is used as element type of solutions etc.
                                           RealT = real(solver1), uEltype = RealT,
                                           initial_cache1 = NamedTuple(),
                                           initial_cache2 = NamedTuple())
    cache1 = (; create_cache(mesh, equations1, solver1, RealT, uEltype)...,
              initial_cache1...)
    cache2 = (; create_cache(mesh, equations2, solver2, RealT, uEltype)...,
              initial_cache2...)
    _boundary_conditions1 = digest_boundary_conditions(boundary_conditions1, mesh,
                                                       solver1,
                                                       cache1)
    _boundary_conditions2 = digest_boundary_conditions(boundary_conditions2,
                                                       mesh, solver2, cache2)

    check_periodicity_mesh_boundary_conditions(mesh, _boundary_conditions1)

    SemidiscretizationHyperbolicSplit{typeof(mesh), typeof(equations1),
                                      typeof(equations2),
                                      typeof(initial_condition),
                                      typeof(_boundary_conditions1),
                                      typeof(_boundary_conditions2),
                                      typeof(source_terms1), typeof(source_terms2),
                                      typeof(solver1),
                                      typeof(solver2), typeof(cache1),
                                      typeof(cache2)}(mesh, equations1,
                                                      equations2,
                                                      initial_condition,
                                                      _boundary_conditions1,
                                                      _boundary_conditions2,
                                                      source_terms1,
                                                      source_terms2,
                                                      solver1,
                                                      solver2,
                                                      cache1,
                                                      cache2)
end

function Base.show(io::IO, ::MIME"text/plain",
                   semi::SemidiscretizationHyperbolicSplit)
    @nospecialize semi # reduce precompilation time

    if get(io, :compact, false)
        show(io, semi)
    else
        summary_header(io, "SemidiscretizationHyperbolicSplit")
        summary_line(io, "#spatial dimensions", ndims(semi.equations1))
        summary_line(io, "mesh", semi.mesh)
        summary_line(io, "hyperbolic equations 1", semi.equations1 |> typeof |> nameof)
        summary_line(io, "hyperbolic equations 2",
                     semi.equations2 |> typeof |> nameof)
        summary_line(io, "initial condition", semi.initial_condition)

        summary_line(io, "source terms 1", semi.source_terms1)
        summary_line(io, "source terms 2", semi.source_terms2)
        summary_line(io, "solver 1", semi.solver1 |> typeof |> nameof)
        summary_line(io, "solver 2", semi.solver2 |> typeof |> nameof)
        summary_line(io, "total #DOFs per field", ndofs(semi))
        summary_footer(io)
    end
end

# retain dispatch on hyperbolic equations only
@inline function mesh_equations_solver_cache(semi::SemidiscretizationHyperbolicSplit)
    @unpack mesh, equations1, solver1, cache1 = semi
    return mesh, equations1, solver1, cache1
end

function compute_coefficients(t, semi::SemidiscretizationHyperbolicSplit)
    # Call `compute_coefficients` in `src/semidiscretization/semidiscretization.jl`
    compute_coefficients(semi.initial_condition, t, semi)
end

"""
semidiscretize(semi::SemidiscretizationHyperbolicParabolic, tspan)

Wrap the semidiscretization `semi` as a split ODE problem in the time interval `tspan`
that can be passed to `solve` from the [SciML ecosystem](https://diffeq.sciml.ai/latest/).
The parabolic right-hand side is the first function of the split ODE problem and
will be used by default by the implicit part of IMEX methods from the
SciML ecosystem.
"""
function Trixi.semidiscretize(semi::SemidiscretizationHyperbolicSplit, tspan;
                              reset_threads = true)
    # Optionally reset Polyester.jl threads. See
    # https://github.com/trixi-framework/Trixi.jl/issues/1583
    # https://github.com/JuliaSIMD/Polyester.jl/issues/30
    if reset_threads
        Trixi.Polyester.reset_threads!()
    end
    u0_ode = compute_coefficients(first(tspan), semi)
    # TODO: MPI, do we want to synchronize loading and print debug statements, e.g. using
    #       mpi_isparallel() && MPI.Barrier(mpi_comm())
    #       See https://github.com/trixi-framework/Trixi.jl/issues/328
    iip = true # is-inplace, i.e., we modify a vector when calling rhs_parabolic!, rhs!
    # Note that the IMEX time integration methods of OrdinaryDiffEq.jl treat the
    # first function implicitly and the second one explicitly. Thus, we pass the
    # stiffer parabolic function first.
    return SplitODEProblem{iip}(rhs_stiff!, rhs!, u0_ode, tspan, semi)
end

function rhs_stiff!(du_ode, u_ode, semi::SemidiscretizationHyperbolicSplit, t)
    @unpack mesh, equations1, initial_condition, boundary_conditions1, source_terms1, solver1, cache1 = semi

    u = wrap_array(u_ode, mesh, equations1, solver1, cache1)
    du = wrap_array(du_ode, mesh, equations1, solver1, cache1)

    # TODO: Taal decide, do we need to pass the mesh?
    time_start = time_ns()
    @trixi_timeit timer() "rhs! stiff" rhs!(du, u, t, mesh, equations1,
                                            boundary_conditions1, source_terms1,
                                            solver1,
                                            cache1)
    runtime = time_ns() - time_start
    put!(semi.performance_counter.counters[1], runtime)

    return nothing
end

function rhs!(du_ode, u_ode, semi::SemidiscretizationHyperbolicSplit, t)
    @unpack mesh, equations2, initial_condition, boundary_conditions2, source_terms2, solver2, cache2 = semi

    u = wrap_array(u_ode, mesh, equations2, solver2, cache2)
    du = wrap_array(du_ode, mesh, equations2, solver2, cache2)

    # TODO: Taal decide, do we need to pass the mesh?
    time_start = time_ns()
    @trixi_timeit timer() "rhs!" rhs!(du, u, t, mesh, equations2,
                                      boundary_conditions2, source_terms2,
                                      solver2,
                                      cache2)
    runtime = time_ns() - time_start
    put!(semi.performance_counter.counters[2], runtime)

    return nothing
end
# TODO: eventually to fix.
function calc_error_norms(func, u_ode, t, analyzer,
                          semi::SemidiscretizationHyperbolicSplit,
                          cache_analysis)
    @unpack mesh, equations1, initial_condition, solver1, cache1 = semi
    u = wrap_array(u_ode, mesh, equations1, solver1, cache1)

    calc_error_norms(func, u, t, analyzer, mesh, equations1, initial_condition, solver1,
                     cache1, cache_analysis)
end
end # @muladd
