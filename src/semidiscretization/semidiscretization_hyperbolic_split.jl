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
                                         SourceTerms, Solver1, Solver2,
                                         Cache1, Cache2} <:
       AbstractSemidiscretization
    mesh::Mesh

    equations1::Equations1
    equations2::Equations2

    # This guy is a bit messy since we abuse it as some kind of "exact solution"
    # although this doesn't really exist...
    initial_condition::InitialCondition

    boundary_conditions1::BoundaryConditions1
    boundary_conditions2::BoundaryConditions2

    source_terms::SourceTerms

    solver1::Solver1
    solver2::Solver2

    cache1::Cache1
    cache2::Cache2

    performance_counter::PerformanceCounterList{2}

    function SemidiscretizationHyperbolicSplit{Mesh, Equations1, Equations2,
                                               InitialCondition, BoundaryConditions1,
                                               BoundaryConditions2,
                                               SourceTerms, Solver1,
                                               Solver2, Cache1,
                                               Cache2}(mesh::Mesh,
                                                       equations1::Equations1,
                                                       equations2::Equations2,
                                                       initial_condition::InitialCondition,
                                                       boundary_conditions1::BoundaryConditions1,
                                                       boundary_conditions2::BoundaryConditions2,
                                                       source_terms::SourceTerms,
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
                                                                              SourceTerms,
                                                                              Solver1,
                                                                              Solver2,
                                                                              Cache1,
                                                                              Cache2
                                                                              }
        @assert ndims(mesh) == ndims(equations1)

        # Todo: assert nvariables(equations)==nvariables(equations_parabolic)

        performance_counter = PerformanceCounterList{2}(false)

        new(mesh, equations1, equations2, initial_condition,
            boundary_conditions1, boundary_conditions2,
            source_terms, solver1, solver2, cache1, cache2,
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
                                           source_terms = nothing,
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

    return SemidiscretizationHyperbolicSplit(mesh, equations1,
                                             equations2,
                                             initial_condition, solver1, solver2;
                                             source_terms,
                                             boundary_conditions1 = boundary_conditions1,
                                             boundary_conditions2 = boundary_conditions2,
                                             RealT, uEltype,
                                             initial_cache1 = initial_hyperbolic_cache1,
                                             initial_cache2 = initial_hyperbolic_cache2)
end

function SemidiscretizationHyperbolicSplit(mesh, equations1, equations2,
                                           initial_condition, solver1, solver2;
                                           source_terms = nothing,
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
                                      typeof(source_terms), typeof(solver1),
                                      typeof(solver2), typeof(cache1),
                                      typeof(cache2)}(mesh, equations1,
                                                      equations2,
                                                      initial_condition,
                                                      _boundary_conditions1,
                                                      _boundary_conditions2,
                                                      source_terms,
                                                      solver1,
                                                      solver2,
                                                      cache1,
                                                      cache2)
end

# Create a new semidiscretization but change some parameters compared to the input.
# `Base.similar` follows a related concept but would require us to `copy` the `mesh`,
# which would impact the performance. Instead, `SciMLBase.remake` has exactly the
# semantics we want to use here. In particular, it allows us to re-use mutable parts,
# e.g. `remake(semi).mesh === semi.mesh`.
function remake(semi::SemidiscretizationHyperbolicSplit;
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
    SemidiscretizationHyperbolicParabolic(mesh, equations, equations_parabolic,
                                          initial_condition, solver; solver_parabolic,
                                          source_terms, boundary_conditions,
                                          boundary_conditions_parabolic, uEltype)
end

function Base.show(io::IO, semi::SemidiscretizationHyperbolicSplit)
    @nospecialize semi # reduce precompilation time

    print(io, "SemidiscretizationHyperbolicSplit(")
    print(io, semi.mesh)
    print(io, ", ", semi.equations1)
    print(io, ", ", semi.equations2)
    print(io, ", ", semi.initial_condition)
    print(io, ", ", semi.boundary_conditions1)
    print(io, ", ", semi.boundary_conditions2)
    print(io, ", ", semi.source_terms)
    print(io, ", ", semi.solver1)
    print(io, ", ", semi.solver2)
    print(io, ", cache(")
    for (idx, key) in enumerate(keys(semi.cache1))
        idx > 1 && print(io, " ")
        print(io, key)
    end
    print(io, "))")
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

        # print_boundary_conditions(io, semi)

        summary_line(io, "source terms", semi.source_terms)
        summary_line(io, "solver 1", semi.solver1 |> typeof |> nameof)
        summary_line(io, "solver 2", semi.solver2 |> typeof |> nameof)
        summary_line(io, "total #DOFs per field", ndofs(semi))
        summary_footer(io)
    end
end

@inline Base.ndims(semi::SemidiscretizationHyperbolicSplit) = ndims(semi.mesh)

@inline function nvariables(semi::SemidiscretizationHyperbolicSplit)
    nvariables(semi.equations1)
end

@inline Base.real(semi::SemidiscretizationHyperbolicSplit) = real(semi.solver)

# retain dispatch on hyperbolic equations only
@inline function mesh_equations_solver_cache(semi::SemidiscretizationHyperbolicSplit)
    @unpack mesh, equations1, solver1, cache1 = semi
    return mesh, equations1, solver1, cache1
end

function compute_coefficients(t, semi::SemidiscretizationHyperbolicSplit)
    # Call `compute_coefficients` in `src/semidiscretization/semidiscretization.jl`
    compute_coefficients(semi.initial_condition, t, semi)
end

function compute_coefficients!(u_ode, t, semi::SemidiscretizationHyperbolicSplit)
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
function semidiscretize(semi::SemidiscretizationHyperbolicSplit, tspan;
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
    # Note that the IMEX time integration methods of OrdinaryDiffEq.jl treat the
    # first function implicitly and the second one explicitly. Thus, we pass the
    # stiffer parabolic function first.
    return SplitODEProblem{iip}(rhs1!, rhs2!, u0_ode, tspan, semi)
end

function rhs1!(du_ode, u_ode, semi::SemidiscretizationHyperbolicSplit, t)
    @unpack mesh, equations1, initial_condition, boundary_conditions1, source_terms, solver1, cache1 = semi

    u = wrap_array(u_ode, mesh, equations1, solver1, cache1)
    du = wrap_array(du_ode, mesh, equations1, solver1, cache1)

    # TODO: Taal decide, do we need to pass the mesh?
    time_start = time_ns()
    @trixi_timeit timer() "rhs!" rhs!(du, u, t, mesh, equations1, initial_condition,
                                      boundary_conditions1, source_terms, solver1,
                                      cache1)
    runtime = time_ns() - time_start
    put!(semi.performance_counter.counters[1], runtime)

    return nothing
end

function rhs2!(du_ode, u_ode, semi::SemidiscretizationHyperbolicSplit, t)
    @unpack mesh, equations2, initial_condition, boundary_conditions2, source_terms, solver2, cache2 = semi

    u = wrap_array(u_ode, mesh, equations2, solver2, cache2)
    du = wrap_array(du_ode, mesh, equations2, solver2, cache2)

    # TODO: Taal decide, do we need to pass the mesh?
    time_start = time_ns()
    @trixi_timeit timer() "rhs!" rhs!(du, u, t, mesh, equations2, initial_condition,
                                      boundary_conditions2, source_terms, solver2,
                                      cache2)
    runtime = time_ns() - time_start
    put!(semi.performance_counter.counters[2], runtime)

    return nothing
end
end # @muladd
