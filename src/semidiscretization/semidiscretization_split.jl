# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
	SemidiscretizationHyperbolicSplit

A struct containing everything needed to describe a spatial semidiscretization
of a split-rhs corresponding to a hyperbolic conservation/balance law.
"""
struct SemidiscretizationHyperbolicSplit{Mesh, EquationsStiff, EquationsNonStiff,
                                         InitialCondition,
                                         BoundaryConditionsStiff,
                                         BoundaryConditionsNonStiff,
                                         SourceTermsStiff, SourceTermsNonStiff,
                                         SolverStiff, SolverNonStiff,
                                         CacheStiff, CacheNonStiff} <:
       AbstractSemidiscretization
    mesh::Mesh

    equations_stiff::EquationsStiff
    equations_nonstiff::EquationsNonStiff

    initial_condition::InitialCondition

    boundary_conditions_stiff::BoundaryConditionsStiff
    boundary_conditions_nonstiff::BoundaryConditionsNonStiff

    source_terms_stiff::SourceTermsStiff
    source_terms_nonstiff::SourceTermsNonStiff

    solver_stiff::SolverStiff
    solver_nonstiff::SolverNonStiff

    cache_stiff::CacheStiff
    cache_nonstiff::CacheNonStiff

    performance_counter::PerformanceCounterList{2}
end

"""
	SemidiscretizationHyperbolicSplit(mesh, equations::Tuple, 
                                  initial_condition,
                                  solver_stiff, solver_nonstiff;
                                  source_terms=(nothing, nothing),
                                  boundary_conditions=(boundary_condition_periodic, boundary_condition_periodic),
                                  RealT=real(solver), uEltype=RealT,
                                  initial_caches=(NamedTuple(), NamedTuple()))

Construct a semidiscretization of a hyperbolic-split PDE.
"""
function SemidiscretizationHyperbolicSplit(mesh, equations::Tuple,
                                           initial_condition, solvers::Tuple;
                                           source_terms = (nothing, nothing),
                                           boundary_conditions = (boundary_condition_periodic,
                                                                  boundary_condition_periodic),
                                           # `RealT` is used as real type for node locations etc.
                                           # while `uEltype` is used as element type of solutions etc.
                                           RealT = real(first(solvers)),
                                           uEltype = RealT,
                                           initial_caches = (NamedTuple(),
                                                             NamedTuple()))
    solver_stiff, solver_nonstiff = solvers
    equations_stiff, equations_nonstiff = equations
    @assert ndims(mesh) == ndims(equations_stiff)
    @assert ndims(mesh) == ndims(equations_nonstiff)
    @assert nvariables(equations_stiff) == nvariables(equations_nonstiff)

    boundary_conditions_stiff, boundary_conditions_nonstiff = boundary_conditions
    initial_cache_stiff, initial_cache_nonstiff = initial_caches
    source_terms_stiff, source_terms_nonstiff = source_terms

    cache_stiff = (;
                   create_cache(mesh, equations_stiff, solver_stiff, RealT, uEltype)...,
                   initial_cache_stiff...)
    cache_nonstiff = (;
                      create_cache(mesh, equations_nonstiff, solver_nonstiff, RealT,
                                   uEltype)...,
                      initial_cache_nonstiff...)

    _boundary_conditions_stiff = digest_boundary_conditions(boundary_conditions_stiff,
                                                            mesh,
                                                            solver_stiff,
                                                            cache_stiff)
    _boundary_conditions_nonstiff = digest_boundary_conditions(boundary_conditions_nonstiff,
                                                               mesh, solver_nonstiff,
                                                               cache_nonstiff)

    check_periodicity_mesh_boundary_conditions(mesh, _boundary_conditions_stiff)

    performance_counter = PerformanceCounterList{2}(false)

    SemidiscretizationHyperbolicSplit{typeof(mesh),
                                      typeof(equations_stiff),
                                      typeof(equations_nonstiff),
                                      typeof(initial_condition),
                                      typeof(_boundary_conditions_stiff),
                                      typeof(_boundary_conditions_nonstiff),
                                      typeof(source_terms_stiff),
                                      typeof(source_terms_nonstiff),
                                      typeof(solver_stiff), typeof(solver_nonstiff),
                                      typeof(cache_stiff),
                                      typeof(cache_nonstiff)}(mesh, equations_stiff,
                                                              equations_nonstiff,
                                                              initial_condition,
                                                              _boundary_conditions_stiff,
                                                              _boundary_conditions_nonstiff,
                                                              source_terms_stiff,
                                                              source_terms_nonstiff,
                                                              solver_stiff,
                                                              solver_nonstiff,
                                                              cache_stiff,
                                                              cache_nonstiff,
                                                              performance_counter)
end

function Base.show(io::IO, ::MIME"text/plain",
                   semi::SemidiscretizationHyperbolicSplit)
    @nospecialize semi # reduce precompilation time

    if get(io, :compact, false)
        show(io, semi)
    else
        summary_header(io, "SemidiscretizationHyperbolicSplit")
        summary_line(io, "#spatial dimensions", ndims(semi.equations_stiff))
        summary_line(io, "mesh", semi.mesh)
        summary_line(io, "hyperbolic equations stiff",
                     semi.equations_stiff |> typeof |> nameof)
        summary_line(io, "hyperbolic equations nonstiff",
                     semi.equations_nonstiff |> typeof |> nameof)
        summary_line(io, "initial condition", semi.initial_condition)

        summary_line(io, "source terms stiff", semi.source_terms_stiff)
        summary_line(io, "source terms nonstiff", semi.source_terms_nonstiff)
        summary_line(io, "solver stiff", semi.solver_stiff |> typeof |> nameof)
        summary_line(io, "solver nonstiff", semi.solver_nonstiff |> typeof |> nameof)
        summary_line(io, "total #DOFs per field", ndofs(semi))
        summary_footer(io)
    end
end

@inline Base.ndims(semi::SemidiscretizationHyperbolicSplit) = ndims(semi.mesh)

# retain dispatch on hyperbolic non-stiff equations only
@inline function mesh_equations_solver_cache(semi::SemidiscretizationHyperbolicSplit)
    @unpack mesh, equations_nonstiff, solver_nonstiff, cache_nonstiff = semi
    return mesh, equations_nonstiff, solver_nonstiff, cache_nonstiff
end

function compute_coefficients(t, semi::SemidiscretizationHyperbolicSplit)
    # Call `compute_coefficients` in `src/semidiscretization/semidiscretization.jl`
    compute_coefficients(semi.initial_condition, t, semi)
end

"""
    semidiscretize(semi::SemidiscretizationHyperbolicSplit, tspan)

Wrap the semidiscretization `semi` as a split ODE problem in the time interval `tspan`
that can be passed to `solve` from the [SciML ecosystem](https://diffeq.sciml.ai/latest/).
The stiff hyperbolic right-hand side is the first function of the split ODE problem and
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
    # stiffer function first.
    return SplitODEProblem{iip}(rhs_stiff!, rhs!, u0_ode, tspan, semi)
end

function rhs_stiff!(du_ode, u_ode, semi::SemidiscretizationHyperbolicSplit, t)
    @unpack mesh, equations_stiff, initial_condition, boundary_conditions_stiff, source_terms_stiff, solver_stiff, cache_stiff = semi

    u = wrap_array(u_ode, mesh, equations_stiff, solver_stiff, cache_stiff)
    du = wrap_array(du_ode, mesh, equations_stiff, solver_stiff, cache_stiff)

    # TODO: Taal decide, do we need to pass the mesh?
    time_start = time_ns()
    @trixi_timeit timer() "rhs! stiff" rhs!(du, u, t, mesh, equations_stiff,
                                            boundary_conditions_stiff,
                                            source_terms_stiff,
                                            solver_stiff,
                                            cache_stiff)
    runtime = time_ns() - time_start
    put!(semi.performance_counter.counters[1], runtime)

    return nothing
end

# nonstiff `rhs!`
function rhs!(du_ode, u_ode, semi::SemidiscretizationHyperbolicSplit, t)
    @unpack mesh, equations_nonstiff, initial_condition, boundary_conditions_nonstiff, source_terms_nonstiff, solver_nonstiff, cache_nonstiff = semi

    u = wrap_array(u_ode, mesh, equations_nonstiff, solver_nonstiff, cache_nonstiff)
    du = wrap_array(du_ode, mesh, equations_nonstiff, solver_nonstiff, cache_nonstiff)

    # TODO: Taal decide, do we need to pass the mesh?
    time_start = time_ns()
    @trixi_timeit timer() "rhs! nonstiff" rhs!(du, u, t, mesh, equations_nonstiff,
                                               boundary_conditions_nonstiff,
                                               source_terms_nonstiff,
                                               solver_nonstiff,
                                               cache_nonstiff)
    runtime = time_ns() - time_start
    put!(semi.performance_counter.counters[2], runtime)

    return nothing
end

# Here we only pass the nonstiff solver and cache to the calc_error_norms function,
# since they are needed only for auxiliary functions, such as get_node_vars, etc.
function calc_error_norms(func, u_ode, t, analyzer,
                          semi::SemidiscretizationHyperbolicSplit,
                          cache_analysis)
    @unpack mesh, equations_nonstiff, initial_condition, solver_nonstiff, cache_nonstiff = semi
    u = wrap_array(u_ode, mesh, equations_nonstiff, solver_nonstiff, cache_nonstiff)

    calc_error_norms(func, u, t, analyzer, mesh, equations_nonstiff, initial_condition,
                     solver_nonstiff,
                     cache_nonstiff, cache_analysis)
end
end # @muladd
