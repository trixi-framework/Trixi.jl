# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    SemidiscretizationHyperbolic

A struct containing everything needed to describe a spatial semidiscretization
of a hyperbolic conservation law.
"""
mutable struct SemidiscretizationHyperbolic{Mesh, Equations, InitialCondition,
                                            BoundaryConditions,
                                            SourceTerms, Solver, Cache} <:
               AbstractSemidiscretization
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

    function SemidiscretizationHyperbolic{Mesh, Equations, InitialCondition,
                                          BoundaryConditions, SourceTerms, Solver,
                                          Cache}(mesh::Mesh, equations::Equations,
                                                 initial_condition::InitialCondition,
                                                 boundary_conditions::BoundaryConditions,
                                                 source_terms::SourceTerms,
                                                 solver::Solver,
                                                 cache::Cache) where {Mesh, Equations,
                                                                      InitialCondition,
                                                                      BoundaryConditions,
                                                                      SourceTerms,
                                                                      Solver,
                                                                      Cache}
        performance_counter = PerformanceCounter()

        new(mesh, equations, initial_condition, boundary_conditions, source_terms,
            solver, cache, performance_counter)
    end
end

"""
    SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                 source_terms=nothing,
                                 boundary_conditions=boundary_condition_periodic,
                                 RealT=real(solver),
                                 uEltype=RealT,
                                 initial_cache=NamedTuple())

Construct a semidiscretization of a hyperbolic PDE.
"""
function SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                      source_terms = nothing,
                                      boundary_conditions = boundary_condition_periodic,
                                      # `RealT` is used as real type for node locations etc.
                                      # while `uEltype` is used as element type of solutions etc.
                                      RealT = real(solver), uEltype = RealT,
                                      initial_cache = NamedTuple())
    @assert ndims(mesh) == ndims(equations)

    cache = (; create_cache(mesh, equations, solver, RealT, uEltype)...,
             initial_cache...)
    _boundary_conditions = digest_boundary_conditions(boundary_conditions, mesh, solver,
                                                      cache)

    check_periodicity_mesh_boundary_conditions(mesh, _boundary_conditions)

    SemidiscretizationHyperbolic{typeof(mesh), typeof(equations),
                                 typeof(initial_condition),
                                 typeof(_boundary_conditions), typeof(source_terms),
                                 typeof(solver), typeof(cache)}(mesh, equations,
                                                                initial_condition,
                                                                _boundary_conditions,
                                                                source_terms, solver,
                                                                cache)
end

# Create a new semidiscretization but change some parameters compared to the input.
# `Base.similar` follows a related concept but would require us to `copy` the `mesh`,
# which would impact the performance. Instead, `SciMLBase.remake` has exactly the
# semantics we want to use here. In particular, it allows us to re-use mutable parts,
# e.g. `remake(semi).mesh === semi.mesh`.
function remake(semi::SemidiscretizationHyperbolic; uEltype = real(semi.solver),
                mesh = semi.mesh,
                equations = semi.equations,
                initial_condition = semi.initial_condition,
                solver = semi.solver,
                source_terms = semi.source_terms,
                boundary_conditions = semi.boundary_conditions)
    # TODO: Which parts do we want to `remake`? At least the solver needs some
    #       special care if shock-capturing volume integrals are used (because of
    #       the indicators and their own caches...).
    SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                 source_terms, boundary_conditions, uEltype)
end

# general fallback
function digest_boundary_conditions(boundary_conditions, mesh, solver, cache)
    boundary_conditions
end

# general fallback
function digest_boundary_conditions(boundary_conditions::BoundaryConditionPeriodic,
                                    mesh, solver, cache)
    boundary_conditions
end

# resolve ambiguities with definitions below
function digest_boundary_conditions(boundary_conditions::BoundaryConditionPeriodic,
                                    mesh::Union{TreeMesh{1}, StructuredMesh{1}}, solver,
                                    cache)
    boundary_conditions
end

function digest_boundary_conditions(boundary_conditions::BoundaryConditionPeriodic,
                                    mesh::Union{TreeMesh{2}, StructuredMesh{2}}, solver,
                                    cache)
    boundary_conditions
end

function digest_boundary_conditions(boundary_conditions::BoundaryConditionPeriodic,
                                    mesh::Union{TreeMesh{3}, StructuredMesh{3}}, solver,
                                    cache)
    boundary_conditions
end

# allow passing a single BC that get converted into a tuple of BCs
# on (mapped) hypercube domains
function digest_boundary_conditions(boundary_conditions,
                                    mesh::Union{TreeMesh{1}, StructuredMesh{1}}, solver,
                                    cache)
    (; x_neg = boundary_conditions, x_pos = boundary_conditions)
end

function digest_boundary_conditions(boundary_conditions,
                                    mesh::Union{TreeMesh{2}, StructuredMesh{2}}, solver,
                                    cache)
    (; x_neg = boundary_conditions, x_pos = boundary_conditions,
     y_neg = boundary_conditions, y_pos = boundary_conditions)
end

function digest_boundary_conditions(boundary_conditions,
                                    mesh::Union{TreeMesh{3}, StructuredMesh{3}}, solver,
                                    cache)
    (; x_neg = boundary_conditions, x_pos = boundary_conditions,
     y_neg = boundary_conditions, y_pos = boundary_conditions,
     z_neg = boundary_conditions, z_pos = boundary_conditions)
end

# allow passing a tuple of BCs that get converted into a named tuple to make it
# self-documenting on (mapped) hypercube domains
function digest_boundary_conditions(boundary_conditions::NTuple{2, Any},
                                    mesh::Union{TreeMesh{1}, StructuredMesh{1}}, solver,
                                    cache)
    (; x_neg = boundary_conditions[1], x_pos = boundary_conditions[2])
end

function digest_boundary_conditions(boundary_conditions::NTuple{4, Any},
                                    mesh::Union{TreeMesh{2}, StructuredMesh{2}}, solver,
                                    cache)
    (; x_neg = boundary_conditions[1], x_pos = boundary_conditions[2],
     y_neg = boundary_conditions[3], y_pos = boundary_conditions[4])
end

function digest_boundary_conditions(boundary_conditions::NTuple{6, Any},
                                    mesh::Union{TreeMesh{3}, StructuredMesh{3}}, solver,
                                    cache)
    (; x_neg = boundary_conditions[1], x_pos = boundary_conditions[2],
     y_neg = boundary_conditions[3], y_pos = boundary_conditions[4],
     z_neg = boundary_conditions[5], z_pos = boundary_conditions[6])
end

# allow passing named tuples of BCs constructed in an arbitrary order
# on (mapped) hypercube domains
function digest_boundary_conditions(boundary_conditions::NamedTuple{Keys, ValueTypes},
                                    mesh::Union{TreeMesh{1}, StructuredMesh{1}}, solver,
                                    cache) where {Keys, ValueTypes <: NTuple{2, Any}}
    @unpack x_neg, x_pos = boundary_conditions
    (; x_neg, x_pos)
end

function digest_boundary_conditions(boundary_conditions::NamedTuple{Keys, ValueTypes},
                                    mesh::Union{TreeMesh{2}, StructuredMesh{2}}, solver,
                                    cache) where {Keys, ValueTypes <: NTuple{4, Any}}
    @unpack x_neg, x_pos, y_neg, y_pos = boundary_conditions
    (; x_neg, x_pos, y_neg, y_pos)
end

function digest_boundary_conditions(boundary_conditions::NamedTuple{Keys, ValueTypes},
                                    mesh::Union{TreeMesh{3}, StructuredMesh{3}}, solver,
                                    cache) where {Keys, ValueTypes <: NTuple{6, Any}}
    @unpack x_neg, x_pos, y_neg, y_pos, z_neg, z_pos = boundary_conditions
    (; x_neg, x_pos, y_neg, y_pos, z_neg, z_pos)
end

# sort the boundary conditions from a dictionary and into tuples
function digest_boundary_conditions(boundary_conditions::Dict, mesh, solver, cache)
    UnstructuredSortedBoundaryTypes(boundary_conditions, cache)
end

function digest_boundary_conditions(boundary_conditions::AbstractArray, mesh, solver,
                                    cache)
    throw(ArgumentError("Please use a (named) tuple instead of an (abstract) array to supply multiple boundary conditions (to improve performance)."))
end

# No checks for these meshes yet available
function check_periodicity_mesh_boundary_conditions(mesh::Union{P4estMesh,
                                                                UnstructuredMesh2D,
                                                                T8codeMesh,
                                                                DGMultiMesh},
                                                    boundary_conditions)
end

# No actions needed for periodic boundary conditions
function check_periodicity_mesh_boundary_conditions(mesh::Union{TreeMesh,
                                                                StructuredMesh},
                                                    boundary_conditions::BoundaryConditionPeriodic)
end

function check_periodicity_mesh_boundary_conditions_x(mesh, x_neg, x_pos)
    if isperiodic(mesh, 1) &&
       (x_neg != BoundaryConditionPeriodic() ||
        x_pos != BoundaryConditionPeriodic())
        @error "For periodic mesh non-periodic boundary conditions in x-direction are supplied."
    end
end

function check_periodicity_mesh_boundary_conditions_y(mesh, y_neg, y_pos)
    if isperiodic(mesh, 2) &&
       (y_neg != BoundaryConditionPeriodic() ||
        y_pos != BoundaryConditionPeriodic())
        @error "For periodic mesh non-periodic boundary conditions in y-direction are supplied."
    end
end

function check_periodicity_mesh_boundary_conditions_z(mesh, z_neg, z_pos)
    if isperiodic(mesh, 3) &&
       (z_neg != BoundaryConditionPeriodic() ||
        z_pos != BoundaryConditionPeriodic())
        @error "For periodic mesh non-periodic boundary conditions in z-direction are supplied."
    end
end

function check_periodicity_mesh_boundary_conditions(mesh::Union{TreeMesh{1},
                                                                StructuredMesh{1}},
                                                    boundary_conditions::Union{NamedTuple,
                                                                               Tuple})
    check_periodicity_mesh_boundary_conditions_x(mesh, boundary_conditions[1],
                                                 boundary_conditions[2])
end

function check_periodicity_mesh_boundary_conditions(mesh::Union{TreeMesh{2},
                                                                StructuredMesh{2},
                                                                StructuredMeshView{2}},
                                                    boundary_conditions::Union{NamedTuple,
                                                                               Tuple})
    check_periodicity_mesh_boundary_conditions_x(mesh, boundary_conditions[1],
                                                 boundary_conditions[2])
    check_periodicity_mesh_boundary_conditions_y(mesh, boundary_conditions[3],
                                                 boundary_conditions[4])
end

function check_periodicity_mesh_boundary_conditions(mesh::Union{TreeMesh{3},
                                                                StructuredMesh{3}},
                                                    boundary_conditions::Union{NamedTuple,
                                                                               Tuple})
    check_periodicity_mesh_boundary_conditions_x(mesh, boundary_conditions[1],
                                                 boundary_conditions[2])
    check_periodicity_mesh_boundary_conditions_y(mesh, boundary_conditions[3],
                                                 boundary_conditions[4])
    check_periodicity_mesh_boundary_conditions_z(mesh, boundary_conditions[5],
                                                 boundary_conditions[6])
end

function Base.show(io::IO, semi::SemidiscretizationHyperbolic)
    @nospecialize semi # reduce precompilation time

    print(io, "SemidiscretizationHyperbolic(")
    print(io, semi.mesh)
    print(io, ", ", semi.equations)
    print(io, ", ", semi.initial_condition)
    print(io, ", ", semi.boundary_conditions)
    print(io, ", ", semi.source_terms)
    print(io, ", ", semi.solver)
    print(io, ", cache(")
    for (idx, key) in enumerate(keys(semi.cache))
        idx > 1 && print(io, " ")
        print(io, key)
    end
    print(io, "))")
end

function Base.show(io::IO, ::MIME"text/plain", semi::SemidiscretizationHyperbolic)
    @nospecialize semi # reduce precompilation time

    if get(io, :compact, false)
        show(io, semi)
    else
        summary_header(io, "SemidiscretizationHyperbolic")
        summary_line(io, "#spatial dimensions", ndims(semi.equations))
        summary_line(io, "mesh", semi.mesh)
        summary_line(io, "equations", semi.equations |> typeof |> nameof)
        summary_line(io, "initial condition", semi.initial_condition)

        print_boundary_conditions(io, semi)

        summary_line(io, "source terms", semi.source_terms)
        summary_line(io, "solver", semi.solver |> typeof |> nameof)
        summary_line(io, "total #DOFs per field", ndofsglobal(semi))
        summary_footer(io)
    end
end

# type alias for dispatch in printing of boundary conditions
#! format: off
const SemiHypMeshBCSolver{Mesh, BoundaryConditions, Solver} =
        SemidiscretizationHyperbolic{Mesh,
                                     Equations,
                                     InitialCondition,
                                     BoundaryConditions,
                                     SourceTerms,
                                     Solver} where {Equations,
                                                    InitialCondition,
                                                    SourceTerms}
#! format: on

# generic fallback: print the type of semi.boundary_condition.
function print_boundary_conditions(io, semi::SemiHypMeshBCSolver)
    summary_line(io, "boundary conditions", typeof(semi.boundary_conditions))
end

function print_boundary_conditions(io,
                                   semi::SemiHypMeshBCSolver{<:Any,
                                                             <:UnstructuredSortedBoundaryTypes})
    @unpack boundary_conditions = semi
    @unpack boundary_dictionary = boundary_conditions
    summary_line(io, "boundary conditions", length(boundary_dictionary))
    for (boundary_name, boundary_condition) in boundary_dictionary
        summary_line(increment_indent(io), boundary_name, typeof(boundary_condition))
    end
end

function print_boundary_conditions(io, semi::SemiHypMeshBCSolver{<:Any, <:NamedTuple})
    @unpack boundary_conditions = semi
    summary_line(io, "boundary conditions", length(boundary_conditions))
    bc_names = keys(boundary_conditions)
    for (i, bc_name) in enumerate(bc_names)
        summary_line(increment_indent(io), String(bc_name),
                     typeof(boundary_conditions[i]))
    end
end

function print_boundary_conditions(io,
                                   semi::SemiHypMeshBCSolver{<:Union{TreeMesh,
                                                                     StructuredMesh},
                                                             <:Union{Tuple, NamedTuple,
                                                                     AbstractArray}})
    summary_line(io, "boundary conditions", 2 * ndims(semi))
    bcs = semi.boundary_conditions

    summary_line(increment_indent(io), "negative x", bcs[1])
    summary_line(increment_indent(io), "positive x", bcs[2])
    if ndims(semi) > 1
        summary_line(increment_indent(io), "negative y", bcs[3])
        summary_line(increment_indent(io), "positive y", bcs[4])
    end
    if ndims(semi) > 2
        summary_line(increment_indent(io), "negative z", bcs[5])
        summary_line(increment_indent(io), "positive z", bcs[6])
    end
end

@inline Base.ndims(semi::SemidiscretizationHyperbolic) = ndims(semi.mesh)

@inline nvariables(semi::SemidiscretizationHyperbolic) = nvariables(semi.equations)

@inline Base.real(semi::SemidiscretizationHyperbolic) = real(semi.solver)

@inline function mesh_equations_solver_cache(semi::SemidiscretizationHyperbolic)
    @unpack mesh, equations, solver, cache = semi
    return mesh, equations, solver, cache
end

function calc_error_norms(func, u_ode, t, analyzer, semi::SemidiscretizationHyperbolic,
                          cache_analysis)
    @unpack mesh, equations, initial_condition, solver, cache = semi
    u = wrap_array(u_ode, mesh, equations, solver, cache)

    calc_error_norms(func, u, t, analyzer, mesh, equations, initial_condition, solver,
                     cache, cache_analysis)
end

function compute_coefficients(t, semi::SemidiscretizationHyperbolic)
    # Call `compute_coefficients` in `src/semidiscretization/semidiscretization.jl`
    compute_coefficients(semi.initial_condition, t, semi)
end

function compute_coefficients!(u_ode, t, semi::SemidiscretizationHyperbolic)
    compute_coefficients!(u_ode, semi.initial_condition, t, semi)
end

function rhs!(du_ode, u_ode, semi::SemidiscretizationHyperbolic, t)
    @unpack mesh, equations, boundary_conditions, source_terms, solver, cache = semi

    u = wrap_array(u_ode, mesh, equations, solver, cache)
    du = wrap_array(du_ode, mesh, equations, solver, cache)

    # TODO: Taal decide, do we need to pass the mesh?
    time_start = time_ns()
    @trixi_timeit timer() "rhs!" rhs!(du, u, t, mesh, equations,
                                      boundary_conditions, source_terms, solver, cache)
    runtime = time_ns() - time_start
    put!(semi.performance_counter, runtime)

    return nothing
end
end # @muladd
