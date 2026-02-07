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
    const initial_condition::InitialCondition

    const boundary_conditions::BoundaryConditions
    const source_terms::SourceTerms
    const solver::Solver
    cache::Cache
    performance_counter::PerformanceCounter
end
# We assume some properties of the fields of the semidiscretization, e.g.,
# the `equations` and the `mesh` should have the same dimension. We check these
# properties in the outer constructor defined below. While we could ensure
# them even better in an inner constructor, we do not use this approach to
# simplify the integration with Adapt.jl for GPU usage, see
# https://github.com/trixi-framework/Trixi.jl/pull/2677#issuecomment-3591789921

"""
    SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                 source_terms=nothing,
                                 boundary_conditions,
                                 RealT=real(solver),
                                 uEltype=RealT)

Construct a semidiscretization of a hyperbolic PDE.

Boundary conditions must be provided explicitly either as a `NamedTuple` or as a
single boundary condition that gets applied to all boundaries.
"""
function SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
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

    performance_counter = PerformanceCounter()

    return SemidiscretizationHyperbolic{typeof(mesh), typeof(equations),
                                        typeof(initial_condition),
                                        typeof(_boundary_conditions),
                                        typeof(source_terms),
                                        typeof(solver), typeof(cache)}(mesh, equations,
                                                                       initial_condition,
                                                                       _boundary_conditions,
                                                                       source_terms,
                                                                       solver, cache,
                                                                       performance_counter)
end

# @eval due to @muladd
@eval Adapt.@adapt_structure(SemidiscretizationHyperbolic)

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
    return SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                        source_terms, boundary_conditions, uEltype)
end

# general fallback
function digest_boundary_conditions(boundary_conditions, mesh, solver, cache)
    return boundary_conditions
end

# general fallback
function digest_boundary_conditions(boundary_conditions::BoundaryConditionPeriodic,
                                    mesh, solver, cache)
    return boundary_conditions
end

# resolve ambiguities with definitions below
function digest_boundary_conditions(boundary_conditions::BoundaryConditionPeriodic,
                                    mesh::Union{TreeMesh{1}, StructuredMesh{1}}, solver,
                                    cache)
    return boundary_conditions
end

function digest_boundary_conditions(boundary_conditions::BoundaryConditionPeriodic,
                                    mesh::Union{TreeMesh{2}, StructuredMesh{2}}, solver,
                                    cache)
    return boundary_conditions
end

function digest_boundary_conditions(boundary_conditions::BoundaryConditionPeriodic,
                                    mesh::Union{TreeMesh{3}, StructuredMesh{3}}, solver,
                                    cache)
    return boundary_conditions
end

# allow passing a single BC that get converted into a tuple of BCs
# on (mapped) hypercube domains
function digest_boundary_conditions(boundary_conditions,
                                    mesh::Union{TreeMesh{1}, StructuredMesh{1}}, solver,
                                    cache)
    return (; x_neg = boundary_conditions, x_pos = boundary_conditions)
end

function digest_boundary_conditions(boundary_conditions,
                                    mesh::Union{TreeMesh{2}, StructuredMesh{2}}, solver,
                                    cache)
    return (; x_neg = boundary_conditions, x_pos = boundary_conditions,
            y_neg = boundary_conditions, y_pos = boundary_conditions)
end

function digest_boundary_conditions(boundary_conditions,
                                    mesh::Union{TreeMesh{3}, StructuredMesh{3}}, solver,
                                    cache)
    return (; x_neg = boundary_conditions, x_pos = boundary_conditions,
            y_neg = boundary_conditions, y_pos = boundary_conditions,
            z_neg = boundary_conditions, z_pos = boundary_conditions)
end

# allow passing named tuples of BCs constructed in an arbitrary order
# on (mapped) hypercube domains
function digest_boundary_conditions(boundary_conditions::NamedTuple{Keys, ValueTypes},
                                    mesh::Union{TreeMesh{1}, StructuredMesh{1}}, solver,
                                    cache) where {Keys, ValueTypes <: NTuple{2, Any}}
    @unpack x_neg, x_pos = boundary_conditions
    return (; x_neg, x_pos)
end

function digest_boundary_conditions(boundary_conditions::NamedTuple{Keys, ValueTypes},
                                    mesh::Union{TreeMesh{2}, StructuredMesh{2}}, solver,
                                    cache) where {Keys, ValueTypes <: NTuple{4, Any}}
    @unpack x_neg, x_pos, y_neg, y_pos = boundary_conditions
    return (; x_neg, x_pos, y_neg, y_pos)
end

function digest_boundary_conditions(boundary_conditions::NamedTuple{Keys, ValueTypes},
                                    mesh::Union{TreeMesh{3}, StructuredMesh{3}}, solver,
                                    cache) where {Keys, ValueTypes <: NTuple{6, Any}}
    @unpack x_neg, x_pos, y_neg, y_pos, z_neg, z_pos = boundary_conditions
    return (; x_neg, x_pos, y_neg, y_pos, z_neg, z_pos)
end

# If a NamedTuple is passed with not the same number of BCs, ensure that the keys are correct.
# For periodic boundary parts, the keys can be missing and get filled with `boundary_condition_periodic`.
function digest_boundary_conditions(boundary_conditions::NamedTuple,
                                    mesh::Union{TreeMesh{1}, StructuredMesh{1}}, solver,
                                    cache)
    x_neg, x_pos = get_periodicity_boundary_conditions_x(boundary_conditions, mesh)
    return (; x_neg, x_pos)
end

function digest_boundary_conditions(boundary_conditions::NamedTuple,
                                    mesh::Union{TreeMesh{2}, StructuredMesh{2}}, solver,
                                    cache)
    x_neg, x_pos = get_periodicity_boundary_conditions_x(boundary_conditions, mesh)
    y_neg, y_pos = get_periodicity_boundary_conditions_y(boundary_conditions, mesh)
    return (; x_neg, x_pos, y_neg, y_pos)
end

function digest_boundary_conditions(boundary_conditions::NamedTuple,
                                    mesh::Union{TreeMesh{3}, StructuredMesh{3}}, solver,
                                    cache)
    x_neg, x_pos = get_periodicity_boundary_conditions_x(boundary_conditions, mesh)
    y_neg, y_pos = get_periodicity_boundary_conditions_y(boundary_conditions, mesh)
    z_neg, z_pos = get_periodicity_boundary_conditions_z(boundary_conditions, mesh)
    return (; x_neg, x_pos, y_neg, y_pos, z_neg, z_pos)
end

# Allow NamedTuple for P4estMesh, UnstructuredMesh2D, and T8codeMesh
function digest_boundary_conditions(boundary_conditions::NamedTuple,
                                    mesh::Union{P4estMesh, UnstructuredMesh2D,
                                                T8codeMesh},
                                    solver, cache)
    return UnstructuredSortedBoundaryTypes(boundary_conditions, cache)
end

# allow passing a single BC that get converted into a named tuple of BCs
# on (mapped) hypercube domains
function digest_boundary_conditions(boundary_conditions,
                                    mesh::Union{P4estMesh{2}, UnstructuredMesh2D,
                                                T8codeMesh{2}}, solver,
                                    cache)
    return (; x_neg = boundary_conditions, x_pos = boundary_conditions,
            y_neg = boundary_conditions, y_pos = boundary_conditions)
end

function digest_boundary_conditions(boundary_conditions,
                                    mesh::Union{P4estMesh{3}, T8codeMesh{3}}, solver,
                                    cache)
    return (; x_neg = boundary_conditions, x_pos = boundary_conditions,
            y_neg = boundary_conditions, y_pos = boundary_conditions,
            z_neg = boundary_conditions, z_pos = boundary_conditions)
end

# add methods for every dimension to resolve ambiguities
function digest_boundary_conditions(boundary_conditions::AbstractArray,
                                    mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                                    solver, cache)
    throw(ArgumentError("Please use a named tuple instead of an (abstract) array to supply multiple boundary conditions (to improve performance)."))
end

function digest_boundary_conditions(boundary_conditions::AbstractArray,
                                    mesh::Union{TreeMesh{2}, StructuredMesh{2}},
                                    solver, cache)
    throw(ArgumentError("Please use a named tuple instead of an (abstract) array to supply multiple boundary conditions (to improve performance)."))
end

function digest_boundary_conditions(boundary_conditions::AbstractArray,
                                    mesh::Union{TreeMesh{3}, StructuredMesh{3}},
                                    solver, cache)
    throw(ArgumentError("Please use a named tuple instead of an (abstract) array to supply multiple boundary conditions (to improve performance)."))
end

function digest_boundary_conditions(boundary_conditions::Tuple,
                                    mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                                    solver, cache)
    throw(ArgumentError("Please use a named tuple instead of a tuple to supply multiple boundary conditions."))
end

function digest_boundary_conditions(boundary_conditions::Tuple,
                                    mesh::Union{TreeMesh{2}, StructuredMesh{2}},
                                    solver, cache)
    throw(ArgumentError("Please use a named tuple instead of a tuple to supply multiple boundary conditions."))
end

function digest_boundary_conditions(boundary_conditions::Tuple,
                                    mesh::Union{TreeMesh{3}, StructuredMesh{3}},
                                    solver, cache)
    throw(ArgumentError("Please use a named tuple instead of a tuple to supply multiple boundary conditions."))
end

function get_periodicity_boundary_conditions_x(boundary_conditions, mesh)
    if isperiodic(mesh, 1)
        if :x_neg in keys(boundary_conditions) &&
           boundary_conditions.x_neg != boundary_condition_periodic ||
           :x_pos in keys(boundary_conditions) &&
           boundary_conditions.x_pos != boundary_condition_periodic
            throw(ArgumentError("For periodic mesh non-periodic boundary conditions in x-direction are supplied."))
        end
        x_neg = x_pos = boundary_condition_periodic
    else
        required = (:x_neg, :x_pos)
        if !all(in(keys(boundary_conditions)), required)
            throw(ArgumentError("NamedTuple of boundary conditions for 1-dimensional (non-periodic) mesh must have keys $(required), got $(keys(boundary_conditions))"))
        end
        @unpack x_neg, x_pos = boundary_conditions
    end
    return x_neg, x_pos
end

function get_periodicity_boundary_conditions_y(boundary_conditions, mesh)
    if isperiodic(mesh, 2)
        if :y_neg in keys(boundary_conditions) &&
           boundary_conditions.y_neg != boundary_condition_periodic ||
           :y_pos in keys(boundary_conditions) &&
           boundary_conditions.y_pos != boundary_condition_periodic
            throw(ArgumentError("For periodic mesh non-periodic boundary conditions in y-direction are supplied."))
        end
        y_neg = y_pos = boundary_condition_periodic
    else
        required = (:y_neg, :y_pos)
        if !all(in(keys(boundary_conditions)), required)
            throw(ArgumentError("NamedTuple of boundary conditions for 2-dimensional (non-periodic) mesh must have keys $(required), got $(keys(boundary_conditions))"))
        end
        @unpack y_neg, y_pos = boundary_conditions
    end
    return y_neg, y_pos
end

function get_periodicity_boundary_conditions_z(boundary_conditions, mesh)
    if isperiodic(mesh, 3)
        if :z_neg in keys(boundary_conditions) &&
           boundary_conditions.z_neg != boundary_condition_periodic ||
           :z_pos in keys(boundary_conditions) &&
           boundary_conditions.z_pos != boundary_condition_periodic
            throw(ArgumentError("For periodic mesh non-periodic boundary conditions in z-direction are supplied."))
        end
        z_neg = z_pos = boundary_condition_periodic
    else
        required = (:z_neg, :z_pos)
        if !all(in(keys(boundary_conditions)), required)
            throw(ArgumentError("NamedTuple of boundary conditions for 3-dimensional (non-periodic) mesh must have keys $(required), got $(keys(boundary_conditions))"))
        end
        @unpack z_neg, z_pos = boundary_conditions
    end
    return z_neg, z_pos
end

# No checks for these meshes yet available
function check_periodicity_mesh_boundary_conditions(mesh::Union{P4estMesh,
                                                                P4estMeshView,
                                                                UnstructuredMesh2D,
                                                                T8codeMesh,
                                                                DGMultiMesh},
                                                    boundary_conditions)
    return nothing
end

function check_periodicity_mesh_boundary_conditions(mesh::Union{TreeMesh,
                                                                StructuredMesh,
                                                                StructuredMeshView},
                                                    boundary_conditions::BoundaryConditionPeriodic)
    if !isperiodic(mesh)
        throw(ArgumentError("Periodic boundary condition supplied for non-periodic mesh."))
    end
    return nothing
end

function check_periodicity_mesh_boundary_conditions_x(mesh, x_neg, x_pos)
    if isperiodic(mesh, 1) &&
       (x_neg != boundary_condition_periodic ||
        x_pos != boundary_condition_periodic)
        throw(ArgumentError("For periodic mesh non-periodic boundary conditions in x-direction are supplied."))
    end
    if !isperiodic(mesh, 1) &&
       (x_neg == boundary_condition_periodic ||
        x_pos == boundary_condition_periodic)
        throw(ArgumentError("For non-periodic mesh periodic boundary conditions in x-direction are supplied."))
    end
    return nothing
end

function check_periodicity_mesh_boundary_conditions_y(mesh, y_neg, y_pos)
    if isperiodic(mesh, 2) &&
       (y_neg != boundary_condition_periodic ||
        y_pos != boundary_condition_periodic)
        throw(ArgumentError("For periodic mesh non-periodic boundary conditions in y-direction are supplied."))
    end
    if !isperiodic(mesh, 2) &&
       (y_neg == boundary_condition_periodic ||
        y_pos == boundary_condition_periodic)
        throw(ArgumentError("For non-periodic mesh periodic boundary conditions in y-direction are supplied."))
    end
    return nothing
end

function check_periodicity_mesh_boundary_conditions_z(mesh, z_neg, z_pos)
    if isperiodic(mesh, 3) &&
       (z_neg != boundary_condition_periodic ||
        z_pos != boundary_condition_periodic)
        throw(ArgumentError("For periodic mesh non-periodic boundary conditions in z-direction are supplied."))
    end
    if !isperiodic(mesh, 3) &&
       (z_neg == boundary_condition_periodic ||
        z_pos == boundary_condition_periodic)
        throw(ArgumentError("For non-periodic mesh periodic boundary conditions in z-direction are supplied."))
    end
    return nothing
end

function check_periodicity_mesh_boundary_conditions(mesh::Union{TreeMesh{1},
                                                                StructuredMesh{1}},
                                                    boundary_conditions::NamedTuple)
    return check_periodicity_mesh_boundary_conditions_x(mesh, boundary_conditions.x_neg,
                                                        boundary_conditions.x_pos)
end

function check_periodicity_mesh_boundary_conditions(mesh::Union{TreeMesh{2},
                                                                StructuredMesh{2},
                                                                StructuredMeshView{2}},
                                                    boundary_conditions::NamedTuple)
    check_periodicity_mesh_boundary_conditions_x(mesh, boundary_conditions.x_neg,
                                                 boundary_conditions.x_pos)
    return check_periodicity_mesh_boundary_conditions_y(mesh, boundary_conditions.y_neg,
                                                        boundary_conditions.y_pos)
end

function check_periodicity_mesh_boundary_conditions(mesh::Union{TreeMesh{3},
                                                                StructuredMesh{3}},
                                                    boundary_conditions::NamedTuple)
    check_periodicity_mesh_boundary_conditions_x(mesh, boundary_conditions.x_neg,
                                                 boundary_conditions.x_pos)
    check_periodicity_mesh_boundary_conditions_y(mesh, boundary_conditions.y_neg,
                                                 boundary_conditions.y_pos)
    return check_periodicity_mesh_boundary_conditions_z(mesh, boundary_conditions.z_neg,
                                                        boundary_conditions.z_pos)
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
    return nothing
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
    return summary_line(io, "boundary conditions", typeof(semi.boundary_conditions))
end

function print_boundary_conditions(io,
                                   semi::SemiHypMeshBCSolver{<:Any,
                                                             <:UnstructuredSortedBoundaryTypes})
    @unpack boundary_conditions = semi.boundary_conditions
    summary_line(io, "boundary conditions", length(boundary_conditions))
    for (boundary_name, boundary_condition) in pairs(boundary_conditions)
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
                                                             <:Union{NamedTuple,
                                                                     AbstractArray}})
    summary_line(io, "boundary conditions", 2 * ndims(semi))
    bcs = semi.boundary_conditions

    summary_line(increment_indent(io), "negative x", bcs.x_neg)
    summary_line(increment_indent(io), "positive x", bcs.x_pos)
    if ndims(semi) > 1
        summary_line(increment_indent(io), "negative y", bcs.y_neg)
        summary_line(increment_indent(io), "positive y", bcs.y_pos)
    end
    if ndims(semi) > 2
        summary_line(increment_indent(io), "negative z", bcs.z_neg)
        summary_line(increment_indent(io), "positive z", bcs.z_pos)
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

    return calc_error_norms(func, u, t, analyzer, mesh, equations, initial_condition,
                            solver,
                            cache, cache_analysis)
end

function compute_coefficients(t, semi::SemidiscretizationHyperbolic)
    # Call `compute_coefficients` in `src/semidiscretization/semidiscretization.jl`
    return compute_coefficients(semi.initial_condition, t, semi)
end

function compute_coefficients!(u_ode, t, semi::SemidiscretizationHyperbolic)
    return compute_coefficients!(u_ode, semi.initial_condition, t, semi)
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
