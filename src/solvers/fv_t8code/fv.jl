# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct FV{RealT <: Real, SurfaceFlux}
    order::Integer
    surface_flux::SurfaceFlux

    function FV(; order = 1, RealT = Float64, surface_flux = flux_central)
        new{RealT, typeof(surface_flux)}(order, surface_flux)
    end
end

function Base.show(io::IO, solver::FV)
    @nospecialize solver # reduce precompilation time

    print(io, "FV(")
    print(io, "order $(solver.order)")
    print(io, ", ", solver.surface_flux)
    print(io, ")")
end

function Base.show(io::IO, mime::MIME"text/plain", solver::FV)
    @nospecialize solver # reduce precompilation time

    if get(io, :compact, false)
        show(io, solver)
    else
        summary_header(io, "FV{" * string(real(solver)) * "}")
        summary_line(io, "order", solver.order)
        summary_line(io, "surface flux", solver.surface_flux)
        summary_footer(io)
    end
end

Base.summary(io::IO, solver::FV) = print(io, "FV(order=$(solver.order))")

@inline Base.real(solver::FV{RealT}) where {RealT} = RealT

@inline ndofs(mesh, solver::FV, cache) = ncells(mesh)

@inline nelements(mesh::T8codeMesh, solver::FV, cache) = ncells(mesh)
@inline function ndofsglobal(mesh, solver::FV, cache)
    nelementsglobal(mesh, solver, cache)
end

@inline function eachelement(mesh, solver::FV, cache)
    Base.OneTo(nelements(mesh, solver, cache))
end

@inline eachinterface(solver::FV, cache) = Base.OneTo(ninterfaces(solver, cache))
@inline eachboundary(solver::FV, cache) = Base.OneTo(nboundaries(solver, cache))

@inline function nelementsglobal(mesh, solver::FV, cache)
    if mpi_isparallel()
        Int(t8_forest_get_global_num_elements(mesh.forest))
    else
        nelements(mesh, solver, cache)
    end
end

@inline ninterfaces(solver::FV, cache) = ninterfaces(cache.interfaces)
@inline nboundaries(solver::FV, cache) = nboundaries(cache.boundaries)

@inline function get_node_coords(x, equations, solver::FV, indices...)
    SVector(ntuple(@inline(idx->x[idx, indices...]), Val(ndims(equations))))
end

@inline function get_node_vars(u, equations, solver::FV, element)
    SVector(ntuple(@inline(v->u[v, element]), Val(nvariables(equations))))
end

@inline function set_node_vars!(u, u_node, equations, solver::FV, element)
    for v in eachvariable(equations)
        u[v, element] = u_node[v]
    end
    return nothing
end

@inline function get_surface_node_vars(u, equations, solver::FV, indices...)
    # There is a cut-off at `n == 10` inside of the method
    # `ntuple(f::F, n::Integer) where F` in Base at ntuple.jl:17
    # in Julia `v1.5`, leading to type instabilities if
    # more than ten variables are used. That's why we use
    # `Val(...)` below.
    u_ll = SVector(ntuple(@inline(v->u[1, v, indices...]), Val(nvariables(equations))))
    u_rr = SVector(ntuple(@inline(v->u[2, v, indices...]), Val(nvariables(equations))))
    return u_ll, u_rr
end

function allocate_coefficients(mesh::T8codeMesh, equations, solver::FV, cache)
    # We must allocate a `Vector` in order to be able to `resize!` it (AMR).
    # cf. wrap_array
    zeros(eltype(cache.elements),
          nvariables(equations) * nelements(mesh, solver, cache))
end

# General fallback
@inline function wrap_array(u_ode::AbstractVector, mesh::AbstractMesh, equations,
                            solver::FV, cache)
    wrap_array_native(u_ode, mesh, equations, solver, cache)
end

# Like `wrap_array`, but guarantees to return a plain `Array`, which can be better
# for interfacing with external C libraries (MPI, HDF5, visualization),
# writing solution files etc.
@inline function wrap_array_native(u_ode::AbstractVector, mesh::AbstractMesh, equations,
                                   solver::FV, cache)
    @boundscheck begin
        @assert length(u_ode) ==
                nvariables(equations) * nelements(mesh, solver, cache)
    end
    unsafe_wrap(Array{eltype(u_ode), 2}, pointer(u_ode),
                (nvariables(equations), nelements(mesh, solver, cache)))
end

function compute_coefficients!(u, func, t, mesh::T8codeMesh,
                               equations, solver::FV, cache)
    for element in eachelement(mesh, solver, cache)
        x_node = get_node_coords(cache.elements.midpoint, equations, solver, element)
        u_node = func(x_node, t, equations)
        set_node_vars!(u, u_node, equations, solver, element)
    end
end

function create_cache(mesh::T8codeMesh, equations::AbstractEquations, solver::FV, ::Any,
                      ::Type{uEltype}) where {uEltype <: Real}
    count_required_surfaces!(mesh)

    # After I saved some data (e.g. normal) in the interfaces and boundaries,
    # the element data structure is not used anymore after this `create_cache` routine.
    # Possible to remove it and directly save the data in interface, boundars (and mortar) data structure?
    elements = init_elements(mesh, equations, solver, uEltype)
    interfaces = init_interfaces(mesh, equations, solver, uEltype)
    boundaries = init_boundaries(mesh, equations, solver, uEltype)
    # mortars = init_mortars(mesh, equations, solver, uEltype)

    fill_mesh_info_fv!(mesh, interfaces, boundaries,
                       mesh.boundary_names)

    # Temporary solution array to allow exchange between MPI ranks.
    u_tmp = init_solution!(mesh, equations)

    cache = (; elements, interfaces, boundaries, u_tmp)

    return cache
end

function rhs!(du, u, t, mesh::T8codeMesh, equations,
              initial_condition, boundary_conditions, source_terms::Source,
              solver::FV, cache) where {Source}
    # Reset du
    @trixi_timeit timer() "reset ∂u/∂t" du.=zero(eltype(du))

    # Exchange solution between MPI ranks
    @trixi_timeit timer() "exchange_solution!" exchange_solution!(u, mesh, equations,
                                                                  solver, cache)

    @trixi_timeit timer() "gradient reconstruction" calc_gradient_reconstruction!(u,
                                                                                  mesh,
                                                                                  equations,
                                                                                  solver,
                                                                                  cache)

    # Prolong solution to interfaces
    @trixi_timeit timer() "prolong2interfaces" begin
        prolong2interfaces!(cache, mesh, equations, solver)
    end

    # Calculate interface fluxes
    @trixi_timeit timer() "interface flux" begin
        calc_interface_flux!(du, mesh, have_nonconservative_terms(equations), equations,
                             solver, cache)
    end

    # Prolong solution to boundaries
    @trixi_timeit timer() "prolong2boundaries!" begin
        prolong2boundaries!(cache, mesh, equations, solver)
    end

    # Calculate boundary fluxes
    @trixi_timeit timer() "calc boundary flux" begin
        calc_boundary_flux!(du, cache, t, boundary_conditions, mesh,
                            equations, solver)
    end

    @trixi_timeit timer() "volume" begin
        for element in eachelement(mesh, solver, cache)
            volume = cache.elements.volume[element]
            for v in eachvariable(equations)
                du[v, element] = (1 / volume) * du[v, element]
            end
        end
    end

    return nothing
end

function calc_gradient_reconstruction!(u, mesh, equations, solver, cache)
    if solver.order == 1
        return nothing
    elseif solver.order > 2
        error("Order $(solver.order) is not supported yet!")
    end

    (; elements) = cache
    (; reconstruction_stencil, reconstruction_gradient) = elements

    # A         N x 2 Matrix, where N is the number of stencil neighbors
    # A^T A     2 x 2 Matrix
    # b         N     Vector
    # A^T b     2     Vector

    # Matrix/vector notation
    # A^T A = [a b; b c]
    # (A^T A)^-1 = determinant_factor * [c -b; -b, a]

    # A^T b = [d; e]
    # Note: A^T b depends on the variable v. Using structure [d/e, v]
    d = zeros(eltype(u), size(u, 1))
    e = zeros(eltype(u), size(u, 1))

    for element in eachelement(mesh, solver, cache)
        n_stencil_neighbors = length(reconstruction_stencil[element])
        coordinates_element = get_node_coords(elements.midpoint, equations, solver,
                                              element)

        # Reset variables
        a = zero(eltype(u))
        b = zero(eltype(u))
        c = zero(eltype(u))
        # A^T b = [d; e]
        # Note: A^T b depends on the variable v. Using structure [d/e, v]
        d .= zero(eltype(u))
        e .= zero(eltype(u))
        for i in 1:n_stencil_neighbors
            neighbor = reconstruction_stencil[element][i]
            coordinates_neighbor = get_node_coords(elements.midpoint, equations, solver,
                                                   neighbor)

            a += (coordinates_neighbor[1] - coordinates_element[1])^2
            b += (coordinates_neighbor[1] - coordinates_element[1]) *
                 (coordinates_neighbor[2] - coordinates_element[2])
            c += (coordinates_neighbor[2] - coordinates_element[2])^2

            for v in eachvariable(equations)
                d[v] += (coordinates_neighbor[1] - coordinates_element[1]) *
                        (u[v, neighbor] - u[v, element])
                e[v] += (coordinates_neighbor[2] - coordinates_element[2]) *
                        (u[v, neighbor] - u[v, element])
            end
        end

        # Divide by determinant
        AT_A_determinant = a * c - b^2
        a *= 1 / AT_A_determinant
        b *= 1 / AT_A_determinant
        c *= 1 / AT_A_determinant

        # Solving least square problem
        for v in eachvariable(equations)
            reconstruction_gradient[1, v, element] = c * d[v] - b * e[v]
            reconstruction_gradient[2, v, element] = -b * d[v] + a * e[v]
        end
    end

    return nothing
end

function prolong2interfaces!(cache, mesh::T8codeMesh, equations, solver::FV)
    (; elements, interfaces, u_tmp) = cache
    (; midpoint, face_midpoints, reconstruction_gradient) = elements

    for interface in eachinterface(solver, cache)
        element = interfaces.neighbor_ids[1, interface]
        neighbor = interfaces.neighbor_ids[2, interface]
        if solver.order == 1
            for v in eachvariable(equations)
                interfaces.u[1, v, interface] = u_tmp[element].u[v]
                interfaces.u[2, v, interface] = u_tmp[neighbor].u[v]
            end
        elseif solver.order == 2
            face_element = interfaces.faces[1, interface]
            face_neighbor = interfaces.faces[2, interface]

            face_midpoint_element = get_node_coords(face_midpoints, equations, solver,
                                                    face_element, element)
            face_midpoint_neighbor = get_node_coords(face_midpoints, equations, solver,
                                                     face_neighbor, neighbor)

            midpoint_element = get_node_coords(midpoint, equations, solver, element)
            midpoint_neighbor = get_node_coords(midpoint, equations, solver, neighbor)

            vector_element = face_midpoint_element .- midpoint_element
            vector_neighbor = face_midpoint_neighbor .- midpoint_neighbor
            for v in eachvariable(equations)
                gradient_v_element = get_node_coords(reconstruction_gradient, equations,
                                                     solver, v, element)
                gradient_v_neighbor = get_node_coords(reconstruction_gradient,
                                                      equations, solver, v, neighbor)
                interfaces.u[1, v, interface] = u_tmp[element].u[v] +
                                                dot(gradient_v_element, vector_element)
                interfaces.u[2, v, interface] = u_tmp[neighbor].u[v] +
                                                dot(gradient_v_neighbor,
                                                    vector_neighbor)
            end
        else
            error("Order $(solver.order) is not supported.")
        end
    end

    return nothing
end

function calc_interface_flux!(du, mesh::T8codeMesh,
                              nonconservative_terms::False, equations,
                              solver::FV, cache)
    (; surface_flux) = solver
    (; elements, interfaces) = cache

    for interface in eachinterface(solver, cache)
        element = interfaces.neighbor_ids[1, interface]
        neighbor = interfaces.neighbor_ids[2, interface]
        face = interfaces.faces[1, interface]

        # TODO: Save normal and face_areas in interface?
        normal = get_node_coords(elements.face_normals, equations, solver,
                                 face, element)
        u_ll, u_rr = get_surface_node_vars(interfaces.u, equations, solver,
                                           interface)
        flux = surface_flux(u_ll, u_rr, normal, equations)

        for v in eachvariable(equations)
            flux_ = elements.face_areas[face, element] * flux[v]
            du[v, element] -= flux_
            if !is_ghost_cell(neighbor, mesh)
                du[v, neighbor] += flux_
            end
        end
    end

    return nothing
end

function prolong2boundaries!(cache, mesh::T8codeMesh, equations, solver::FV)
    (; boundaries, u_tmp) = cache

    for boundary in eachboundary(solver, cache)
        element = boundaries.neighbor_ids[boundary]
        if solver.order == 1
            for v in eachvariable(equations)
                boundaries.u[v, boundary] = u_tmp[element].u[v]
            end
        else
            error("Order $(solver.order) is not supported.")
        end
    end

    return nothing
end

function calc_boundary_flux!(du, cache, t,
                             boundary_condition::BoundaryConditionPeriodic,
                             mesh::T8codeMesh,
                             equations, solver::FV)
    @assert isempty(eachboundary(solver, cache))
end

# Function barrier for type stability
function calc_boundary_flux!(du, cache, t, boundary_conditions,
                             mesh::T8codeMesh,
                             equations, solver::FV)
    @unpack boundary_condition_types, boundary_indices = boundary_conditions

    calc_boundary_flux_by_type!(du, cache, t, boundary_condition_types,
                                boundary_indices, mesh, equations, solver)
    return nothing
end

# Iterate over tuples of boundary condition types and associated indices
# in a type-stable way using "lispy tuple programming".
function calc_boundary_flux_by_type!(du, cache, t, BCs::NTuple{N, Any},
                                     BC_indices::NTuple{N, Vector{Int}},
                                     mesh::T8codeMesh,
                                     equations, solver::FV) where {N}
    # Extract the boundary condition type and index vector
    boundary_condition = first(BCs)
    boundary_condition_indices = first(BC_indices)
    # Extract the remaining types and indices to be processed later
    remaining_boundary_conditions = Base.tail(BCs)
    remaining_boundary_condition_indices = Base.tail(BC_indices)

    # process the first boundary condition type
    calc_boundary_flux!(du, cache, t, boundary_condition, boundary_condition_indices,
                        mesh, equations, solver)

    # recursively call this method with the unprocessed boundary types
    calc_boundary_flux_by_type!(du, cache, t, remaining_boundary_conditions,
                                remaining_boundary_condition_indices,
                                mesh, equations, solver)

    return nothing
end

# terminate the type-stable iteration over tuples
function calc_boundary_flux_by_type!(du, cache, t, BCs::Tuple{}, BC_indices::Tuple{},
                                     mesh::T8codeMesh,
                                     equations, solver::FV)
    nothing
end

function calc_boundary_flux!(du, cache, t, boundary_condition::BC, boundary_indexing,
                             mesh::T8codeMesh,
                             equations, solver::FV) where {BC}
    (; elements, boundaries) = cache
    (; surface_flux) = solver

    for local_index in eachindex(boundary_indexing)
        # Use the local index to get the global boundary index from the pre-sorted list
        boundary = boundary_indexing[local_index]

        # Get information on the adjacent element, compute the surface fluxes,
        # and store them
        element = boundaries.neighbor_ids[boundary]
        face = boundaries.faces[boundary]

        # TODO: Save normal and face_areas in interface?
        normal = get_node_coords(cache.elements.face_normals, equations, solver,
                                 face, element)

        u_inner = get_node_vars(boundaries.u, equations, solver, boundary)

        # Coordinates at boundary node
        face_midpoint = get_node_coords(cache.elements.face_midpoints, equations,
                                        solver, face, element)

        flux = boundary_condition(u_inner, normal, face_midpoint, t, surface_flux,
                                  equations)
        for v in eachvariable(equations)
            flux_ = elements.face_areas[face, element] * flux[v]
            du[v, element] -= flux_
        end
    end

    return nothing
end

function get_element_variables!(element_variables, u,
                                mesh, equations,
                                solver::FV, cache)
    return nothing
end

function get_node_variables!(node_variables, mesh,
                             equations, solver::FV, cache)
    return nothing
end

function SolutionAnalyzer(solver::FV; kwargs...)
end

function create_cache_analysis(analyzer, mesh,
                               equations, solver::FV, cache,
                               RealT, uEltype)
end

function T8codeMesh(cmesh::Ptr{t8_cmesh}, solver::DG; kwargs...)
    T8codeMesh(cmesh; kwargs...)
end

function T8codeMesh(cmesh::Ptr{t8_cmesh}, solver::FV; kwargs...)
    T8codeMesh(cmesh; polydeg = 0, kwargs...)
end

# Container data structures
include("containers.jl")
end # @muladd
