# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct FV{SlopeLimiter, SurfaceFlux}
    order::Integer
    slope_limiter::SlopeLimiter
    surface_flux::SurfaceFlux

    function FV(; order = 1, slope_limiter = "TODO", surface_flux = flux_central)
        new{typeof(slope_limiter), typeof(surface_flux)}(order, slope_limiter,
                                                         surface_flux)
    end
end

function Base.show(io::IO, solver::FV)
    @nospecialize solver # reduce precompilation time

    print(io, "FV(")
    print(io, "order $(solver.order)")
    if solver.order > 1
        print(io, ", ", solver.slope_limiter)
    end
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
        if solver.order > 1
            summary_line(io, "slope limiter", solver.slope_limiter)
        end
        summary_line(io, "surface flux", solver.surface_flux)
        summary_footer(io)
    end
end

@inline Base.real(solver::FV) = Float64 # TODO
@inline ndofs(mesh, solver::FV, cache) = nelementsglobal(mesh, solver, cache)

@inline function ndofsglobal(mesh, solver::FV, cache)
    ndofs(mesh, solver, cache)
end

function compute_coefficients!(u, func, t, mesh::AbstractMesh, equations,
                               solver::FV,
                               cache)
    for element in eachelement(mesh, solver)    # TODO: Does @threaded work with mpi?
        x_node = SVector(cache.elements[element].midpoint) # Save t8code variables as SVector?
        u_node = func(x_node, t, equations)
        set_node_vars!(u, u_node, equations, solver, element)
    end
end

function allocate_coefficients(mesh::AbstractMesh, equations, solver::FV, cache)
    # We must allocate a `Vector` in order to be able to `resize!` it (AMR).
    # cf. wrap_array
    zeros(eltype(cache.elements[1].volume),
          nvariables(equations) * nelements(mesh, solver, cache))
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

function rhs!(du, u, t, mesh, equations, initial_condition, boundary_conditions,
              source_terms::Source, solver::FV, cache) where {Source}
    @trixi_timeit timer() "update neighbor data" exchange_solution!(u, mesh, equations,
                                                                    solver, cache)
    @unpack u_ = cache

    du .= zero(eltype(du))

    @trixi_timeit timer() "reconstruction" reconstruction(u_, mesh, equations, solver, cache)

    @trixi_timeit timer() "update du" begin
        for element in eachelement(mesh, solver)
            @unpack face_normals, num_faces, face_areas, face_connectivity = cache.elements[element]
            for face in 1:num_faces
                neighbor = face_connectivity[face]
                if neighbor < element
                    continue
                end
                normal = SVector(face_normals[2 * face - 1], face_normals[2 * face])
                @trixi_timeit timer() "evaluation" u_element, u_neighbor = evaluate_interface_values(element, neighbor,
                                                                face, normal, u_,
                                                                mesh, solver, cache)
                @trixi_timeit timer() "surface flux" flux = solver.surface_flux(u_element, u_neighbor,
                                                                                normal, equations)
                @trixi_timeit timer() "for loop" for v in eachvariable(equations)
                    flux_ = -face_areas[face] * flux[v]
                    du[v, element] += flux_
                    if neighbor <= mesh.number_elements
                        du[v, neighbor] -= flux_
                    end
                end
            end
        end
        for element in eachelement(mesh, solver)
            @unpack volume = cache.elements[element]
            for v in eachvariable(equations)
                du[v, element] = (1 / volume) * du[v, element]
            end
        end
    end # timer

    return nothing
end

function reconstruction(u_, mesh, equations, solver, cache)
    if solver.order == 1
        return nothing
    elseif solver.order == 2
        linear_reconstruction(u_, mesh, equations, solver, cache)
    else
        error("order $(solver.order) not supported.")
    end

    return nothing
end

function linear_reconstruction(u_, mesh, equations, solver, cache)
    # Approximate slope
    for element in eachelement(mesh, solver, cache)
        @unpack u = u_[element]
        @unpack num_faces, face_connectivity, face_areas, face_normals, midpoint, face_midpoints, volume = cache.elements[element]

        slope = zeros(Cdouble, ndims(mesh))
        for face in 1:num_faces
            neighbor = face_connectivity[face]
            normal = SVector(face_normals[2 * face - 1], face_normals[2 * face])
            face_midpoint = SVector(face_midpoints[2 * face - 1], face_midpoints[2 * face])
            if norm(face_midpoint .- cache.elements[neighbor].midpoint) > norm(cache.elements[neighbor].midpoint .- midpoint)
                ratio = 0.5 # TODO periodic boundary.
            else
                ratio = norm(face_midpoint .- midpoint) /
                        norm(cache.elements[neighbor].midpoint .- midpoint)
            end
            u_face = u .+ (u_[neighbor].u .- u) .* ratio
            slope .+= face_areas[face] .* u_face .* normal
        end
        slope .*= 1 / volume
        u_[element] = T8codeSolutionContainer(u, Tuple(slope))
    end

    exchange_ghost_data(mesh, u_)

    return nothing
end

function evaluate_interface_values(element, neighbor, face, normal, u_, mesh, solver,
                                   cache)
    @unpack elements = cache

    if solver.order == 1
        return SVector(u_[element].u), SVector(u_[neighbor].u)
    elseif solver.order == 2
        @unpack midpoint, face_midpoints = elements[element]
        face_midpoint = SVector(face_midpoints[2 * face - 1], face_midpoints[2 * face])

        face_neighbor = elements[element].neighbor_faces[face]
        face_midpoints_neighbor = elements[neighbor].face_midpoints
        face_midpoint_neighbor = SVector(face_midpoints_neighbor[2 * face_neighbor - 1],
                                         face_midpoints_neighbor[2 * face_neighbor])
        # TODO: Currently, slope only for nvariables=1
        slope = solver.slope_limiter(u_[element].slope, u_[neighbor].slope)
        u1 = SVector(u_[element].u .+ sum(slope .* (face_midpoint .- midpoint)))
        u2 = SVector(u_[neighbor].u .+ sum(slope .* (face_midpoint_neighbor .-
                                                     elements[neighbor].midpoint)))
        return u1, u2
    else
        error("Order $(solver.order) is not supported.")
    end
end

function minmod(slope1::Tuple, slope2::Tuple)
    slope = zeros(length(slope1))
    for d in eachindex(slope1)
        slope[d] = minmod(slope1[d], slope2[2])
    end
    # TODO
    return slope
end

function minmod(slope1, slope2)
    if slope1 > 0 && slope2 > 0
        return min(slope1, slope2)
    elseif slope1 < 0 && slope2 < 0
        return max(slope1, slope2)
    end
    return zero(eltype(slope1))
end

function get_element_variables!(element_variables, u, mesh::T8codeMesh, equations,
                                solver, cache)
    return nothing
end

function SolutionAnalyzer(solver::FV; kwargs...)
end

function create_cache_analysis(analyzer, mesh,
                               equations, solver::FV, cache,
                               RealT, uEltype)
end

# Container data structures
include("containers.jl")
end # @muladd
