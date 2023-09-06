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

    function FV(; order = 1, slope_limiter = average_slope_limiter,
                surface_flux = flux_central)
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

Base.summary(io::IO, solver::FV) = print(io, "FV(order=$(solver.order))")

@inline Base.real(solver::FV) = Float64 # TODO
@inline ndofs(mesh, solver::FV, cache) = nelementsglobal(mesh, solver, cache)

@inline function ndofsglobal(mesh, solver::FV, cache)
    ndofs(mesh, solver, cache)
end

function compute_coefficients!(u, func, t, mesh::T8codeMesh, equations,
                               solver::FV, cache)
    for element in eachelement(mesh, solver, cache)
        x_node = SVector(cache.elements[element].midpoint) # Save t8code variables as SVector?
        u_node = func(x_node, t, equations)
        set_node_vars!(u, u_node, equations, solver, element)
    end
end

function allocate_coefficients(mesh::T8codeMesh, equations, solver::FV, cache)
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

function rhs!(du, u, t, mesh::T8codeMesh, equations, initial_condition,
              boundary_conditions, source_terms::Source, solver::FV,
              cache) where {Source}
    @trixi_timeit timer() "update neighbor data" exchange_solution!(u, mesh, equations,
                                                                    solver, cache)
    @unpack elements, interfaces, u_ = cache

    du .= zero(eltype(du))

    @trixi_timeit timer() "reconstruction" reconstruction(u_, mesh, equations, solver,
                                                          cache)

    @trixi_timeit timer() "evaluation" evaluate_interface_values!(mesh, equations,
                                                                  solver, cache)

    @trixi_timeit timer() "update du" begin
        for interface in eachinterface(solver, cache)
            element = interfaces.neighbor_ids[1, interface]
            neighbor = interfaces.neighbor_ids[2, interface]
            face = interfaces.faces[1, interface]

            # TODO: Save normal and face_areas in interface
            normal = Trixi.get_variable_wrapped(elements[element].face_normals,
                                                equations, face)
            u_ll, u_rr = get_surface_node_vars(interfaces.u, equations, solver,
                                               interface)
            @trixi_timeit timer() "surface flux" flux=solver.surface_flux(u_ll, u_rr,
                                                                          normal,
                                                                          equations)
            @trixi_timeit timer() "for loop" for v in eachvariable(equations)
                flux_ = -elements[element].face_areas[face] * flux[v]
                du[v, element] += flux_
                if neighbor <= mesh.number_elements
                    du[v, neighbor] -= flux_
                end
            end
        end
        for element in eachelement(mesh, solver, cache)
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
    @unpack elements = cache

    slope = zeros(eltype(u_[1].u), nvariables(equations) * ndims(mesh))

    # Approximate slope
    for element in eachelement(mesh, solver, cache)
        @unpack u = u_[element]
        @unpack num_faces, face_connectivity, face_areas, face_normals, midpoint, face_midpoints, volume = cache.elements[element]

        slope .= zero(eltype(slope))
        for face in 1:num_faces
            neighbor = face_connectivity[face]
            normal = Trixi.get_variable_wrapped(face_normals, equations, face)
            face_midpoint = Trixi.get_variable_wrapped(face_midpoints, equations, face)

            face_neighbor = elements[element].neighbor_faces[face]
            face_midpoint_neighbor = Trixi.get_variable_wrapped(elements[neighbor].face_midpoints,
                                                                equations,
                                                                face_neighbor)
            if face_midpoint != face_midpoint_neighbor
                # Periodic boundary
                # - The face_midpoint must be synchronous at each side of the mesh.
                #   Is it possible to have shifted faces?
                # - Distance is implemented as the sum of the two distances to the face_midpoint.
                #   In general, this is not the actual distance.
                distance = norm(face_midpoint .- midpoint) +
                           norm(face_midpoint_neighbor .- elements[neighbor].midpoint)
            else
                distance = norm(elements[neighbor].midpoint .- midpoint)
            end
            slope_ = (u_[neighbor].u .- u) ./ distance
            u_face = u .+ slope_ .* norm(face_midpoint .- midpoint)

            for v in eachvariable(equations)
                for d in eachindex(normal)
                    slope[(v - 1) * ndims(mesh) + d] += face_areas[face] * u_face[v] *
                                                        normal[d]
                end
            end
        end
        slope .*= 1 / volume
        u_[element] = T8codeSolutionContainer(u, Tuple(slope))
    end

    exchange_ghost_data(mesh, u_)

    return nothing
end

function evaluate_interface_values!(mesh, equations, solver, cache)
    @unpack elements, interfaces, u_ = cache

    for interface in eachinterface(solver, cache)
        element = interfaces.neighbor_ids[1, interface]
        neighbor = interfaces.neighbor_ids[2, interface]
        if solver.order == 1
            for v in eachvariable(equations)
                interfaces.u[1, v, interface] = u_[element].u[v]
                interfaces.u[2, v, interface] = u_[neighbor].u[v]
            end
        elseif solver.order == 2
            @unpack midpoint, face_midpoints = elements[element]
            face = interfaces.faces[1, interface]
            face_neighbor = interfaces.faces[2, interface]

            face_midpoint = Trixi.get_variable_wrapped(face_midpoints, equations, face)
            face_midpoints_neighbor = elements[neighbor].face_midpoints
            face_midpoint_neighbor = Trixi.get_variable_wrapped(face_midpoints_neighbor,
                                                                equations,
                                                                face_neighbor)

            for v in eachvariable(equations)
                s1 = Trixi.get_variable_wrapped(u_[element].slope, equations, v)
                s2 = Trixi.get_variable_wrapped(u_[neighbor].slope, equations, v)

                s1 = dot(s1,
                         (face_midpoint .- midpoint) ./ norm(face_midpoint .- midpoint))
                s2 = dot(s2,
                         (elements[neighbor].midpoint .- face_midpoint_neighbor) ./
                         norm(elements[neighbor].midpoint .- face_midpoint_neighbor))
                # Is it useful to compare such slopes in different directions? Alternatively, one could use the normal vector.
                # But this is again not useful, since u_face would use the slope in normal direction. I think it looks good the way it is.

                slope_v = solver.slope_limiter(s1, s2)
                interfaces.u[1, v, interface] = u_[element].u[v] +
                                                slope_v *
                                                norm(face_midpoint .- midpoint)
                interfaces.u[2, v, interface] = u_[neighbor].u[v] -
                                                slope_v *
                                                norm(elements[neighbor].midpoint .-
                                                     face_midpoint_neighbor)
            end
        else
            error("Order $(solver.order) is not supported.")
        end
    end

    return nothing
end

function average_slope_limiter(s1, s2)
    return 0.5 * s1 + 0.5 * s2
end

function minmod(s...)
    if all(s .> 0)
        return minimum(s)
    elseif all(s .< 0)
        return maximum(s)
    end
    return zero(eltype(s[1]))
end

function monotonized_central(s1, s2)
    return minmod(2 * s1, (s1 + s2) / 2, 2 * s2)
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
