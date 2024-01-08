# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct FV{SurfaceFlux}
    order::Integer
    surface_flux::SurfaceFlux

    function FV(; order = 1, surface_flux = flux_central)
        new{typeof(surface_flux)}(order, surface_flux)
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

@inline Base.real(solver::FV) = Float64 # TODO
@inline ndofs(mesh, solver::FV, cache) = nelementsglobal(mesh, solver, cache)

@inline function ndofsglobal(mesh, solver::FV, cache)
    ndofs(mesh, solver, cache)
end

function compute_coefficients!(u, func, t, mesh::T8codeFVMesh, equations,
                               solver::FV, cache)
    for element in eachelement(mesh, solver, cache)
        x_node = SVector(cache.elements[element].midpoint) # Save t8code variables as SVector?
        u_node = func(x_node, t, equations)
        set_node_vars!(u, u_node, equations, solver, element)
    end
end

function allocate_coefficients(mesh::T8codeFVMesh, equations, solver::FV, cache)
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

function rhs!(du, u, t, mesh::T8codeFVMesh, equations, initial_condition,
              boundary_conditions, source_terms::Source, solver::FV,
              cache) where {Source}
    @trixi_timeit timer() "update neighbor data" exchange_solution!(u, mesh, equations,
                                                                    solver, cache)
    @unpack elements, interfaces, u_ = cache

    du .= zero(eltype(du))

    @trixi_timeit timer() "evaluation" evaluate_interface_values!(mesh, equations,
                                                                  solver, cache)

    @trixi_timeit timer() "update du" begin
        for interface in eachinterface(solver, cache)
            element = interfaces.neighbor_ids[1, interface]
            neighbor = interfaces.neighbor_ids[2, interface]
            face = interfaces.faces[1, interface]

            # TODO: Save normal and face_areas in interface?
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
        else
            error("Order $(solver.order) is not supported.")
        end
    end

    return nothing
end

function get_element_variables!(element_variables, u, mesh::T8codeFVMesh, equations,
                                solver, cache)
    return nothing
end

function get_node_variables!(node_variables, mesh::T8codeFVMesh, equations, solver,
                             cache)
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
