# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct FVMuscl{SurfaceFlux, Limiter}
    surface_flux::SurfaceFlux
    limiter::Limiter

    function FVMuscl(; surface_flux, limiter = "TODO")
        new{typeof(surface_flux), typeof(limiter)}(surface_flux, limiter)
    end
end

function Base.show(io::IO, solver::FVMuscl)
    @nospecialize solver # reduce precompilation time

    print(io, "FV(")
    print(io, solver.surface_flux)
    print(io, ", ", solver.limiter)
    print(io, ")")
end

function Base.show(io::IO, mime::MIME"text/plain", solver::FVMuscl)
    @nospecialize solver # reduce precompilation time

    if get(io, :compact, false)
        show(io, solver)
    else
        summary_header(io, "FV{" * string(real(solver)) * "}")
        summary_line(io, "surface flux", solver.surface_flux)
        summary_line(io, "order", 1)
        summary_line(io, "limiter", solver.limiter)
        summary_footer(io)
    end
end

@inline Base.real(solver::FVMuscl) = Float64 # TODO
@inline ndofs(mesh, solver::FVMuscl, cache) = nelements_global(mesh, solver, cache)

@inline function eachelement(mesh::T8codeMesh, solver)
    Base.OneTo(mesh.number_elements)
end

function compute_coefficients!(u, func, t, mesh::AbstractMesh, equations,
                               solver::FVMuscl,
                               cache)
    @threaded for element in eachelement(mesh, solver)
        x_node = cache.elements[element].midpoint
        u_node = func(x_node, t, equations)
        u[element] = u_node
    end
end

function allocate_coefficients(mesh::AbstractMesh, equations, solver::FVMuscl, cache)
    # We must allocate a `Vector` in order to be able to `resize!` it (AMR).
    # cf. wrap_array
    zeros(eltype(cache.elements[1].volume),
          nvariables(equations) * nelements(mesh, solver, cache))
end

# General fallback
@inline function wrap_array(u_ode::AbstractVector, mesh::AbstractMesh, equations,
                            solver::FVMuscl, cache)
    wrap_array_native(u_ode, mesh, equations, solver, cache)
end

# Like `wrap_array`, but guarantees to return a plain `Array`, which can be better
# for interfacing with external C libraries (MPI, HDF5, visualization),
# writing solution files etc.
@inline function wrap_array_native(u_ode::AbstractVector, mesh::AbstractMesh, equations,
                                   solver::FVMuscl, cache)
    @boundscheck begin
        @assert length(u_ode) ==
                nvariables(equations) * nelements(mesh, solver, cache)
    end
    unsafe_wrap(Array{eltype(u_ode), 2}, pointer(u_ode),
                (nvariables(equations), nelements(mesh, solver, cache)))
end

function rhs!(du, u, t, mesh, equations, initial_condition, boundary_conditions, source_terms::Source, solver::FVMuscl, cache) where {Source}
    u_ = update_solution(mesh, u, solver, cache)

    du .= zero(eltype(du))

	for element in eachelement(mesh, solver)
		@unpack volume, face_normals, num_faces, face_areas, face_connectivity = cache.elements[element]
		for face in 1:num_faces
			neighbor = face_connectivity[face]
			normal = @views([face_normals[2 * face - 1 : 2 * face]...]) # Unfortunaly, flux() requires an Vector and no Tuple
            du[element] += - face_areas[face] * solver.surface_flux(u_[element].u[1]#=TODO for system of equations=#,
                                                                    u_[neighbor].u[1], normal, equations)
		end
		du[element] = (1 / volume) * du[element]
	end

	return nothing
end

function get_element_variables!(element_variables, u, mesh::T8codeMesh, equations, solver, cache)
    return nothing
end

# Container data structures
include("containers.jl")
end # @muladd
