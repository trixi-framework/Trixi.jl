# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

struct FVMuscl{SurfaceFlux, Limiter}
    surface_flux::SurfaceFlux
    limiter::Limiter
    
    function FVMuscl(; surface_flux, limiter="TODO")
        new{typeof(surface_flux), typeof(limiter)}(surface_flux, limiter)
    end
end

@inline Base.real(solver::FVMuscl) = Float64 # TODO
@inline ndofs(mesh, solver::FVMuscl, cache) = nelements_global(mesh, solver, cache)

@inline nelements(mesh, solver::FVMuscl, cache) = nelements(solver, cache)
@inline nelements(solver::FVMuscl, cache) = nelements(cache.elements)

function compute_coefficients!(u, func, t, mesh::AbstractMesh, equations, solver::FVMuscl,
                               cache)
    @threaded for element in eachelement(mesh, solver, cache)
        x_node = mesh.elements[element].midpoint
        u_node = func(x_node, t, equations)
        u[element] = u_node
    end
end

function allocate_coefficients(mesh::AbstractMesh, equations, solver::FVMuscl, cache)
    # We must allocate a `Vector` in order to be able to `resize!` it (AMR).
    # cf. wrap_array
    zeros(eltype(mesh.elements[1].volume),
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

# Container data structures
include("containers.jl")
end # @muladd
