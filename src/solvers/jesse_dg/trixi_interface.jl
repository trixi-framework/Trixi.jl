
include("unstructured_mesh.jl")

# =================== threading utilities ===================
@inline function tmap!(f,out,x)
        Trixi.@threaded for i in eachindex(x)
            @inbounds out[i] = f(x[i])
        end
end
## workaround for matmul! with threads https://discourse.julialang.org/t/odd-benchmarktools-timings-using-threads-and-octavian/59838/5
@inline function bmap!(f,out,x)
        @batch for i in eachindex(x)
                @inbounds out[i] = f(x[i])
        end
end
# ==========================================================

Base.real(rd::RefElemData) = Float64 # is this for DiffEq.jl?
Trixi.ndofs(mesh::UnstructuredMesh, rd::RefElemData, cache) = length(rd.r)*cache.md.K
Trixi.wrap_array(u_ode::StructArray, semi::Trixi.AbstractSemidiscretization) = u_ode
Trixi.wrap_array(u_ode::StructArray, mesh::UnstructuredMesh, equations, solver, cache) = u_ode
Trixi.ndims(mesh::UnstructuredMesh{NDIMS}) where {NDIMS} = NDIMS


# Enable timings, see src/auxiliary/auxiliary.jl
timeit_debug_enabled() = false
