function build_hSBP_ops(rd::RefElemData{1})
        @unpack M,Dr,Vq,Pq,Vf,wf,nrJ = rd
        Qr = Pq'*M*Dr*Pq
        Ef = Vf*Pq
        Br = diagm(wf.*nrJ)
        Qrh = .5*[Qr-Qr' Ef'*Br;
                -Br*Ef  Br]
        Vh = [Vq;Vf]
        Ph = M\transpose(Vh)
        VhP = Vh*Pq

        # make skew symmetric versions of the operators"
        Qrhskew = .5*(Qrh-transpose(Qrh))
        return Qrhskew,VhP,Ph
    end

function build_hSBP_ops(rd::RefElemData{2})
    @unpack M,Dr,Ds,Vq,Pq,Vf,wf,nrJ,nsJ = rd
    Qr = Pq'*M*Dr*Pq
    Qs = Pq'*M*Ds*Pq
    Ef = Vf*Pq
    Br = diagm(wf.*nrJ)
    Bs = diagm(wf.*nsJ)
    Qrh = .5*[Qr-Qr' Ef'*Br;
            -Br*Ef  Br]
    Qsh = .5*[Qs-Qs' Ef'*Bs;
            -Bs*Ef  Bs]
    Vh = [Vq;Vf]
    Ph = M\transpose(Vh)
    VhP = Vh*Pq

    # make skew symmetric versions of the operators"
    Qrhskew = .5*(Qrh-transpose(Qrh))
    Qshskew = .5*(Qsh-transpose(Qsh))
    return Qrhskew,Qshskew,VhP,Ph
end

# accumulate Q.*F into rhs
function hadsum_ATr!(rhs,ATr,F,u; skip_index=(i,j)->false)
    rows,cols = axes(ATr)
    for i in cols
        ui = u[i]
        val_i = rhs[i]
        for j in rows
            if !skip_index(i,j)
                val_i += ATr[j,i].*F(ui,u[j]) # breaks for tuples, OK for StaticArrays
            end
        end
        rhs[i] = val_i # why not .= here?
    end
end


# mesh data structure
struct UnstructuredMesh{NDIMS,Tv,Ti}
        VXYZ::NTuple{NDIMS,Tv}
        EToV::Matrix{Ti}
end
function Base.show(io::IO, mesh::UnstructuredMesh{NDIMS}) where {NDIMS}
        @nospecialize mesh
        println("Unstructured mesh in $NDIMS dimensions.")
end



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





## ====== workaround for StructArrays/DiffEq.jl from https://github.com/SciML/OrdinaryDiffEq.jl/issues/1386
function RecursiveArrayTools.recursivecopy(a::AbstractArray{T,N}) where {T<:AbstractArray,N}
        if ArrayInterface.ismutable(a)
                b = similar(a)
                map!(recursivecopy,b,a)
        else
                ArrayInterface.restructure(a,map(recursivecopy,a))
        end
end
ArrayInterface.ismutable(x::StructArray) = true
## ====== end workaround ===========





################## interface stuff #################

Base.real(rd::RefElemData) = Float64 # is this for DiffEq.jl?

Trixi.ndims(mesh::UnstructuredMesh) = length(mesh.VXYZ)
function Trixi.allocate_coefficients(mesh::UnstructuredMesh,
                    equations, rd::RefElemData, cache)
    @unpack md = cache
    NVARS = nvariables(equations) # TODO: replace with static type info?
    return StructArray([SVector{NVARS}(zeros(NVARS)) for i in axes(md.x,1), j in axes(md.x,2)])
end
function Trixi.compute_coefficients!(u::StructArray, initial_condition, t,
                                     mesh::UnstructuredMesh, equations, rd::RefElemData, cache)
    for i = 1:length(cache.md.x) # loop over nodes
        xyz_i = getindex.(cache.md.xyz,i)
        u[i] = initial_condition(xyz_i,t,equations) # interpolate
    end
end
Trixi.wrap_array(u_ode::StructArray, semi::Trixi.AbstractSemidiscretization) = u_ode
Trixi.wrap_array(u_ode::StructArray, mesh::UnstructuredMesh, equations, solver, cache) = u_ode
Trixi.ndofs(mesh::UnstructuredMesh, rd::RefElemData, cache) = length(rd.r)*cache.md.K

# Enable timings, see src/auxiliary/auxiliary.jl
timeit_debug_enabled() = true
