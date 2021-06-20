# !!! warning "Experimental features"

# todo: replace mul! by Octavian.matmul!
# out <- A*x
mul_by!(A) = let A=A 
    @inline (out,x)->mul!(out,A,x) 
end

# out <- out + A * x 
mul_by_accum!(A) = let A=A 
    @inline (out,x)->mul!(out,A,x,one(eltype(out)),one(eltype(out))) 
end

#  out <- out + α * A * x 
mul_by_accum!(A,α) = let A=A 
    @inline (out,x)->mul!(out,A,x,α,one(eltype(out))) 
end

const ModalDG{Dims,Elem} = DG{<:RefElemData{Dims,Elem,Polynomial},Tuple{}} where {Dims,Elem}
const DGWeakForm{Dims,ElemType} = DG{<:RefElemData{Dims,ElemType},Mortar,<:SurfaceIntegralWeakForm{<:FluxPlusDissipation},
                       <:VolumeIntegralWeakForm} where {Mortar}

# is this necessary? seems to be for printing?
Base.real(rd::RefElemData{Dims,Elem,ApproxType,Nfaces,RealT}) where {Dims,Elem,ApproxType,Nfaces,RealT} = RealT
function Trixi.ndofs(mesh::AbstractMeshData, dg::DG{<:RefElemData}, cache)
    rd = dg.basis
    return rd.Np * mesh.md.num_elements
end
Trixi.wrap_array(u_ode::StructArray, semi::Trixi.AbstractSemidiscretization) = u_ode
Trixi.wrap_array(u_ode::StructArray, mesh::AbstractMeshData, equations, dg::DG{<:RefElemData}, cache) = u_ode

function allocate_coefficients(mesh::AbstractMeshData, equations, dg::DG{<:RefElemData}, cache)
    md = mesh.md
    nvars = nvariables(equations) 
    return StructArray{SVector{nvars,real(dg)}}(ntuple(_->similar(md.x),nvars))
end

function compute_coefficients!(u::StructArray, initial_condition, t,
                               mesh::AbstractMeshData, equations, dg::DG{<:RefElemData}, cache)
    md = mesh.md
    for i in Base.OneTo(ndofs(mesh,dg,cache)) 
        xyz_i = getindex.(md.xyz,i)
        u[i] = initial_condition(xyz_i,t,equations) # todo: switch to L2 projection?
    end
end

function prolong2interfaces!(cache, u, mesh::AbstractMeshData, equations, 
                             surface_integral, dg::ModalDG)
    rd = dg.basis        
    @unpack u_face_values = cache
    StructArrays.foreachfield(mul_by!(rd.Vf), u_face_values, u)
end

function create_cache(mesh::VertexMappedMesh, equations, dg::DG, RealT, uEltype) where {DG <: DGWeakForm{2,Tri}}

    rd = dg.basis
    md = mesh.md
    @unpack M,Dr,Ds,wq,Vq = rd

    # ∫f(u) * dv/dx_i = (Vq*Dr)'*diagm(wq)*(rxJ.*f(Vq*u)) + (Vq*Ds)'*diagm(wq)*(sxJ.*f(Vq*u))
    invMQrTrW = -M\((Vq*Dr)'*diagm(wq))
    invMQsTrW = -M\((Vq*Ds)'*diagm(wq))

    nvars = nvariables(equations)

    # local volume quadrature storage
    u_values = StructArray{SVector{nvars,RealT}}(ntuple(_->zeros(rd.Nq,md.num_elements),nvars))
    u_face_values = StructArray{SVector{nvars,RealT}}(ntuple(_->zeros(rd.Nfq,md.num_elements),nvars))

    # local storage for fluxes
    f_values = StructArray{SVector{nvars,RealT}}(ntuple(_->zeros(rd.Nq),nvars))
    f_face_values = StructArray{SVector{nvars,RealT}}(ntuple(_->zeros(rd.Nfq),nvars))

    invJ = inv.(md.J)
    return (; md, invMQrTrW,invMQsTrW, invJ,
            u_values, f_values, u_face_values, f_face_values)
end

function calc_volume_integral!(du,u::StructArray,nonconservative_terms::Val{false}, 
                               volume_integral::VolumeIntegralWeakForm,
                               mesh::VertexMappedMesh, equations, dg::ModalDG{2}, cache)

    rd = dg.basis  
    md = mesh.md
    @unpack invMQrTrW,invMQsTrW,u_values,f_values = cache
    @unpack rxJ,sxJ,ryJ,syJ = md

    # interpolate to quadrature points
    StructArrays.foreachfield(mul_by!(rd.Vq),u_values,u)

    # todo: dispatch on curved/non-curved mesh types. This only works for affine meshes (accessing rxJ[1,e],...)
    for e in Base.OneTo(md.num_elements)
        f_values .= flux.(view(u_values,:,e),1,equations)
        StructArrays.foreachfield(mul_by_accum!(invMQrTrW,rxJ[1,e]),view(du,:,e),f_values)
        StructArrays.foreachfield(mul_by_accum!(invMQsTrW,sxJ[1,e]),view(du,:,e),f_values)

        f_values .= flux.(view(u_values,:,e),2,equations)
        StructArrays.foreachfield(mul_by_accum!(invMQrTrW,ryJ[1,e]),view(du,:,e),f_values)
        StructArrays.foreachfield(mul_by_accum!(invMQsTrW,syJ[1,e]),view(du,:,e),f_values)
    end
end

function calc_surface_integral!(du, u, mesh::VertexMappedMesh, equations, 
                                surface_integral::SurfaceIntegralWeakForm{<:FluxPlusDissipation}, 
                                dg::ModalDG{2}, cache) 
    rd = dg.basis
    md = mesh.md
    @unpack LIFT = rd
    @unpack mapP,nxJ,nyJ,sJ = md
    @unpack u_face_values, f_face_values = cache
    @unpack surface_flux = surface_integral
    @unpack numerical_flux, dissipation = surface_flux
    
    for e in Base.OneTo(md.num_elements) # todo: replace with eachelement(...)?
        for i in Base.OneTo(rd.Nfq)
            uM = u_face_values[i,e]
            uP = u_face_values[mapP[i,e]]
            normal = SVector{2}(nxJ[i,e],nyJ[i,e]) / sJ[i,e]
            f_face_values[i] = numerical_flux(uM, uP, 1, equations) * nxJ[i,e] + 
                               numerical_flux(uM, uP, 2, equations) * nyJ[i,e] - 
                               dissipation(uM, uP, normal, equations) * sJ[i,e] # orientation = 1 = arbitrary
        end
        StructArrays.foreachfield(mul_by_accum!(LIFT),view(du,:,e),f_face_values)
    end
end

# todo: specialize for modal DG on curved meshes using WADG
negate_and_invert_jacobian!(du, mesh::AbstractMeshData, equations, 
                            dg::DG{<:RefElemData}, cache) = du .*= cache.invJ

calc_sources!(du, u, t, source_terms::Nothing, mesh::AbstractMeshData, equations::AbstractEquations{Dims,NVARS}, 
              dg::DG{<:RefElemData}, cache) where {Dims,NVARS} = nothing

# uses quadrature + projection to compute source terms. 
# todo: use interpolation here instead of quadrature? 
function calc_sources!(du, u, t, source_terms, mesh::AbstractMeshData, equations::AbstractEquations{2,NVARS}, 
                       dg::DG{<:RefElemData}, cache) where {NVARS}
    rd = dg.basis
    @unpack Pq = rd
    @unpack u_values, f_values = cache
    md = mesh.md
    for e in Base.OneTo(md.num_elements)
        u_e = view(u_values, :, e) # u_values should already be computed from volume kernel
        xy_e = StructArray{SVector{2,Float64}}(view.(md.xyzq, :, e))
        f_values .= source_terms.(u_e, xy_e, t, equations) # reuse f_values 
        StructArrays.foreachfield(mul_by_accum!(Pq), view(du,:,e), f_values)
    end
end

function calc_boundary_states!(cache,t,boundary_conditions,mesh,equations,dg::DG{<:RefElemData})
    return nothing
end

function rhs!(du, u, t, mesh, equations, 
              initial_condition, boundary_conditions, source_terms,
              dg::DG{<:RefElemData}, cache)

    @trixi_timeit timer() "Reset du/dt" fill!(du,zero(eltype(du)))

    @trixi_timeit timer() "calc_volume_integral!" calc_volume_integral!(du, u, dg.volume_integral, 
                                                                        mesh, equations, dg, cache)
    @trixi_timeit timer() "prolong2interfaces!" prolong2interfaces!(cache, u, mesh, equations, dg.surface_integral, dg)

    # todo: implement boundary conditions
    @trixi_timeit timer() "calc_boundary_states!" calc_boundary_states!(cache, t, boundary_conditions, mesh, equations, dg)

    @trixi_timeit timer() "calc_surface_integral!" calc_surface_integral!(du, u, dg.surface_integral, mesh, equations, dg, cache)
    
    @trixi_timeit timer() "invert_jacobian" invert_jacobian!(du, mesh, equations, dg, cache)

    @trixi_timeit timer() "calc_sources!" calc_sources!(du, u, t, source_terms, mesh, equations, dg, cache)

    return nothing
end

