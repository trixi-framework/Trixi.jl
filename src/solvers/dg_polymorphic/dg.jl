# todo: replace mul! by Octavian.matmul!

# C <- A*x
mul_by!(A) = let A=A 
    (out,x)->mul!(out,A,x) 
end

# C <- C + A * x 
mul_by_accum!(A) = let A=A 
    (out,x)->mul!(out,A,x,one(eltype(out)),one(eltype(out))) 
end

#  C <- C + α * A * x 
mul_by_accum!(A,α) = let A=A 
    (out,x)->mul!(out,A,x,α,one(eltype(out))) 
end

const ModalDG{Dims,Elem} = DG{<:RefElemData{Dims,Elem,Polynomial},Tuple{}} where {Dims,Elem}
# const SBPDG{Dims,Elem} = DG{<:RefElemData{Dims,Elem,SBP},Tuple{}} where {Dims,Elem}

# is this necessary? seems to be for printing?
Base.real(rd::RefElemData{Dims,Elem,ApproxType,Nfaces,RealT}) where {Dims,Elem,ApproxType,Nfaces,RealT} = RealT
Trixi.ndofs(md::MeshData, rd::RefElemData, cache) = rd.Np * md.num_elements
Trixi.ndims(md::MeshData{NDIMS}) where {NDIMS} = NDIMS
Trixi.wrap_array(u_ode::StructArray, semi::Trixi.AbstractSemidiscretization) = u_ode
Trixi.wrap_array(u_ode::StructArray, md::MeshData, equations, solver, cache) = u_ode

function allocate_coefficients(md::MeshData, equations, dg::DG{<:RefElemData}, cache)
    nvars = nvariables(equations) 
    return StructArray{SVector{nvars,real(dg)}}(ntuple(_->similar(md.x),nvars))
end

function compute_coefficients!(u::StructArray, initial_condition, t,
                               md::MeshData, equations, dg::DG{<:RefElemData}, cache)
    for i in Base.OneTo(length(md.x)) # loop over nodes
        xyz_i = getindex.(md.xyz,i)
        u[i] = initial_condition(xyz_i,t,equations) # interpolate - add projection later?
    end
end

# todo: implement for FluxDifferencing and WeakForm
const DG_WeakForm = DG{<:RefElemData,Mortar,<:SurfaceIntegralWeakForm,<:VolumeIntegralWeakForm} where {Mortar}

function create_cache(md::MeshData, equations, dg::DG, RealT, uEltype) where {DG <: DG_WeakForm}
    rd = dg.basis
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
    return (; invMQrTrW,invMQsTrW, u_values, f_values, u_face_values, f_face_values, invJ)
end

function calc_volume_integral!(du,u::StructArray, md::MeshData{2},
                               nonconservative_terms::Val{false}, equations,
                               volume_integral::VolumeIntegralWeakForm, dg::ModalDG{2}, cache)

    @unpack invMQrTrW,invMQsTrW,u_values,f_values = cache
    rd = dg.basis  
    @unpack rxJ,sxJ,ryJ,syJ = md

    # interpolate to quadrature points
    StructArrays.foreachfield(mul_by!(rd.Vq),u_values,u)

    # todo: need to dispatch on curved/non-curved meshes. Maybe put Meshdata in cache instead?
    for e in Base.OneTo(md.num_elements)
        f_values .= flux.(view(u_values,:,e),1,equations)
        StructArrays.foreachfield(mul_by_accum!(invMQrTrW,rxJ[1,e]),view(du,:,e),f_values)
        StructArrays.foreachfield(mul_by_accum!(invMQsTrW,sxJ[1,e]),view(du,:,e),f_values)

        f_values .= flux.(view(u_values,:,e),2,equations)
        StructArrays.foreachfield(mul_by_accum!(invMQrTrW,ryJ[1,e]),view(du,:,e),f_values)
        StructArrays.foreachfield(mul_by_accum!(invMQsTrW,syJ[1,e]),view(du,:,e),f_values)
    end
end

function prolong2interfaces!(cache, u, md::MeshData, equations, 
                             surface_integral, dg::ModalDG)    
    rd = dg.basis        
    @unpack u_face_values = cache
    StructArrays.foreachfield(mul_by!(rd.Vf), u_face_values, u)
end

function calc_surface_integral!(du, u, md::MeshData, equations, 
                                surface_integral::SurfaceIntegralWeakForm, dg::ModalDG, cache) 
    rd = dg.basis
    @unpack LIFT = rd
    @unpack mapP,nxJ,nyJ,sJ = md
    @unpack u_face_values, f_face_values = cache
    @unpack surface_flux = surface_integral
    for e in Base.OneTo(md.num_elements) # todo: replace with eachelement(...)?
        for i in Base.OneTo(rd.Nfq)
            uM = u_face_values[i,e]
            uP = u_face_values[mapP[i,e]]
            f_face_values[i] = surface_flux(uM, uP, 1, equations)*nxJ[i,e] + 
                               surface_flux(uM, uP, 2, equations)*nyJ[i,e]
            # todo: add dissipation * sJ term
        end
        StructArrays.foreachfield(mul_by_accum!(LIFT),view(du,:,e),f_face_values)
    end
end

# todo: specialize. this doesn't work for modal DG on curved meshes. 
invert_jacobian!(du, md::MeshData, equations, dg::DG{<:RefElemData}, cache) = du .*= cache.invJ

function rhs!(du, u, t,
              mesh::MeshData, equations, 
              initial_condition, boundary_conditions, source_terms,
              dg::DG{<:RefElemData}, cache)

    fill!(du,zero(eltype(du)))
    calc_volume_integral!(du, u, mesh, Trixi.have_nonconservative_terms(equations), equations,
                          dg.volume_integral, dg, cache)
    prolong2interfaces!(cache, u, mesh, equations, dg.surface_integral, dg)
    calc_surface_integral!(du, u, mesh, equations, dg.surface_integral, dg, cache)
    invert_jacobian!(du, mesh, equations, dg, cache)

    return nothing
end

