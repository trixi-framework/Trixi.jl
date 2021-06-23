# !!! warning "Experimental features"

# todo: replace mul! by Octavian.matmul!
# todo: add @threaded 

# out <- A*x
mul_by!(A) = let A = A 
    @inline (out, x)->mul!(out, A, x) 
end

# out <- out + A * x 
mul_by_accum!(A) = let A = A 
    @inline (out, x)->mul!(out, A, x, one(eltype(out)), one(eltype(out))) 
end

#  out <- out + α * A * x 
mul_by_accum!(A, α) = let A = A 
    @inline (out, x)->mul!(out, A, x, α, one(eltype(out))) 
end

const DGWeakForm{Dims, ElemType} = DG{<:RefElemData{Dims, ElemType}, Mortar, 
                                      <:SurfaceIntegralWeakForm,
                                      <:VolumeIntegralWeakForm} where {Mortar}

# this is necessary for pretty printing
Base.real(rd::RefElemData{Dims, Elem, ApproxType, Nfaces, RealT}) where {Dims, Elem, ApproxType, Nfaces, RealT} = RealT

function Trixi.ndofs(mesh::AbstractMeshData, dg::DG{<:RefElemData}, cache)
    rd = dg.basis
    return rd.Np * mesh.md.num_elements
end
Trixi.wrap_array(u_ode::StructArray, semi::Trixi.AbstractSemidiscretization) = u_ode
Trixi.wrap_array(u_ode::StructArray, mesh::AbstractMeshData, equations, dg::DG{<:RefElemData}, cache) = u_ode

# interface with semidiscretization_hyperbolic
function Trixi.digest_boundary_conditions(boundary_conditions::NamedTuple{Keys,ValueTypes}, 
                                          mesh::AbstractMeshData, dg::DG{<:RefElemData}, cache) where {Keys,ValueTypes<:Tuple{Any,Any}}
    return boundary_conditions
end

function allocate_coefficients(mesh::AbstractMeshData, equations, dg::DG{<:RefElemData}, cache)
    md = mesh.md
    nvars = nvariables(equations) 
    return StructArray{SVector{nvars, real(dg)}}(ntuple(_->similar(md.x),nvars))
end

function compute_coefficients!(u::StructArray, initial_condition, t,
                               mesh::AbstractMeshData{Dim}, equations, dg::DG{<:RefElemData{Dim}}, cache) where {Dim}
    md = mesh.md
    rd = dg.basis
    @unpack u_values = cache

    num_quad_nodes = length(u_values) 
    for i in Base.OneTo(num_quad_nodes) 
        xyz_i = SVector{Dim}(getindex.(md.xyzq, i))
        u_values[i] = initial_condition(xyz_i, t, equations) 
    end

    # compute L2 projection
    StructArrays.foreachfield(mul_by!(rd.Pq), u, u_values)
end

# interpolates from solution coefficients to face quadrature points
function prolong2interfaces!(cache, u, mesh::AbstractMeshData, equations, 
                             surface_integral, dg::DG{<:RefElemData})
    rd = dg.basis        
    @unpack u_face_values = cache
    StructArrays.foreachfield(mul_by!(rd.Vf), u_face_values, u)
end

function create_cache(mesh::VertexMappedMesh, equations, dg::DG, RealT, uEltype) where {DG <: DGWeakForm{2}}

    rd = dg.basis
    md = mesh.md
    # mass matrix, differentiation matrices
    @unpack M, Dr, Ds = rd
    # volume quadrature weights, volume interpolation matrix
    @unpack wq, Vq = rd 

    # ∫f(u) * dv/dx_i = (Vq*Dr)'*diagm(wq)*(rxJ.*f(Vq*u)) + (Vq*Ds)'*diagm(wq)*(sxJ.*f(Vq*u))
    invMQrTrW = -M\((Vq*Dr)'*diagm(wq))
    invMQsTrW = -M\((Vq*Ds)'*diagm(wq))

    nvars = nvariables(equations)

    # storage for volume quadrature values, face quadrature values, flux values
    u_values = StructArray{SVector{nvars, RealT}}(ntuple(_->zeros(rd.Nq, md.num_elements), nvars))
    u_face_values = StructArray{SVector{nvars, RealT}}(ntuple(_->zeros(rd.Nfq, md.num_elements), nvars))
    flux_face_values = similar(u_face_values)

    # local storage for fluxes
    flux_values = StructArray{SVector{nvars,RealT}}(ntuple(_->zeros(rd.Nq), nvars))
    
    return (; md, invMQrTrW, invMQsTrW, invJ=inv.(md.J),
            u_values, flux_values, u_face_values, flux_face_values)
end

function calc_volume_integral!(du,u::StructArray, volume_integral::VolumeIntegralWeakForm,
                               mesh::VertexMappedMesh, equations, dg::DG{<:RefElemData{2}}, cache)

    rd = dg.basis  
    md = mesh.md
    @unpack invMQrTrW, invMQsTrW, u_values, flux_values = cache
    @unpack rxJ, sxJ, ryJ, syJ = md # geometric terms

    # interpolate to quadrature points
    StructArrays.foreachfield(mul_by!(rd.Vq), u_values, u)

    # todo: dispatch on curved/non-curved mesh types. This only works for affine meshes (accessing rxJ[1,e],...)
    for e in Base.OneTo(md.num_elements)
        flux_values .= flux.(view(u_values,:,e), 1, equations)
        StructArrays.foreachfield(mul_by_accum!(invMQrTrW, rxJ[1,e]), view(du,:,e), flux_values)
        StructArrays.foreachfield(mul_by_accum!(invMQsTrW, sxJ[1,e]), view(du,:,e), flux_values)

        flux_values .= flux.(view(u_values,:,e),2,equations)
        StructArrays.foreachfield(mul_by_accum!(invMQrTrW, ryJ[1,e]), view(du,:,e), flux_values)
        StructArrays.foreachfield(mul_by_accum!(invMQsTrW, syJ[1,e]), view(du,:,e), flux_values)
    end
end

# calc_interface_flux!(cache, dg.surface_integral, mesh, equations, dg)
function calc_interface_flux!(cache, surface_integral::SurfaceIntegralWeakForm, 
                              mesh::VertexMappedMesh, equations, 
                              dg::DG{<:RefElemData{2}}) 

    @unpack surface_flux = surface_integral
    md = mesh.md
    @unpack mapM, mapP, nxJ, nyJ, sJ = md 
    @unpack u_face_values, flux_face_values = cache 

    num_face_nodes = length(u_face_values)
    for i in Base.OneTo(num_face_nodes)

        # inner (idM -> minus) and outer (idP -> plus) indices 
        idM, idP = mapM[i], mapP[i]        
        uM = u_face_values[idM]
            
        # compute flux if node is not a boundary node
        if idM != idP
            uP = u_face_values[idP]
            normal = SVector{2}(nxJ[idM], nyJ[idM]) / sJ[idM]            
            flux_face_values[idM] = surface_flux(uM, uP, normal, equations) * sJ[idM]
        end
    end
end

# assumes cache.flux_face_values is computed and filled with 
# for polyomial discretizations, use dense LIFT matrix for surface contributions.
function calc_surface_integral!(du, u, surface_integral::SurfaceIntegralWeakForm, 
                                mesh::VertexMappedMesh, equations, 
                                dg::DG{<:RefElemData{2}}, cache) 
    rd = dg.basis
    StructArrays.foreachfield(mul_by_accum!(rd.LIFT), du, cache.flux_face_values)
end

# Specialize for nodal SBP discretizations. Uses that Vf*u = u[Fmask,:] 
function prolong2interfaces!(cache, u, mesh::AbstractMeshData, equations, surface_integral, 
                             dg::DG{<:RefElemData{Dim, <:AbstractElemShape, SBP}}) where {Dim}
    rd = dg.basis        
    @unpack Fmask = rd
    @unpack u_face_values = cache
    StructArrays.foreachfield((out, u)->out .= view(u, Fmask, :), u_face_values, u)
end

# # Specialize for nodal SBP discretizations. Uses that du = LIFT*u is equivalent to 
# # du[Fmask,:] .= u ./ rd.wq[rd.Fmask] 
function calc_surface_integral!(du, u, surface_integral::SurfaceIntegralWeakForm, 
                                mesh::VertexMappedMesh, equations, 
                                dg::DG{<:RefElemData{2,<:AbstractElemShape, SBP}}, cache) 
    rd = dg.basis
    md = mesh.md
    @unpack flux_face_values = cache
    for e in Base.OneTo(md.num_elements)
        for i in Base.OneTo(rd.Nfq)
            du[rd.Fmask[i],e] += flux_face_values[i,e] * rd.wf[i] / rd.wq[rd.Fmask[i]]
        end        
    end
end  

# do nothing for periodic (default) boundary conditions
calc_boundary_flux!(cache, t, boundary_conditions::BoundaryConditionPeriodic, 
                    mesh, equations, dg::DG{<:RefElemData}) = nothing

# "lispy tuple programming" instead of for loop for type stability
function calc_boundary_flux!(cache, t, boundary_conditions, mesh, equations, dg::DG{<:RefElemData})       

    # peel off first boundary condition
    calc_single_boundary_flux!(cache, t, first(boundary_conditions), first(keys(boundary_conditions)), 
                               mesh, equations, dg)

    # recurse on the remainder of the boundary conditions                            
    calc_boundary_flux!(cache, t, Base.tail(boundary_conditions), mesh, equations, dg)
end

# terminate recursion
calc_boundary_flux!(cache, t, boundary_conditions::NamedTuple{(),Tuple{}}, 
                    mesh, equations, dg::DG{<:RefElemData}) = nothing

function calc_single_boundary_flux!(cache, t, boundary_condition, boundary_key, 
                                    mesh, equations, dg::DG{<:RefElemData{Dim}}) where {Dim}
    
    rd = dg.basis
    md = mesh.md
    @unpack u_face_values, flux_face_values = cache    
    @unpack xyzf, nxyzJ, sJ = md
    @unpack surface_flux = dg.surface_integral

    # reshape face/normal arrays to have size = (num_points_on_face, num_faces_total).
    # mesh.boundary_faces indexes into the columns of these face-reshaped arrays.
    num_pts_per_face = rd.Nfq ÷ rd.Nfaces
    num_faces_total = rd.Nfaces * md.num_elements     
    reshape_by_face(u) = reshape(u, num_pts_per_face, num_faces_total)
    u_face_values = reshape_by_face(u_face_values)
    flux_face_values = reshape_by_face(flux_face_values)
    sJ            = reshape_by_face(sJ)
    nxyzJ, xyzf   = reshape_by_face.(nxyzJ), reshape_by_face.(xyzf) # broadcast over nxyzJ::NTuple{Dim,Matrix}
        
    # loop through boundary faces, which correspond to columns of reshaped u_face_values, ...
    for f in mesh.boundary_faces[boundary_key]
        for i in Base.OneTo(num_pts_per_face)            
            face_normal = SVector{Dim}(getindex.(nxyzJ, i, f)) / sJ[i,f]
            face_coordinates = SVector{Dim}(getindex.(xyzf, i, f))
            flux_face_values[i,f] = boundary_condition(u_face_values[i,f], 
                                                    face_normal, face_coordinates, t,
                                                    surface_flux, equations) * sJ[i,f]
        end
    end
    
    # Note: modifying the values of the reshaped array modifies the values of cache.flux_face_values. 
    # However, we don't have to re-reshape, since cache.flux_face_values still retains its original shape.
end
   
# todo: specialize for modal DG on curved meshes using WADG
function invert_jacobian!(du, mesh::Mesh, equations, dg::DG{<:RefElemData}, 
                                  cache) where {Mesh <: AbstractMeshData} 
    for i in Base.OneTo(ndofs(mesh,dg,cache))
        du[i] *= -cache.invJ[i]
    end
end

calc_sources!(du, u, t, source_terms::Nothing, 
              mesh::VertexMappedMesh, equations, dg::DG{<:RefElemData}, cache) = nothing

# uses quadrature + projection to compute source terms. 
# todo: use interpolation here instead of quadrature? Would be cheaper. 
function calc_sources!(du, u, t, source_terms, 
                       mesh::VertexMappedMesh, equations, dg::DG{<:RefElemData}, cache) 

    rd = dg.basis
    md = mesh.md
    @unpack Pq = rd
    @unpack u_values, flux_values = cache
    for e in Base.OneTo(md.num_elements)
        u_e = view(u_values, :, e) # u_values should already be computed from volume kernel

        # todo: why do these lines allocate?
        for i in Base.OneTo(rd.Nq)
            flux_values[i] = source_terms(u_e[i], getindex.(md.xyzq, i, e), t, equations) 
        end
        StructArrays.foreachfield(mul_by_accum!(Pq), view(du, :, e), flux_values)
    end
end

function rhs!(du, u, t, mesh, equations, 
              initial_condition, boundary_conditions, source_terms,
              dg::DG{<:RefElemData}, cache)

    @trixi_timeit timer() "Reset du/dt" fill!(du,zero(eltype(du)))

    @trixi_timeit timer() "calc_volume_integral!" calc_volume_integral!(du, u, dg.volume_integral, 
                                                                        mesh, equations, dg, cache)

    @trixi_timeit timer() "prolong2interfaces!" prolong2interfaces!(cache, u, mesh, equations, dg.surface_integral, dg)

    @trixi_timeit timer() "calc_interface_flux!" calc_interface_flux!(cache, dg.surface_integral, mesh, equations, dg)

    @trixi_timeit timer() "calc_boundary_flux!" calc_boundary_flux!(cache, t, boundary_conditions, mesh, equations, dg)

    @trixi_timeit timer() "calc_surface_integral!" calc_surface_integral!(du, u, dg.surface_integral, mesh, equations, dg, cache)
    
    @trixi_timeit timer() "invert_jacobian" invert_jacobian!(du, mesh, equations, dg, cache)

    @trixi_timeit timer() "calc_sources!" calc_sources!(du, u, t, source_terms, mesh, equations, dg, cache)

    return nothing
end

