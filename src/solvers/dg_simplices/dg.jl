# !!! warning "Experimental features"

# out <- A*x
mul_by!(A) = let A = A 
  @inline (out, x)->matmul!(out, A, x) 
end

# specialize for SBP operators since matmul! doesn't work for `UniformScaling` types
mul_by!(A::UniformScaling) = let A = A
  @inline (out, x)->out .= x
end
  
# # Todo: simplices. Use `matmul!` for the following 2 functions until 5-arg `matmul!` once 
# the hanging bug is fixed (see https://github.com/JuliaLinearAlgebra/Octavian.jl/issues/103). 

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

eachdim(mesh::AbstractMeshData{Dim}, dg::DG{<:RefElemData{Dim}}, cache) where {Dim} = Base.OneTo(Dim)

# iteration over all elements in a mesh
ndofs(mesh::AbstractMeshData, dg::DG{<:RefElemData}, cache) = dg.basis.Np * mesh.md.num_elements
eachelement(mesh::AbstractMeshData, dg::DG{<:RefElemData}, cache) = Base.OneTo(mesh.md.num_elements)

# iteration over quantities in a single element
each_face_node(mesh::AbstractMeshData, dg::DG{<:RefElemData}, cache) = Base.OneTo(dg.basis.Nfq)
each_quad_node(mesh::AbstractMeshData, dg::DG{<:RefElemData}, cache) = Base.OneTo(dg.basis.Nq)

# iteration over quantities over the entire mesh (dofs, quad nodes, face nodes). 
each_dof_global(mesh::AbstractMeshData, dg::DG{<:RefElemData}, cache) = Base.OneTo(ndofs(mesh, dg, cache))
each_quad_node_global(mesh::AbstractMeshData, dg::DG{<:RefElemData}, cache) = Base.OneTo(dg.basis.Nq * mesh.md.num_elements)
each_face_node_global(mesh::AbstractMeshData, dg::DG{<:RefElemData}, cache) = Base.OneTo(dg.basis.Nfq * mesh.md.num_elements)

# interface with semidiscretization_hyperbolic
wrap_array(u_ode::StructArray, mesh::AbstractMeshData, equations, dg::DG{<:RefElemData}, cache) = u_ode
function digest_boundary_conditions(boundary_conditions::NamedTuple{Keys,ValueTypes}, mesh::AbstractMeshData, 
                                    dg::DG{<:RefElemData}, cache) where {Keys,ValueTypes<:NTuple{N,Any}} where {N}
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

  @threaded for i in each_quad_node_global(mesh, dg, cache)    
    u_values[i] = initial_condition(getindex.(md.xyzq, i), t, equations) 
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

function create_cache(mesh::VertexMappedMesh, equations, dg::DG, 
                      RealT, uEltype) where {DG <: DGWeakForm{Dim}} where {Dim}

  rd = dg.basis
  md = mesh.md

  # volume quadrature weights, volume interpolation matrix
  @unpack wq, Vq = rd 

  # mass matrix, differentiation matrices
  @unpack M, Drst = rd

  # ∫f(u) * dv/dx_i = ∑_j (Vq*D_i)'*diagm(wq)*(rstxyzJ[i,j].*f(Vq*u))
  invMQrstTrW = map(D -> -M\((Vq*D)'*diagm(wq)), Drst)

  nvars = nvariables(equations)

  # Todo: simplices. Factor common storage into a struct (MeshDataCache?) for reuse across solvers?
  # storage for volume quadrature values, face quadrature values, flux values
  u_values = StructArray{SVector{nvars, uEltype}}(ntuple(_->zeros(rd.Nq, md.num_elements), nvars))
  u_face_values = StructArray{SVector{nvars, uEltype}}(ntuple(_->zeros(rd.Nfq, md.num_elements), nvars))
  flux_face_values = similar(u_face_values)

  # local storage for fluxes
  flux_values_threaded = [StructArray{SVector{nvars, uEltype}}(ntuple(_->zeros(rd.Nq), nvars)) for _ in 1:Threads.nthreads()]
  
  return (; md, invMQrstTrW, invJ = inv.(md.J),
      u_values, flux_values_threaded, u_face_values, flux_face_values)
end

function calc_volume_integral!(du,u::StructArray, volume_integral::VolumeIntegralWeakForm,
                 mesh::VertexMappedMesh, equations, dg::DG{<:RefElemData{Dim}}, cache) where {Dim}

  rd = dg.basis  
  md = mesh.md
  @unpack invMQrstTrW, u_values, flux_values_threaded = cache
  @unpack rstxyzJ = md # geometric terms

  # interpolate to quadrature points
  StructArrays.foreachfield(mul_by!(rd.Vq), u_values, u)

  # Todo: simplices. Dispatch on curved/non-curved mesh types, this code only works for affine meshes (accessing rxJ[1,e],...)
  @threaded for e in eachelement(mesh, dg, cache)
    
    flux_values = flux_values_threaded[Threads.threadid()]
    for i in eachdim(mesh, dg, cache) 
      flux_values .= flux.(view(u_values,:,e), i, equations)
      for j in eachdim(mesh, dg, cache)
        StructArrays.foreachfield(mul_by_accum!(invMQrstTrW[j], rstxyzJ[i,j][1,e]), 
                                  view(du,:,e), flux_values)
      end
    end
  end
end

function calc_interface_flux!(cache, surface_integral::SurfaceIntegralWeakForm, 
                mesh::VertexMappedMesh, equations, dg::DG{<:RefElemData{Dim}}) where {Dim}

  @unpack surface_flux = surface_integral
  md = mesh.md
  @unpack mapM, mapP, nxyzJ, Jf = md 
  @unpack u_face_values, flux_face_values = cache 

  @threaded for face_node_index in each_face_node_global(mesh, dg, cache)

    # inner (idM -> minus) and outer (idP -> plus) indices 
    idM, idP = mapM[face_node_index], mapP[face_node_index]
    uM = u_face_values[idM]
      
    # compute flux if node is not a boundary node
    if idM != idP
      uP = u_face_values[idP]
      normal = SVector{Dim}(getindex.(nxyzJ, idM)) / Jf[idM]      
      flux_face_values[idM] = surface_flux(uM, uP, normal, equations) * Jf[idM]
    end
  end
end

# assumes cache.flux_face_values is computed and filled with 
# for polyomial discretizations, use dense LIFT matrix for surface contributions.
function calc_surface_integral!(du, u, surface_integral::SurfaceIntegralWeakForm, 
                mesh::VertexMappedMesh, equations, 
                dg::DG{<:RefElemData}, cache) 
  rd = dg.basis
  StructArrays.foreachfield(mul_by_accum!(rd.LIFT), du, cache.flux_face_values)
end

# Specialize for nodal SBP discretizations. Uses that Vf*u = u[Fmask,:] 
function prolong2interfaces!(cache, u, mesh::AbstractMeshData, equations, surface_integral, 
                             dg::DG{<:RefElemData{Dim, <:AbstractElemShape, <:SBP}}) where {Dim}
  rd = dg.basis    
  @unpack Fmask = rd
  @unpack u_face_values = cache
  StructArrays.foreachfield((out, u)->out .= view(u, Fmask, :), u_face_values, u)
end

# Specialize for nodal SBP discretizations. Uses that du = LIFT*u is equivalent to 
# du[Fmask,:] .= u ./ rd.wq[rd.Fmask] 
function calc_surface_integral!(du, u, surface_integral::SurfaceIntegralWeakForm, 
                                mesh::VertexMappedMesh, equations, 
                                dg::DG{<:RefElemData{Dim,<:AbstractElemShape, <:SBP}}, cache) where {Dim}
  rd = dg.basis
  md = mesh.md
  @unpack flux_face_values = cache
  @threaded for e in eachelement(mesh, dg, cache)
    for i in each_face_node(mesh, dg, cache)
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
  @unpack xyzf, nxyzJ, Jf = md
  @unpack surface_flux = dg.surface_integral

  # reshape face/normal arrays to have size = (num_points_on_face, num_faces_total).
  # mesh.boundary_faces indexes into the columns of these face-reshaped arrays.
  num_pts_per_face = rd.Nfq ÷ rd.Nfaces
  num_faces_total = rd.Nfaces * md.num_elements   
  reshape_by_face(u) = reshape(u, num_pts_per_face, num_faces_total)
  u_face_values = reshape_by_face(u_face_values)
  flux_face_values = reshape_by_face(flux_face_values)
  Jf = reshape_by_face(Jf)
  nxyzJ, xyzf = reshape_by_face.(nxyzJ), reshape_by_face.(xyzf) # broadcast over nxyzJ::NTuple{Dim,Matrix}
    
  # loop through boundary faces, which correspond to columns of reshaped u_face_values, ...
  for f in mesh.boundary_faces[boundary_key]
    for i in Base.OneTo(num_pts_per_face)      
      face_normal = SVector{Dim}(getindex.(nxyzJ, i, f)) / Jf[i,f]
      face_coordinates = SVector{Dim}(getindex.(xyzf, i, f))
      flux_face_values[i,f] = boundary_condition(u_face_values[i,f], 
                          face_normal, face_coordinates, t,
                          surface_flux, equations) * Jf[i,f]
    end
  end
  
  # Note: modifying the values of the reshaped array modifies the values of cache.flux_face_values. 
  # However, we don't have to re-reshape, since cache.flux_face_values still retains its original shape.
end
   
# Todo: simplices. Specialize for modal DG on curved meshes using WADG
function invert_jacobian!(du, mesh::Mesh, equations, dg::DG{<:RefElemData}, 
                          cache) where {Mesh <: AbstractMeshData} 
  @threaded for i in each_dof_global(mesh, dg, cache)
    du[i] *= -cache.invJ[i]
  end
end

calc_sources!(du, u, t, source_terms::Nothing, 
              mesh::VertexMappedMesh, equations, dg::DG{<:RefElemData}, cache) = nothing

# uses quadrature + projection to compute source terms. 
function calc_sources!(du, u, t, source_terms::SourceTerms, 
                       mesh::VertexMappedMesh, equations, dg::DG{<:RefElemData}, cache) where {SourceTerms}

  rd = dg.basis
  md = mesh.md
  @unpack Pq = rd
  @unpack u_values, flux_values_threaded = cache
  @threaded for e in eachelement(mesh, dg, cache)

    flux_values = flux_values_threaded[Threads.threadid()]

    u_e = view(u_values, :, e) # u_values should already be computed from volume kernel

    for i in each_quad_node(mesh, dg, cache)
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
