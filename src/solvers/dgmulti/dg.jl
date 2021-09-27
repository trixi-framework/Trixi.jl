# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


# out <- A*x
mul_by!(A) = @inline (out, x)->matmul!(out, A, x)

# out <- out + A * x
mul_by_accum!(A) = @inline (out, x)->matmul!(out, A, x, one(eltype(out)), one(eltype(out)))

#  out <- out + α * A * x
mul_by_accum!(A, α) = @inline (out, x)->matmul!(out, A, x, α, one(eltype(out)))

# specialize for SBP operators since `matmul!` doesn't work for `UniformScaling` types.
struct MulByUniformScaling end
struct MulByAccumUniformScaling end
mul_by!(A::UniformScaling) = MulByUniformScaling()
mul_by_accum!(A::UniformScaling) = MulByAccumUniformScaling()

# StructArray fallback
@inline apply_to_each_field(f::F, args::Vararg{Any, N}) where {F, N} = StructArrays.foreachfield(f, args...)

# specialize for UniformScaling types: works for either StructArray{SVector} or Matrix{SVector}
# solution storage formats.
@inline apply_to_each_field(f::MulByUniformScaling, out, x, args...) = copy!(out, x)
@inline function apply_to_each_field(f::MulByAccumUniformScaling, out, x, args...)
  for (i, x_i) in enumerate(x)
    out[i] = out[i] + x_i
  end
end

@inline eachdim(mesh) = Base.OneTo(ndims(mesh))

# iteration over all elements in a mesh
@inline ndofs(mesh::AbstractMeshData, dg::DGMulti, cache) = dg.basis.Np * mesh.md.num_elements
@inline eachelement(mesh::AbstractMeshData, dg::DGMulti, cache) = Base.OneTo(mesh.md.num_elements)

# iteration over quantities in a single element
@inline nnodes(basis::RefElemData) = basis.Np
@inline each_face_node(mesh::AbstractMeshData, dg::DGMulti, cache) = Base.OneTo(dg.basis.Nfq)
@inline each_quad_node(mesh::AbstractMeshData, dg::DGMulti, cache) = Base.OneTo(dg.basis.Nq)

# iteration over quantities over the entire mesh (dofs, quad nodes, face nodes).
@inline each_dof_global(mesh::AbstractMeshData, dg::DGMulti, cache) = Base.OneTo(ndofs(mesh, dg, cache))
@inline each_quad_node_global(mesh::AbstractMeshData, dg::DGMulti, cache) = Base.OneTo(dg.basis.Nq * mesh.md.num_elements)
@inline each_face_node_global(mesh::AbstractMeshData, dg::DGMulti, cache) = Base.OneTo(dg.basis.Nfq * mesh.md.num_elements)

# interface with semidiscretization_hyperbolic
wrap_array(u_ode, mesh::AbstractMeshData, equations, dg::DGMulti, cache) = u_ode
wrap_array_native(u_ode, mesh::AbstractMeshData, equations, dg::DGMulti, cache) = u_ode
function digest_boundary_conditions(boundary_conditions::NamedTuple{Keys,ValueTypes}, mesh::AbstractMeshData,
                                    dg::DGMulti, cache) where {Keys,ValueTypes<:NTuple{N,Any}} where {N}
  return boundary_conditions
end

# Allocate nested array type for DGMulti solution storage.
function allocate_nested_array(uEltype, nvars, array_dimensions, dg)
  # store components as separate arrays, combine via StructArrays
  return StructArray{SVector{nvars, uEltype}}(ntuple(_->zeros(uEltype, array_dimensions...), nvars))
end

function create_cache(mesh::VertexMappedMesh, equations, dg::DGMultiWeakForm, RealT, uEltype)

  rd = dg.basis
  md = mesh.md

  # volume quadrature weights, volume interpolation matrix, mass matrix, differentiation matrices
  @unpack wq, Vq, M, Drst = rd

  # ∫f(u) * dv/dx_i = ∑_j (Vq*Drst[i])'*diagm(wq)*(rstxyzJ[i,j].*f(Vq*u))
  weak_differentiation_matrices = map(D -> -M \ ((Vq * D)' * diagm(wq)), Drst)

  nvars = nvariables(equations)

  # storage for volume quadrature values, face quadrature values, flux values
  u_values = allocate_nested_array(uEltype, nvars, size(md.xq), dg)
  u_face_values = allocate_nested_array(uEltype, nvars, size(md.xf), dg)
  flux_face_values = allocate_nested_array(uEltype, nvars, size(md.xf), dg)
  if typeof(rd.approximationType) <: SBP
    lift_scalings = rd.wf ./ rd.wq[rd.Fmask] # lift scalings for diag-norm SBP operators
  else
    lift_scalings = nothing
  end

  # local storage for volume integral and source computations
  local_values_threaded = [allocate_nested_array(uEltype, nvars, (rd.Nq,), dg) for _ in 1:Threads.nthreads()]

  return (; md, weak_differentiation_matrices, invJ = inv.(md.J), lift_scalings,
            u_values, u_face_values, flux_face_values,
            local_values_threaded)
end

function allocate_coefficients(mesh::AbstractMeshData, equations, dg::DGMulti, cache)
  return allocate_nested_array(real(dg), nvariables(equations), size(mesh.md.x), dg)
end

function compute_coefficients!(u, initial_condition, t,
                               mesh::AbstractMeshData, equations, dg::DGMulti, cache)
  md = mesh.md
  rd = dg.basis
  @unpack u_values = cache

  # evaluate the initial condition at quadrature points
  @threaded for i in each_quad_node_global(mesh, dg, cache)
    u_values[i] = initial_condition(getindex.(md.xyzq, i), t, equations)
  end

  # multiplying by Pq computes the L2 projection
  apply_to_each_field(mul_by!(rd.Pq), u, u_values)
end

# estimates the timestep based on polynomial degree and mesh. Does not account for physics (e.g.,
# computes an estimate of `dt` based on the advection equation with constant unit advection speed).
function estimate_dt(mesh::AbstractMeshData, dg::DGMulti)
  rd = dg.basis # RefElemData
  return StartUpDG.estimate_h(rd, mesh.md) / StartUpDG.inverse_trace_constant(rd)
end

# for the stepsize callback
function max_dt(u, t, mesh::AbstractMeshData,
                constant_speed::Val{false}, equations, dg::DGMulti{NDIMS}, cache) where {NDIMS}

  @unpack md = mesh
  rd = dg.basis

  dt_min = Inf
  for e in eachelement(mesh, dg, cache)
    h_e = StartUpDG.estimate_h(e, rd, md)
    max_speeds = ntuple(_->nextfloat(zero(t)), NDIMS)
    for i in Base.OneTo(rd.Np) # loop over nodes
      lambda_i = max_abs_speeds(u[i, e], equations)
      max_speeds = max.(max_speeds, lambda_i)
    end
    dt_min = min(dt_min, h_e / sum(max_speeds))
  end
  # This mimics `max_dt` for `TreeMesh`, except that `nnodes(dg)` is replaced by
  # `polydeg+1`. This is because `nnodes(dg)` returns the total number of
  # multi-dimensional nodes for DGMulti solver types, while `nnodes(dg)` returns
  # the number of 1D nodes for `DGSEM` solvers.
  polydeg = rd.N
  return 2 * dt_min / (polydeg + 1)
end

# interpolates from solution coefficients to face quadrature points
function prolong2interfaces!(cache, u, mesh::AbstractMeshData, equations,
                             surface_integral, dg::DGMulti)
  rd = dg.basis
  @unpack u_face_values = cache
  apply_to_each_field(mul_by!(rd.Vf), u_face_values, u)
end

function calc_volume_integral!(du, u, volume_integral::VolumeIntegralWeakForm,
                               mesh::VertexMappedMesh,
                               have_nonconservative_terms::Val{false}, equations,
                               dg::DGMulti, cache)

  rd = dg.basis
  md = mesh.md
  @unpack weak_differentiation_matrices, u_values, local_values_threaded = cache
  @unpack rstxyzJ = md # geometric terms

  # interpolate to quadrature points
  apply_to_each_field(mul_by!(rd.Vq), u_values, u)

  # Todo: DGMulti. Dispatch on curved/non-curved mesh types, this code only works for affine meshes (accessing rxJ[1,e],...)
  @threaded for e in eachelement(mesh, dg, cache)

    flux_values = local_values_threaded[Threads.threadid()]
    for i in eachdim(mesh)
      flux_values .= flux.(view(u_values,:,e), i, equations)
      for j in eachdim(mesh)
        apply_to_each_field(mul_by_accum!(weak_differentiation_matrices[j], rstxyzJ[i,j][1,e]),
                            view(du,:,e), flux_values)
      end
    end
  end
end

function calc_interface_flux!(cache, surface_integral::SurfaceIntegralWeakForm,
                              mesh::VertexMappedMesh,
                              have_nonconservative_terms::Val{false}, equations,
                              dg::DGMulti{NDIMS}) where {NDIMS}

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
      normal = SVector{NDIMS}(getindex.(nxyzJ, idM)) / Jf[idM]
      flux_face_values[idM] = surface_flux(uM, uP, normal, equations) * Jf[idM]
    end
  end
end

function calc_interface_flux!(cache, surface_integral::SurfaceIntegralWeakForm,
                              mesh::VertexMappedMesh,
                              have_nonconservative_terms::Val{true}, equations,
                              dg::DGMulti{NDIMS}) where {NDIMS}

  flux_conservative, flux_nonconservative = surface_integral.surface_flux
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
      normal = SVector{NDIMS}(getindex.(nxyzJ, idM)) / Jf[idM]
      conservative_part = flux_conservative(uM, uP, normal, equations)

      # Two notes on the use of `flux_nonconservative`:
      # 1. In contrast to other mesh types, only one nonconservative part needs to be
      #    computed since we loop over the elements, not the unique interfaces.
      # 2. In general, nonconservative fluxes can depend on both the contravariant
      #    vectors (normal direction) at the current node and the averaged ones. However,
      #    both are the same at watertight interfaces, so we pass `normal` twice.
      nonconservative_part = flux_nonconservative(uM, uP, normal, normal, equations)
      # The factor 0.5 is necessary for the nonconservative fluxes based on the
      # interpretation of global SBP operators.
      flux_face_values[idM] = (conservative_part + 0.5 * nonconservative_part) * Jf[idM]
    end
  end
end

# assumes cache.flux_face_values is computed and filled with
# for polyomial discretizations, use dense LIFT matrix for surface contributions.
function calc_surface_integral!(du, u, surface_integral::SurfaceIntegralWeakForm,
                                mesh::VertexMappedMesh, equations,
                                dg::DGMulti, cache)
  rd = dg.basis
  apply_to_each_field(mul_by_accum!(rd.LIFT), du, cache.flux_face_values)
end

# Specialize for nodal SBP discretizations. Uses that Vf*u = u[Fmask,:]
function prolong2interfaces!(cache, u, mesh::AbstractMeshData, equations, surface_integral,
                             dg::DGMulti{NDIMS, <:AbstractElemShape, <:SBP}) where {NDIMS}
  rd = dg.basis
  @unpack Fmask = rd
  @unpack u_face_values = cache
  @threaded for e in eachelement(mesh, dg, cache)
    for (i,fid) in enumerate(Fmask)
      u_face_values[i, e] = u[fid, e]
    end
  end
end

# Specialize for nodal SBP discretizations. Uses that du = LIFT*u is equivalent to
# du[Fmask,:] .= u ./ rd.wq[rd.Fmask]
function calc_surface_integral!(du, u, surface_integral::SurfaceIntegralWeakForm,
                                mesh::VertexMappedMesh, equations,
                                dg::DGMulti{NDIMS,<:AbstractElemShape, <:SBP}, cache) where {NDIMS}
  rd = dg.basis
  md = mesh.md
  @unpack flux_face_values, lift_scalings = cache
  @threaded for e in eachelement(mesh, dg, cache)
    for i in each_face_node(mesh, dg, cache)
      fid = rd.Fmask[i]
      du[fid, e] = du[fid, e] + flux_face_values[i,e] * lift_scalings[i]
    end
  end
end

# do nothing for periodic (default) boundary conditions
calc_boundary_flux!(cache, t, boundary_conditions::BoundaryConditionPeriodic,
                    mesh, equations, dg::DGMulti) = nothing

# "lispy tuple programming" instead of for loop for type stability
function calc_boundary_flux!(cache, t, boundary_conditions, mesh, equations, dg::DGMulti)

  # peel off first boundary condition
  calc_single_boundary_flux!(cache, t, first(boundary_conditions), first(keys(boundary_conditions)),
                 mesh, equations, dg)

  # recurse on the remainder of the boundary conditions
  calc_boundary_flux!(cache, t, Base.tail(boundary_conditions), mesh, equations, dg)
end

# terminate recursion
calc_boundary_flux!(cache, t, boundary_conditions::NamedTuple{(),Tuple{}},
                    mesh, equations, dg::DGMulti) = nothing

function calc_single_boundary_flux!(cache, t, boundary_condition, boundary_key,
                                    mesh, equations, dg::DGMulti{NDIMS}) where {NDIMS}

  rd = dg.basis
  md = mesh.md
  @unpack u_face_values, flux_face_values = cache
  @unpack xyzf, nxyzJ, Jf = md
  @unpack surface_flux = dg.surface_integral

  # reshape face/normal arrays to have size = (num_points_on_face, num_faces_total).
  # mesh.boundary_faces indexes into the columns of these face-reshaped arrays.
  num_pts_per_face = rd.Nfq ÷ rd.Nfaces
  num_faces_total = rd.Nfaces * md.num_elements

  # This function was originally defined as
  # `reshape_by_face(u) = reshape(view(u, :), num_pts_per_face, num_faces_total)`.
  # This results in allocations due to https://github.com/JuliaLang/julia/issues/36313.
  # To avoid allocations, we use Tim Holy's suggestion:
  # https://github.com/JuliaLang/julia/issues/36313#issuecomment-782336300.
  reshape_by_face(u) = Base.ReshapedArray(u, (num_pts_per_face, num_faces_total), ())

  u_face_values = reshape_by_face(u_face_values)
  flux_face_values = reshape_by_face(flux_face_values)
  Jf = reshape_by_face(Jf)
  nxyzJ, xyzf = reshape_by_face.(nxyzJ), reshape_by_face.(xyzf) # broadcast over nxyzJ::NTuple{NDIMS,Matrix}

  # loop through boundary faces, which correspond to columns of reshaped u_face_values, ...
  for f in mesh.boundary_faces[boundary_key]
    for i in Base.OneTo(num_pts_per_face)
      face_normal = SVector{NDIMS}(getindex.(nxyzJ, i, f)) / Jf[i,f]
      face_coordinates = SVector{NDIMS}(getindex.(xyzf, i, f))
      flux_face_values[i,f] = boundary_condition(u_face_values[i,f],
                          face_normal, face_coordinates, t,
                          surface_flux, equations) * Jf[i,f]
    end
  end

  # Note: modifying the values of the reshaped array modifies the values of cache.flux_face_values.
  # However, we don't have to re-reshape, since cache.flux_face_values still retains its original shape.
end


# Todo: DGMulti. Specialize for modal DG on curved meshes using WADG
function invert_jacobian!(du, mesh::Mesh, equations, dg::DGMulti,
                          cache) where {Mesh <: AbstractMeshData}
  @threaded for i in each_dof_global(mesh, dg, cache)
    du[i] *= -cache.invJ[i]
  end
end

# Multiple calc_sources! to resolve method ambiguities
calc_sources!(du, u, t, source_terms::Nothing,
              mesh, equations, dg::DGMulti, cache) = nothing
calc_sources!(du, u, t, source_terms::Nothing,
              mesh, equations, dg::DGMultiFluxDiff{<:SBP}, cache) = nothing

# uses quadrature + projection to compute source terms.
function calc_sources!(du, u, t, source_terms,
                       mesh, equations, dg::DGMulti, cache)

  rd = dg.basis
  md = mesh.md
  @unpack Pq = rd
  @unpack u_values, local_values_threaded = cache
  @threaded for e in eachelement(mesh, dg, cache)

    source_values = local_values_threaded[Threads.threadid()]

    u_e = view(u_values, :, e) # u_values should already be computed from volume kernel

    for i in each_quad_node(mesh, dg, cache)
      source_values[i] = source_terms(u_e[i], getindex.(md.xyzq, i, e), t, equations)
    end
    apply_to_each_field(mul_by_accum!(Pq), view(du, :, e), source_values)
  end
end

function rhs!(du, u, t, mesh, equations,
              initial_condition, boundary_conditions::BC, source_terms::Source,
              dg::DGMulti, cache) where {BC, Source}

  @trixi_timeit timer() "Reset du/dt" fill!(du,zero(eltype(du)))

  @trixi_timeit timer() "calc_volume_integral!" calc_volume_integral!(du, u, dg.volume_integral,
                                    mesh, have_nonconservative_terms(equations), equations, dg, cache)

  @trixi_timeit timer() "prolong2interfaces!" prolong2interfaces!(cache, u, mesh, equations, dg.surface_integral, dg)

  @trixi_timeit timer() "calc_interface_flux!" calc_interface_flux!(cache, dg.surface_integral, mesh,
                                                                    have_nonconservative_terms(equations),
                                                                    equations, dg)

  @trixi_timeit timer() "calc_boundary_flux!" calc_boundary_flux!(cache, t, boundary_conditions, mesh, equations, dg)

  @trixi_timeit timer() "calc_surface_integral!" calc_surface_integral!(du, u, dg.surface_integral, mesh, equations, dg, cache)

  @trixi_timeit timer() "invert_jacobian" invert_jacobian!(du, mesh, equations, dg, cache)

  @trixi_timeit timer() "calc_sources!" calc_sources!(du, u, t, source_terms, mesh, equations, dg, cache)

  return nothing
end


end # @muladd
