function create_cache_parabolic(mesh::DGMultiMesh,
                                equations_hyperbolic::AbstractEquations,
                                equations_parabolic::AbstractEquationsParabolic,
                                dg::DGMulti, parabolic_scheme, RealT, uEltype)
  # default to taking derivatives of all hyperbolic terms
  # TODO: parabolic; utilize the parabolic variables in `equations_parabolic` to reduce memory usage in the parabolic cache
  nvars = nvariables(equations_hyperbolic)

  @unpack M, Drst = dg.basis
  weak_differentiation_matrices = map(A -> -M \ (A' * M), Drst)

  # u_transformed stores "transformed" variables for computing the gradient
  @unpack md = mesh
  u_transformed = allocate_nested_array(uEltype, nvars, size(md.x), dg)
  gradients = ntuple(_ -> similar(u_transformed), ndims(mesh))
  flux_viscous = similar.(gradients)

  u_face_values = allocate_nested_array(uEltype, nvars, size(md.xf), dg)
  scalar_flux_face_values = similar(u_face_values)
  gradients_face_values = ntuple(_ -> similar(u_face_values), ndims(mesh))

  local_u_values_threaded = [similar(u_transformed, dg.basis.Nq) for _ in 1:Threads.nthreads()]
  local_flux_viscous_threaded = [ntuple(_ -> similar(u_transformed, dg.basis.Nq), ndims(mesh)) for _ in 1:Threads.nthreads()]
  local_flux_face_values_threaded = [similar(scalar_flux_face_values[:, 1]) for _ in 1:Threads.nthreads()]

  # precompute 1 / h for penalty terms
  inv_h = similar(mesh.md.Jf)
  J = dg.basis.Vf * mesh.md.J # interp to face nodes
  for e in eachelement(mesh, dg)
    for i in each_face_node(mesh, dg)
      inv_h[i, e] = mesh.md.Jf[i, e] / J[i, e]
    end
  end

  return (; u_transformed, gradients, flux_viscous,
            weak_differentiation_matrices, inv_h,
            u_face_values, gradients_face_values, scalar_flux_face_values,
            local_u_values_threaded, local_flux_viscous_threaded, local_flux_face_values_threaded)
end

# Transform solution variables prior to taking the gradient
# (e.g., conservative to primitive variables). Defaults to doing nothing.
# TODO: can we avoid copying data?
function transform_variables!(u_transformed, u, mesh, equations_parabolic::AbstractEquationsParabolic,
                              dg::DGMulti, parabolic_scheme, cache, cache_parabolic)
  @threaded for i in eachindex(u)
    u_transformed[i] = gradient_variable_transformation(equations_parabolic)(u[i], equations_parabolic)
  end
end

# interpolates from solution coefficients to face quadrature points
# We pass the `surface_integral` argument solely for dispatch
function prolong2interfaces!(u_face_values, u, mesh::DGMultiMesh, equations::AbstractEquationsParabolic,
                             surface_integral, dg::DGMulti, cache)
  apply_to_each_field(mul_by!(dg.basis.Vf), u_face_values, u)
end

function calc_gradient_surface_integral(gradients, u, scalar_flux_face_values,
                                        mesh, equations::AbstractEquationsParabolic,
                                        dg::DGMulti, cache, cache_parabolic)
  @unpack local_flux_face_values_threaded = cache_parabolic
  @threaded for e in eachelement(mesh, dg)
    local_flux_values = local_flux_face_values_threaded[Threads.threadid()]
    for dim in eachdim(mesh)
      for i in eachindex(local_flux_values)
        # compute flux * (nx, ny, nz)
        local_flux_values[i] = scalar_flux_face_values[i, e] * mesh.md.nxyzJ[dim][i, e]
      end
      apply_to_each_field(mul_by_accum!(dg.basis.LIFT), view(gradients[dim], :, e), local_flux_values)
    end
  end
end

function calc_gradient!(gradients, u::StructArray, t, mesh::DGMultiMesh,
                        equations::AbstractEquationsParabolic,
                        boundary_conditions, dg::DGMulti, cache, cache_parabolic)

  @unpack weak_differentiation_matrices = cache_parabolic

  for dim in eachindex(gradients)
    reset_du!(gradients[dim], dg)
  end

  # compute volume contributions to gradients
  @threaded for e in eachelement(mesh, dg)
    for i in eachdim(mesh), j in eachdim(mesh)
      dxidxhatj = mesh.md.rstxyzJ[i, j][1, e] # TODO: DGMulti. Assumes mesh is affine here.
      apply_to_each_field(mul_by_accum!(weak_differentiation_matrices[j], dxidxhatj),
                          view(gradients[i], :, e), view(u, :, e))
    end
  end

  @unpack u_face_values = cache_parabolic
  prolong2interfaces!(u_face_values, u, mesh, equations, dg.surface_integral, dg, cache)

  # compute fluxes at interfaces
  @unpack scalar_flux_face_values = cache_parabolic
  @unpack mapM, mapP, Jf = mesh.md
  @threaded for face_node_index in each_face_node_global(mesh, dg)
    idM, idP = mapM[face_node_index], mapP[face_node_index]
    uM = u_face_values[idM]
    uP = u_face_values[idP]
    scalar_flux_face_values[idM] = 0.5 * (uP + uM) # TODO: use strong/weak formulation for curved meshes?
  end

  calc_boundary_flux!(scalar_flux_face_values, u_face_values, t, Gradient(), boundary_conditions,
                      mesh, equations, dg, cache, cache_parabolic)

  # compute surface contributions
  calc_gradient_surface_integral(gradients, u, scalar_flux_face_values,
                                 mesh, equations, dg, cache, cache_parabolic)

  for dim in eachdim(mesh)
    invert_jacobian!(gradients[dim], mesh, equations, dg, cache; scaling=1.0)
  end
end

# do nothing for periodic domains
function calc_boundary_flux!(flux, u, t, operator_type, ::BoundaryConditionPeriodic,
                             mesh, equations::AbstractEquationsParabolic, dg::DGMulti,
                             cache, cache_parabolic)
  return nothing
end

# "lispy tuple programming" instead of for loop for type stability
function calc_boundary_flux!(flux, u, t, operator_type, boundary_conditions,
                             mesh, equations, dg::DGMulti, cache, cache_parabolic)

  # peel off first boundary condition
  calc_single_boundary_flux!(flux, u, t, operator_type, first(boundary_conditions), first(keys(boundary_conditions)),
                             mesh, equations, dg, cache, cache_parabolic)

  # recurse on the remainder of the boundary conditions
  calc_boundary_flux!(flux, u, t, operator_type, Base.tail(boundary_conditions),
                      mesh, equations, dg, cache, cache_parabolic)
end

# terminate recursion
calc_boundary_flux!(flux, u, t, operator_type, boundary_conditions::NamedTuple{(),Tuple{}},
                    mesh, equations, dg::DGMulti, cache, cache_parabolic) = nothing

# TODO: DGMulti. Decide if we want to use the input `u_face_values` (currently unused)
function calc_single_boundary_flux!(flux_face_values, u_face_values, t,
                                    operator_type, boundary_condition, boundary_key,
                                    mesh, equations, dg::DGMulti{NDIMS}, cache, cache_parabolic) where {NDIMS}
  rd = dg.basis
  md = mesh.md

  num_pts_per_face = rd.Nfq ÷ rd.Nfaces
  @unpack xyzf, nxyzJ, Jf = md
  for f in mesh.boundary_faces[boundary_key]
    for i in Base.OneTo(num_pts_per_face)

      # reverse engineer element + face node indices (avoids reshaping arrays)
      e = ((f-1) ÷ rd.Nfaces) + 1
      fid = i + ((f-1) % rd.Nfaces) * num_pts_per_face

      face_normal = SVector{NDIMS}(getindex.(nxyzJ, fid, e)) / Jf[fid,e]
      face_coordinates = SVector{NDIMS}(getindex.(xyzf, fid, e))

      # for both the gradient and the divergence, the boundary flux is scalar valued.
      # for the gradient, it is the solution; for divergence, it is the normal flux.
      flux_face_values[fid,e] = boundary_condition(flux_face_values[fid,e], u_face_values[fid,e],
                                                   face_normal, face_coordinates, t,
                                                   operator_type, equations)
    end
  end
  return nothing
end

function calc_viscous_fluxes!(flux_viscous, u, gradients, mesh::DGMultiMesh,
                              equations::AbstractEquationsParabolic,
                              dg::DGMulti, cache, cache_parabolic)

  for dim in eachdim(mesh)
    reset_du!(flux_viscous[dim], dg)
  end

  @unpack local_flux_viscous_threaded, local_u_values_threaded = cache_parabolic

  @threaded for e in eachelement(mesh, dg)

    # reset local storage for each element
    local_flux_viscous = local_flux_viscous_threaded[Threads.threadid()]
    local_u_values = local_u_values_threaded[Threads.threadid()]
    fill!(local_u_values, zero(eltype(local_u_values)))
    for dim in eachdim(mesh)
      fill!(local_flux_viscous[dim], zero(eltype(local_flux_viscous[dim])))
    end

    # interpolate u and gradient to quadrature points, store in `local_flux_viscous`
    apply_to_each_field(mul_by!(dg.basis.Vq), local_u_values, view(u, :, e)) # TODO: DGMulti. Specialize for nodal collocation methods (SBP, GaussSBP)
    for dim in eachdim(mesh)
      apply_to_each_field(mul_by!(dg.basis.Vq), local_flux_viscous[dim], view(gradients[dim], :, e))
    end

    # compute viscous flux at quad points
    for i in eachindex(local_u_values)
      u_i = local_u_values[i]
      gradients_i = getindex.(local_flux_viscous, i)
      for dim in eachdim(mesh)
        flux_viscous_i = flux(u_i, gradients_i, dim, equations)
        setindex!(local_flux_viscous[dim], flux_viscous_i, i)
      end
    end

    # project back to the DG approximation space
    for dim in eachdim(mesh)
      apply_to_each_field(mul_by!(dg.basis.Pq), view(flux_viscous[dim], :, e), local_flux_viscous[dim])
    end
  end
end

# no penalization for a BR1 parabolic solver
function calc_viscous_penalty!(scalar_flux_face_values, u_face_values, t, boundary_conditions,
                               mesh, equations::AbstractEquationsParabolic, dg::DGMulti,
                               parabolic_scheme::ViscousFormulationBassiRebay1, cache, cache_parabolic)
  return nothing
end

function calc_viscous_penalty!(scalar_flux_face_values, u_face_values, t, boundary_conditions,
                               mesh, equations::AbstractEquationsParabolic, dg::DGMulti,
                               parabolic_scheme, cache, cache_parabolic)
  # compute fluxes at interfaces
  @unpack scalar_flux_face_values, inv_h = cache_parabolic
  @unpack mapM, mapP = mesh.md
  @threaded for face_node_index in each_face_node_global(mesh, dg)
    idM, idP = mapM[face_node_index], mapP[face_node_index]
    uM, uP = u_face_values[idM], u_face_values[idP]
    inv_h_face = inv_h[face_node_index]
    scalar_flux_face_values[idM] = scalar_flux_face_values[idM] + penalty(uP, uM, inv_h_face, equations, parabolic_scheme)
  end
  return nothing
end


function calc_divergence!(du, u::StructArray, t, flux_viscous, mesh::DGMultiMesh,
                          equations::AbstractEquationsParabolic,
                          boundary_conditions, dg::DGMulti, parabolic_scheme, cache, cache_parabolic)

  @unpack weak_differentiation_matrices = cache_parabolic

  reset_du!(du, dg)

  # compute volume contributions to divergence
  @threaded for e in eachelement(mesh, dg)
    for i in eachdim(mesh), j in eachdim(mesh)
      dxidxhatj = mesh.md.rstxyzJ[i, j][1, e] # assumes mesh is affine
      apply_to_each_field(mul_by_accum!(weak_differentiation_matrices[j], dxidxhatj),
                                view(du, :, e), view(flux_viscous[i], :, e))
    end
  end

  # interpolates from solution coefficients to face quadrature points
  flux_viscous_face_values = cache_parabolic.gradients_face_values # reuse storage
  for dim in eachdim(mesh)
    prolong2interfaces!(flux_viscous_face_values[dim], flux_viscous[dim], mesh, equations,
                        dg.surface_integral, dg, cache)
  end

  # compute fluxes at interfaces
  @unpack scalar_flux_face_values = cache_parabolic
  @unpack mapM, mapP, nxyzJ = mesh.md
  @threaded for face_node_index in each_face_node_global(mesh, dg, cache, cache_parabolic)
    idM, idP = mapM[face_node_index], mapP[face_node_index]

    # compute f(u, ∇u) ⋅ n
    flux_face_value = zero(eltype(scalar_flux_face_values))
    for dim in eachdim(mesh)
      uM = flux_viscous_face_values[dim][idM]
      uP = flux_viscous_face_values[dim][idP]
      # TODO: use strong/weak formulation to ensure stability on curved meshes?
      flux_face_value = flux_face_value + 0.5 * (uP + uM) * nxyzJ[dim][face_node_index]
    end
    scalar_flux_face_values[idM] = flux_face_value
  end

  calc_boundary_flux!(scalar_flux_face_values, cache_parabolic.u_face_values, t, Divergence(),
                      boundary_conditions, mesh, equations, dg, cache, cache_parabolic)

  calc_viscous_penalty!(scalar_flux_face_values, cache_parabolic.u_face_values, t,
                        boundary_conditions, mesh, equations, dg, parabolic_scheme,
                        cache, cache_parabolic)

  # surface contributions
  apply_to_each_field(mul_by_accum!(dg.basis.LIFT), du, scalar_flux_face_values)

  # Note: we do not flip the sign of the geometric Jacobian here.
  # This is because the parabolic fluxes are assumed to be of the form
  #   `du/dt + df/dx = dg/dx + source(x,t)`,
  # where f(u) is the inviscid flux and g(u) is the viscous flux.
  invert_jacobian!(du, mesh, equations, dg, cache; scaling=1.0)
end

# assumptions: parabolic terms are of the form div(f(u, grad(u))) and
# will be discretized first order form as follows:
#               1. compute grad(u)
#               2. compute f(u, grad(u))
#               3. compute div(u)
# boundary conditions will be applied to both grad(u) and div(u).
function rhs_parabolic!(du, u, t, mesh::DGMultiMesh, equations_parabolic::AbstractEquationsParabolic,
                        initial_condition, boundary_conditions, source_terms,
                        dg::DGMulti, parabolic_scheme, cache, cache_parabolic)

  reset_du!(du, dg)

  @unpack u_transformed, gradients, flux_viscous = cache_parabolic
  transform_variables!(u_transformed, u, mesh, equations_parabolic,
                       dg, parabolic_scheme, cache, cache_parabolic)

  calc_gradient!(gradients, u_transformed, t, mesh, equations_parabolic,
                 boundary_conditions, dg, cache, cache_parabolic)

  calc_viscous_fluxes!(flux_viscous, u_transformed, gradients,
                       mesh, equations_parabolic, dg, cache, cache_parabolic)

  calc_divergence!(du, u_transformed, t, flux_viscous, mesh, equations_parabolic,
                   boundary_conditions, dg, parabolic_scheme, cache, cache_parabolic)

  return nothing

end
