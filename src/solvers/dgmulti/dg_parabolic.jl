function create_cache(mesh::DGMultiMesh, equations::AbstractParabolicEquations,
                      dg::DGMultiWeakForm, RealT, uEltype)
  nvars = nvariables(equations)

  # u_parabolic stores "transformed" variables for computing the gradient
  @unpack md = mesh
  u_transformed = allocate_nested_array(uEltype, nvars, size(md.x), dg)
  u_grad = ntuple(_ -> similar(u_transformed), ndims(mesh))
  u_face_values = allocate_nested_array(uEltype, nvars, size(md.xf), dg)
  grad_u_face_values = ntuple(_ -> similar(u_face_values), ndims(mesh))

  viscous_flux = similar.(u_grad)
  flux_face_values = allocate_nested_array(uEltype, nvars, size(md.xf), dg)

  local_viscous_flux_threaded = [ntuple(_ -> similar(u_transformed, dg.basis.Nq), ndims(mesh)) for _ in 1:Threads.nthreads()]
  local_flux_values_threaded = [similar(flux_face_values[:, 1]) for _ in 1:Threads.nthreads()]

  return (; u_transformed, u_grad, viscous_flux, u_face_values, grad_u_face_values, flux_face_values,
            local_viscous_flux_threaded, local_flux_values_threaded)
end

# Transform variables prior to taking the gradient. Defaults to doing nothing.
# TODO: can we avoid copying data?
function transform_variables!(u_transformed, u, equations)
  @threaded for i in eachindex(u)
    u_transformed[i] = u[i]
  end
end

# interpolates from solution coefficients to face quadrature points
function prolong2interfaces!(u_face_values, u, mesh::DGMultiMesh, equations::AbstractParabolicEquations,
                             surface_integral, dg::DGMulti, cache)
  apply_to_each_field(mul_by!(dg.basis.Vf), u_face_values, u)
end

function calc_gradient_surface_integral(u_grad, u, flux_face_values,
                                        mesh, equations::AbstractParabolicEquations,
                                        dg::DGMulti, cache, parabolic_cache)
  @unpack local_flux_values_threaded = parabolic_cache
  @threaded for e in eachelement(mesh, dg)
    local_flux_values = local_flux_values_threaded[Threads.threadid()]
    for dim in eachdim(mesh)
      for i in eachindex(local_flux_values)
        # compute [u] * (nx, ny, nz)
        local_flux_values[i] = flux_face_values[i, e] * mesh.md.nxyzJ[dim][i, e]
      end
      apply_to_each_field(mul_by_accum!(dg.basis.LIFT), view(u_grad[dim], :, e), local_flux_values)
    end
  end
end

function calc_gradient!(u_grad, u::StructArray, t, mesh::DGMultiMesh,
                        equations::AbstractParabolicEquations,
                        boundary_conditions, dg::DGMulti, cache, parabolic_cache)

  for dim in 1:length(u_grad)
    reset_du!(u_grad[dim], dg)
  end

  # compute volume contributions to gradients
  @threaded for e in eachelement(mesh, dg)
    for i in eachdim(mesh), j in eachdim(mesh)
      dxidxhatj = mesh.md.rstxyzJ[i, j][1, e] # assumes mesh is affine
      apply_to_each_field(mul_by_accum!(dg.basis.Drst[j], dxidxhatj),
                          view(u_grad[i], :, e), view(u, :, e))
    end
  end

  prolong2interfaces!(cache.u_face_values, u,
                      mesh, equations, dg.surface_integral, dg, cache)

  # compute fluxes at interfaces
  @unpack u_face_values, flux_face_values = cache
  @unpack mapM, mapP, Jf = mesh.md
  @threaded for face_node_index in each_face_node_global(mesh, dg)
    idM, idP = mapM[face_node_index], mapP[face_node_index]
    uM = u_face_values[idM]
    # compute flux if node is not a boundary node
    if idM != idP
      uP = u_face_values[idP]
      # {u} - u = 0.5 * (uP - uM)
      flux_face_values[idM] = 0.5 * (uP - uM) # TODO: use strong/weak formulation?
    end
  end

  calc_boundary_flux!(flux_face_values, u, t, Gradient(), boundary_conditions,
                      mesh, equations, dg, cache, parabolic_cache)

  # compute surface contributions
  calc_gradient_surface_integral(u_grad, u, flux_face_values,
                                 mesh, equations, dg, cache, parabolic_cache)

  for dim in eachdim(mesh)
    invert_jacobian!(u_grad[dim], mesh, equations, dg, cache; scaling=1.0)
  end
end

# operator types used for dispatch on boundary fluxes
struct Gradient end
struct Divergence end

# do nothing for periodic domains
function calc_boundary_flux!(flux, u, t, operator_type, ::BoundaryConditionPeriodic,
                             mesh, equations::AbstractParabolicEquations, dg::DGMulti,
                             cache, parabolic_cache)
  return nothing
end

# "lispy tuple programming" instead of for loop for type stability
function calc_boundary_flux!(flux, u, t, operator_type, boundary_conditions,
                             mesh, equations, dg::DGMulti, cache, parabolic_cache)
  # peel off first boundary condition
  calc_single_boundary_flux!(flux, u, t, operator_type, first(boundary_conditions), first(keys(boundary_conditions)),
                             mesh, equations, dg, cache, parabolic_cache)

  # recurse on the remainder of the boundary conditions
  calc_boundary_flux!(flux, u, t, operator_type, Base.tail(boundary_conditions),
                      mesh, equations, dg, cache, parabolic_cache)
end

# terminate recursion
calc_boundary_flux!(flux, u, t, operator_type, boundary_conditions::NamedTuple{(),Tuple{}},
                    mesh, equations, dg::DGMulti, cache, parabolic_cache) = nothing

# TODO: finish
function calc_single_boundary_flux!(flux, u, t, operator_type, boundary_condition,
                                    boundary_key, mesh, equations, dg::DGMulti,
                                    cache, parabolic_cache)
  return nothing
end

function calc_viscous_fluxes!(viscous_flux, u, u_grad, mesh::DGMultiMesh,
                              equations::AbstractParabolicEquations,
                              dg::DGMulti, cache, parabolic_cache)

  for dim in eachdim(mesh)
    reset_du!(viscous_flux[dim], dg)
  end

  # @threaded for i in eachindex(u)
  #   for dim in eachdim(mesh)
  #     viscous_flux[dim][i] = equations.diffusivity * u_grad[dim][i]
  #   end
  # end

  @unpack local_viscous_flux_threaded = parabolic_cache
  local_u_values_threaded = parabolic_cache.local_flux_values_threaded

  @threaded for e in eachelement(mesh, dg)

    # reset local storage for each element
    local_viscous_flux = local_viscous_flux_threaded[Threads.threadid()]
    local_u_values = local_u_values_threaded[Threads.threadid()]
    fill!(local_u_values, zero(eltype(local_u_values)))
    for dim in eachdim(mesh)
      fill!(local_viscous_flux[dim], zero(eltype(local_viscous_flux[dim])))
    end

    # interpolate u and gradient to quadrature points, store in `local_viscous_flux`
    apply_to_each_field(mul_by!(dg.basis.Vq), local_u_values, view(u, :, e)) # TODO: can we avoid this when we don't need it?
    for dim in eachdim(mesh)
      apply_to_each_field(mul_by!(dg.basis.Vq), local_viscous_flux[dim], view(u_grad[dim], :, e))
    end

    # compute viscous flux at quad points
    for i in eachindex(local_u_values)
      u_i = local_u_values[i]
      u_grad_i = getindex.(local_viscous_flux, i) # TODO: check if this allocates. Shouldn't for tuples or SVector...
      viscous_flux_i = flux(u_i, u_grad_i, equations)
      setindex!.(local_viscous_flux, viscous_flux_i, i)
    end

    # project back to the DG approximation space
    for dim in eachdim(mesh)
      apply_to_each_field(mul_by!(dg.basis.Pq), view(viscous_flux[dim], :, e), local_viscous_flux[dim])
    end
  end
end

function calc_divergence!(du, u::StructArray, t, viscous_flux, mesh::DGMultiMesh,
                          equations::AbstractParabolicEquations,
                          boundary_conditions, dg::DGMulti, cache, parabolic_cache)

  reset_du!(du, dg)

  # compute volume contributions to divergence
  @threaded for e in eachelement(mesh, dg)
    for i in eachdim(mesh), j in eachdim(mesh)
      dxidxhatj = mesh.md.rstxyzJ[i, j][1, e] # assumes mesh is affine
      apply_to_each_field(mul_by_accum!(dg.basis.Drst[j], dxidxhatj),
                                view(du, :, e), view(viscous_flux[i], :, e))
    end
  end

  # interpolates from solution coefficients to face quadrature points
  @unpack grad_u_face_values = parabolic_cache
  for dim in eachdim(mesh)
    prolong2interfaces!(grad_u_face_values[dim], viscous_flux[dim], mesh, equations,
                        dg.surface_integral, dg, cache)
  end

  # compute fluxes at interfaces
  @unpack grad_u_face_values, flux_face_values = parabolic_cache
  @unpack mapM, mapP, nxyzJ = mesh.md
  @threaded for face_node_index in each_face_node_global(mesh, dg, cache, parabolic_cache)
    idM, idP = mapM[face_node_index], mapP[face_node_index]

    # compute f(u, ∇u) ⋅ n
    flux_face_value = zero(eltype(flux_face_values))
    for dim in eachdim(mesh)
      uM = grad_u_face_values[dim][idM]
      # compute flux if node is not a boundary node
      if idM != idP
        uP = grad_u_face_values[dim][idP]
        # TODO: use strong/weak formulation?
        flux_face_value = flux_face_value + 0.5 * (uP - uM) * nxyzJ[dim][face_node_index]
      end
    end
    flux_face_values[idM] = flux_face_value
  end

  calc_boundary_flux!(flux_face_values, u, t, Divergence(), boundary_conditions,
                      mesh, equations, dg, cache, parabolic_cache)

  # surface contributions
  apply_to_each_field(mul_by_accum!(dg.basis.LIFT), du, flux_face_values)

  invert_jacobian!(du, mesh, equations, dg, cache; scaling=1.0)
end

# assumptions: parabolic terms are of the form div(f(u, grad(u))) and
# will be discretized first order form as follows:
#               1. compute grad(u)
#               2. compute f(u, grad(u))
#               3. compute div(u)
# boundary conditions will be applied to both grad(u) and div(u).
function rhs!(du, u, t, mesh::DGMultiMesh, parabolic_equations::AbstractParabolicEquations,
              initial_condition, boundary_conditions, source_terms,
              dg::DGMulti, cache, parabolic_cache)

  reset_du!(du, dg)

  @unpack u_transformed, u_grad, viscous_flux = parabolic_cache
  transform_variables!(u_transformed, u, parabolic_equations)

  calc_gradient!(u_grad, u_transformed, t, mesh, parabolic_equations,
                 boundary_conditions, dg, cache, parabolic_cache)

  calc_viscous_fluxes!(viscous_flux, u_transformed, u_grad,
                       mesh, parabolic_equations, dg, cache, parabolic_cache)

  calc_divergence!(du, u_transformed, t, viscous_flux, mesh, parabolic_equations,
                   boundary_conditions, dg, cache, parabolic_cache)

  return nothing

end
