
abstract type AbstractParabolicEquations{NDIMS, NVARS} <: AbstractEquations{NDIMS, NVARS} end

struct ScalarDiffusion2D{T} <: AbstractParabolicEquations{2, 1}
  diffusivity::T
end

# no orientation here since the flux is vector-valued
function flux(u, grad_u, equations::ScalarDiffusion2D)
  dudx, dudy = grad_u
  return equations.diffusivity * dudx, equations.diffusivity * dudy
end

function create_cache(mesh::DGMultiMesh, equations::AbstractParabolicEquations,
                      dg::DGMultiWeakForm, RealT, uEltype)
  nvars = nvariables(equations)

  # u_parabolic stores "transformed" variables for
  @unpack md = mesh
  u_transformed = allocate_nested_array(uEltype, nvars, size(md.xq), dg)
  u_grad = ntuple(_ -> similar(u_transformed), ndims(mesh))
  u_face_values = allocate_nested_array(uEltype, nvars, size(md.xf), dg)
  flux_face_values = allocate_nested_array(uEltype, nvars, size(md.xf), dg)
  return (; u_transformed, u_grad, viscous_flux=similar.(u_grad), u_face_values,
            flux_face_values, local_flux_values = similar(flux_face_values[:, 1]))
end

# Transform variables prior to taking the gradient. Defaults to doing nothing.
# TODO: can we avoid copying data?
function transform_variables(u_transformed, u, equations)
  @threaded for i in eachindex(u)
    u_transformed[i] = u[i]
  end
end

# interpolates from solution coefficients to face quadrature points
function prolong2interfaces!(u_face_values, u, mesh::DGMultiMesh, equations::AbstractParabolicEquations,
                             surface_integral, dg::DGMulti, cache)
  rd = dg.basis
  apply_to_each_field(mul_by!(rd.Vf), u_face_values, u)
end

function calc_gradient_surface_integral(u_grad, u, flux_face_values,
                                        mesh, equations::AbstractParabolicEquations,
                                        dg::DGMulti, cache, parabolic_cache)
  @unpack local_flux_values = parabolic_cache
  @threaded for e in eachelement(mesh, dg)
    for dim in eachdim(mesh)
      for i in eachindex(local_flux_values)
        # compute [u] * (nx, ny, nz)
        local_flux_values[i] = flux_face_values[i, e] * mesh.md.nxyzJ[dim][i, e]
      end
      apply_to_each_field(mul_by_accum!(dg.basis.LIFT), view(u_grad[dim], :, e), local_flux_values)
    end
  end
end

function calc_gradient!(u_grad, u::StructArray, mesh::DGMultiMesh,
                        equations::AbstractParabolicEquations,
                        boundary_conditions, dg::DGMulti, cache, parabolic_cache)

  for dim in 1:length(u_grad)
    reset_du!(u_grad[dim], dg)
  end

  # compute volume contributions to gradients
  @threaded for e in eachelement(mesh, dg)
    for i in eachdim(mesh), j in eachdim(mesh)
      dxidxhatj = mesh.md.rstxyzJ[i, j][1, e] # assumes mesh is affine
      StructArrays.foreachfield(mul_by_accum!(dg.basis.Drst[j], dxidxhatj),
                                view(u_grad[i], :, e), view(u, :, e))
    end
  end

  prolong2interfaces!(cache.u_face_values, u, mesh, equations, dg.surface_integral, dg, cache)

  # compute fluxes at interfaces
  @unpack u_face_values, flux_face_values = cache
  @unpack mapM, mapP, Jf = mesh.md
  @threaded for face_node_index in each_face_node_global(mesh, dg, cache, parabolic_cache)
    idM, idP = mapM[face_node_index], mapP[face_node_index]
    uM = u_face_values[idM]
    # compute flux if node is not a boundary node
    if idM != idP
      uP = u_face_values[idP]
      flux_face_values[idM] = 0.5 * (uP - uM) # TODO: use strong/weak formulation?
    end
  end

  # compute surface contributions
  calc_gradient_surface_integral(u_grad, u, flux_face_values,
                                 mesh, equations, dg, cache, parabolic_cache)

  calc_gradient_boundary_integral!(u_grad, u, mesh, equations, boundary_conditions, dg, cache, parabolic_cache)

  for dim in eachdim(mesh)
    invert_jacobian!(u_grad[dim], mesh, equations, dg, cache)
  end
end

# This routine differs from the one in dgmulti/dg.jl in that we do not negate the result.
function invert_jacobian!(du, mesh::DGMultiMesh, equations::AbstractParabolicEquations,
                          dg::DGMulti, cache)
  @threaded for i in Base.OneTo(length(du))
    du[i] = du[i] * cache.invJ[i]
  end
end

# do nothing for periodic domains
function calc_gradient_boundary_integral!(du, u, mesh, equations, ::BoundaryConditionPeriodic,
                                          dg, cache, parabolic_cache)
  return nothing
end

function calc_viscous_fluxes!(u_flux, u, grad_u, mesh::DGMultiMesh,
                              equations::AbstractParabolicEquations,
                              dg::DGMulti, cache, parabolic_cache)
  @threaded for i in eachindex(u)
    u_flux[i] = flux(u, grad_u, equations)
  end
end

# function calc_divergence!(du, u::StructArray, mesh::DGMultiMesh,
#                           equations::AbstractParabolicEquations,
#                           boundary_conditions, dg::DGMulti, cache, parabolic_cache)
#   calc_divergence_volume_integral!(du, u, mesh, equations, dg, cache, parabolic_cache)
#   calc_divergence_surface_integral!(du, u, mesh, equations, dg, cache, parabolic_cache)
#   calc_divergence_boundary_integral!(du, u, mesh, equations, boundary_conditions, dg, cache, parabolic_cache)
# end

# # assumptions: parabolic terms are of the form div(f(u, grad(u))) and
# # will be discretized in first order form
# #               - compute grad(u)
# #               - compute f(u, grad(u))
# #               - compute div(u)
# # boundary conditions will be applied to both grad(u) and div(u).
# function rhs!(du, u, mesh::DGMultiMesh, equations::AbstractParabolicEquations,
#               initial_condition, boundary_conditions, source_terms,
#               dg::DGMulti, cache, parabolic_cache)
#   @unpack u_transformed, grad_u, viscous_flux = parabolic_cache
#   transform_variables!(u_transformed, u, equations)
#   calc_gradient!(grad_u, u_transformed, mesh, equations,
#                  boundary_conditions, dg, cache, parabolic_cache)
#   calc_viscous_fluxes!(viscous_flux, u_transformed, grad_u,
#                        mesh, equations, dg, cache, parabolic_cache)
#   calc_divergence!(du, grad_u, mesh, equations, boundary_conditions, dg, cache, parabolic_cache)
# end
