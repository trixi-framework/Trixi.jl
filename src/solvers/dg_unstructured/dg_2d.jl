
using UnPack
using TimerOutputs

# this type of unstructured mesh is really a sophisticated DG "container" so I am not sure where it goes
include("unstructured_mesh.jl")

# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::UnstructuredMesh, equations::Trixi.AbstractEquations,
                      dg::Trixi.DG, RealT)

  # extract the elements and interfaces out of the mesh container and into cache
  cache = (; mesh.elements, mesh.interfaces, mesh.boundaries)

  return cache
end

@inline ndofs(mesh::UnstructuredMesh, dg::DG, cache) = nelements(cache.elements) * nnodes(dg)^ndims(mesh)

# @inline function wrap_array(u_ode::AbstractVector, mesh::TreeMesh{2}, equations, dg::DG, cache)
#   @boundscheck begin
#     @assert length(u_ode) == nvariables(equations) * nnodes(dg)^ndims(mesh) * nelements(dg, cache)
#   end
#   # We would like to use
#   #   reshape(u_ode, (nvariables(equations), nnodes(dg), nnodes(dg), nelements(dg, cache)))
#   # but that results in
#   #   ERROR: LoadError: cannot resize array with shared data
#   # when we resize! `u_ode` during AMR.
#
#   # The following version is fast and allows us to `resize!(u_ode, ...)`.
#   # OBS! Remember to `GC.@preserve` temporaries such as copies of `u_ode`
#   #      and other stuff that is only used indirectly via `wrap_array` afterwards!
#   unsafe_wrap(Array{eltype(u_ode), ndims(mesh)+2}, pointer(u_ode),
#               (nvariables(equations), nnodes(dg), nnodes(dg), nelements(dg, cache)))
# end
#
#
# function compute_coefficients!(u, func, t, mesh::TreeMesh{2}, equations, dg::DG, cache)
#
#   @threaded for element in eachelement(dg, cache)
#     for j in eachnode(dg), i in eachnode(dg)
#       x_node = get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, element)
#       u_node = func(x_node, t, equations)
#       set_node_vars!(u, u_node, equations, dg, i, j, element)
#     end
#   end
# end
#

# TODO: Taal discuss/refactor timer, allowing users to pass a custom timer?

function rhs!(du::AbstractArray{<:Any,4}, u, t,
              mesh::UnstructuredMesh, equations,
              initial_condition, boundary_conditions, source_terms,
              dg::DG, cache)
  # Reset du
  @timeit_debug timer() "reset ∂u/∂t" du .= zero(eltype(du))

  # Calculate volume integral
  @timeit_debug timer() "volume integral" calc_volume_integral!(du, u, Trixi.have_nonconservative_terms(equations), equations,
                                                                dg.volume_integral, dg, cache)

  # Prolong solution to interfaces
  @timeit_debug timer() "prolong2interfaces" prolong2interfaces!(cache, u, equations, dg)

  # Calculate interface fluxes
  @timeit_debug timer() "interface flux" calc_interface_flux!(cache.elements.surface_flux_values,
                                                              Trixi.have_nonconservative_terms(equations),
                                                              equations,
                                                              dg)

  # Prolong solution to boundaries
  @timeit_debug timer() "prolong2boundaries" prolong2boundaries!(cache, u, equations, dg)

  # Calculate boundary fluxes
  #  TODO: remove initial condition as an input argument here, only needed for hacky BCs
  @timeit_debug timer() "boundary flux" calc_boundary_flux!(cache, t, boundary_conditions, equations, dg, initial_condition)

  # Calculate surface integrals
  @timeit_debug timer() "surface integral" calc_surface_integral!(du, equations, dg, cache)

  # Apply Jacobian from mapping to reference element
  @timeit_debug timer() "Jacobian" apply_jacobian!(du, equations, dg, cache)

  # Calculate source terms
  @timeit_debug timer() "source terms" calc_sources!(du, u, t, source_terms, equations, dg, cache)

  return nothing
end


function calc_volume_integral!(du::AbstractArray{<:Any,4}, u,
                               nonconservative_terms::Val{false}, equations,
                               volume_integral::VolumeIntegralWeakForm,
                               dg::DGSEM, cache)
  @unpack geometry = cache.elements
  @unpack derivative_dhat = dg.basis

#  @threaded for element in eachelement(cache.mesh.elements)
  for element in eachelement(cache.elements)
    for j in eachnode(dg), i in eachnode(dg)
      u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)

      # compute the physical fluxes in each Cartesian direction
      x_flux = flux(u_node, 1, equations)
      y_flux = flux(u_node, 2, equations)

      # compute the contravariant flux in the x-direction
      flux1  = geometry[element].Y_eta[i,j] * x_flux - geometry[element].X_eta[i,j] * y_flux
      for ii in eachnode(dg)
        integral_contribution = derivative_dhat[ii, i] * flux1
        Trixi.add_to_node_vars!(du, integral_contribution, equations, dg, ii, j, element)
      end

      # compute the contravariant flux in the y-direction
      flux2  = -geometry[element].Y_xi[i,j] * x_flux + geometry[element].X_xi[i,j] * y_flux
      for jj in eachnode(dg)
        integral_contribution = derivative_dhat[jj, j] * flux2
        Trixi.add_to_node_vars!(du, integral_contribution, equations, dg, i, jj, element)
      end
    end
  end

  return nothing
end


# prolong the solution into the convenience array in the interior interface container
function prolong2interfaces!(cache, u::AbstractArray{<:Any,4}, equations, dg::DG)
  @unpack interfaces = cache

#  @threaded for interface in eachinterface(interfaces)
  for interface in eachinterface(interfaces)
    primary_element   = interfaces.element_ids[1, interface]
    secondary_element = interfaces.element_ids[2, interface]

    primary_side   = interfaces.element_side_ids[1, interface]
    secondary_side = interfaces.element_side_ids[2, interface]

    if primary_side == 1
      for i in eachnode(dg), v in eachvariable(equations)
        interfaces.u[1, v, i, interface] = u[v, i, 1, primary_element]
      end
    elseif primary_side == 2
      for i in eachnode(dg), v in eachvariable(equations)
        interfaces.u[1, v, i, interface] = u[v, nnodes(dg), i, primary_element]
      end
    elseif primary_side == 3
      for i in eachnode(dg), v in eachvariable(equations)
        interfaces.u[1, v, i, interface] = u[v, i, nnodes(dg), primary_element]
      end
    else # primary_side == 4
      for i in eachnode(dg), v in eachvariable(equations)
        interfaces.u[1, v, i, interface] = u[v, 1, i, primary_element]
      end
    end

    if secondary_side == 1
      for i in eachnode(dg), v in eachvariable(equations)
        interfaces.u[2, v, i, interface] = u[v, i, 1, secondary_element]
      end
    elseif secondary_side == 2
      for i in eachnode(dg), v in eachvariable(equations)
        interfaces.u[2, v, i, interface] = u[v, nnodes(dg), i, secondary_element]
      end
    elseif secondary_side == 3
      for i in eachnode(dg), v in eachvariable(equations)
        interfaces.u[2, v, i, interface] = u[v, i, nnodes(dg), secondary_element]
      end
    else # secondary_side == 4
      for i in eachnode(dg), v in eachvariable(equations)
        interfaces.u[2, v, i, interface] = u[v, 1, i, secondary_element]
      end
    end
  end

  return nothing
end


function calc_interface_flux!(surface_flux_values::AbstractArray{<:Any,4},
                              nonconservative_terms::Val{false}, equations, dg::DG)
  @unpack surface_flux = dg
  @unpack u, start_index, inc_index, element_ids, element_side_ids = cache.interfaces
  @unpack geometry = cache.elements


#  @threaded for interface in eachinterface(cache.mesh.interfaces)
  for interface in eachinterface(cache.interfaces)
    # Get neighboring elements
    primary_element   = element_ids[1, interface]
    secondary_element = element_ids[2, interface]

    # Get the local side id on which to compute the flux
    primary_side   = element_side_ids[1, interface]
    secondary_side = element_side_ids[2, interface]

    # initial index for the coordinate system on the secondary element
    j = start_index[interface]

    # loop through the primary element coordinate system and compute the interface coupling
    for i in eachnode(dg)
      # pull the primary and secondary states from the boundary u values
      u_ll = u[1, :, i, interface]
      u_rr = u[2, :, j, interface]

      # pull the directional vectors and scaling factors
      #   Note! this assumes a conforming approximation, more must be done in terms of the normals
      #         and tangents for hanging nodes and other non-conforming approximation spaces
      n_vec   = geometry[primary_element].normals[i, primary_side, :]
      t_vec   = geometry[primary_element].tangents[i, primary_side, :]
      scal_ll = geometry[primary_element].scaling[i, primary_side]
      scal_rr = geometry[secondary_element].scaling[j, secondary_side]

      # rotate states
      u_tilde_ll = rotate_solution(u_ll, n_vec, t_vec, equations)
      u_tilde_rr = rotate_solution(u_rr, n_vec, t_vec, equations)

      # Call pointwise Riemann solver in the rotated direction
      flux_tilde = surface_flux(u_tilde_ll, u_tilde_rr, 1, equations)

      # backrotate the flux into the original direction
      flux = backrotate_flux(flux_tilde, n_vec, t_vec, equations)

      # Scale the flux appropriately and copy back to primary/secondary element storage
      # Note the sign change for the normal flux in the secondary element!
      for v in eachvariable(equations)
        surface_flux_values[v, i, primary_side  , primary_element  ] =  flux[v] * scal_ll
        surface_flux_values[v, j, secondary_side, secondary_element] = -flux[v] * scal_rr
      end

      # increment the index of the coordinate system in the secondary element
      j += inc_index[interface]
    end
  end

  return nothing
end


##
# TODO: move these rotation routines into compressible_euler_2d.jl as they are equation dependent
@inline function rotate_solution(u, normal, tangent, equations::CompressibleEulerEquations2D)

  u_tilde1 = u[1]
  u_tilde2 = u[2] * normal[1]  + u[3] * normal[2]
  u_tilde3 = u[2] * tangent[1] + u[3] * tangent[2]
  u_tilde4 = u[4]

  return SVector(u_tilde1, u_tilde2, u_tilde3, u_tilde4)
end


@inline function backrotate_flux(f_tilde, normal, tangent, equations::CompressibleEulerEquations2D)

  f1 = f_tilde[1]
  f2 = f_tilde[2] * normal[1] + f_tilde[3] * tangent[1]
  f3 = f_tilde[2] * normal[2] + f_tilde[3] * tangent[2]
  f4 = f_tilde[4]

  return SVector(f1, f2, f3, f4)
end


function prolong2boundaries!(cache, u::AbstractArray{<:Any,4}, equations, dg::DG)
  @unpack boundaries = cache

#  @threaded for boundary in eachboundary(boundaries)
  for boundary in eachboundary(boundaries)
    element = boundaries.element_id[boundary]
    side    = boundaries.element_side_id[boundary]

    if side == 1
      for l in eachnode(dg), v in eachvariable(equations)
        boundaries.u[1, v, l, boundary] = u[v, l, 1, element]
      end
    elseif side == 2
      for l in eachnode(dg), v in eachvariable(equations)
        boundaries.u[1, v, l, boundary] = u[v, nnodes(dg), l, element]
      end
    elseif side == 3
      for l in eachnode(dg), v in eachvariable(equations)
        boundaries.u[1, v, l, boundary] = u[v, l, nnodes(dg), element]
      end
    else # side == 4
      for l in eachnode(dg), v in eachvariable(equations)
        boundaries.u[1, v, l, boundary] = u[v, 1, l, element]
      end
    end
  end

  return nothing
end


# TODO: Taal dimension agnostic
function calc_boundary_flux!(cache, t, boundary_condition::Trixi.BoundaryConditionPeriodic,
                             equations::Trixi.AbstractEquations{2}, dg::DG, initial_condition)
  @assert isempty(eachboundary(cache.boundaries))
end


# TODO: last argument for convenience, cleanup later with better boundar condition handling
function calc_boundary_flux!(cache, t, boundary_condition, equations, dg::DG, initial_condition)

  @unpack surface_flux = dg
  @unpack geometry, surface_flux_values = cache.elements
  @unpack u, element_id, element_side_id, name  = cache.boundaries

#  @threaded for boundary in eachboundary(cache.mesh.boundaries)
  for boundary in eachboundary(cache.boundaries)
    # Get the element and side IDs on the primary element
    primary_element = element_id[boundary]
    primary_side    = element_side_id[boundary]

    for i in eachnode(dg)
      # hacky way to set "exact solution" boundary conditions. Only used to test the orientation
      # for a mesh with flipped elements
      u_ext = initial_condition( ( geometry[primary_element].x_bndy[i, primary_side],
                                   geometry[primary_element].y_bndy[i, primary_side] ) ,
                                   t, equations)

      # pull the left state from the boundary u values on the primary element as well as the
      # directional vectors and scaling
      #   Note! this assumes a conforming approximation, more must be done in terms of the normals
      #         and tangents for hanging nodes and other non-conforming approximation spaces
      u_ll  = u[1, :, i, boundary]
      n_vec = geometry[primary_element].normals[i, primary_side, :]
      t_vec = geometry[primary_element].tangents[i, primary_side, :]
      scal  = geometry[primary_element].scaling[i, primary_side]

      # rotate states
      u_tilde_ll  = rotate_solution(u_ll , n_vec, t_vec, equations)
      u_tilde_ext = rotate_solution(u_ext, n_vec, t_vec, equations)

      # Call pointwise Riemann solver in the rotated direction
      flux_tilde = surface_flux(u_tilde_ll, u_tilde_ext, 1, equations)

      # backrotate the flux into the original direction
      flux = backrotate_flux(flux_tilde, n_vec, t_vec, equations)

      # Scale the flux appropriately and copy back to primary element storage
      for v in eachvariable(equations)
        surface_flux_values[v, i, primary_side, primary_element] = flux[v] * scal
      end
    end
  end

  return nothing
end


function calc_surface_integral!(du::AbstractArray{<:Any,4}, equations, dg::DGSEM, cache)
  @unpack boundary_interpolation = dg.basis
  @unpack surface_flux_values = cache.elements

#  @threaded for element in eachelement(cache.mesh.elements)
  for element in eachelement(cache.elements)
    for l in eachnode(dg), v in eachvariable(equations)
      # surface contribution along local sides 2 and 4 (fixed x and y varies)
      du[v, 1,          l, element] += ( surface_flux_values[v, l, 4, element]
                                          * boundary_interpolation[1, 1] )
      du[v, nnodes(dg), l, element] += ( surface_flux_values[v, l, 2, element]
                                          * boundary_interpolation[nnodes(dg), 2] )
      # surface contribution along local sides 1 and 3 (fixed y and x varies)
      du[v, l, 1,          element] += ( surface_flux_values[v, l, 1, element]
                                          * boundary_interpolation[1, 1] )
      du[v, l, nnodes(dg), element] += ( surface_flux_values[v, l, 3, element]
                                          * boundary_interpolation[nnodes(dg), 2] )
    end
  end

  return nothing
end


function apply_jacobian!(du::AbstractArray{<:Any,4}, equations, dg::DG, cache)
  @unpack geometry = cache.elements

#  @threaded for element in eachelement(cache.mesh.elements)
  for element in eachelement(cache.elements)
    for j in eachnode(dg), i in eachnode(dg)
      factor = -geometry[element].invJac[i,j]
      for v in eachvariable(equations)
        du[v, i, j, element] *= factor
      end
    end
  end

  return nothing
end


# TODO: Taal dimension agnostic
function calc_sources!(du::AbstractArray{<:Any,4}, u, t, source_terms::Nothing, equations, dg::DG, cache)
  return nothing
end


function calc_sources!(du::AbstractArray{<:Any,4}, u, t, source_terms, equations, dg::DG, cache)
  @unpack geometry = cache.elements

#  @threaded for element in eachelement(cache.mesh.elements)
  for element in eachelement(cache.elements)
    for j in eachnode(dg), i in eachnode(dg)
      u_local = Trixi.get_node_vars(u, equations, dg, i, j, element)
      x_local = ( geometry[element].x[i,j] , geometry[element].y[i,j] )
      du_local = source_terms(u_local, x_local, t, equations)
      Trixi.add_to_node_vars!(du, du_local, equations, dg, i, j, element)
    end
  end

  return nothing
end
