
using UnPack
using TimerOutputs

# everything related to a DG semidiscretization in 2D,
# currently limited to Lobatto-Legendre nodes


# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::UnstructuredQuadMesh, equations::Trixi.AbstractEquations{2},
                      dg::Trixi.DG, RealT)

  # for now this is overkill but I kept the cache structure for later improvements

  # mesh including all the element geometries and interface coupling already constructed

  # save it into the cache
  cache = (; mesh)

  return cache
end


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
              mesh::UnstructuredQuadMesh, equations,
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
  #  TODO: Separate the physical boundaries from the interior boundaries. Currently this routine
  #        computes the numerical flux at all interfaces
  #  TODO: remove the last three arguments once proper boundary treatment is figured out, for now
  #        we use them to test the general flipped orientation mesh
  @timeit_debug timer() "interface flux" calc_interface_flux!(cache,
                                                              Trixi.have_nonconservative_terms(equations),
                                                              equations,
                                                              dg,
                                                              boundary_conditions,
                                                              initial_condition,
                                                              t)

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
  @unpack mesh = cache
  @unpack derivative_dhat = dg.basis

#  @threaded for element in eachelement(dg, cache)
  for element in eachelement(mesh)
    for j in eachnode(dg), i in eachnode(dg)
      u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)

      # compute the physical fluxes in each Cartesian direction
      x_flux = flux(u_node, 1, equations)
      y_flux = flux(u_node, 2, equations)

      # compute the contravariant flux in the x-direction
      flux1  = (   mesh.elements[element].geometry.Y_eta[i,j] * x_flux
                 - mesh.elements[element].geometry.X_eta[i,j] * y_flux )
      for ii in eachnode(dg)
        integral_contribution = derivative_dhat[ii, i] * flux1
        Trixi.add_to_node_vars!(du, integral_contribution, equations, dg, ii, j, element)
      end

      # compute the contravariant flux in the y-direction
      flux2  = ( - mesh.elements[element].geometry.Y_xi[i,j] * x_flux
                 + mesh.elements[element].geometry.X_xi[i,j] * y_flux )
      for jj in eachnode(dg)
        integral_contribution = derivative_dhat[jj, j] * flux2
        Trixi.add_to_node_vars!(du, integral_contribution, equations, dg, i, jj, element)
      end
    end
  end

  return nothing
end


# prolong the solution on each element into the convenience array in the element container
# note that this prolongs to ALL interfaces, i.e., on the physical boundaries and the interior
function prolong2interfaces!(cache, u::AbstractArray{<:Any,4}, equations, dg::DG)
  @unpack mesh = cache
  @unpack elements = mesh

#  @threaded for eID in eachelement(mesh)
  for k in eachelement(mesh)
    for j in eachnode(dg), v in eachvariable(equations)
      elements[k].surface_u_values[v, j, 2] = u[v, nnodes(dg), j, k]
      elements[k].surface_u_values[v, j, 4] = u[v,          1, j, k]
    end

    for i in eachnode(dg), v in eachvariable(equations)
      elements[k].surface_u_values[v, i, 1] = u[v, i,          1, k]
      elements[k].surface_u_values[v, i, 3] = u[v, i, nnodes(dg), k]
    end
  end

  return nothing
end


# for the time being this computes ALL of the interfaces fluxes
# handles the physical boundaries in a hacky way
function calc_interface_flux!(cache, nonconservative_terms::Val{false}, equations, dg::DG,
                              boundary_condition, initial_condition, t_loc)
  @unpack surface_flux = dg
  @unpack interfaces, elements = cache.mesh

#  @threaded for interface in eachinterface(cache.mesh)
  for interface in eachinterface(cache.mesh)
    # Get neighboring elements
    primary_id   = interfaces[interface].element_ids[1]
    secondary_id = interfaces[interface].element_ids[2]

    # Get the local side id on which to compute the flux
    primary_side   = interfaces[interface].element_side_ids[1]
    secondary_side = interfaces[interface].element_side_ids[2]

    if interfaces[interface].interface_type == "Interior"
      j = interfaces[interface].start_index
      for i in eachnode(dg)
        # pull the left and right states from the boundary u values
        u_ll = elements[primary_id].surface_u_values[:, i, primary_side]
        u_rr = elements[secondary_id].surface_u_values[:, j, secondary_side]
        # pull the directional vectors and scaling factors
        #   TODO: this assumes a conforming approximation, more must be done in terms of the normals
        #         and tangents for hanging nodes and other non-conforming approximation spaces
        n_vec   = elements[primary_id].geometry.normals[i, primary_side, :]
        t_vec   = elements[primary_id].geometry.tangents[i, primary_side, :]
        scal_ll = elements[primary_id].geometry.scaling[i, primary_side]
        scal_rr = elements[secondary_id].geometry.scaling[j, secondary_side]

        # rotate states
        u_tilde_ll = rotate_solution(u_ll, n_vec, t_vec, equations)
        u_tilde_rr = rotate_solution(u_rr, n_vec, t_vec, equations)

        # Call pointwise Riemann solver in the rotated direction
        flux_tilde = surface_flux(u_tilde_ll, u_tilde_rr, 1, equations)

        # backrotate the flux into the original direction
        flux = backrotate_flux(flux_tilde, n_vec, t_vec, equations)

        # Scale the flux appropriately and copy back to left and right element storage
        # Note the sign change for the normal flux in the secondary element!
        for v in eachvariable(equations)
          elements[primary_id].surface_flux_values[v, i, primary_side]     =  flux[v] * scal_ll
          elements[secondary_id].surface_flux_values[v, j, secondary_side] = -flux[v] * scal_rr
        end

        # increment the index of the coordinate system in the secondary element
        j += interfaces[interface].inc_index
      end
    else # interface_type == "Boundary"
      # Get the element and side IDs on the primary element
      primary_id   = interfaces[interface].element_ids[1]
      primary_side = interfaces[interface].element_side_ids[1]

      # Get the total number of elements in the mesh
      K = nelements(cache.mesh)

      for i in eachnode(dg)
        #  TODO: this needs generalized to handle arbitrary boundaries. For instance, right now the
        #        coordinate system of the periodic elements cannot be flipped in the current implementation
        #  TODO: separate the physical boundary flux computations into another routine to match the
        #        the existing Trixi ecosystem
        if typeof(boundary_condition) == Trixi.BoundaryConditionPeriodic
          # hacky way to set periodic BCs similar to a structured mesh
          if elements[primary_id].bndy_names[primary_side] == "Bottom"
            secondary_id   = primary_id + (K - convert(Int64, sqrt(K)))
            secondary_side = 3
          elseif elements[primary_id].bndy_names[primary_side] == "Top"
            secondary_id   = primary_id - (K - convert(Int64, sqrt(K)))
            secondary_side = 1
          elseif elements[primary_id].bndy_names[primary_side] == "Left"
            secondary_id   = primary_id + (convert(Int64, sqrt(K)) - 1)
            secondary_side = 2
          elseif elements[primary_id].bndy_names[primary_side] == "Right"
            secondary_id   = primary_id - (convert(Int64, sqrt(K)) - 1)
            secondary_side = 4
          end
          u_rr = elements[secondary_id].surface_u_values[:, i, secondary_side]
        else
          # hacky way to set "exact solution" boundary conditions. Only used to test the orientation
          # for a mesh with flipped elements
          u_rr = initial_condition( ( elements[primary_id].geometry.x_bndy[i, primary_side],
                                      elements[primary_id].geometry.y_bndy[i, primary_side] ) ,
                                      t_loc, equations)
        end

        # pull the left state from the boundary u values on the primary element as well as the
        # directional vectors and scaling
        #   TODO: this assumes a conforming approximation, more must be done in terms of the normals
        #         and tangents for hanging nodes and other non-conforming approximation spaces
        u_ll  = elements[primary_id].surface_u_values[:, i, primary_side]
        n_vec = elements[primary_id].geometry.normals[i, primary_side, :]
        t_vec = elements[primary_id].geometry.tangents[i, primary_side, :]
        scal  = elements[primary_id].geometry.scaling[i, primary_side]

        # rotate states
        u_tilde_ll = rotate_solution(u_ll, n_vec, t_vec, equations)
        u_tilde_rr = rotate_solution(u_rr, n_vec, t_vec, equations)

        # Call pointwise Riemann solver in the rotated direction
        flux_tilde = surface_flux(u_tilde_ll, u_tilde_rr, 1, equations)

        # backrotate the flux into the original direction
        flux = backrotate_flux(flux_tilde, n_vec, t_vec, equations)

        # Scale the flux appropriately and copy back to left and right element storage
        for v in eachvariable(equations)
          elements[primary_id].surface_flux_values[v, i, primary_side] = flux[v] * scal
        end
      end
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

# function prolong2boundaries!(cache, u::AbstractArray{<:Any,4}, equations, dg::DG)
#   @unpack boundaries = cache
#   @unpack orientations, neighbor_sides = boundaries
#
#   @threaded for boundary in eachboundary(dg, cache)
#     element = boundaries.neighbor_ids[boundary]
#
#     if orientations[boundary] == 1
#       # boundary in x-direction
#       if neighbor_sides[boundary] == 1
#         # element in -x direction of boundary
#         for l in eachnode(dg), v in eachvariable(equations)
#           boundaries.u[1, v, l, boundary] = u[v, nnodes(dg), l, element]
#         end
#       else # Element in +x direction of boundary
#         for l in eachnode(dg), v in eachvariable(equations)
#           boundaries.u[2, v, l, boundary] = u[v, 1,          l, element]
#         end
#       end
#     else # if orientations[boundary] == 2
#       # boundary in y-direction
#       if neighbor_sides[boundary] == 1
#         # element in -y direction of boundary
#         for l in eachnode(dg), v in eachvariable(equations)
#           boundaries.u[1, v, l, boundary] = u[v, l, nnodes(dg), element]
#         end
#       else
#         # element in +y direction of boundary
#         for l in eachnode(dg), v in eachvariable(equations)
#           boundaries.u[2, v, l, boundary] = u[v, l, 1,          element]
#         end
#       end
#     end
#   end
#
#   return nothing
# end


# # TODO: Taal dimension agnostic
# function calc_boundary_flux!(cache, t, boundary_condition::BoundaryConditionPeriodic,
#                              equations::AbstractEquations{2}, dg::DG)
#   @assert isempty(eachboundary(dg, cache))
# end
#
# # TODO: Taal dimension agnostic
# function calc_boundary_flux!(cache, t, boundary_condition,
#                              equations::AbstractEquations{2}, dg::DG)
#   @unpack surface_flux_values = cache.elements
#   @unpack n_boundaries_per_direction = cache.boundaries
#
#   # Calculate indices
#   lasts = accumulate(+, n_boundaries_per_direction)
#   firsts = lasts - n_boundaries_per_direction .+ 1
#
#   # Calc boundary fluxes in each direction
#   for direction in eachindex(firsts)
#     calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_condition,
#                                      equations, dg, cache,
#                                      direction, firsts[direction], lasts[direction])
#   end
# end
#
# function calc_boundary_flux!(cache, t, boundary_conditions::Union{NamedTuple,Tuple},
#                              equations::AbstractEquations{2}, dg::DG)
#   @unpack surface_flux_values = cache.elements
#   @unpack n_boundaries_per_direction = cache.boundaries
#
#   # Calculate indices
#   lasts = accumulate(+, n_boundaries_per_direction)
#   firsts = lasts - n_boundaries_per_direction .+ 1
#
#   # Calc boundary fluxes in each direction
#   calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions[1],
#                                    equations, dg, cache, 1, firsts[1], lasts[1])
#   calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions[2],
#                                    equations, dg, cache, 2, firsts[2], lasts[2])
#   calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions[3],
#                                    equations, dg, cache, 3, firsts[3], lasts[3])
#   calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions[4],
#                                    equations, dg, cache, 4, firsts[4], lasts[4])
# end
#
# function calc_boundary_flux_by_direction!(surface_flux_values::AbstractArray{<:Any,4}, t,
#                                           boundary_condition, equations, dg::DG, cache,
#                                           direction, first_boundary, last_boundary)
#   @unpack surface_flux = dg
#   @unpack u, neighbor_ids, neighbor_sides, node_coordinates, orientations = cache.boundaries
#
#   @threaded for boundary in first_boundary:last_boundary
#     # Get neighboring element
#     neighbor = neighbor_ids[boundary]
#
#     for i in eachnode(dg)
#       # Get boundary flux
#       u_ll, u_rr = get_surface_node_vars(u, equations, dg, i, boundary)
#       if neighbor_sides[boundary] == 1 # Element is on the left, boundary on the right
#         u_inner = u_ll
#       else # Element is on the right, boundary on the left
#         u_inner = u_rr
#       end
#       x = get_node_coords(node_coordinates, equations, dg, i, boundary)
#       flux = boundary_condition(u_inner, orientations[boundary], direction, x, t, surface_flux,
#                                 equations)
#
#       # Copy flux to left and right element storage
#       for v in eachvariable(equations)
#         surface_flux_values[v, i, direction, neighbor] = flux[v]
#       end
#     end
#   end
#
#   return nothing
# end
#
#
# @inline function calc_fstar!(destination::AbstractArray{<:Any,2}, equations, dg::DGSEM,
#                              u_interfaces, interface, orientation)
#   @unpack surface_flux = dg
#
#   for i in eachnode(dg)
#     # Call pointwise two-point numerical flux function
#     u_ll, u_rr = get_surface_node_vars(u_interfaces, equations, dg, i, interface)
#     flux = surface_flux(u_ll, u_rr, orientation, equations)
#
#     # Copy flux to left and right element storage
#     set_node_vars!(destination, flux, equations, dg, i)
#   end
#
#   return nothing
# end


function calc_surface_integral!(du::AbstractArray{<:Any,4}, equations, dg::DGSEM, cache)
  @unpack boundary_interpolation = dg.basis
  @unpack mesh = cache

#  @threaded for element in eachelement(mesh)
  for element in eachelement(mesh)
    for l in eachnode(dg)
      for v in eachvariable(equations)
        # surface contribution along local sides 2 and 4 (fixed x and y varies)
        du[v, 1,          l, element] += ( mesh.elements[element].surface_flux_values[v, l, 4]
                                            * boundary_interpolation[1, 1] )
        du[v, nnodes(dg), l, element] += ( mesh.elements[element].surface_flux_values[v, l, 2]
                                            * boundary_interpolation[nnodes(dg), 2] )
        # surface contribution along local sides 1 and 3 (fixed y and x varies)
        du[v, l, 1,          element] += ( mesh.elements[element].surface_flux_values[v, l, 1]
                                            * boundary_interpolation[1, 1] )
        du[v, l, nnodes(dg), element] += ( mesh.elements[element].surface_flux_values[v, l, 3]
                                            * boundary_interpolation[nnodes(dg), 2] )
      end
    end
  end

  return nothing
end


function apply_jacobian!(du::AbstractArray{<:Any,4}, equations, dg::DG, cache)

#  @threaded for element in eachelement(cache.mesh)
  for element in eachelement(cache.mesh)
    for j in eachnode(dg), i in eachnode(dg)
      factor = -cache.mesh.elements[element].geometry.invJac[i,j]
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

  x_local = zeros(2)
#  @threaded for element in eachelement(cache.mesh)
  for element in eachelement(cache.mesh)
    for j in eachnode(dg), i in eachnode(dg)
      u_local = Trixi.get_node_vars(u, equations, dg, i, j, element)
      #x_local = get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, element)
      x_local[1] = cache.mesh.elements[element].geometry.x[i,j]
      x_local[2] = cache.mesh.elements[element].geometry.y[i,j]
      du_local = source_terms(u_local, x_local, t, equations)
      Trixi.add_to_node_vars!(du, du_local, equations, dg, i, j, element)
    end
  end

  return nothing
end
