module DgSolver

include("interpolation.jl")
include("dg_containers.jl")

using ...Trixi
using ..Solvers # Use everything to allow method extension via "function <parent_module>.<method>"
using ...Equations: AbstractEquation, initial_conditions, calcflux!, riemann!, sources, calc_max_dt
import ...Equations: nvariables # Import to allow method extension
using ...Auxiliary: timer
using ...Mesh: TreeMesh
using ...Mesh.Trees: leaf_cells, length_at_cell, n_directions, has_neighbor,
                     opposite_direction, has_coarse_neighbor, has_child
using .Interpolation: interpolate_nodes, calc_dhat,
                      polynomial_interpolation_matrix, calc_lhat, gauss_lobatto_nodes_weights
using StaticArrays: SVector, SMatrix, MMatrix, MArray
using TimerOutputs: @timeit
using Printf: @sprintf, @printf

export Dg
export set_initial_conditions
export nvariables
export equations
export polydeg
export rhs!
export calc_dt
export calc_error_norms
export analyze_solution


# Main DG data structure that contains all relevant data for the DG solver
struct Dg{Eqn <: AbstractEquation, V, N, Np1, NAna, NAnap1} <: AbstractSolver
  equations::Eqn
  elements::ElementContainer{V, N}
  n_elements::Int

  surfaces::SurfaceContainer{V, N}
  n_surfaces::Int

  nodes::SVector{Np1}
  weights::SVector{Np1}
  dhat::SMatrix{Np1, Np1}
  lhat::SMatrix{Np1, 2}

  analysis_nodes::SVector{NAnap1}
  analysis_weights::SVector{NAnap1}
  analysis_weights_volume::SVector{NAnap1}
  analysis_vandermonde::SMatrix{NAnap1, Np1}
  analysis_total_volume::Float64
end


# Convenience constructor to create DG solver instance
function Dg(equation::AbstractEquation{V}, mesh::TreeMesh, N::Int) where V
  # Get cells for which an element needs to be created (i.e., all leaf cells)
  leaf_cell_ids = leaf_cells(mesh.tree)
  n_elements = length(leaf_cell_ids)

  # Initialize elements
  elements = ElementContainer{V, N}(n_elements)
  elements.cell_ids .= leaf_cell_ids

  # Initialize surfaces
  n_surfaces = count_required_surfaces(mesh, leaf_cell_ids)
  @assert n_surfaces == 2*n_elements ("For 2D and periodic domains and conforming elements, "
                                      * "n_surf must be the same as 2*n_elem")
  surfaces = SurfaceContainer{V, N}(n_surfaces)

  # Connect surfaces and elements
  init_connectivity!(elements, surfaces, mesh)

  # Initialize interpolation data structures
  n_nodes = N + 1
  nodes, weights = gauss_lobatto_nodes_weights(n_nodes)
  dhat = calc_dhat(nodes, weights)
  lhat = zeros(n_nodes, 2)
  lhat[:, 1] = calc_lhat(-1.0, nodes, weights)
  lhat[:, 2] = calc_lhat( 1.0, nodes, weights)

  # Initialize data structures for error analysis (by default, we use twice the
  # number of analysis nodes as the normal solution)
  NAna = 2 * (n_nodes) - 1
  analysis_nodes, analysis_weights = gauss_lobatto_nodes_weights(NAna + 1)
  analysis_weights_volume = analysis_weights
  analysis_vandermonde = polynomial_interpolation_matrix(nodes, analysis_nodes)
  analysis_total_volume = mesh.tree.length_level_0^ndim

  # Create actual DG solver instance
  dg = Dg{typeof(equation), V, N, n_nodes, NAna, NAna + 1}(
      equation, elements, n_elements, surfaces, n_surfaces, nodes, weights,
      dhat, lhat, analysis_nodes, analysis_weights, analysis_weights_volume,
      analysis_vandermonde, analysis_total_volume)

  # Calculate inverse Jacobian and node coordinates
  for element_id in 1:n_elements
    # Get cell id
    cell_id = leaf_cell_ids[element_id]

    # Get cell length
    dx = length_at_cell(mesh.tree, cell_id)

    # Calculate inverse Jacobian as 1/(h/2)
    dg.elements.inverse_jacobian[element_id] = 2/dx

    # Calculate node coordinates
    for j = 1:n_nodes
      for i = 1:n_nodes
        dg.elements.node_coordinates[1, i, j, element_id] = (
            mesh.tree.coordinates[1, cell_id] + dx/2 * nodes[i])
        dg.elements.node_coordinates[2, i, j, element_id] = (
            mesh.tree.coordinates[2, cell_id] + dx/2 * nodes[j])
      end
    end
  end

  return dg
end


# Return polynomial degree for a DG solver
@inline polydeg(::Dg{Eqn, V, N}) where {Eqn, V, N} = N


# Return number of nodes in one direction
@inline nnodes(::Dg{Eqn, V, N}) where {Eqn, V, N} = N + 1


# Return system of equations instance for a DG solver
@inline Solvers.equations(dg::Dg) = dg.equations


# Return number of variables for the system of equations in use
@inline nvariables(dg::Dg) = nvariables(equations(dg))


# Return number of degrees of freedom
@inline Solvers.ndofs(dg::Dg) = dg.n_elements * (polydeg(dg) + 1)^ndim


# Count the number of surfaces that need to be created
function count_required_surfaces(mesh::TreeMesh, cell_ids)
  count = 0

  # Iterate over all cells
  for cell_id in cell_ids
    for dir in 1:n_directions(mesh.tree)
      # Only count surfaces in positive direction to avoid double counting
      if dir % 2 == 0
        count += 1
        continue
      end
    end
  end

  return count
end


# Initialize connectivity between elements and surfaces
function init_connectivity!(elements, surfaces, mesh)
  # Construct cell -> element mapping for easier algorithm implementation
  tree = mesh.tree
  c2e = zeros(Int, length(tree))
  for element_id in 1:nelements(elements)
    c2e[elements.cell_ids[element_id]] = element_id
  end

  # Reset surface count
  count = 0

  # Iterate over all elements to find neighbors and to connect via surfaces
  for element_id in 1:nelements(elements)
    # Get cell id
    cell_id = elements.cell_ids[element_id]

    # Find neighbor cell in positive x- and y-direction (+x -> 2, +y -> 4)
    for direction in [2, 4]
      if has_neighbor(tree, cell_id, direction)
        # Find direct neighbor
        neighbor_cell_id = tree.neighbor_ids[direction, cell_id]

        # Check if neighbor has children - if yes, find appropriate child neighbor
        if has_child(tree, neighbor_cell_id, opposite_direction(direction))
          neighbor_cell_id = tree.child_ids[opposite_direction(direction), neighbor_cell_id]
          error("this cannot happen for conforming meshes")
        end
      elseif has_coarse_neighbor(tree, cell_id, direction)
        neighbor_cell_id = tree.neighbor_ids[direction, tree.parent_ids[cell_id]]
        error("this cannot happen for conforming meshes")
      else
        error("this cannot happen with periodic BCs")
      end

      # Create surface between elements (1 -> "left" of surface, 2 -> "right" of surface)
      count += 1
      surfaces.neighbor_ids[2, count] = c2e[neighbor_cell_id]
      surfaces.neighbor_ids[1, count] = element_id

      # Set orientation (x -> 1, y -> 2)
      surfaces.orientations[count] = div(direction, 2)

      # Set surface ids in elements
      elements.surface_ids[direction, element_id] = count
      elements.surface_ids[opposite_direction(direction), c2e[neighbor_cell_id]] = count
    end
  end

  @assert count == nsurfaces(surfaces) ("Actual surface count ($count) does not match " *
                                        "expectations $(nsurfaces(surfaces))")
end


# Calculate L2/Linf error norms based on "exact solution"
function calc_error_norms(dg::Dg, t::Float64)
  # Gather necessary information
  equation = equations(dg)
  n_nodes_analysis = length(dg.analysis_nodes)

  # Set up data structures
  l2_error = zeros(nvariables(equation))
  linf_error = zeros(nvariables(equation))
  u_exact = zeros(nvariables(equation))

  # Iterate over all elements for error calculations
  for element_id = 1:dg.n_elements
    # Interpolate solution and node locations to analysis nodes
    u = interpolate_nodes(dg.elements.u[:, :, :, element_id],
                          dg.analysis_vandermonde, nvariables(equation))
    x = interpolate_nodes(dg.elements.node_coordinates[:, :, :, element_id],
                          dg.analysis_vandermonde, ndim)

    # Calculate errors at each analysis node
    weights = dg.analysis_weights_volume
    jacobian_volume = (1 / dg.elements.inverse_jacobian[element_id])^ndim
    for j = 1:n_nodes_analysis
      for i = 1:n_nodes_analysis
        u_exact = initial_conditions(equation, x[:, i, j], t)
        diff = similar(u_exact)
        @. diff = u_exact - u[:, i, j]
        @. l2_error += diff^2 * weights[i] * weights[j] * jacobian_volume
        @. linf_error = max(linf_error, abs(diff))
      end
    end
  end

  # For L2 error, divide by total volume
  @. l2_error = sqrt(l2_error / dg.analysis_total_volume)

  return l2_error, linf_error
end


# Calculate error norms and print information for user
function Solvers.analyze_solution(dg, time::Real, dt::Real, step::Integer,
                                  runtime_absolute::Real, runtime_relative::Real)
  equation = equations(dg)

  l2_error, linf_error = calc_error_norms(dg, time)

  println()
  println("-"^80)
  println(" Simulation running '$(equation.name)' with N = $(polydeg(dg))")
  println("-"^80)
  println(" #timesteps:    " * @sprintf("% 14d", step))
  println(" dt:            " * @sprintf("%10.8e", dt))
  println(" sim. time:     " * @sprintf("%10.8e", time))
  println(" run time:      " * @sprintf("%10.8e s", runtime_absolute))
  println(" Time/DOF/step: " * @sprintf("%10.8e s", runtime_relative))
  print(" Variable:    ")
  for v in 1:nvariables(equation)
    @printf("  %-14s", equation.varnames_cons[v])
  end
  println()
  print(" L2 error:    ")
  for v in 1:nvariables(equation)
    @printf("  %10.8e", l2_error[v])
  end
  println()
  print(" Linf error:  ")
  for v in 1:nvariables(equation)
    @printf("  %10.8e", linf_error[v])
  end
  println()
  println()
end


# Call equation-specific initial conditions functions and apply to all elements
function Solvers.set_initial_conditions(dg::Dg, time::Float64)
  equation = equations(dg)

  for element_id = 1:dg.n_elements
    for j = 1:nnodes(dg)
      for i = 1:nnodes(dg)
        dg.elements.u[:, i, j, element_id] .= initial_conditions(
            equation, dg.elements.node_coordinates[:, i, j, element_id], time)
      end
    end
  end
end


# Calculate time derivative
function Solvers.rhs!(dg::Dg, t_stage)
  # Reset u_t
  @timeit timer() "reset ∂u/∂t" dg.elements.u_t .= 0.0

  # Calculate volume integral
  @timeit timer() "volume integral" calc_volume_integral!(dg, dg.elements.u_t, dg.dhat)

  # Prolong solution to surfaces
  @timeit timer() "prolong2surfaces" prolong2surfaces!(dg)

  # Calculate surface fluxes
  @timeit timer() "surface flux" calc_surface_flux!(dg.elements.surface_flux,
                                                    dg.surfaces.neighbor_ids, dg.surfaces.u, dg, 
                                                    dg.surfaces.orientations)

  # Calculate surface integrals
  @timeit timer() "surface integral" calc_surface_integral!(dg, dg.elements.u_t,
                                                            dg.elements.surface_flux, 
                                                            dg.lhat, dg.elements.surface_ids)

  # Apply Jacobian from mapping to reference element
  @timeit timer() "Jacobian" apply_jacobian!(dg)

  # Calculate source terms
  @timeit timer() "source terms" calc_sources!(dg, t_stage)
end


# Calculate volume integral and update u_t
function calc_volume_integral!(dg, u_t::Array{Float64, 4}, dhat::SMatrix)
  #=@inbounds Threads.@threads for element_id = 1:dg.n_elements=#
  for element_id = 1:dg.n_elements
    # Calculate volume fluxes
    f1 = Array{Float64, 3}(undef, nvariables(dg), nnodes(dg), nnodes(dg))
    f2 = Array{Float64, 3}(undef, nvariables(dg), nnodes(dg), nnodes(dg))
    calcflux!(f1, f2, equations(dg), dg.elements.u, element_id, nnodes(dg))

    # Calculate volume integral
    for j = 1:nnodes(dg)
      for i = 1:nnodes(dg)
        for v = 1:nvariables(dg)
          for l = 1:nnodes(dg)
            u_t[v, i, j, element_id] += dhat[i, l] * f1[v, l, j] + dhat[j, l] * f2[v, i, l]
          end
        end
      end
    end
  end
end


# Prolong solution to surfaces (for GL nodes: just a copy)
function prolong2surfaces!(dg)
  equation = equations(dg)

  for s = 1:dg.n_surfaces
    left_element_id = dg.surfaces.neighbor_ids[1, s]
    right_element_id = dg.surfaces.neighbor_ids[2, s]
    for l = 1:nnodes(dg)
      for v = 1:nvariables(dg)
        if dg.surfaces.orientations[s] == 1
          # Surface in x-direction
          dg.surfaces.u[1, v, l, s] = dg.elements.u[v, nnodes(dg), l, left_element_id]
          dg.surfaces.u[2, v, l, s] = dg.elements.u[v,          1, l, right_element_id]
        else
          # Surface in y-direction
          dg.surfaces.u[1, v, l, s] = dg.elements.u[v, l, nnodes(dg), left_element_id]
          dg.surfaces.u[2, v, l, s] = dg.elements.u[v, l,          1, right_element_id]
        end
      end
    end
  end
end


# Calculate and store fluxes across surfaces
function calc_surface_flux!(surface_flux::Array{Float64, 4}, neighbor_ids::Matrix{Int},
                            u_surfaces::Array{Float64, 4}, dg,
                            orientations::Vector{Int})
  #=@inbounds Threads.@threads for s = 1:dg.n_surfaces=#
  for s = 1:dg.n_surfaces
    # Calculate flux
    fs = Matrix{Float64}(undef, nvariables(dg), nnodes(dg))
    riemann!(fs, u_surfaces, s, equations(dg), nnodes(dg), orientations)

    # Get neighboring cells
    left_neighbor_id  = neighbor_ids[1, s]
    right_neighbor_id = neighbor_ids[2, s]

    # Determine surface direction with respect to elements:
    # orientation = 1: left -> 2, right -> 1
    # orientation = 2: left -> 4, right -> 3
    left_neighbor_direction = 2 * orientations[s]
    right_neighbor_direction = 2 * orientations[s] - 1

    # Copy flux to left and right element storage
    surface_flux[:, :, left_neighbor_direction,  left_neighbor_id]  .= fs
    surface_flux[:, :, right_neighbor_direction, right_neighbor_id] .= fs
  end
end


# Calculate surface integrals and update u_t
function calc_surface_integral!(dg, u_t::Array{Float64, 4}, surface_flux::Array{Float64, 4},
                                lhat::SMatrix, surface_ids::Matrix{Int})
  for element_id = 1:dg.n_elements
    for l = 1:nnodes(dg)
      for v = 1:nvariables(dg)
        # surface at -x
        u_t[v, 1,          l, element_id] -= surface_flux[v, l, 1, element_id] * lhat[1,          1]
        # surface at +x
        u_t[v, nnodes(dg), l, element_id] += surface_flux[v, l, 2, element_id] * lhat[nnodes(dg), 2]
        # surface at -y
        u_t[v, l, 1,          element_id] -= surface_flux[v, l, 3, element_id] * lhat[1,          1]
        # surface at +y
        u_t[v, l, nnodes(dg), element_id] += surface_flux[v, l, 4, element_id] * lhat[nnodes(dg), 2]
      end
    end
  end
end


# Apply Jacobian from mapping to reference element
function apply_jacobian!(dg)
  for element_id = 1:dg.n_elements
    for j = 1:nnodes(dg)
      for i = 1:nnodes(dg)
        for v = 1:nvariables(dg)
          dg.elements.u_t[v, i, j, element_id] *= -dg.elements.inverse_jacobian[element_id]
        end
      end
    end
  end
end


# Calculate source terms and apply them to u_t
function calc_sources!(dg::Dg, t)
  equation = equations(dg)
  if equation.sources == "none"
    return
  end

  for element_id = 1:dg.n_elements
    sources(equations(dg), dg.elements.u_t, dg.elements.u,
            dg.elements.node_coordinates, element_id, t, nnodes(dg))
  end
end


# Calculate stable time step size
function Solvers.calc_dt(dg::Dg, cfl)
  min_dt = Inf
  for element_id = 1:dg.n_elements
    dt = calc_max_dt(equations(dg), dg.elements.u, element_id, nnodes(dg),
                     dg.elements.inverse_jacobian[element_id], cfl)
    min_dt = min(min_dt, dt)
  end

  return min_dt
end

end # module
