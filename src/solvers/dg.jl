module DgSolver

include("interpolation.jl")
include("dg_containers.jl")
include("l2mortar.jl")

using ...Trixi
using ..Solvers # Use everything to allow method extension via "function <parent_module>.<method>"
using ...Equations: AbstractEquation, initial_conditions, calcflux!, calcflux_twopoint!,
                    riemann!, sources, calc_max_dt,
	            cons2entropy,cons2indicator
import ...Equations: nvariables # Import to allow method extension
using ...Auxiliary: timer, parameter
using ...Mesh: TreeMesh
using ...Mesh.Trees: leaf_cells, leaf_cells_by_domain, length_at_cell, n_directions, has_neighbor,
                     opposite_direction, has_coarse_neighbor, has_child, has_children
using .Interpolation: interpolate_nodes, calc_dhat, calc_dsplit,
                      polynomial_interpolation_matrix, calc_lhat, gauss_lobatto_nodes_weights,
		      vandermonde_legendre, nodal2modal
import .L2Mortar # Import to satisfy Gregor
using ...Parallel: n_domains, domain_id, is_parallel, Request, Irecv!, @mpi_parallel, @mpi_root,
                   Isend, comm, Waitall!, Allreduce!, is_mpi_root, mpi_println, Allgather!

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
export calc_entropy_timederivative
export analyze_solution


# Main DG data structure that contains all relevant data for the DG solver
struct Dg{Eqn <: AbstractEquation, V, N, Np1, NAna, NAnap1} <: AbstractSolver
  equations::Eqn
  elements::ElementContainer{V, N}
  n_elements::Int

  surfaces::SurfaceContainer{V, N}
  n_surfaces::Int

  mpi_surfaces::MpiSurfaceContainer{V, N}
  n_mpi_surfaces::Int

  l2mortars::L2MortarContainer{V, N}
  n_l2mortars::Int

  nodes::SVector{Np1}
  weights::SVector{Np1}
  inverse_weights::SVector{Np1}
  inverse_vandermonde_legendre::SMatrix{Np1, Np1}
  lhat::SMatrix{Np1, 2}

  volume_integral_type::Symbol
  differentiation_operator::SMatrix{Np1, Np1}

  l2mortar_forward_upper::SMatrix{Np1, Np1}
  l2mortar_forward_lower::SMatrix{Np1, Np1}
  l2mortar_reverse_upper::SMatrix{Np1, Np1}
  l2mortar_reverse_lower::SMatrix{Np1, Np1}

  analysis_nodes::SVector{NAnap1}
  analysis_weights::SVector{NAnap1}
  analysis_weights_volume::SVector{NAnap1}
  analysis_vandermonde::SMatrix{NAnap1, Np1}
  analysis_total_volume::Float64

  neighbor_domains::Vector{Int}
  mpi_surfaces_by_domain::Vector{Vector{Int}}
  mpi_send_buffers::Vector{Vector{Float64}}
  mpi_recv_buffers::Vector{Vector{Float64}}
  mpi_send_requests::Vector{Request}
  mpi_recv_requests::Vector{Request}
  n_elements_by_domain::Vector{Int}
end


# Convenience constructor to create DG solver instance
function Dg(equation::AbstractEquation{V}, mesh::TreeMesh, N::Int) where V
  # Get cells for which an element needs to be created (i.e., all domain-local leaf cells)
  leaf_cell_ids = leaf_cells_by_domain(mesh.tree, domain_id())
  n_elements = length(leaf_cell_ids)

  # Initialize elements
  elements = ElementContainer{V, N}(n_elements)
  elements.cell_ids .= leaf_cell_ids

  # Initialize surfaces
  n_surfaces = count_required_surfaces(mesh, leaf_cell_ids)
  surfaces = SurfaceContainer{V, N}(n_surfaces)

  # Initialize MPI surfaces
  n_mpi_surfaces = count_required_mpi_surfaces(mesh, leaf_cell_ids)
  mpi_surfaces = MpiSurfaceContainer{V, N}(n_mpi_surfaces)

  # Initialize L2 mortars
  n_l2mortars = count_required_l2mortars(mesh, leaf_cell_ids)
  l2mortars = L2MortarContainer{V, N}(n_l2mortars)

  # Sanity check
  if n_l2mortars == 0 && !is_parallel()
    @assert n_surfaces == 2*n_elements ("For 2D and periodic domains and conforming elements, "
                                        * "n_surf must be the same as 2*n_elem")
  end

  # Connect elements with surfaces and l2mortars
  init_surface_connectivity!(elements, surfaces, mesh)
  init_mpi_surface_connectivity!(elements, mpi_surfaces, mesh)
  init_l2mortar_connectivity!(elements, l2mortars, mesh)

  # Initialize interpolation data structures
  n_nodes = N + 1
  nodes, weights = gauss_lobatto_nodes_weights(n_nodes)
  inverse_weights = 1 ./ weights
  _, inverse_vandermonde_legendre = vandermonde_legendre(nodes)
  lhat = zeros(n_nodes, 2)
  lhat[:, 1] = calc_lhat(-1.0, nodes, weights)
  lhat[:, 2] = calc_lhat( 1.0, nodes, weights)

  # Initialize differentiation operator
  volume_integral_type = Symbol(parameter("volume_integral_type", "weak_form",
                                          valid=["weak_form", "split_form", "shock_capturing"]))
  if volume_integral_type == :weak_form
    differentiation_operator = calc_dhat(nodes, weights)
  else
    # Transposed dsplit for efficiency
    differentiation_operator = transpose(calc_dsplit(nodes, weights))
  end

  # Initialize L2 mortar projection operators
  l2mortar_forward_upper = L2Mortar.calc_forward_upper(n_nodes)
  l2mortar_forward_lower = L2Mortar.calc_forward_lower(n_nodes)
  l2mortar_reverse_upper = L2Mortar.calc_reverse_upper(n_nodes)
  l2mortar_reverse_lower = L2Mortar.calc_reverse_lower(n_nodes)

  # Initialize data structures for error analysis (by default, we use twice the
  # number of analysis nodes as the normal solution)
  NAna = 2 * (n_nodes) - 1
  analysis_nodes, analysis_weights = gauss_lobatto_nodes_weights(NAna + 1)
  analysis_weights_volume = analysis_weights
  analysis_vandermonde = polynomial_interpolation_matrix(nodes, analysis_nodes)
  analysis_total_volume = mesh.tree.length_level_0^ndim

  # Initialize data structures for parallelization
  if is_parallel()
    # Determine unique list of neighbor domains
    neighbor_domains = Int[]
    for s in 1:n_mpi_surfaces
      neighbor_cell_id = mpi_surfaces.neighbor_cell_ids[s]
      neighbor_domain = mesh.tree.domain_ids[neighbor_cell_id]
      push!(neighbor_domains, neighbor_domain)
    end
    neighbor_domains = unique(sort(neighbor_domains))

    # Find all MPI surfaces
    mpi_surfaces_by_domain = [Int[] for d in 1:length(neighbor_domains)]
    for (idx, d) in enumerate(neighbor_domains)
      # First, determine all MPI surfaces for a given domain
      for s in 1:n_mpi_surfaces
        neighbor_cell_id = mpi_surfaces.neighbor_cell_ids[s]
        neighbor_domain = mesh.tree.domain_ids[neighbor_cell_id]
        if neighbor_domain == d
          push!(mpi_surfaces_by_domain[idx], s)
        end
      end

      # Then, sort in a globally unique way (by cell_id of "left" neighbor cell)
      sort!(mpi_surfaces_by_domain[idx], by =
            function(surface_id)
              if mpi_surfaces.element_sides[surface_id] == 1
                element_id = mpi_surfaces.element_ids[surface_id]
                return elements.cell_ids[element_id]
              else
                neighbor_cell_id = mpi_surfaces.neighbor_cell_ids[surface_id]
                return neighbor_cell_id
              end
            end)
    end

    # Sanity check: The total count of MPI surfaces by domain must match the number of MPI surfaces
    @assert nmpisurfaces(mpi_surfaces) == sum(length(v) for v in mpi_surfaces_by_domain) (
        "Total number of mpi_surfaces_by_domain " *
        "($(sum(length(v) for v in mpi_surfaces_by_domain))) does " *
        "not match actual number of surfaces ($(nmpisurfaces(mpi_surfaces)))")

    # Initialize buffers and requests
    mpi_send_buffers = Vector{Vector{Float64}}(undef, length(neighbor_domains))
    mpi_recv_buffers = Vector{Vector{Float64}}(undef, length(neighbor_domains))
    for (idx, d) in enumerate(neighbor_domains)
      buffer_size = length(mpi_surfaces_by_domain[idx]) * n_nodes * V
      mpi_send_buffers[idx] = Vector{Float64}(undef, buffer_size)
      mpi_recv_buffers[idx] = Vector{Float64}(undef, buffer_size)
    end
    mpi_send_requests = Vector{Request}(undef, length(neighbor_domains))
    mpi_recv_requests = Vector{Request}(undef, length(neighbor_domains))

    # Count number of elements on each domain
    n_elements_by_domain = Vector{Int}(undef, n_domains())
    n_elements_by_domain[domain_id() + 1] = n_elements
    Allgather!(n_elements_by_domain, 1, comm())

    # Sanity check: total number of elements matches number of leaf cells
    @assert sum(n_elements_by_domain) == length(leaf_cells(mesh.tree)) (
        "Total number of elements does not match total number of leaf cells.")
  else
    # Set sensible defaults for serial execution
    neighbor_domains = Int[]
    mpi_surfaces_by_domain = Int[]
    mpi_send_buffers = Vector{Vector{Float64}}()
    mpi_recv_buffers = Vector{Vector{Float64}}()
    mpi_send_requests = Vector{Request}()
    mpi_recv_requests = Vector{Request}()
    n_elements_by_domain = Int[n_elements]
  end

  # Create actual DG solver instance
  dg = Dg{typeof(equation), V, N, n_nodes, NAna, NAna + 1}(
      equation,
      elements, n_elements,
      surfaces, n_surfaces,
      mpi_surfaces, n_mpi_surfaces,
      l2mortars, n_l2mortars,
      nodes, weights, inverse_weights, inverse_vandermonde_legendre, lhat,
      volume_integral_type, differentiation_operator,
      l2mortar_forward_upper, l2mortar_forward_lower,
      l2mortar_reverse_upper, l2mortar_reverse_lower,
      analysis_nodes, analysis_weights, analysis_weights_volume,
      analysis_vandermonde, analysis_total_volume,
      neighbor_domains, mpi_surfaces_by_domain,
      mpi_send_buffers, mpi_recv_buffers, mpi_send_requests, mpi_recv_requests,
      n_elements_by_domain)

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
    for direction in 1:n_directions(mesh.tree)
      # Only count surfaces in positive direction to avoid double counting
      if direction % 2 == 1
        continue
      end

      # If no neighbor exists, current cell is small and thus we need a mortar
      if !has_neighbor(mesh.tree, cell_id, direction)
        continue
      end

      # Skip if neighbor has children
      neighbor_cell_id = mesh.tree.neighbor_ids[direction, cell_id]
      if has_children(mesh.tree, neighbor_cell_id)
        continue
      end

      # Skip if neighbor is from different domain -> requires MPI surface
      if mesh.tree.domain_ids[neighbor_cell_id] != domain_id()
        continue
      end

      count += 1
    end
  end

  return count
end


# Count the number of MPI surfaces that need to be created
function count_required_mpi_surfaces(mesh::TreeMesh, cell_ids)
  count = 0

  # Iterate over all cells
  for cell_id in cell_ids
    for direction in 1:n_directions(mesh.tree)
      # If no neighbor exists, current cell is small and thus we need a mortar
      if !has_neighbor(mesh.tree, cell_id, direction)
        continue
      end

      # Skip if neighbor has children
      neighbor_cell_id = mesh.tree.neighbor_ids[direction, cell_id]
      if has_children(mesh.tree, neighbor_cell_id)
        continue
      end

      # Skip if neighbor is from same domain -> requires normal surface
      if mesh.tree.domain_ids[neighbor_cell_id] == domain_id()
        continue
      end

      count += 1
    end
  end

  return count
end


# Count the number of L2 mortars that need to be created
function count_required_l2mortars(mesh::TreeMesh, cell_ids)
  count = 0

  # Iterate over all cells and count mortars from perspective of coarse cells
  for cell_id in cell_ids
    for direction in 1:n_directions(mesh.tree)
      # If no neighbor exists, cell is small with large neighbor -> do nothing
      if !has_neighbor(mesh.tree, cell_id, direction)
        continue
      end

      # If neighbor has no children, this is a conforming interface -> do nothing
      neighbor_cell_id = mesh.tree.neighbor_ids[direction, cell_id]
      if !has_children(mesh.tree, neighbor_cell_id)
        continue
      end

      count +=1
    end
  end

  return count
end


# Initialize connectivity between elements and surfaces
function init_surface_connectivity!(elements, surfaces, mesh)
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

    # Loop over directions
    for direction in 1:n_directions(mesh.tree)
      # Only create surfaces in positive direction
      if direction % 2 == 1
        continue
      end

      # If no neighbor exists, current cell is small and thus we need a mortar
      if !has_neighbor(mesh.tree, cell_id, direction)
        continue
      end

      # Skip if neighbor has children
      neighbor_cell_id = mesh.tree.neighbor_ids[direction, cell_id]
      if has_children(mesh.tree, neighbor_cell_id)
        continue
      end

      # Skip if neighbor is from different domain -> requires MPI surface
      if mesh.tree.domain_ids[neighbor_cell_id] != domain_id()
        continue
      end

      # Create surface between elements (1 -> "left" of surface, 2 -> "right" of surface)
      count += 1
      surfaces.neighbor_ids[2, count] = c2e[neighbor_cell_id]
      surfaces.neighbor_ids[1, count] = element_id

      # Set orientation (x -> 1, y -> 2)
      if direction in [1, 2]
        surfaces.orientations[count] = 1
      else
        surfaces.orientations[count] = 2
      end
    end
  end

  @assert count == nsurfaces(surfaces) ("Actual surface count ($count) does not match " *
                                        "expectations $(nsurfaces(surfaces))")
end


# Initialize connectivity between elements and MPI surfaces
function init_mpi_surface_connectivity!(elements, mpi_surfaces, mesh)
  # Construct cell -> element mapping for easier algorithm implementation
  tree = mesh.tree
  c2e = zeros(Int, length(tree))
  for element_id in 1:nelements(elements)
    c2e[elements.cell_ids[element_id]] = element_id
  end

  # Reset MPI surface count
  count = 0

  # Iterate over all elements to find neighbors and to connect via MPI surfaces
  for element_id in 1:nelements(elements)
    # Get cell id
    cell_id = elements.cell_ids[element_id]

    # Loop over directions
    for direction in 1:n_directions(mesh.tree)
      # If no neighbor exists, current cell is small and thus we need a mortar
      if !has_neighbor(mesh.tree, cell_id, direction)
        continue
      end

      # Skip if neighbor has children
      neighbor_cell_id = mesh.tree.neighbor_ids[direction, cell_id]
      if has_children(mesh.tree, neighbor_cell_id)
        continue
      end

      # Skip if neighbor is from same domain -> requires normal surface
      if mesh.tree.domain_ids[neighbor_cell_id] == domain_id()
        continue
      end

      # Create MPI surface (element sides: 1 -> "left" of surface, 2 -> "right" of surface)
      count += 1
      mpi_surfaces.element_ids[count] = element_id
      mpi_surfaces.neighbor_cell_ids[count] = neighbor_cell_id
      if direction in [2, 4]
        mpi_surfaces.element_sides[count] = 1
      else
        mpi_surfaces.element_sides[count] = 2
      end

      # Set orientation (x -> 1, y -> 2)
      if direction in [1, 2]
        mpi_surfaces.orientations[count] = 1
      else
        mpi_surfaces.orientations[count] = 2
      end
    end
  end

  @assert count == nmpisurfaces(mpi_surfaces) ("Actual mpi_surface count ($count) does not match " *
                                               "expectations $(nmpisurfaces(mpi_surfaces))")
end


# Initialize connectivity between elements and L2 mortars
function init_l2mortar_connectivity!(elements, l2mortars, mesh)
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

    for direction in 1:n_directions(mesh.tree)
      # If no neighbor exists, cell is small with large neighbor -> do nothing
      if !has_neighbor(mesh.tree, cell_id, direction)
        continue
      end

      # If neighbor has no children, this is a conforming interface -> do nothing
      neighbor_cell_id = mesh.tree.neighbor_ids[direction, cell_id]
      if !has_children(mesh.tree, neighbor_cell_id)
        continue
      end

      # Create mortar between elements:
      # 1 -> small element in negative coordinate direction
      # 2 -> small element in positive coordinate direction
      # 3 -> large element
      count += 1
      l2mortars.neighbor_ids[3, count] = element_id
      if direction == 1
        l2mortars.neighbor_ids[1, count] = c2e[mesh.tree.child_ids[2, neighbor_cell_id]]
        l2mortars.neighbor_ids[2, count] = c2e[mesh.tree.child_ids[4, neighbor_cell_id]]
      elseif direction == 2
        l2mortars.neighbor_ids[1, count] = c2e[mesh.tree.child_ids[1, neighbor_cell_id]]
        l2mortars.neighbor_ids[2, count] = c2e[mesh.tree.child_ids[3, neighbor_cell_id]]
      elseif direction == 3
        l2mortars.neighbor_ids[1, count] = c2e[mesh.tree.child_ids[3, neighbor_cell_id]]
        l2mortars.neighbor_ids[2, count] = c2e[mesh.tree.child_ids[4, neighbor_cell_id]]
      elseif direction == 4
        l2mortars.neighbor_ids[1, count] = c2e[mesh.tree.child_ids[1, neighbor_cell_id]]
        l2mortars.neighbor_ids[2, count] = c2e[mesh.tree.child_ids[2, neighbor_cell_id]]
      else
        error("should not happen")
      end

      # Set large side, which denotes the direction (1 -> negative, 2 -> positive) of the large side
      if direction in [2, 4]
        l2mortars.large_sides[count] = 1
      else
        l2mortars.large_sides[count] = 2
      end

      # Set orientation (x -> 1, y -> 2)
      if direction in [1, 2]
        l2mortars.orientations[count] = 1
      else
        l2mortars.orientations[count] = 2
      end
    end
  end

  @assert count == nl2mortars(l2mortars) ("Actual l2mortar count ($count) does not match " *
                                          "expectations $(nl2mortars(l2mortars))")
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

  # Collect global information
  @mpi_parallel Allreduce!(l2_error, +, comm())
  @mpi_parallel Allreduce!(linf_error, max, comm())

  # For L2 error, divide by total volume
  @. l2_error = sqrt(l2_error / dg.analysis_total_volume)

  return l2_error, linf_error
end

# Calculate L2/Linf error norms based on "exact solution"
function calc_entropy_timederivative(dg::Dg, t::Float64)
  # Gather necessary information
  equation = equations(dg)
  n_nodes = nnodes(dg)
  # Compute entropy variables for all elements and nodes with current solution u
  duds = cons2entropy(equation,dg.elements.u,n_nodes,dg.n_elements)
  # Compute ut = rhs(u) with current solution u
  Solvers.rhs!(dg, t)
  # Quadrature weights
  weights = dg.weights
  # Integrate over all elements to get the total semi-discrete entropy update
  dsdu_ut = 0.0
  for element_id = 1:dg.n_elements
    jacobian_volume = (1 / dg.elements.inverse_jacobian[element_id])^ndim
    for j = 1:n_nodes
      for i = 1:n_nodes
         dsdu_ut += jacobian_volume*weights[i]*weights[j]*sum(duds[:,i,j,element_id].*dg.elements.u_t[:,i,j,element_id])
      end
    end
  end
  # Normalize with total volume
  dsdu_ut = dsdu_ut/dg.analysis_total_volume
  return dsdu_ut
end

# Calculate error norms and print information for user
function Solvers.analyze_solution(dg, time::Real, dt::Real, step::Integer,
                                  runtime_absolute::Real, runtime_relative::Real)
  equation = equations(dg)

  l2_error, linf_error = calc_error_norms(dg, time)
  duds_ut = calc_entropy_timederivative(dg, time)

  if is_mpi_root()
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
    print(" Semi-discrete Entropy update:  ")
    @printf("  %10.8e", duds_ut)
    println()
    println()
  end
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

  # Prolong solution to MPI surfaces and start data exchange
  @mpi_parallel @timeit timer() "prolong2mpisurfaces" prolong2mpisurfaces!(dg)

  # Calculate volume integral
  @timeit timer() "volume integral" calc_volume_integral!(dg)

  # Prolong solution to surfaces
  @timeit timer() "prolong2surfaces" prolong2surfaces!(dg)

  # Calculate surface fluxes
  @timeit timer() "surface flux" calc_surface_flux!(dg.elements.surface_flux,
                                                    dg.surfaces.neighbor_ids, dg.surfaces.u, dg, 
                                                    dg.surfaces.orientations)

  # Prolong solution to L2 mortars
  @timeit timer() "prolong2l2mortars" prolong2l2mortars!(dg)

  # Calculate mortar fluxes
  @timeit timer() "l2mortar flux" calc_l2mortar_flux!(dg.elements.surface_flux,
                                                      dg.l2mortars.neighbor_ids,
                                                      dg.l2mortars.u_lower,
                                                      dg.l2mortars.u_upper,
                                                      dg, dg.l2mortars.orientations)

  # Finish data exchange and calculate MPI surface fluxes
  @mpi_parallel @timeit timer() "MPI surface flux" (
      calc_mpi_surface_flux!(dg.elements.surface_flux, dg.mpi_surfaces.element_ids,
                            dg.mpi_surfaces.element_sides, dg.mpi_surfaces.u, dg,
                            dg.mpi_surfaces.orientations))

  # Calculate surface integrals
  @timeit timer() "surface integral" calc_surface_integral!(dg, dg.elements.u_t,
                                                            dg.elements.surface_flux, dg.lhat)

  # Apply Jacobian from mapping to reference element
  @timeit timer() "Jacobian" apply_jacobian!(dg)

  # Calculate source terms
  @timeit timer() "source terms" calc_sources!(dg, t_stage)
end


# Calculate volume integral and update u_t
function calc_volume_integral!(dg)
  calc_volume_integral!(dg, Val(dg.volume_integral_type), dg.elements.u_t,
                        dg.differentiation_operator)
end


# Calculate volume integral (DGSEM in weak form)
function calc_volume_integral!(dg, ::Val{:weak_form}, u_t::Array{Float64, 4}, dhat::SMatrix)
  #=@inbounds Threads.@threads for element_id = 1:dg.n_elements=#
  for element_id in 1:dg.n_elements
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


# Calculate volume integral (DGSEM in split form)
function calc_volume_integral!(dg, ::Val{:split_form}, u_t::Array{Float64, 4},
                               dsplit_transposed::SMatrix)
  #=@inbounds Threads.@threads for element_id = 1:dg.n_elements=#
  for element_id in 1:dg.n_elements
    # Calculate volume fluxes (one more dimension than weak form)
    f1 = MArray{Tuple{nvariables(dg), nnodes(dg), nnodes(dg), nnodes(dg)}, Float64}(undef)
    f2 = MArray{Tuple{nvariables(dg), nnodes(dg), nnodes(dg), nnodes(dg)}, Float64}(undef)
    calcflux_twopoint!(f1, f2, equations(dg), dg.elements.u, element_id, nnodes(dg))

    # Calculate volume integral
    for j = 1:nnodes(dg)
      for i = 1:nnodes(dg)
        for v = 1:nvariables(dg)
          for l = 1:nnodes(dg)
            u_t[v, i, j, element_id] += (dsplit_transposed[l, i] * f1[v, l, i, j] +
                                         dsplit_transposed[l, j] * f2[v, l, i, j])
          end
        end
      end
    end
  end
end


# Calculate volume integral (DGSEM in split form with shock capturing)
function calc_volume_integral!(dg, ::Val{:shock_capturing}, u_t::Array{Float64, 4},
                               dsplit_transposed::SMatrix)
  calc_volume_integral!(dg, Val(:shock_capturing), u_t, dsplit_transposed,
                        dg.inverse_weights)
end

function calc_volume_integral!(dg, ::Val{:shock_capturing}, u_t::Array{Float64, 4},
                               dsplit_transposed::SMatrix, inverse_weights::SVector)
  # Calculate blending factors α: u = u_DG * (1 - α) + u_FV * α
  @timeit timer() "blending factors" begin
    alpha, element_ids_dg, element_ids_dgfv = calc_blending_factors(dg, dg.elements.u)
  end

  # Loop over pure DG elements
  #=@inbounds Threads.@threads for element_id = 1:dg.n_elements=#
  @timeit timer() "pure DG" for element_id in element_ids_dg
    # Calculate volume fluxes (one more dimension than weak form)
    f1 = MArray{Tuple{nvariables(dg), nnodes(dg), nnodes(dg), nnodes(dg)}, Float64}(undef)
    f2 = MArray{Tuple{nvariables(dg), nnodes(dg), nnodes(dg), nnodes(dg)}, Float64}(undef)
    calcflux_twopoint!(f1, f2, equations(dg), dg.elements.u, element_id, nnodes(dg))

    # Calculate volume integral
    for j = 1:nnodes(dg)
      for i = 1:nnodes(dg)
        for v = 1:nvariables(dg)
          for l = 1:nnodes(dg)
            u_t[v, i, j, element_id] += (dsplit_transposed[l, i] * f1[v, l, i, j] +
                                         dsplit_transposed[l, j] * f2[v, l, i, j])
          end
        end
      end
    end
  end

  # Loop over blended DG-FV elements
  #=@inbounds Threads.@threads for element_id = 1:dg.n_elements=#
  @timeit timer() "blended DG-FV" for element_id in element_ids_dgfv
    # Calculate volume fluxes (one more dimension than weak form)
    f1 = MArray{Tuple{nvariables(dg), nnodes(dg), nnodes(dg), nnodes(dg)}, Float64}(undef)
    f2 = MArray{Tuple{nvariables(dg), nnodes(dg), nnodes(dg), nnodes(dg)}, Float64}(undef)
    calcflux_twopoint!(f1, f2, equations(dg), dg.elements.u, element_id, nnodes(dg))

    # Calculate DG volume integral contribution
    for j = 1:nnodes(dg)
      for i = 1:nnodes(dg)
        for v = 1:nvariables(dg)
          for l = 1:nnodes(dg)
            u_t[v, i, j, element_id] += ((1 - alpha[element_id]) *
                                         (dsplit_transposed[l, i] * f1[v, l, i, j] +
                                          dsplit_transposed[l, j] * f2[v, l, i, j]))
          end
        end
      end
    end

    # Calculate volume fluxes (one more dimension than weak form)
    fstar1 = MArray{Tuple{nvariables(dg), nnodes(dg)+1, nnodes(dg)}, Float64}(undef)
    fstar2 = MArray{Tuple{nvariables(dg), nnodes(dg), nnodes(dg)+1}, Float64}(undef)
    calcflux_fv!(fstar1, fstar2, equations(dg), dg.elements.u, element_id, nnodes(dg))

    # Calculate FV volume integral contribution
    for j = 1:nnodes(dg)
      for i = 1:nnodes(dg)
        for v = 1:nvariables(dg)
          u_t[v, i, j, element_id] += ((alpha[element_id])
                                       *(inverse_weights[i]*(fstar1[v, i+1, j] - fstar1[v,i,j]) +
                                         inverse_weights[j]*(fstar2[v, i, j+1] - fstar2[v,i,j])))

        end
      end
    end
  end
end


# Calculate 2D two-point flux (element version)
@inline function calcflux_fv!(fstar1::AbstractArray{Float64},
                              fstar2::AbstractArray{Float64},
                              equation,
                              u::AbstractArray{Float64},
                              element_id::Int, n_nodes::Int)

  u_leftright=MMatrix{2,nvariables(equation), Float64}(undef)

  fstar1[:,1,:]       = 0.0
  fstar1[:,n_nodes+1,:] = 0.0
  for j = 1:n_nodes
    for i = 2:n_nodes
      u_leftright[1,:] = u[:,i-1,j,element_id]
      u_leftright[2,:] = u[:,i,j,element_id]
      @views riemann!(fstar1[:,i,j],u_leftright,equation,1) 
    end
  end
  fstar2[:,:,1]       = 0.0
  fstar2[:,:,n_nodes+1] = 0.0
  for j = 2:n_nodes
    for i = 1:n_nodes
      u_leftright[1,:] = u[:,i,j-1,element_id]
      u_leftright[2,:] = u[:,i,j,element_id]
      @views riemann!(fstar2[:,i,j],u_leftright,equation,2) 
    end
  end
end


# Prolong solution to MPI surfaces: copy local data, start sending via MPI
function prolong2mpisurfaces!(dg)
  # Start receiving data
  for (idx, d) in enumerate(dg.neighbor_domains)
    dg.mpi_recv_requests[idx] = Irecv!(dg.mpi_recv_buffers[idx], d, 1000, comm())
  end

  # Prolong local data
  for s = 1:dg.n_mpi_surfaces
    element_id = dg.mpi_surfaces.element_ids[s]
    element_side = dg.mpi_surfaces.element_sides[s]
    if element_side == 1
      node_id = nnodes(dg)
    else
      node_id = 1
    end
    for l = 1:nnodes(dg)
      for v = 1:nvariables(dg)
        if dg.mpi_surfaces.orientations[s] == 1
          # Surface in x-direction
          dg.mpi_surfaces.u[element_side, v, l, s] = dg.elements.u[v, node_id, l, element_id]
        else
          # Surface in y-direction
          dg.mpi_surfaces.u[element_side, v, l, s] = dg.elements.u[v, l, node_id, element_id]
        end
      end
    end
  end

  # Copy data from MPI surfaces to send buffers
  block_size = nvariables(dg) * nnodes(dg)
  for d in 1:length(dg.neighbor_domains)
    for (idx, s) in enumerate(dg.mpi_surfaces_by_domain[d])
      element_id = dg.mpi_surfaces.element_ids[s]
      element_side = dg.mpi_surfaces.element_sides[s]
      @views dg.mpi_send_buffers[d][(1:block_size) .+ (idx-1)*block_size] = (
          dg.mpi_surfaces.u[element_side, :, :, s][:])
    end
  end

  # Start sending data
  for (idx, d) in enumerate(dg.neighbor_domains)
    dg.mpi_send_requests[idx] = Isend(dg.mpi_send_buffers[idx], d, 1000, comm())
  end
end


# Calculate and store fluxes across surfaces
function calc_mpi_surface_flux!(surface_flux::Array{Float64, 4},
                                element_ids::Vector{Int},
                                element_sides::Vector{Int},
                                u_surfaces::Array{Float64, 4}, dg,
                                orientations::Vector{Int})
  # Finish receiving data
  Waitall!(dg.mpi_recv_requests)

  # Copy data from receive buffers to MPI surfaces
  block_size = nvariables(dg) * nnodes(dg)
  for d in 1:length(dg.neighbor_domains)
    for (idx, s) in enumerate(dg.mpi_surfaces_by_domain[d])
      element_id = dg.mpi_surfaces.element_ids[s]
      element_side = dg.mpi_surfaces.element_sides[s]
      @views dg.mpi_surfaces.u[3 - element_side, :, :, s][:] = (
          dg.mpi_recv_buffers[d][(1:block_size) .+ (idx-1)*block_size])
    end
  end

  #=@inbounds Threads.@threads for s = 1:dg.n_mpi_surfaces=#
  for s = 1:dg.n_mpi_surfaces
    # Calculate flux
    fs = Matrix{Float64}(undef, nvariables(dg), nnodes(dg))
    riemann!(fs, u_surfaces, s, equations(dg), nnodes(dg), orientations)

    # Get element information
    element_id  = element_ids[s]
    element_side  = element_sides[s]

    # Determine surface direction with respect to elements:
    # orientation = 1: left -> 2, right -> 1
    # orientation = 2: left -> 4, right -> 3
    if element_side == 1
      element_direction = 2 * orientations[s]
    else
      element_direction = 2 * orientations[s] - 1
    end

    # Copy flux to left and right element storage
    surface_flux[:, :, element_direction,  element_id]  .= fs
  end

  # Finish sending data
  Waitall!(dg.mpi_send_requests)
end


# Prolong solution to surfaces (for GL nodes: just a copy)
function prolong2surfaces!(dg)
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


# Prolong solution to L2 mortars
function prolong2l2mortars!(dg)
  for m = 1:dg.n_l2mortars
    large_element_id = dg.l2mortars.neighbor_ids[3, m]
    upper_element_id = dg.l2mortars.neighbor_ids[2, m]
    lower_element_id = dg.l2mortars.neighbor_ids[1, m]

    # Copy solution small to small
    if dg.l2mortars.large_sides[m] == 1 # -> small elements on right side
      if dg.l2mortars.orientations[m] == 1
        # L2 mortars in x-direction
        @views dg.l2mortars.u_upper[2, :, :, m] .= dg.elements.u[:, 1, :, upper_element_id]
        @views dg.l2mortars.u_lower[2, :, :, m] .= dg.elements.u[:, 1, :, lower_element_id]
      else
        # L2 mortars in y-direction
        @views dg.l2mortars.u_upper[2, :, :, m] .= dg.elements.u[:, :, 1, upper_element_id]
        @views dg.l2mortars.u_lower[2, :, :, m] .= dg.elements.u[:, :, 1, lower_element_id]
      end
    else # large_sides[m] == 2 -> small elements on left side
      if dg.l2mortars.orientations[m] == 1
        # L2 mortars in x-direction
        @views dg.l2mortars.u_upper[1, :, :, m] .= dg.elements.u[:, nnodes(dg), :, upper_element_id]
        @views dg.l2mortars.u_lower[1, :, :, m] .= dg.elements.u[:, nnodes(dg), :, lower_element_id]
      else
        # L2 mortars in y-direction
        @views dg.l2mortars.u_upper[1, :, :, m] .= dg.elements.u[:, :, nnodes(dg), upper_element_id]
        @views dg.l2mortars.u_lower[1, :, :, m] .= dg.elements.u[:, :, nnodes(dg), lower_element_id]
      end
    end

    # Local storage for surface data of large element
    u_large = zeros(nvariables(dg), nnodes(dg))

    # Interpolate large element face data to small surface locations
    for v = 1:nvariables(dg)
      if dg.l2mortars.large_sides[m] == 1 # -> large element on left side
        if dg.l2mortars.orientations[m] == 1
          # L2 mortars in x-direction
          u_large[v, :] = dg.elements.u[v, nnodes(dg), :, large_element_id]
        else
          # L2 mortars in y-direction
          u_large[v, :] = dg.elements.u[v, :, nnodes(dg), large_element_id]
        end
        @views dg.l2mortars.u_upper[1, v, :, m] .= dg.l2mortar_forward_upper * u_large[v, :]
        @views dg.l2mortars.u_lower[1, v, :, m] .= dg.l2mortar_forward_lower * u_large[v, :]
      else # large_sides[m] == 2 -> large element on right side
        if dg.l2mortars.orientations[m] == 1
          # L2 mortars in x-direction
          u_large[v, :] = dg.elements.u[v, 1, :, large_element_id]
        else
          # L2 mortars in y-direction
          u_large[v, :] = dg.elements.u[v, :, 1, large_element_id]
        end
        @views dg.l2mortars.u_upper[2, v, :, m] .= dg.l2mortar_forward_upper * u_large[v, :]
        @views dg.l2mortars.u_lower[2, v, :, m] .= dg.l2mortar_forward_lower * u_large[v, :]
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

    # Get neighboring elements
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


# Calculate and store fluxes across L2 mortars
function calc_l2mortar_flux!(surface_flux::Array{Float64, 4}, neighbor_ids::Matrix{Int},
                             u_lower::Array{Float64, 4}, u_upper::Array{Float64, 4}, dg,
                             orientations::Vector{Int})
  #=@inbounds Threads.@threads for m = 1:dg.n_l2mortars=#
  for m = 1:dg.n_l2mortars
    large_element_id = dg.l2mortars.neighbor_ids[3, m]
    upper_element_id = dg.l2mortars.neighbor_ids[2, m]
    lower_element_id = dg.l2mortars.neighbor_ids[1, m]

    # Calculate fluxes
    f_upper = Matrix{Float64}(undef, nvariables(dg), nnodes(dg))
    f_lower = Matrix{Float64}(undef, nvariables(dg), nnodes(dg))
    riemann!(f_upper, u_upper, m, equations(dg), nnodes(dg), orientations)
    riemann!(f_lower, u_lower, m, equations(dg), nnodes(dg), orientations)

    # Copy flux small to small
    if dg.l2mortars.large_sides[m] == 1 # -> small elements on right side
      if dg.l2mortars.orientations[m] == 1
        # L2 mortars in x-direction
        surface_flux[:, :, 1, upper_element_id] .= f_upper
        surface_flux[:, :, 1, lower_element_id] .= f_lower
      else
        # L2 mortars in y-direction
        surface_flux[:, :, 3, upper_element_id] .= f_upper
        surface_flux[:, :, 3, lower_element_id] .= f_lower
      end
    else # large_sides[m] == 2 -> small elements on left side
      if dg.l2mortars.orientations[m] == 1
        # L2 mortars in x-direction
        surface_flux[:, :, 2, upper_element_id] .= f_upper
        surface_flux[:, :, 2, lower_element_id] .= f_lower
      else
        # L2 mortars in y-direction
        surface_flux[:, :, 4, upper_element_id] .= f_upper
        surface_flux[:, :, 4, lower_element_id] .= f_lower
      end
    end

    # Project small fluxes to large element
    for v = 1:nvariables(dg)
      large_surface_flux = (dg.l2mortar_reverse_upper * f_upper[v, :] +
                            dg.l2mortar_reverse_lower * f_lower[v, :])
      if dg.l2mortars.large_sides[m] == 1 # -> large element on left side
        if dg.l2mortars.orientations[m] == 1
          # L2 mortars in x-direction
          surface_flux[v, :, 2, large_element_id] .= large_surface_flux
        else
          # L2 mortars in y-direction
          surface_flux[v, :, 4, large_element_id] .= large_surface_flux
        end
      else # large_sides[m] == 2 -> large element on right side
        if dg.l2mortars.orientations[m] == 1
          # L2 mortars in x-direction
          surface_flux[v, :, 1, large_element_id] .= large_surface_flux
        else
          # L2 mortars in y-direction
          surface_flux[v, :, 3, large_element_id] .= large_surface_flux
        end
      end
    end
  end
end


# Calculate surface integrals and update u_t
function calc_surface_integral!(dg, u_t::Array{Float64, 4}, surface_flux::Array{Float64, 4},
                                lhat::SMatrix)
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

  @mpi_parallel begin
    # This is necessary since MPI.jl's Allreduce! only works with array-like structures
    temp = [min_dt]
    Allreduce!(temp, min, comm())
    min_dt = temp[1]
  end

  return min_dt
end


# Calculate blending factors for shock capturing
function calc_blending_factors(dg, u::AbstractArray{Float64})
  # Calculate blending factor
  alpha = similar(dg.elements.inverse_jacobian)
  indicator = zeros(1, nnodes(dg), nnodes(dg))
  threshold = 0.5 * 10^(-1.8 * (nnodes(dg))^0.25)
  parameter_s = log((1 - 0.0001)/0.0001)
  alpha_min = 0.001
  alpha_max = 0.5

  for element_id in 1:dg.n_elements
    # Calculate indicator variables at Gauss-Lobatto nodes
    for i in 1:nnodes(dg)
      for j in 1:nnodes(dg)
        @views indicator[1, i, j] = cons2indicator(equations(dg), u[:, i, j, element_id])
      end
    end

    # Convert to modal representation
    modal = nodal2modal(indicator, dg.inverse_vandermonde_legendre)

    # Calculate total energies for all modes, without highest, without two highest
    total_energy = sum(modal.^2)
    total_energy_clip1 = sum(modal[:, 1:nnodes(dg)-1, 1:nnodes(dg)-1].^2)
    total_energy_clip2 = sum(modal[:, 1:nnodes(dg)-2, 1:nnodes(dg)-2].^2)

    # Calculate energy in lower modes
    energy = max((total_energy - total_energy_clip1)/total_energy,
                 (total_energy_clip1 - total_energy_clip2)/total_energy_clip1)

    alpha[element_id] = 1/(1 + exp(-parameter_s/threshold * (energy - threshold)))

    # Take care of the case close to pure DG
    if (alpha[element_id] < alpha_min)
      alpha[element_id] = 0.
    end

    # Take care of the case close to pure FV
    if (alpha[element_id] > 1-alpha_min)
      alpha[element_id] = 1.
    end

    # Clip the maximum amount of FV allowed
    alpha[element_id] = max(alpha_max, alpha[element_id])
  end

  # Diffuse alpha values by setting each alpha to at least 50% of neighboring elements' alpha
  # Loop over surfaces
  for surface_id in 1:dg.n_surfaces
    # Get neighboring element ids
    left = dg.surfaces.neighbor_ids[1, surface_id]
    right = dg.surfaces.neighbor_ids[2, surface_id]

    # Apply smoothing
    alpha[left] = max(alpha[left], 0.5 * alpha[right])
    alpha[right] = max(alpha[right], 0.5 * alpha[left])
  end
 
  # Loop over mortars
  # TODO: Gregor, please check if this implementation makes sense
  for l2mortar_id in 1:dg.n_l2mortars
    # Get neighboring element ids
    lower = dg.l2mortars.neighbor_ids[1, l2mortar_id]
    upper = dg.l2mortars.neighbor_ids[2, l2mortar_id]
    large = dg.l2mortars.neighbor_ids[3, l2mortar_id]

    # Apply smoothing
    alpha[lower] = max(alpha[lower], 0.5 * alpha[large])
    alpha[upper] = max(alpha[upper], 0.5 * alpha[large])
    alpha[large] = max(alpha[large], 0.5 * alpha[lower])
    alpha[large] = max(alpha[large], 0.5 * alpha[upper])
  end

  # Clip blending factor for values close to zero (-> pure DG)
  dg_only = isapprox.(alpha, 0, atol=1e-12)
  element_ids_dg = collect(1:dg.n_elements)[dg_only .== 1]
  element_ids_dgfv = collect(1:dg.n_elements)[dg_only .!= 1]

  return alpha, element_ids_dg, element_ids_dgfv
end

end # module
