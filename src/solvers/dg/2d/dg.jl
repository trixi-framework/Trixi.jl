# Main DG data structure that contains all relevant data for the DG solver
mutable struct Dg2D{Eqn<:AbstractEquation, NVARS, POLYDEG,
                  SurfaceFlux, VolumeFlux, InitialConditions, SourceTerms,
                  MortarType, VolumeIntegralType,
                  VectorNnodes, MatrixNnodes, MatrixNnodes2,
                  VectorAnalysisNnodes, AnalysisVandermonde} <: AbstractDg{2, POLYDEG}
  equations::Eqn

  surface_flux_function::SurfaceFlux
  volume_flux_function::VolumeFlux

  initial_conditions::InitialConditions
  source_terms::SourceTerms

  elements::ElementContainer2D{NVARS, POLYDEG}
  n_elements::Int

  interfaces::InterfaceContainer2D{NVARS, POLYDEG}
  n_interfaces::Int

  boundaries::BoundaryContainer2D{NVARS, POLYDEG}
  n_boundaries::Int

  mortar_type::MortarType
  l2mortars::L2MortarContainer2D{NVARS, POLYDEG}
  n_l2mortars::Int
  ecmortars::EcMortarContainer2D{NVARS, POLYDEG}
  n_ecmortars::Int

  nodes::VectorNnodes
  weights::VectorNnodes
  inverse_weights::VectorNnodes
  inverse_vandermonde_legendre::MatrixNnodes
  lhat::MatrixNnodes2

  volume_integral_type::VolumeIntegralType
  dhat::MatrixNnodes
  dsplit::MatrixNnodes
  dsplit_transposed::MatrixNnodes

  mortar_forward_upper::MatrixNnodes
  mortar_forward_lower::MatrixNnodes
  l2mortar_reverse_upper::MatrixNnodes
  l2mortar_reverse_lower::MatrixNnodes
  ecmortar_reverse_upper::MatrixNnodes
  ecmortar_reverse_lower::MatrixNnodes

  analysis_nodes::VectorAnalysisNnodes
  analysis_weights::VectorAnalysisNnodes
  analysis_weights_volume::VectorAnalysisNnodes
  analysis_vandermonde::AnalysisVandermonde
  analysis_total_volume::Float64
  analysis_quantities::Vector{Symbol}
  save_analysis::Bool
  analysis_filename::String

  shock_indicator_variable::Symbol
  shock_alpha_max::Float64
  shock_alpha_min::Float64
  shock_alpha_smooth::Bool
  amr_indicator::Symbol
  amr_alpha_max::Float64
  amr_alpha_min::Float64
  amr_alpha_smooth::Bool

  element_variables::Dict{Symbol, Union{Vector{Float64}, Vector{Int}}}
  cache::Dict{Symbol, Any}
  thread_cache::Any # to make fully-typed output more readable

  initial_state_integrals::Vector{Float64}
end


# Convenience constructor to create DG solver instance
function Dg2D(equation::AbstractEquation{NDIMS, NVARS}, surface_flux_function, volume_flux_function, initial_conditions, source_terms, mesh::TreeMesh{NDIMS}, POLYDEG) where {NDIMS, NVARS}
  # Get cells for which an element needs to be created (i.e., all leaf cells)
  leaf_cell_ids = leaf_cells(mesh.tree)

  # Initialize element container
  elements = init_elements(leaf_cell_ids, mesh, Val(NVARS), Val(POLYDEG))
  n_elements = nelements(elements)

  # Initialize interface container
  interfaces = init_interfaces(leaf_cell_ids, mesh, Val(NVARS), Val(POLYDEG), elements)
  n_interfaces = ninterfaces(interfaces)

  # Initialize boundaries
  boundaries = init_boundaries(leaf_cell_ids, mesh, Val(NVARS), Val(POLYDEG), elements)
  n_boundaries = nboundaries(boundaries)

  # Initialize mortar containers
  mortar_type = Val(Symbol(parameter("mortar_type", "l2", valid=["l2", "ec"])))
  l2mortars, ecmortars = init_mortars(leaf_cell_ids, mesh, Val(NVARS), Val(POLYDEG), elements, mortar_type)
  n_l2mortars = nmortars(l2mortars)
  n_ecmortars = nmortars(ecmortars)

  # Sanity checks
  if isperiodic(mesh.tree) && n_l2mortars == 0 && n_ecmortars == 0
    @assert n_interfaces == 2*n_elements ("For 2D and periodic domains and conforming elements, "
                                        * "n_surf must be the same as 2*n_elem")
  end

  # Initialize interpolation data structures
  n_nodes = POLYDEG + 1
  nodes, weights = gauss_lobatto_nodes_weights(n_nodes)
  inverse_weights = 1 ./ weights
  _, inverse_vandermonde_legendre = vandermonde_legendre(nodes)
  lhat = zeros(n_nodes, 2)
  lhat[:, 1] = calc_lhat(-1.0, nodes, weights)
  lhat[:, 2] = calc_lhat( 1.0, nodes, weights)

  # Initialize differentiation operator
  volume_integral_type = Val(Symbol(parameter("volume_integral_type", "weak_form",
                                              valid=["weak_form", "split_form", "shock_capturing"])))
  # FIXME: This should be removed as soon as it possible to set solver-specific parameters
  if equation isa HyperbolicDiffusionEquations2D && globals[:euler_gravity]
    volume_integral_type = Val(:weak_form)
  end
  dhat = calc_dhat(nodes, weights)
  dsplit = calc_dsplit(nodes, weights)
  dsplit_transposed = transpose(calc_dsplit(nodes, weights))

  # Initialize L2 mortar projection operators
  mortar_forward_upper = calc_forward_upper(n_nodes)
  mortar_forward_lower = calc_forward_lower(n_nodes)
  l2mortar_reverse_upper = calc_reverse_upper(n_nodes, Val(:gauss))
  l2mortar_reverse_lower = calc_reverse_lower(n_nodes, Val(:gauss))
  ecmortar_reverse_upper = calc_reverse_upper(n_nodes, Val(:gauss_lobatto))
  ecmortar_reverse_lower = calc_reverse_lower(n_nodes, Val(:gauss_lobatto))

  # Initialize data structures for error analysis (by default, we use twice the
  # number of analysis nodes as the normal solution)
  analysis_polydeg = 2 * (n_nodes) - 1
  analysis_nodes, analysis_weights = gauss_lobatto_nodes_weights(analysis_polydeg + 1)
  analysis_weights_volume = analysis_weights
  analysis_vandermonde = polynomial_interpolation_matrix(nodes, analysis_nodes)
  analysis_total_volume = mesh.tree.length_level_0^ndims(mesh)

  # Store which quantities should be analyzed in `analyze_solution`
  if parameter_exists("extra_analysis_quantities")
    extra_analysis_quantities = Symbol.(parameter("extra_analysis_quantities"))
  else
    extra_analysis_quantities = Symbol[]
  end
  analysis_quantities = vcat(collect(Symbol.(default_analysis_quantities(equation))),
                             extra_analysis_quantities)

  # If analysis should be saved to file, create file with header
  save_analysis = parameter("save_analysis", false)
  if save_analysis
    # Create output directory (if it does not exist)
    output_directory = parameter("output_directory", "out")
    mkpath(output_directory)

    # Determine filename
    analysis_filename = joinpath(output_directory, "analysis.dat")

    # Open file and write header
    save_analysis_header(analysis_filename, analysis_quantities, equation)
  else
    analysis_filename = ""
  end

  # Initialize AMR
  amr_indicator = Symbol(parameter("amr_indicator", "n/a",
                                   valid=["n/a", "gauss", "isentropic_vortex", "blast_wave", "khi", "blob", "sedov_self_gravity"]))

  # Initialize storage for element variables
  element_variables = Dict{Symbol, Union{Vector{Float64}, Vector{Int}}}()
  if amr_indicator === :khi || amr_indicator === :blob
    element_variables[:amr_indicator_values] = zeros(n_elements)
  end
  # maximum and minimum alpha for shock capturing
  shock_alpha_max = parameter("shock_alpha_max", 0.5)
  shock_alpha_min = parameter("shock_alpha_min", 0.001)
  shock_alpha_smooth = parameter("shock_alpha_smooth", true)

  # variable used to compute the shock capturing indicator
  shock_indicator_variable = Symbol(parameter("shock_indicator_variable", "density_pressure",
                                          valid=["density", "density_pressure", "pressure"]))

  # maximum and minimum alpha for amr control
  amr_alpha_max = parameter("amr_alpha_max", 0.5)
  amr_alpha_min = parameter("amr_alpha_min", 0.001)
  amr_alpha_smooth = parameter("amr_alpha_smooth", false)

  # Initialize element variables such that they are available in the first solution file
  if volume_integral_type === Val(:shock_capturing)
    element_variables[:blending_factor] = zeros(n_elements)
  end

  # Initialize storage for the cache
  cache = Dict{Symbol, Any}()
  thread_cache = create_thread_cache_2d(NVARS, POLYDEG+1)

  # Store initial state integrals for conservation error calculation
  initial_state_integrals = Vector{Float64}()

  # Create actual DG solver instance
  dg = Dg2D(
      equation,
      surface_flux_function, volume_flux_function,
      initial_conditions, source_terms,
      elements, n_elements,
      interfaces, n_interfaces,
      boundaries, n_boundaries,
      mortar_type,
      l2mortars, n_l2mortars,
      ecmortars, n_ecmortars,
      SVector{POLYDEG+1}(nodes), SVector{POLYDEG+1}(weights), SVector{POLYDEG+1}(inverse_weights),
      SMatrix{POLYDEG+1,POLYDEG+1}(inverse_vandermonde_legendre), SMatrix{POLYDEG+1,2}(lhat),
      volume_integral_type,
      SMatrix{POLYDEG+1,POLYDEG+1}(dhat), SMatrix{POLYDEG+1,POLYDEG+1}(dsplit), SMatrix{POLYDEG+1,POLYDEG+1}(dsplit_transposed),
      SMatrix{POLYDEG+1,POLYDEG+1}(mortar_forward_upper), SMatrix{POLYDEG+1,POLYDEG+1}(mortar_forward_lower),
      SMatrix{POLYDEG+1,POLYDEG+1}(l2mortar_reverse_upper), SMatrix{POLYDEG+1,POLYDEG+1}(l2mortar_reverse_lower),
      SMatrix{POLYDEG+1,POLYDEG+1}(ecmortar_reverse_upper), SMatrix{POLYDEG+1,POLYDEG+1}(ecmortar_reverse_lower),
      SVector{analysis_polydeg+1}(analysis_nodes), SVector{analysis_polydeg+1}(analysis_weights), SVector{analysis_polydeg+1}(analysis_weights_volume),
      SMatrix{analysis_polydeg+1,POLYDEG+1}(analysis_vandermonde), analysis_total_volume,
      analysis_quantities, save_analysis, analysis_filename,
      shock_indicator_variable, shock_alpha_max, shock_alpha_min, shock_alpha_smooth,
      amr_indicator, amr_alpha_max, amr_alpha_min, amr_alpha_smooth,
      element_variables, cache, thread_cache,
      initial_state_integrals)

  return dg
end


function create_thread_cache_2d(n_variables, n_nodes)
  # Type alias only for convenience
  A4d = Array{Float64, 4}
  A3dp1_x = Array{Float64, 3}
  A3dp1_y = Array{Float64, 3}
  MA2d = MArray{Tuple{n_variables, n_nodes}, Float64}
  A2d = Array{Float64, 2}

  # Pre-allocate data structures to speed up computation (thread-safe)
  f1_threaded      = A4d[A4d(undef, n_variables, n_nodes, n_nodes, n_nodes)
                          for _ in 1:Threads.nthreads()]
  f2_threaded      = A4d[A4d(undef, n_variables, n_nodes, n_nodes, n_nodes)
                          for _ in 1:Threads.nthreads()]
  fstar1_threaded  = A3dp1_x[A3dp1_x(undef, n_variables, n_nodes+1, n_nodes)
                             for _ in 1:Threads.nthreads()]
  fstar2_threaded  = A3dp1_y[A3dp1_y(undef, n_variables, n_nodes, n_nodes+1)
                             for _ in 1:Threads.nthreads()]
  fstar_upper_threaded = [MA2d(undef) for _ in 1:Threads.nthreads()]
  fstar_lower_threaded = [MA2d(undef) for _ in 1:Threads.nthreads()]
  u_large_threaded = A2d[A2d(undef, n_variables, n_nodes)
                         for _ in 1:Threads.nthreads()]

  return (; f1_threaded, f2_threaded, fstar1_threaded, fstar2_threaded,
            fstar_upper_threaded, fstar_lower_threaded, u_large_threaded)
end


# Count the number of interfaces that need to be created
function count_required_interfaces(mesh::TreeMesh{2}, cell_ids)
  count = 0

  # Iterate over all cells
  for cell_id in cell_ids
    for direction in 1:n_directions(mesh.tree)
      # Only count interfaces in positive direction to avoid double counting
      if direction % 2 == 1
        continue
      end

      # If no neighbor exists, current cell is small or at boundary and thus we need a mortar
      if !has_neighbor(mesh.tree, cell_id, direction)
        continue
      end

      # Skip if neighbor has children
      neighbor_id = mesh.tree.neighbor_ids[direction, cell_id]
      if has_children(mesh.tree, neighbor_id)
        continue
      end

      count += 1
    end
  end

  return count
end


# Count the number of boundaries that need to be created
function count_required_boundaries(mesh::TreeMesh{2}, cell_ids)
  count = 0

  # Iterate over all cells
  for cell_id in cell_ids
    for direction in 1:n_directions(mesh.tree)
      # If neighbor exists, current cell is not at a boundary
      if has_neighbor(mesh.tree, cell_id, direction)
        continue
      end

      # If coarse neighbor exists, current cell is not at a boundary
      if has_coarse_neighbor(mesh.tree, cell_id, direction)
        continue
      end

      # No neighbor exists in this direction -> must be a boundary
      count += 1
    end
  end

  return count
end


# Count the number of mortars that need to be created
function count_required_mortars(mesh::TreeMesh{2}, cell_ids)
  count = 0

  # Iterate over all cells and count mortars from perspective of coarse cells
  for cell_id in cell_ids
    for direction in 1:n_directions(mesh.tree)
      # If no neighbor exists, cell is small with large neighbor or at boundary -> do nothing
      if !has_neighbor(mesh.tree, cell_id, direction)
        continue
      end

      # If neighbor has no children, this is a conforming interface -> do nothing
      neighbor_id = mesh.tree.neighbor_ids[direction, cell_id]
      if !has_children(mesh.tree, neighbor_id)
        continue
      end

      count +=1
    end
  end

  return count
end


# Create element container, initialize element data, and return element container for further use
#
# NVARS: number of variables
# POLYDEG: polynomial degree
function init_elements(cell_ids, mesh::TreeMesh{2}, ::Val{NVARS}, ::Val{POLYDEG}) where {NVARS, POLYDEG}
  # Initialize container
  n_elements = length(cell_ids)
  elements = ElementContainer2D{NVARS, POLYDEG}(n_elements)

  # Store cell ids
  elements.cell_ids .= cell_ids

  # Determine node locations
  n_nodes = POLYDEG + 1
  nodes, _ = gauss_lobatto_nodes_weights(n_nodes)

  # Calculate inverse Jacobian and node coordinates
  for element_id in 1:nelements(elements)
    # Get cell id
    cell_id = cell_ids[element_id]

    # Get cell length
    dx = length_at_cell(mesh.tree, cell_id)

    # Calculate inverse Jacobian as 1/(h/2)
    elements.inverse_jacobian[element_id] = 2/dx

    # Calculate node coordinates
    for j in 1:n_nodes
      for i in 1:n_nodes
        elements.node_coordinates[1, i, j, element_id] = (
            mesh.tree.coordinates[1, cell_id] + dx/2 * nodes[i])
        elements.node_coordinates[2, i, j, element_id] = (
            mesh.tree.coordinates[2, cell_id] + dx/2 * nodes[j])
      end
    end
  end

  return elements
end


# Create interface container, initialize interface data, and return interface container for further use
#
# NVARS: number of variables
# POLYDEG: polynomial degree
function init_interfaces(cell_ids, mesh::TreeMesh{2}, ::Val{NVARS}, ::Val{POLYDEG}, elements) where {NVARS, POLYDEG}
  # Initialize container
  n_interfaces = count_required_interfaces(mesh, cell_ids)
  interfaces = InterfaceContainer2D{NVARS, POLYDEG}(n_interfaces)

  # Connect elements with interfaces
  init_interface_connectivity!(elements, interfaces, mesh)

  return interfaces
end


# Create boundaries container, initialize boundary data, and return boundaries container
#
# NVARS: number of variables
# POLYDEG: polynomial degree
function init_boundaries(cell_ids, mesh::TreeMesh{2}, ::Val{NVARS}, ::Val{POLYDEG}, elements) where {NVARS, POLYDEG}
  # Initialize container
  n_boundaries = count_required_boundaries(mesh, cell_ids)
  boundaries = BoundaryContainer2D{NVARS, POLYDEG}(n_boundaries)

  # Connect elements with boundaries
  init_boundary_connectivity!(elements, boundaries, mesh)

  return boundaries
end


# Create mortar container, initialize mortar data, and return mortar container for further use
#
# NVARS: number of variables
# POLYDEG: polynomial degree
function init_mortars(cell_ids, mesh::TreeMesh{2}, ::Val{NVARS}, ::Val{POLYDEG}, elements, mortar_type) where {NVARS, POLYDEG}
  # Initialize containers
  n_mortars = count_required_mortars(mesh, cell_ids)
  if mortar_type === Val(:l2)
    n_l2mortars = n_mortars
    n_ecmortars = 0
  elseif mortar_type === Val(:ec)
    n_l2mortars = 0
    n_ecmortars = n_mortars
  else
    error("unknown mortar type '$(mortar_type)'")
  end
  l2mortars = L2MortarContainer2D{NVARS, POLYDEG}(n_l2mortars)
  ecmortars = EcMortarContainer2D{NVARS, POLYDEG}(n_ecmortars)

  # Connect elements with interfaces and l2mortars
  if mortar_type === Val(:l2)
    init_mortar_connectivity!(elements, l2mortars, mesh)
  elseif mortar_type === Val(:ec)
    init_mortar_connectivity!(elements, ecmortars, mesh)
  else
    error("unknown mortar type '$(mortar_type)'")
  end

  return l2mortars, ecmortars
end


# Initialize connectivity between elements and interfaces
function init_interface_connectivity!(elements, interfaces, mesh::TreeMesh{2})
  # Construct cell -> element mapping for easier algorithm implementation
  tree = mesh.tree
  c2e = zeros(Int, length(tree))
  for element_id in 1:nelements(elements)
    c2e[elements.cell_ids[element_id]] = element_id
  end

  # Reset interface count
  count = 0

  # Iterate over all elements to find neighbors and to connect via interfaces
  for element_id in 1:nelements(elements)
    # Get cell id
    cell_id = elements.cell_ids[element_id]

    # Loop over directions
    for direction in 1:n_directions(mesh.tree)
      # Only create interfaces in positive direction
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

      # Create interface between elements (1 -> "left" of interface, 2 -> "right" of interface)
      count += 1
      interfaces.neighbor_ids[2, count] = c2e[neighbor_cell_id]
      interfaces.neighbor_ids[1, count] = element_id

      # Set orientation (x -> 1, y -> 2)
      interfaces.orientations[count] = div(direction, 2)
    end
  end

  @assert count == ninterfaces(interfaces) ("Actual interface count ($count) does not match " *
                                            "expectations $(ninterfaces(interfaces))")
end


# Initialize connectivity between elements and boundaries
function init_boundary_connectivity!(elements, boundaries, mesh::TreeMesh{2})
  # Reset boundaries count
  count = 0

  # Iterate over all elements to find missing neighbors and to connect to boundaries
  for element_id in 1:nelements(elements)
    # Get cell id
    cell_id = elements.cell_ids[element_id]

    # Loop over directions
    for direction in 1:n_directions(mesh.tree)
      # If neighbor exists, current cell is not at a boundary
      if has_neighbor(mesh.tree, cell_id, direction)
        continue
      end

      # If coarse neighbor exists, current cell is not at a boundary
      if has_coarse_neighbor(mesh.tree, cell_id, direction)
        continue
      end

      # Create boundary
      count += 1

      # Set neighbor element id
      boundaries.neighbor_ids[count] = element_id

      # Set neighbor side, which denotes the direction (1 -> negative, 2 -> positive) of the element
      if direction in (2, 4)
        boundaries.neighbor_sides[count] = 1
      else
        boundaries.neighbor_sides[count] = 2
      end

      # Set orientation (x -> 1, y -> 2)
      if direction in (1, 2)
        boundaries.orientations[count] = 1
      else
        boundaries.orientations[count] = 2
      end

      # Store node coordinates
      enc = elements.node_coordinates
      if direction == 1 # -x direction
        boundaries.node_coordinates[:, :, count] .= enc[:, 1,   :,   element_id]
      elseif direction == 2 # +x direction
        boundaries.node_coordinates[:, :, count] .= enc[:, end, :,   element_id]
      elseif direction == 3 # -y direction
        boundaries.node_coordinates[:, :, count] .= enc[:, :,   1,   element_id]
      elseif direction == 4 # +y direction
        boundaries.node_coordinates[:, :, count] .= enc[:, :,   end, element_id]
      else
        error("should not happen")
      end
    end
  end

  @assert count == nboundaries(boundaries) ("Actual boundaries count ($count) does not match " *
                                            "expectations $(nboundaries(boundaries))")
end


# Initialize connectivity between elements and mortars
function init_mortar_connectivity!(elements, mortars, mesh::TreeMesh{2})
  # Construct cell -> element mapping for easier algorithm implementation
  tree = mesh.tree
  c2e = zeros(Int, length(tree))
  for element_id in 1:nelements(elements)
    c2e[elements.cell_ids[element_id]] = element_id
  end

  # Reset interface count
  count = 0

  # Iterate over all elements to find neighbors and to connect via interfaces
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
      mortars.neighbor_ids[3, count] = element_id
      if direction == 1
        mortars.neighbor_ids[1, count] = c2e[mesh.tree.child_ids[2, neighbor_cell_id]]
        mortars.neighbor_ids[2, count] = c2e[mesh.tree.child_ids[4, neighbor_cell_id]]
      elseif direction == 2
        mortars.neighbor_ids[1, count] = c2e[mesh.tree.child_ids[1, neighbor_cell_id]]
        mortars.neighbor_ids[2, count] = c2e[mesh.tree.child_ids[3, neighbor_cell_id]]
      elseif direction == 3
        mortars.neighbor_ids[1, count] = c2e[mesh.tree.child_ids[3, neighbor_cell_id]]
        mortars.neighbor_ids[2, count] = c2e[mesh.tree.child_ids[4, neighbor_cell_id]]
      elseif direction == 4
        mortars.neighbor_ids[1, count] = c2e[mesh.tree.child_ids[1, neighbor_cell_id]]
        mortars.neighbor_ids[2, count] = c2e[mesh.tree.child_ids[2, neighbor_cell_id]]
      else
        error("should not happen")
      end

      # Set large side, which denotes the direction (1 -> negative, 2 -> positive) of the large side
      if direction in [2, 4]
        mortars.large_sides[count] = 1
      else
        mortars.large_sides[count] = 2
      end

      # Set orientation (x -> 1, y -> 2)
      if direction in [1, 2]
        mortars.orientations[count] = 1
      else
        mortars.orientations[count] = 2
      end
    end
  end

  @assert count == nmortars(mortars) ("Actual mortar count ($count) does not match " *
                                      "expectations $(nmortars(mortars))")
end


"""
    integrate(func, dg::Dg2D, args...; normalize=true)

Call function `func` for each DG node and integrate the result over the computational domain.

The function `func` is called as `func(i, j, element_id, dg, args...)` for each
volume node `(i, j)` and each `element_id`. Additional positional
arguments `args...` are passed along as well. If `normalize` is true, the result
is divided by the total volume of the computational domain.

# Examples
Calculate the integral of the time derivative of the entropy, i.e.,
∫(∂S/∂t)dΩ = ∫(∂S/∂u ⋅ ∂u/∂t)dΩ:
```julia
# Compute entropy variables
entropy_vars = cons2entropy(...)

# Calculate integral of entropy time derivative
dsdu_ut = integrate(dg, entropy_vars, dg.elements.u_t) do i, j, element_id, dg, entropy_vars, u_t
  sum(entropy_vars[:, i, j, element_id] .* u_t[:, i, j, element_id])
end
```
"""
function integrate(func, dg::Dg2D, args...; normalize=true)
  # Initialize integral with zeros of the right shape
  integral = zero(func(1, 1, 1, dg, args...))

  # Use quadrature to numerically integrate over entire domain
  for element_id in 1:dg.n_elements
    jacobian_volume = inv(dg.elements.inverse_jacobian[element_id])^ndims(dg)
    for j in 1:nnodes(dg)
      for i in 1:nnodes(dg)
        integral += jacobian_volume * dg.weights[i] * dg.weights[j] * func(i, j, element_id, dg, args...)
      end
    end
  end

  # Normalize with total volume
  if normalize
    integral = integral/dg.analysis_total_volume
  end

  return integral
end


"""
    integrate(func, u, dg::Dg2D; normalize=true)
    integrate(u, dg::Dg2D; normalize=true)

Call function `func` for each DG node and integrate the result over the computational domain.

The function `func` is called as `func(u_local)` for each volume node `(i, j)`
and each `element_id`, where `u_local` is an `SVector`ized copy of
`u[:, i, j, element_id]`. If `normalize` is true, the result is divided by the
total volume of the computational domain. If `func` is omitted, it defaults to
`identity`.

# Examples
Calculate the integral over all conservative variables:
```julia
state_integrals = integrate(dg.elements.u, dg)
```
"""
function integrate(func, u, dg::Dg2D; normalize=true)
  func_wrapped = function(i, j, element_id, dg, u)
    u_local = get_node_vars(u, dg, i, j, element_id)
    return func(u_local)
  end
  return integrate(func_wrapped, dg, u; normalize=normalize)
end
integrate(u, dg::Dg2D; normalize=true) = integrate(identity, u, dg; normalize=normalize)


# Calculate L2/Linf error norms based on "exact solution"
function calc_error_norms(dg::Dg2D, t::Float64)
  # Gather necessary information
  equation = equations(dg)
  n_nodes_analysis = length(dg.analysis_nodes)

  # Set up data structures
  l2_error = zeros(nvariables(equation))
  linf_error = zeros(nvariables(equation))
  u_exact = zeros(nvariables(equation))

  # Iterate over all elements for error calculations
  for element_id in 1:dg.n_elements
    # Interpolate solution and node locations to analysis nodes
    u = multiply_coordinatewise(dg.elements.u[:, :, :, element_id], dg.analysis_vandermonde)
    x = multiply_coordinatewise(dg.elements.node_coordinates[:, :, :, element_id], dg.analysis_vandermonde)

    # Calculate errors at each analysis node
    weights = dg.analysis_weights_volume
    jacobian_volume = inv(dg.elements.inverse_jacobian[element_id])^ndims(dg)
    for j in 1:n_nodes_analysis, i in 1:n_nodes_analysis
      u_exact = @views dg.initial_conditions(x[:, i, j], t, equation)
      diff = similar(u_exact)
      @views @. diff = u_exact - u[:, i, j]
      @. l2_error += diff^2 * weights[i] * weights[j] * jacobian_volume
      @. linf_error = max(linf_error, abs(diff))
    end
  end

  # For L2 error, divide by total volume
  @. l2_error = sqrt(l2_error / dg.analysis_total_volume)

  return l2_error, linf_error
end


# Integrate ∂S/∂u ⋅ ∂u/∂t over the entire domain
function calc_entropy_timederivative(dg::Dg2D, t)
  # Compute entropy variables for all elements and nodes with current solution u
  dsdu = cons2entropy(dg.elements.u, nnodes(dg), dg.n_elements, equations(dg))

  # Compute ut = rhs(u) with current solution u
  @notimeit timer() rhs!(dg, t)

  # Calculate ∫(∂S/∂u ⋅ ∂u/∂t)dΩ
  dsdu_ut = integrate(dg, dsdu, dg.elements.u_t) do i, j, element_id, dg, dsdu, u_t
    sum(dsdu[:,i,j,element_id].*u_t[:,i,j,element_id])
  end

  return dsdu_ut
end


# Calculate L2/Linf norms of a solenoidal condition ∇ ⋅ B = 0
# OBS! This works only when the problem setup is designed such that ∂B₁/∂x + ∂B₂/∂y = 0. Cannot
#      compute the full 3D divergence from the given data
function calc_mhd_solenoid_condition(dg::Dg2D, t::Float64)
  @assert equations(dg) isa IdealGlmMhdEquations2D "Only relevant for MHD"

  # Local copy of standard derivative matrix
  d = polynomial_derivative_matrix(dg.nodes)
  # Quadrature weights
  weights = dg.weights
  # integrate over all elements to get the divergence-free condition errors
  linf_divb = 0.0
  l2_divb   = 0.0
  for element_id in 1:dg.n_elements
    jacobian_volume = (1.0/dg.elements.inverse_jacobian[element_id])^ndims(dg)
    for j in 1:nnodes(dg)
      for i in 1:nnodes(dg)
        divb   = 0.0
        for k in 1:nnodes(dg)
          divb += d[i,k]*dg.elements.u[6,k,j,element_id] + d[j,k]*dg.elements.u[7,i,k,element_id]
        end
        divb *= dg.elements.inverse_jacobian[element_id]
        linf_divb = max(linf_divb,abs(divb))
        l2_divb += jacobian_volume*weights[i]*weights[j]*divb^2
      end
    end
  end
  l2_divb = sqrt(l2_divb/dg.analysis_total_volume)

  return l2_divb, linf_divb
end



"""
    analyze_solution(dg::Dg2D, mesh::TreeMesh, time, dt, step, runtime_absolute, runtime_relative)

Calculate error norms and other analysis quantities to analyze the solution
during a simulation, and return the L2 and Linf errors. `dg` and `mesh` are the
DG and the mesh instance, respectively. `time`, `dt`, and `step` refer to the
current simulation time, the last time step size, and the current time step
count. The run time (in seconds) is given in `runtime_absolute`, while the
performance index is specified in `runtime_relative`.

**Note:** Keep order of analysis quantities in sync with
          [`save_analysis_header`](@ref) when adding or changing quantities.
"""
function analyze_solution(dg::Dg2D, mesh::TreeMesh, time::Real, dt::Real, step::Integer,
                          runtime_absolute::Real, runtime_relative::Real; solver_gravity=nothing)
  equation = equations(dg)

  # General information
  println()
  println("-"^80)
  println(" Simulation running '", get_name(equation), "' with POLYDEG = ", polydeg(dg))
  println("-"^80)
  println(" #timesteps:     " * @sprintf("% 14d", step) *
          "               " *
          " run time:       " * @sprintf("%10.8e s", runtime_absolute))
  println(" dt:             " * @sprintf("%10.8e", dt) *
          "               " *
          " Time/DOF/step:  " * @sprintf("%10.8e s", runtime_relative))
  println(" sim. time:      " * @sprintf("%10.8e", time))

  # Level information (only show for AMR)
  if parameter("amr_interval", 0) > 0
    levels = Vector{Int}(undef, dg.n_elements)
    for element_id in 1:dg.n_elements
      levels[element_id] = mesh.tree.levels[dg.elements.cell_ids[element_id]]
    end
    min_level = minimum(levels)
    max_level = maximum(levels)

    println(" #elements:      " * @sprintf("% 14d", dg.n_elements))
    for level = max_level:-1:min_level+1
      println(" ├── level $level:    " * @sprintf("% 14d", count(x->x==level, levels)))
    end
    println(" └── level $min_level:    " * @sprintf("% 14d", count(x->x==min_level, levels)))
  end
  println()

  # Open file for appending and store time step and time information
  if dg.save_analysis
    f = open(dg.analysis_filename, "a")
    @printf(f, "% 9d", step)
    @printf(f, "  %10.8e", time)
    @printf(f, "  %10.8e", dt)
  end

  # Calculate and print derived quantities (error norms, entropy etc.)
  # Variable names required for L2 error, Linf error, and conservation error
  if any(q in dg.analysis_quantities for q in
         (:l2_error, :linf_error, :conservation_error, :residual))
    print(" Variable:    ")
    for v in 1:nvariables(equation)
      @printf("   %-14s", varnames_cons(equation)[v])
    end
    println()
  end

  # Calculate L2/Linf errors
  if :l2_error in dg.analysis_quantities || :linf_error in dg.analysis_quantities
    l2_error, linf_error = calc_error_norms(dg, time)
  else
    error("Since `analyze_solution` returns L2/Linf errors, it is an error to not calculate them")
  end

  # L2 error
  if :l2_error in dg.analysis_quantities
    print(" L2 error:    ")
    for v in 1:nvariables(equation)
      @printf("  % 10.8e", l2_error[v])
      dg.save_analysis && @printf(f, "  % 10.8e", l2_error[v])
    end
    println()
  end

  # Linf error
  if :linf_error in dg.analysis_quantities
    print(" Linf error:  ")
    for v in 1:nvariables(equation)
      @printf("  % 10.8e", linf_error[v])
      dg.save_analysis && @printf(f, "  % 10.8e", linf_error[v])
    end
    println()
  end

  # Conservation errror
  if :conservation_error in dg.analysis_quantities
    # Calculate state integrals
    state_integrals = integrate(dg.elements.u, dg)

    # Store initial state integrals at first invocation
    if isempty(dg.initial_state_integrals)
      dg.initial_state_integrals = zeros(nvariables(equation))
      dg.initial_state_integrals .= state_integrals
    end

    print(" |∑U - ∑U₀|:  ")
    for v in 1:nvariables(equation)
      err = abs(state_integrals[v] - dg.initial_state_integrals[v])
      @printf("  % 10.8e", err)
      dg.save_analysis && @printf(f, "  % 10.8e", err)
    end
    println()
  end

  # Residual (defined here as the vector maximum of the absolute values of the time derivatives)
  if :residual in dg.analysis_quantities
    print(" max(|Uₜ|):   ")
    for v in 1:nvariables(equation)
      # Calculate maximum absolute value of Uₜ
      @views res = maximum(abs.(dg.elements.u_t[v, :, :, :]))
      @printf("  % 10.8e", res)
      dg.save_analysis && @printf(f, "  % 10.8e", res)
    end
    println()
  end

  # Entropy time derivative
  if :dsdu_ut in dg.analysis_quantities
    duds_ut = calc_entropy_timederivative(dg, time)
    print(" ∑∂S/∂U ⋅ Uₜ: ")
    @printf("  % 10.8e", duds_ut)
    dg.save_analysis && @printf(f, "  % 10.8e", duds_ut)
    println()
  end

  # Entropy
  if :entropy in dg.analysis_quantities
    s = integrate(dg, dg.elements.u) do i, j, element_id, dg, u
      cons = get_node_vars(u, dg, i, j, element_id)
      return entropy(cons, equations(dg))
    end
    print(" ∑S:          ")
    @printf("  % 10.8e", s)
    dg.save_analysis && @printf(f, "  % 10.8e", s)
    println()
  end

  # Total energy
  if :energy_total in dg.analysis_quantities
    e_total = integrate(dg, dg.elements.u) do i, j, element_id, dg, u
      cons = get_node_vars(u, dg, i, j, element_id)
      return energy_total(cons, equations(dg))
    end
    print(" ∑e_total:    ")
    @printf("  % 10.8e", e_total)
    dg.save_analysis && @printf(f, "  % 10.8e", e_total)
    println()
  end

  # Kinetic energy
  if :energy_kinetic in dg.analysis_quantities
    e_kinetic = integrate(dg, dg.elements.u) do i, j, element_id, dg, u
      cons = get_node_vars(u, dg, i, j, element_id)
      return energy_kinetic(cons, equations(dg))
    end
    print(" ∑e_kinetic:  ")
    @printf("  % 10.8e", e_kinetic)
    dg.save_analysis && @printf(f, "  % 10.8e", e_kinetic)
    println()
  end

  # Internal energy
  if :energy_internal in dg.analysis_quantities
    e_internal = integrate(dg, dg.elements.u) do i, j, element_id, dg, u
      cons = get_node_vars(u, dg, i, j, element_id)
      return energy_internal(cons, equations(dg))
    end
    print(" ∑e_internal: ")
    @printf("  % 10.8e", e_internal)
    dg.save_analysis && @printf(f, "  % 10.8e", e_internal)
    println()
  end

  # Magnetic energy
  if :energy_magnetic in dg.analysis_quantities
    e_magnetic = integrate(dg, dg.elements.u) do i, j, element_id, dg, u
      cons = get_node_vars(u, dg, i, j, element_id)
      return energy_magnetic(cons, equations(dg))
    end
    print(" ∑e_magnetic: ")
    @printf("  % 10.8e", e_magnetic)
    dg.save_analysis && @printf(f, "  % 10.8e", e_magnetic)
    println()
  end

  # Potential energy
  if :energy_potential in dg.analysis_quantities
    # FIXME: This should be implemented properly for multiple coupled solvers
    @assert !isnothing(solver_gravity) "Only works if gravity solver is supplied"
    @assert dg.initial_conditions == initial_conditions_jeans_instability "Only works with Jeans instability setup"

    e_potential = integrate(dg, dg.elements.u, solver_gravity.elements.u) do i, j, element_id, dg, u_euler, u_gravity
      cons_euler = get_node_vars(u_euler, dg, i, j, element_id)
      cons_gravity = get_node_vars(u_gravity, solver_gravity, i, j, element_id)
      # OBS! subtraction is specific to Jeans instability test where rho_0 = 1.5e7
      return (cons_euler[1] - 1.5e7) * cons_gravity[1]
    end
    print(" ∑e_pot:      ")
    @printf("  % 10.8e", e_potential)
    dg.save_analysis && @printf(f, "  % 10.8e", e_potential)
    println()
  end

  # Solenoidal condition ∇ ⋅ B = 0
  if :l2_divb in dg.analysis_quantities || :linf_divb in dg.analysis_quantities
    l2_divb, linf_divb = calc_mhd_solenoid_condition(dg, time)
  end
  # L2 norm of ∇ ⋅ B
  if :l2_divb in dg.analysis_quantities
    print(" L2 ∇ ⋅B:     ")
    @printf("  % 10.8e", l2_divb)
    dg.save_analysis && @printf(f, "  % 10.8e", l2_divb)
    println()
  end
  # Linf norm of ∇ ⋅ B
  if :linf_divb in dg.analysis_quantities
    print(" Linf ∇ ⋅B:   ")
    @printf("  % 10.8e", linf_divb)
    dg.save_analysis && @printf(f, "  % 10.8e", linf_divb)
    println()
  end

  # Cross helicity
  if :cross_helicity in dg.analysis_quantities
    h_c = integrate(dg, dg.elements.u) do i, j, element_id, dg, u
      cons = get_node_vars(u, dg, i, j, element_id)
      return cross_helicity(cons, equations(dg))
    end
    print(" ∑H_c:        ")
    @printf("  % 10.8e", h_c)
    dg.save_analysis && @printf(f, "  % 10.8e", h_c)
    println()
  end

  println("-"^80)
  println()

  # Add line break and close analysis file if it was opened
  if dg.save_analysis
    println(f)
    close(f)
  end

  # Return errors for EOC analysis
  return l2_error, linf_error
end


"""
    save_analysis_header(filename, quantities, equation)

Truncate file `filename` and save a header with the names of the quantities
`quantities` that will subsequently written to `filename` by
[`analyze_solution`](@ref). Since some quantities are equation-specific, the
system of equations instance is passed in `equation`.

**Note:** Keep order of analysis quantities in sync with
          [`analyze_solution`](@ref) when adding or changing quantities.
"""
function save_analysis_header(filename, quantities, equation::AbstractEquation{2})
  open(filename, "w") do f
    @printf(f, "#%-8s", "timestep")
    @printf(f, "  %-14s", "time")
    @printf(f, "  %-14s", "dt")
    if :l2_error in quantities
      for v in varnames_cons(equation)
        @printf(f, "   %-14s", "l2_" * v)
      end
    end
    if :linf_error in quantities
      for v in varnames_cons(equation)
        @printf(f, "   %-14s", "linf_" * v)
      end
    end
    if :conservation_error in quantities
      for v in varnames_cons(equation)
        @printf(f, "   %-14s", "cons_" * v)
      end
    end
    if :residual in quantities
      for v in varnames_cons(equation)
        @printf(f, "   %-14s", "res_" * v)
      end
    end
    if :dsdu_ut in quantities
      @printf(f, "   %-14s", "dsdu_ut")
    end
    if :entropy in quantities
      @printf(f, "   %-14s", "entropy")
    end
    if :energy_total in quantities
      @printf(f, "   %-14s", "e_total")
    end
    if :energy_kinetic in quantities
      @printf(f, "   %-14s", "e_kinetic")
    end
    if :energy_internal in quantities
      @printf(f, "   %-14s", "e_internal")
    end
    if :energy_magnetic in quantities
      @printf(f, "   %-14s", "e_magnetic")
    end
    if :energy_potential in quantities
      @printf(f, "   %-14s", "e_potential")
    end
    if :l2_divb in quantities
      @printf(f, "   %-14s", "l2_divb")
    end
    if :linf_divb in quantities
      @printf(f, "   %-14s", "linf_divb")
    end
    if :cross_helicity in quantities
      @printf(f, "   %-14s", "cross_helicity")
    end
    println(f)
  end
end


# Call equation-specific initial conditions functions and apply to all elements
function set_initial_conditions!(dg::Dg2D, time)
  equation = equations(dg)
  # make sure that the random number generator is reseted and the ICs are reproducible in the julia REPL/interactive mode
  seed!(0)
  for element_id in 1:dg.n_elements
    for j in 1:nnodes(dg)
      for i in 1:nnodes(dg)
        dg.elements.u[:, i, j, element_id] .= dg.initial_conditions(
            dg.elements.node_coordinates[:, i, j, element_id], time, equation)
      end
    end
  end
end


# Calculate time derivative
function rhs!(dg::Dg2D, t_stage)
  # Reset u_t
  @timeit timer() "reset ∂u/∂t" dg.elements.u_t .= 0

  # Calculate volume integral
  @timeit timer() "volume integral" calc_volume_integral!(dg)

  # Prolong solution to interfaces
  @timeit timer() "prolong2interfaces" prolong2interfaces!(dg)

  # Calculate interface fluxes
  @timeit timer() "interface flux" calc_interface_flux!(dg)

  # Prolong solution to boundaries
  @timeit timer() "prolong2boundaries" prolong2boundaries!(dg)

  # Calculate boundary fluxes
  @timeit timer() "boundary flux" calc_boundary_flux!(dg, t_stage)

  # Prolong solution to mortars
  @timeit timer() "prolong2mortars" prolong2mortars!(dg)

  # Calculate mortar fluxes
  @timeit timer() "mortar flux" calc_mortar_flux!(dg)

  # Calculate surface integrals
  @timeit timer() "surface integral" calc_surface_integral!(dg)

  # Apply Jacobian from mapping to reference element
  @timeit timer() "Jacobian" apply_jacobian!(dg)

  # Calculate source terms
  @timeit timer() "source terms" calc_sources!(dg, dg.source_terms, t_stage)
end


# Calculate volume integral and update u_t
calc_volume_integral!(dg::Dg2D) = calc_volume_integral!(dg.elements.u_t, dg.volume_integral_type, dg)


# Calculate 2D twopoint flux (element version)
@inline function calcflux_twopoint!(f1, f2, u, element_id, dg::Dg2D)
  @unpack volume_flux_function = dg

  for j in 1:nnodes(dg)
    for i in 1:nnodes(dg)
      # Set diagonal entries (= regular volume fluxes due to consistency)
      u_node = get_node_vars(u, dg, i, j, element_id)
      flux1 = calcflux(u_node, 1, equations(dg))
      flux2 = calcflux(u_node, 2, equations(dg))
      set_node_vars!(f1, flux1, dg, i, i, j)
      set_node_vars!(f2, flux2, dg, j, i, j)

      # Flux in x-direction
      for l in (i+1):nnodes(dg)
        u_ll = get_node_vars(u, dg, i, j, element_id)
        u_rr = get_node_vars(u, dg, l, j, element_id)
        flux = volume_flux_function(u_ll, u_rr, 1, equations(dg)) # 1-> x-direction
        for v in 1:nvariables(dg)
          f1[v, i, l, j] = f1[v, l, i, j] = flux[v]
        end
      end

      # Flux in y-direction
      for l in (j+1):nnodes(dg)
        u_ll = get_node_vars(u, dg, i, j, element_id)
        u_rr = get_node_vars(u, dg, i, l, element_id)
        flux = volume_flux_function(u_ll, u_rr, 2, equations(dg)) # 2 -> y-direction
        for v in 1:nvariables(dg)
          f2[v, j, i, l] = f2[v, l, i, j] = flux[v]
        end
      end
    end
  end

  calcflux_twopoint_nonconservative!(f1, f2, u, element_id, have_nonconservative_terms(equations(dg)), dg)
end

function calcflux_twopoint_nonconservative!(f1, f2, u, element_id, nonconservative_terms::Val{false}, dg::Dg2D)
  return nothing
end

function calcflux_twopoint_nonconservative!(f1, f2, u, element_id, nonconservative_terms::Val{true}, dg::Dg2D)
  #TODO: Create a unified interface, e.g. using non-symmetric two-point (extended) volume fluxes
  #      For now, just dispatch to an existing function for the IdealMhdEquations
  calcflux_twopoint_nonconservative!(f1, f2, u, element_id, equations(dg), dg)
end


# Calculate volume integral (DGSEM in weak form)
function calc_volume_integral!(u_t, ::Val{:weak_form}, dg::Dg2D)
  @unpack dhat = dg

  Threads.@threads for element_id in 1:dg.n_elements
    # Calculate volume integral
    for j in 1:nnodes(dg)
      for i in 1:nnodes(dg)
        u_node = get_node_vars(dg.elements.u, dg, i, j, element_id)

        flux1 = calcflux(u_node, 1, equations(dg))
        for l in 1:nnodes(dg)
          integral_contribution = dhat[l, i] * flux1
          add_to_node_vars!(u_t, integral_contribution, dg, l, j, element_id)
        end

        flux2 = calcflux(u_node, 2, equations(dg))
        for l in 1:nnodes(dg)
          integral_contribution = dhat[l, j] * flux2
          add_to_node_vars!(u_t, integral_contribution, dg, i, l, element_id)
        end
      end
    end
  end
end


# Calculate volume integral (DGSEM in split form)
# NOTE: The version for nonconservative_terms::Val{true} uses temporary storage and need to allocate.
#       The other version does not need this temporary storage but more assignments.
#       In preliminary tests, the second version is faster.
#       Currently, the nonconservative terms for IdealMhdEquations are only implemented
#       for the first version. That's why we use a dispatch here.
#       We should find a way to generalize the way we handle nonconservative terms,
#       also for the weak form volume integral type.
@inline function calc_volume_integral!(u_t, volume_integral_type::Val{:split_form}, dg::Dg2D)
  calc_volume_integral!(u_t, volume_integral_type, have_nonconservative_terms(equations(dg)), dg.thread_cache, dg)
end


function calc_volume_integral!(u_t, ::Val{:split_form}, nonconservative_terms::Val{false}, cache, dg::Dg2D)
  Threads.@threads for element_id in 1:dg.n_elements
    split_form_kernel!(u_t, element_id, nonconservative_terms, cache, dg)
  end
end

@inline function split_form_kernel!(u_t, element_id, nonconservative_terms::Val{false}, cache, dg::Dg2D, alpha=true)
  @unpack volume_flux_function, dsplit = dg

  # Calculate volume integral in one element
  for j in 1:nnodes(dg), i in 1:nnodes(dg)
    u_node = get_node_vars(dg.elements.u, dg, i, j, element_id)

    # x direction
    # use consistency of the volume flux to make this evaluation cheaper
    flux = calcflux(u_node, 1, equations(dg))
    integral_contribution = alpha * dsplit[i, i] * flux
    add_to_node_vars!(u_t, integral_contribution, dg, i, j, element_id)
    # use symmetry of the volume flux for the remaining terms
    for ii in (i+1):nnodes(dg)
      u_node_ii = get_node_vars(dg.elements.u, dg, ii, j, element_id)
      flux = volume_flux_function(u_node, u_node_ii, 1, equations(dg))
      integral_contribution = alpha * dsplit[i, ii] * flux
      add_to_node_vars!(u_t, integral_contribution, dg, i,  j, element_id)
      integral_contribution = alpha * dsplit[ii, i] * flux
      add_to_node_vars!(u_t, integral_contribution, dg, ii, j, element_id)
    end

    # y direction
    # use consistency of the volume flux to make this evaluation cheaper
    flux = calcflux(u_node, 2, equations(dg))
    integral_contribution = alpha * dsplit[j, j] * flux
    add_to_node_vars!(u_t, integral_contribution, dg, i, j, element_id)
    # use symmetry of the volume flux for the remaining terms
    for jj in (j+1):nnodes(dg)
      u_node_jj = get_node_vars(dg.elements.u, dg, i, jj, element_id)
      flux = volume_flux_function(u_node, u_node_jj, 2, equations(dg))
      integral_contribution = alpha * dsplit[j, jj] * flux
      add_to_node_vars!(u_t, integral_contribution, dg, i, j,  element_id)
      integral_contribution = alpha * dsplit[jj, j] * flux
      add_to_node_vars!(u_t, integral_contribution, dg, i, jj, element_id)
    end
  end
end


function calc_volume_integral!(u_t, ::Val{:split_form}, nonconservative_terms::Val{true}, _, dg::Dg2D)
  # Do not use the thread_cache here since that reduces the performance significantly
  # (which we do not fully understand right now)
  # @unpack f1_threaded, f2_threaded = cache

  # Pre-allocate data structures to speed up computation (thread-safe)
  A4d = MArray{Tuple{nvariables(dg), nnodes(dg), nnodes(dg), nnodes(dg)}, Float64}
  f1_threaded = [A4d(undef) for _ in 1:Threads.nthreads()]
  f2_threaded = [A4d(undef) for _ in 1:Threads.nthreads()]
  cache = (;f1_threaded, f2_threaded)

  Threads.@threads for element_id in 1:dg.n_elements
    split_form_kernel!(u_t, element_id, nonconservative_terms, cache, dg)
  end
end

@inline function split_form_kernel!(u_t, element_id, nonconservative_terms::Val{true}, cache, dg::Dg2D, alpha=true)
  @unpack volume_flux_function, dsplit_transposed = dg
  @unpack f1_threaded, f2_threaded = cache

  # Choose thread-specific pre-allocated container
  f1 = f1_threaded[Threads.threadid()]
  f2 = f2_threaded[Threads.threadid()]

  # Calculate volume fluxes (one more dimension than weak form)
  calcflux_twopoint!(f1, f2, dg.elements.u, element_id, dg)

  # Calculate volume integral in one element
  for j in 1:nnodes(dg), i in 1:nnodes(dg)
    for v in 1:nvariables(dg)
      # Use local accumulator to improve performance
      acc = zero(eltype(u_t))
      for l in 1:nnodes(dg)
        acc += dsplit_transposed[l, i] * f1[v, l, i, j] + dsplit_transposed[l, j] * f2[v, l, i, j]
      end
      u_t[v, i, j, element_id] += alpha * acc
    end
  end
end


# Calculate volume integral (DGSEM in split form with shock capturing)
function calc_volume_integral!(u_t, ::Val{:shock_capturing}, dg::Dg2D)
  # (Re-)initialize element variable storage for blending factor
  if (!haskey(dg.element_variables, :blending_factor) ||
      length(dg.element_variables[:blending_factor]) != dg.n_elements)
    dg.element_variables[:blending_factor] = Vector{Float64}(undef, dg.n_elements)
  end
  if (!haskey(dg.element_variables, :blending_factor_tmp) ||
      length(dg.element_variables[:blending_factor_tmp]) != dg.n_elements)
    dg.element_variables[:blending_factor_tmp] = Vector{Float64}(undef, dg.n_elements)
  end

  # Initialize element variable storage for the cache
  if (!haskey(dg.cache, :element_ids_dg))
    dg.cache[:element_ids_dg] = Int[]
    sizehint!(dg.cache[:element_ids_dg], dg.n_elements)
  end
  if (!haskey(dg.cache, :element_ids_dgfv))
    dg.cache[:element_ids_dgfv] = Int[]
    sizehint!(dg.cache[:element_ids_dgfv], dg.n_elements)
  end

  calc_volume_integral!(u_t, Val(:shock_capturing),
                        dg.element_variables[:blending_factor], dg.element_variables[:blending_factor_tmp],
                        dg.cache[:element_ids_dg], dg.cache[:element_ids_dgfv],
                        dg.thread_cache,
                        dg)
end

function calc_volume_integral!(u_t, ::Val{:shock_capturing}, alpha, alpha_tmp,
                               element_ids_dg, element_ids_dgfv, thread_cache, dg::Dg2D)
  @unpack dsplit_transposed, inverse_weights = dg
  @unpack fstar1_threaded, fstar2_threaded = thread_cache

  # Calculate blending factors α: u = u_DG * (1 - α) + u_FV * α
  @timeit timer() "blending factors" calc_blending_factors!(alpha, alpha_tmp, dg.elements.u,
    dg.shock_alpha_max,
    dg.shock_alpha_min,
    dg.shock_alpha_smooth,
    Val(dg.shock_indicator_variable), dg)

  # Determine element ids for DG-only and blended DG-FV volume integral
  pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha, dg)

  # Loop over pure DG elements
  @timeit timer() "pure DG" Threads.@threads for element_id in element_ids_dg
    split_form_kernel!(u_t, element_id, have_nonconservative_terms(equations(dg)), thread_cache, dg)
  end

  # Loop over blended DG-FV elements
  @timeit timer() "blended DG-FV" Threads.@threads for element_id in element_ids_dgfv
    # Calculate DG volume integral contribution
    split_form_kernel!(u_t, element_id, have_nonconservative_terms(equations(dg)), thread_cache, dg, 1 - alpha[element_id])

    # Calculate FV two-point fluxes
    fstar1 = fstar1_threaded[Threads.threadid()]
    fstar2 = fstar2_threaded[Threads.threadid()]
    calcflux_fv!(fstar1, fstar2, dg.elements.u, element_id, dg)

    # Calculate FV volume integral contribution
    for j in 1:nnodes(dg), i in 1:nnodes(dg)
      for v in 1:nvariables(dg)
        u_t[v, i, j, element_id] += ( alpha[element_id] *
                                        (inverse_weights[i] * (fstar1[v, i+1, j] - fstar1[v, i, j]) +
                                        inverse_weights[j]  * (fstar2[v, i, j+1] - fstar2[v, i, j])) )

      end
    end
  end
end


"""
    calcflux_fv!(fstar1, fstar2, u_leftright, u, element_id, dg::Dg2D)

Calculate the finite volume fluxes inside the elements.

# Arguments
- `fstar1::AbstractArray{T} where T<:Real`:
- `fstar2::AbstractArray{T} where T<:Real`
- `dg::Dg2D`
- `u::AbstractArray{T} where T<:Real`
- `element_id::Integer`
"""
@inline function calcflux_fv!(fstar1, fstar2, u, element_id, dg::Dg2D)
  @unpack surface_flux_function = dg

  fstar1[:, 1,            :] .= zero(eltype(fstar1))
  fstar1[:, nnodes(dg)+1, :] .= zero(eltype(fstar1))

  for j in 1:nnodes(dg)
    for i in 2:nnodes(dg)
      u_ll = get_node_vars(u, dg, i-1, j, element_id)
      u_rr = get_node_vars(u, dg, i,   j, element_id)
      flux = surface_flux_function(u_ll, u_rr, 1, equations(dg)) # orientation 1: x direction
      set_node_vars!(fstar1, flux, dg, i, j)
    end
  end

  fstar2[:, :, 1           ] .= zero(eltype(fstar2))
  fstar2[:, :, nnodes(dg)+1] .= zero(eltype(fstar2))

  for j in 2:nnodes(dg)
    for i in 1:nnodes(dg)
      u_ll = get_node_vars(u, dg, i, j-1, element_id)
      u_rr = get_node_vars(u, dg, i, j,   element_id)
      flux = surface_flux_function(u_ll, u_rr, 2, equations(dg)) # orientation 2: y direction
      set_node_vars!(fstar2, flux, dg, i, j)
    end
  end
end


# Prolong solution to interfaces (for GL nodes: just a copy)
function prolong2interfaces!(dg::Dg2D)
  equation = equations(dg)

  Threads.@threads for s in 1:dg.n_interfaces
    left_element_id = dg.interfaces.neighbor_ids[1, s]
    right_element_id = dg.interfaces.neighbor_ids[2, s]
    if dg.interfaces.orientations[s] == 1
      # interface in x-direction
      for j in 1:nnodes(dg), v in 1:nvariables(dg)
        dg.interfaces.u[1, v, j, s] = dg.elements.u[v, nnodes(dg), j, left_element_id]
        dg.interfaces.u[2, v, j, s] = dg.elements.u[v,          1, j, right_element_id]
      end
    else
      # interface in y-direction
      for i in 1:nnodes(dg), v in 1:nvariables(dg)
        dg.interfaces.u[1, v, i, s] = dg.elements.u[v, i, nnodes(dg), left_element_id]
        dg.interfaces.u[2, v, i, s] = dg.elements.u[v, i,          1, right_element_id]
      end
    end
  end
end


# Prolong solution to boundaries (for GL nodes: just a copy)
function prolong2boundaries!(dg::Dg2D)
  equation = equations(dg)

  for b in 1:dg.n_boundaries
    element_id = dg.boundaries.neighbor_ids[b]
    if dg.boundaries.orientations[b] == 1 # Boundary in x-direction
      if dg.boundaries.neighbor_sides[b] == 1 # Element in -x direction of boundary
        for l in 1:nnodes(dg), v in 1:nvariables(dg)
          dg.boundaries.u[1, v, l, b] = dg.elements.u[v, nnodes(dg), l, element_id]
        end
      else # Element in +x direction of boundary
        for l in 1:nnodes(dg), v in 1:nvariables(dg)
          dg.boundaries.u[2, v, l, b] = dg.elements.u[v, 1,          l, element_id]
        end
      end
    else # Boundary in y-direction
      if dg.boundaries.neighbor_sides[b] == 1 # Element in -y direction of boundary
        for l in 1:nnodes(dg), v in 1:nvariables(dg)
          dg.boundaries.u[1, v, l, b] = dg.elements.u[v, l, nnodes(dg), element_id]
        end
      else # Element in +y direction of boundary
        for l in 1:nnodes(dg), v in 1:nvariables(dg)
          dg.boundaries.u[2, v, l, b] = dg.elements.u[v, l, 1,          element_id]
        end
      end
    end
  end
end


# Prolong solution to mortars (select correct method based on mortar type)
prolong2mortars!(dg::Dg2D) = prolong2mortars!(dg, dg.mortar_type, dg.thread_cache)

# Prolong solution to mortars (l2mortar version)
function prolong2mortars!(dg::Dg2D, ::Val{:l2}, thread_cache)
  equation = equations(dg)

  # Local storage for interface data of large element
  # u_large_threaded = [zeros(nvariables(dg), nnodes(dg)) for _ in 1:Threads.nthreads()]
  @unpack u_large_threaded = thread_cache

  Threads.@threads for m in 1:dg.n_l2mortars
    u_large = u_large_threaded[Threads.threadid()]

    large_element_id = dg.l2mortars.neighbor_ids[3, m]
    upper_element_id = dg.l2mortars.neighbor_ids[2, m]
    lower_element_id = dg.l2mortars.neighbor_ids[1, m]

    # Copy solution small to small
    if dg.l2mortars.large_sides[m] == 1 # -> small elements on right side
      if dg.l2mortars.orientations[m] == 1
        # L2 mortars in x-direction
        for l in 1:nnodes(dg)
          for v in 1:nvariables(dg)
            dg.l2mortars.u_upper[2, v, l, m] = dg.elements.u[v, 1, l, upper_element_id]
            dg.l2mortars.u_lower[2, v, l, m] = dg.elements.u[v, 1, l, lower_element_id]
          end
        end
      else
        # L2 mortars in y-direction
        for l in 1:nnodes(dg)
          for v in 1:nvariables(dg)
            dg.l2mortars.u_upper[2, v, l, m] = dg.elements.u[v, l, 1, upper_element_id]
            dg.l2mortars.u_lower[2, v, l, m] = dg.elements.u[v, l, 1, lower_element_id]
          end
        end
      end
    else # large_sides[m] == 2 -> small elements on left side
      if dg.l2mortars.orientations[m] == 1
        # L2 mortars in x-direction
        for l in 1:nnodes(dg)
          for v in 1:nvariables(dg)
            dg.l2mortars.u_upper[1, v, l, m] = dg.elements.u[v, nnodes(dg), l, upper_element_id]
            dg.l2mortars.u_lower[1, v, l, m] = dg.elements.u[v, nnodes(dg), l, lower_element_id]
          end
        end
      else
        # L2 mortars in y-direction
        for l in 1:nnodes(dg)
          for v in 1:nvariables(dg)
            dg.l2mortars.u_upper[1, v, l, m] = dg.elements.u[v, l, nnodes(dg), upper_element_id]
            dg.l2mortars.u_lower[1, v, l, m] = dg.elements.u[v, l, nnodes(dg), lower_element_id]
          end
        end
      end
    end

    # Interpolate large element face data to small interface locations
    for v in 1:nvariables(dg)
      if dg.l2mortars.large_sides[m] == 1 # -> large element on left side
        if dg.l2mortars.orientations[m] == 1
          # L2 mortars in x-direction
          for l in 1:nnodes(dg)
            u_large[v, l] = dg.elements.u[v, nnodes(dg), l, large_element_id]
          end
        else
          # L2 mortars in y-direction
          for l in 1:nnodes(dg)
            u_large[v, l] = dg.elements.u[v, l, nnodes(dg), large_element_id]
          end
        end
        @views dg.l2mortars.u_upper[1, v, :, m] .= dg.mortar_forward_upper * u_large[v, :]
        @views dg.l2mortars.u_lower[1, v, :, m] .= dg.mortar_forward_lower * u_large[v, :]
      else # large_sides[m] == 2 -> large element on right side
        if dg.l2mortars.orientations[m] == 1
          # L2 mortars in x-direction
          for l in 1:nnodes(dg)
            u_large[v, l] = dg.elements.u[v, 1, l, large_element_id]
          end
        else
          # L2 mortars in y-direction
          for l in 1:nnodes(dg)
            u_large[v, l] = dg.elements.u[v, l, 1, large_element_id]
          end
        end
        @views dg.l2mortars.u_upper[2, v, :, m] .= dg.mortar_forward_upper * u_large[v, :]
        @views dg.l2mortars.u_lower[2, v, :, m] .= dg.mortar_forward_lower * u_large[v, :]
      end
    end
  end
end


# Prolong solution to mortars (ecmortar version)
function prolong2mortars!(dg::Dg2D, ::Val{:ec}, thread_cache)
  equation = equations(dg)

  Threads.@threads for m in 1:dg.n_ecmortars
    large_element_id = dg.ecmortars.neighbor_ids[3, m]
    upper_element_id = dg.ecmortars.neighbor_ids[2, m]
    lower_element_id = dg.ecmortars.neighbor_ids[1, m]

    # Copy solution small to small
    if dg.ecmortars.large_sides[m] == 1 # -> small elements on right side, large element on left
      if dg.ecmortars.orientations[m] == 1
        # L2 mortars in x-direction
        for l in 1:nnodes(dg)
          for v in 1:nvariables(dg)
            dg.ecmortars.u_upper[v, l, m] = dg.elements.u[v, 1,          l, upper_element_id]
            dg.ecmortars.u_lower[v, l, m] = dg.elements.u[v, 1,          l, lower_element_id]
            dg.ecmortars.u_large[v, l, m] = dg.elements.u[v, nnodes(dg), l, large_element_id]
          end
        end
      else
        # L2 mortars in y-direction
        for l in 1:nnodes(dg)
          for v in 1:nvariables(dg)
            dg.ecmortars.u_upper[v, l, m] = dg.elements.u[v, l, 1,          upper_element_id]
            dg.ecmortars.u_lower[v, l, m] = dg.elements.u[v, l, 1,          lower_element_id]
            dg.ecmortars.u_large[v, l, m] = dg.elements.u[v, l, nnodes(dg), large_element_id]
          end
        end
      end
    else # large_sides[m] == 2 -> small elements on left side, large element on right
      if dg.ecmortars.orientations[m] == 1
        # L2 mortars in x-direction
        for l in 1:nnodes(dg)
          for v in 1:nvariables(dg)
            dg.ecmortars.u_upper[v, l, m] = dg.elements.u[v, nnodes(dg), l, upper_element_id]
            dg.ecmortars.u_lower[v, l, m] = dg.elements.u[v, nnodes(dg), l, lower_element_id]
            dg.ecmortars.u_large[v, l, m] = dg.elements.u[v, 1,          l, large_element_id]
          end
        end
      else
        # L2 mortars in y-direction
        for l in 1:nnodes(dg)
          for v in 1:nvariables(dg)
            dg.ecmortars.u_upper[v, l, m] = dg.elements.u[v, l, nnodes(dg), upper_element_id]
            dg.ecmortars.u_lower[v, l, m] = dg.elements.u[v, l, nnodes(dg), lower_element_id]
            dg.ecmortars.u_large[v, l, m] = dg.elements.u[v, l, 1,          large_element_id]
          end
        end
      end
    end
  end
end


"""
    calc_fstar!(destination, u_interfaces_left, u_interfaces_right, interface_id, orientations, dg::Dg2D)

Calculate the surface flux across interface with different states given by
`u_interfaces_left, u_interfaces_right` on both sides (EC mortar version).

# Arguments
- `destination::AbstractArray{T,3} where T<:Real`:
  The array of surface flux values (updated inplace).
- `u_interfaces_left::AbstractArray{T,3} where T<:Real``
- `u_interfaces_right::AbstractArray{T,3} where T<:Real``
- `interface_id::Integer`
- `orientations::Vector{T} where T<:Integer`
- `dg::Dg2D`
"""
function calc_fstar!(destination, u_interfaces_left, u_interfaces_right, interface_id, orientations, dg::Dg2D)
  @unpack surface_flux_function = dg

  # Call pointwise two-point numerical flux function
  # i -> left, j -> right
  for j in 1:nnodes(dg), i in 1:nnodes(dg)
    u_ll = get_node_vars(u_interfaces_left,  dg, i, interface_id)
    u_rr = get_node_vars(u_interfaces_right, dg, j, interface_id)
    flux = surface_flux_function(u_ll, u_rr, orientations[interface_id], equations(dg))

    # Copy flux back to actual flux array
    set_node_vars!(destination, flux, dg, i, j)
  end
end

"""
    calc_fstar!(destination, u_interfaces, interface_id, orientations, dg::Dg2D)

Calculate the surface flux across interface with different states given by
`u_interfaces_left, u_interfaces_right` on both sides (interface version).

# Arguments
- `destination::AbstractArray{T,2} where T<:Real`:
  The array of surface flux values (updated inplace).
- `u_interfaces::AbstractArray{T,4} where T<:Real``
- `interface_id::Integer`
- `orientations::Vector{T} where T<:Integer`
- `dg::Dg2D`
"""
function calc_fstar!(destination, u_interfaces, interface_id, orientations, dg::Dg2D)
  @unpack surface_flux_function = dg

  for i in 1:nnodes(dg)
    # Call pointwise two-point numerical flux function
    u_ll, u_rr = get_surface_node_vars(u_interfaces, dg, i, interface_id)
    flux = surface_flux_function(u_ll, u_rr, orientations[interface_id], equations(dg))

    # Copy flux to left and right element storage
    set_node_vars!(destination, flux, dg, i)
  end
end


# Calculate and store the surface fluxes (standard Riemann and nonconservative parts) at an interface
# OBS! Regarding the nonconservative terms: 1) currently only needed for the MHD equations
#                                           2) not implemented for boundaries
calc_interface_flux!(dg::Dg2D) = calc_interface_flux!(dg.elements.surface_flux_values,
                                                      have_nonconservative_terms(dg.equations), dg)

function calc_interface_flux!(surface_flux_values, nonconservative_terms::Val{false}, dg::Dg2D)
  @unpack surface_flux_function = dg
  @unpack u, neighbor_ids, orientations = dg.interfaces

  Threads.@threads for s in 1:dg.n_interfaces
    # Get neighboring elements
    left_id  = neighbor_ids[1, s]
    right_id = neighbor_ids[2, s]

    # Determine interface direction with respect to elements:
    # orientation = 1: left -> 2, right -> 1
    # orientation = 2: left -> 4, right -> 3
    left_direction  = 2 * orientations[s]
    right_direction = 2 * orientations[s] - 1

    for i in 1:nnodes(dg)
      # Call pointwise Riemann solver
      u_ll, u_rr = get_surface_node_vars(u, dg, i, s)
      flux = surface_flux_function(u_ll, u_rr, orientations[s], equations(dg))

      # Copy flux to left and right element storage
      for v in 1:nvariables(dg)
        surface_flux_values[v, i, left_direction, left_id]  = surface_flux_values[v, i, right_direction, right_id] = flux[v]
      end
    end
  end
end

# Calculate and store Riemann and nonconservative fluxes across interfaces
function calc_interface_flux!(surface_flux_values, nonconservative_terms::Val{true}, dg::Dg2D)
  #TODO temporary workaround while implementing the other stuff
  calc_interface_flux!(surface_flux_values, dg.interfaces.neighbor_ids, dg.interfaces.u,
                       nonconservative_terms, dg.interfaces.orientations, dg)
end

function calc_interface_flux!(surface_flux_values, neighbor_ids,
                              u_interfaces, nonconservative_terms::Val{true},
                              orientations, dg::Dg2D)

  # Type alias only for convenience
  A2d = MArray{Tuple{nvariables(dg), nnodes(dg)}, Float64}

  # Pre-allocate data structures to speed up computation (thread-safe)
  fstar_threaded = [A2d(undef) for _ in 1:Threads.nthreads()]

  noncons_diamond_primary_threaded = [A2d(undef) for _ in 1:Threads.nthreads()]
  noncons_diamond_secondary_threaded = [A2d(undef) for _ in 1:Threads.nthreads()]

  Threads.@threads for s in 1:dg.n_interfaces
    # Choose thread-specific pre-allocated container
    fstar = fstar_threaded[Threads.threadid()]

    noncons_diamond_primary = noncons_diamond_primary_threaded[Threads.threadid()]
    noncons_diamond_secondary = noncons_diamond_secondary_threaded[Threads.threadid()]

    # Calculate flux
    calc_fstar!(fstar, u_interfaces, s, orientations, dg)

    # Compute the nonconservative numerical "flux" along an interface
    # Done twice because left/right orientation matters så
    # 1 -> primary element and 2 -> secondary element
    # See Bohm et al. 2018 for details on the nonconservative diamond "flux"
    for i in 1:nnodes(dg)
      # Call pointwise nonconservative term
      u_ll, u_rr = get_surface_node_vars(u_interfaces, dg, i, s)
      noncons_primary   = noncons_interface_flux!(u_ll, u_rr, orientations[s], equations(dg))
      noncons_secondary = noncons_interface_flux!(u_rr, u_ll, orientations[s], equations(dg))
      # Save to primary and secondary temporay storage
      set_node_vars!(noncons_diamond_primary, noncons_primary, dg, i)
      set_node_vars!(noncons_diamond_secondary, noncons_secondary, dg, i)
    end

    # Get neighboring elements
    left_neighbor_id  = neighbor_ids[1, s]
    right_neighbor_id = neighbor_ids[2, s]

    # Determine interface direction with respect to elements:
    # orientation = 1: left -> 2, right -> 1
    # orientation = 2: left -> 4, right -> 3
    left_neighbor_direction = 2 * orientations[s]
    right_neighbor_direction = 2 * orientations[s] - 1

    # Copy flux to left and right element storage
    for i in 1:nnodes(dg)
      for v in 1:nvariables(dg)
        surface_flux_values[v, i, left_neighbor_direction,  left_neighbor_id]  = (fstar[v, i] +
            noncons_diamond_primary[v, i])
        surface_flux_values[v, i, right_neighbor_direction, right_neighbor_id] = (fstar[v, i] +
            noncons_diamond_secondary[v, i])
      end
    end
  end
end


# Calculate and store boundary flux across domain boundaries
#NOTE: Do we need to dispatch on have_nonconservative_terms(dg.equations)?
calc_boundary_flux!(dg::Dg2D, time) = calc_boundary_flux!(dg.elements.surface_flux_values,
                                                    dg, time)
function calc_boundary_flux!(surface_flux_values, dg::Dg2D, time)
  @unpack surface_flux_function = dg
  @unpack u, neighbor_ids, neighbor_sides, node_coordinates, orientations = dg.boundaries

  Threads.@threads for b in 1:dg.n_boundaries
    # Fill outer boundary state
    # FIXME: This should be replaced by a proper boundary condition
    for i in 1:nnodes(dg)
      @views u[3 - neighbor_sides[b], :, i, b] .= dg.initial_conditions(
        node_coordinates[:, i, b], time, equations(dg))
    end

    # Get neighboring element
    neighbor_id = neighbor_ids[b]

    # Determine boundary direction with respect to elements:
    # orientation = 1: left -> 2, right -> 1
    # orientation = 2: left -> 4, right -> 3
    if orientations[b] == 1 # Boundary in x-direction
      if neighbor_sides[b] == 1 # Element is on the left, boundary on the right
        direction = 2
      else # Element is on the right, boundary on the left
        direction = 1
      end
    else # Boundary in y-direction
      if neighbor_sides[b] == 1 # Element is below, boundary is above
        direction = 4
      else # Element is above, boundary is below
        direction = 3
      end
    end

    for i in 1:nnodes(dg)
      # Call pointwise Riemann solver
      u_ll, u_rr = get_surface_node_vars(u, dg, i, b)
      flux = surface_flux_function(u_ll, u_rr, orientations[b], equations(dg))

      # Copy flux to left and right element storage
      for v in 1:nvariables(dg)
        surface_flux_values[v, i, direction, neighbor_id] = flux[v]
      end
    end
  end
end


# Calculate and store fluxes across mortars (select correct method based on mortar type)
calc_mortar_flux!(dg::Dg2D) = calc_mortar_flux!(dg, dg.mortar_type)

# Calculate and store fluxes and nonconservative terms across L2 mortars
calc_mortar_flux!(dg::Dg2D, mortar_type::Val{:l2}) = calc_mortar_flux!(dg.elements.surface_flux_values, dg, mortar_type,
                                                                       have_nonconservative_terms(dg.equations), dg.l2mortars, dg.thread_cache)
function calc_mortar_flux!(surface_flux_values, dg::Dg2D, mortar_type::Val{:l2},
                           nonconservative_terms::Val{true}, mortars, cache)
  @unpack neighbor_ids, u_lower, u_upper, orientations = mortars
  @unpack fstar_upper_threaded, fstar_lower_threaded = cache

  # Type alias only for convenience
  A2d = MArray{Tuple{nvariables(dg), nnodes(dg)}, Float64}

  noncons_diamond_upper_threaded = [A2d(undef) for _ in 1:Threads.nthreads()]
  noncons_diamond_lower_threaded = [A2d(undef) for _ in 1:Threads.nthreads()]

  Threads.@threads for m in 1:dg.n_l2mortars
    large_element_id = dg.l2mortars.neighbor_ids[3, m]
    upper_element_id = dg.l2mortars.neighbor_ids[2, m]
    lower_element_id = dg.l2mortars.neighbor_ids[1, m]

    # Choose thread-specific pre-allocated container
    fstar_upper = fstar_upper_threaded[Threads.threadid()]
    fstar_lower = fstar_lower_threaded[Threads.threadid()]

    noncons_diamond_upper = noncons_diamond_upper_threaded[Threads.threadid()]
    noncons_diamond_lower = noncons_diamond_lower_threaded[Threads.threadid()]

    # Calculate fluxes
    calc_fstar!(fstar_upper, u_upper, m, orientations, dg)
    calc_fstar!(fstar_lower, u_lower, m, orientations, dg)

    # Compute the nonconservative numerical terms along the upper and lower interface
    # Done twice because left/right orientation matters
    # 1 -> primary element and 2 -> secondary element
    # See Bohm et al. 2018 for details on the nonconservative diamond "flux"
    if dg.l2mortars.large_sides[m] == 1 # -> small elements on right side
      for i in 1:nnodes(dg)
        # pull the left and right solutions
        u_upper_ll, u_upper_rr = get_surface_node_vars(u_upper, dg, i, m)
        u_lower_ll, u_lower_rr = get_surface_node_vars(u_lower, dg, i, m)
        # Call pointwise nonconservative term
        noncons_upper = noncons_interface_flux!(u_upper_ll,u_upper_rr,orientations[m],equations(dg))
        noncons_lower = noncons_interface_flux!(u_lower_ll,u_lower_rr,orientations[m],equations(dg))
        # Save to primary and secondary temporay storage
        set_node_vars!(noncons_diamond_upper, noncons_upper, dg, i)
        set_node_vars!(noncons_diamond_lower, noncons_lower, dg, i)
      end
    else # large_sides[m] == 2 -> small elements on the left
      for i in 1:nnodes(dg)
        # pull the left and right solutions
        u_upper_ll, u_upper_rr = get_surface_node_vars(u_upper, dg, i, m)
        u_lower_ll, u_lower_rr = get_surface_node_vars(u_lower, dg, i, m)
        # Call pointwise nonconservative term
        noncons_upper = noncons_interface_flux!(u_upper_rr,u_upper_ll,orientations[m],equations(dg))
        noncons_lower = noncons_interface_flux!(u_lower_rr,u_lower_ll,orientations[m],equations(dg))
        # Save to primary and secondary temporay storage
        set_node_vars!(noncons_diamond_upper, noncons_upper, dg, i)
        set_node_vars!(noncons_diamond_lower, noncons_lower, dg, i)
      end
    end

    # Copy flux small to small
    if dg.l2mortars.large_sides[m] == 1 # -> small elements on right side
      if dg.l2mortars.orientations[m] == 1
        # L2 mortars in x-direction
        surface_flux_values[:, :, 1, upper_element_id] .= (fstar_upper + noncons_diamond_upper)
        surface_flux_values[:, :, 1, lower_element_id] .= (fstar_lower + noncons_diamond_lower)
      else
        # L2 mortars in y-direction
        surface_flux_values[:, :, 3, upper_element_id] .= (fstar_upper + noncons_diamond_upper)
        surface_flux_values[:, :, 3, lower_element_id] .= (fstar_lower + noncons_diamond_lower)
      end
    else # large_sides[m] == 2 -> small elements on left side
      if dg.l2mortars.orientations[m] == 1
        # L2 mortars in x-direction
        surface_flux_values[:, :, 2, upper_element_id] .= (fstar_upper + noncons_diamond_upper)
        surface_flux_values[:, :, 2, lower_element_id] .= (fstar_lower + noncons_diamond_lower)
      else
        # L2 mortars in y-direction
        surface_flux_values[:, :, 4, upper_element_id] .= (fstar_upper + noncons_diamond_upper)
        surface_flux_values[:, :, 4, lower_element_id] .= (fstar_lower + noncons_diamond_lower)
      end
    end

    # Project small fluxes to large element
    for v in 1:nvariables(dg)
      if dg.l2mortars.large_sides[m] == 1 # -> large element on left side
        @views large_surface_flux_values = (dg.l2mortar_reverse_upper *
                                          (fstar_upper[v, :] + noncons_diamond_upper[v,:]) +
                                            dg.l2mortar_reverse_lower *
                                          (fstar_lower[v, :] + noncons_diamond_lower[v,:]))
        if dg.l2mortars.orientations[m] == 1
          # L2 mortars in x-direction
          surface_flux_values[v, :, 2, large_element_id] .= large_surface_flux_values
        else
          # L2 mortars in y-direction
          surface_flux_values[v, :, 4, large_element_id] .= large_surface_flux_values
        end
      else # large_sides[m] == 2 -> large element on right side
        @views large_surface_flux_values = (dg.l2mortar_reverse_upper *
                                        (fstar_upper[v, :] + noncons_diamond_upper[v,:]) +
                                            dg.l2mortar_reverse_lower *
                                        (fstar_lower[v, :] + noncons_diamond_lower[v,:]))
        if dg.l2mortars.orientations[m] == 1
          # L2 mortars in x-direction
          surface_flux_values[v, :, 1, large_element_id] .= large_surface_flux_values
        else
          # L2 mortars in y-direction
          surface_flux_values[v, :, 3, large_element_id] .= large_surface_flux_values
        end
      end
    end
  end
end

# Calculate and store fluxes across L2 mortars
function calc_mortar_flux!(surface_flux_values, dg::Dg2D, mortar_type::Val{:l2},
                           nonconservative_terms::Val{false}, mortars, cache)
  @unpack neighbor_ids, u_lower, u_upper, orientations = mortars
  @unpack fstar_upper_threaded, fstar_lower_threaded = cache

  Threads.@threads for m in 1:dg.n_l2mortars
    large_element_id = dg.l2mortars.neighbor_ids[3, m]
    upper_element_id = dg.l2mortars.neighbor_ids[2, m]
    lower_element_id = dg.l2mortars.neighbor_ids[1, m]

    # Choose thread-specific pre-allocated container
    fstar_upper = fstar_upper_threaded[Threads.threadid()]
    fstar_lower = fstar_lower_threaded[Threads.threadid()]

    # Calculate fluxes
    calc_fstar!(fstar_upper, u_upper, m, orientations, dg)
    calc_fstar!(fstar_lower, u_lower, m, orientations, dg)

    # Copy flux small to small
    if dg.l2mortars.large_sides[m] == 1 # -> small elements on right side
      if dg.l2mortars.orientations[m] == 1
        # L2 mortars in x-direction
        surface_flux_values[:, :, 1, upper_element_id] .= fstar_upper
        surface_flux_values[:, :, 1, lower_element_id] .= fstar_lower
      else
        # L2 mortars in y-direction
        surface_flux_values[:, :, 3, upper_element_id] .= fstar_upper
        surface_flux_values[:, :, 3, lower_element_id] .= fstar_lower
      end
    else # large_sides[m] == 2 -> small elements on left side
      if dg.l2mortars.orientations[m] == 1
        # L2 mortars in x-direction
        surface_flux_values[:, :, 2, upper_element_id] .= fstar_upper
        surface_flux_values[:, :, 2, lower_element_id] .= fstar_lower
      else
        # L2 mortars in y-direction
        surface_flux_values[:, :, 4, upper_element_id] .= fstar_upper
        surface_flux_values[:, :, 4, lower_element_id] .= fstar_lower
      end
    end

    # Project small fluxes to large element
    for v in 1:nvariables(dg)
      @views large_surface_flux_values = (dg.l2mortar_reverse_upper * fstar_upper[v, :] +
                                          dg.l2mortar_reverse_lower * fstar_lower[v, :])
      if dg.l2mortars.large_sides[m] == 1 # -> large element on left side
        if dg.l2mortars.orientations[m] == 1
          # L2 mortars in x-direction
          surface_flux_values[v, :, 2, large_element_id] .= large_surface_flux_values
        else
          # L2 mortars in y-direction
          surface_flux_values[v, :, 4, large_element_id] .= large_surface_flux_values
        end
      else # large_sides[m] == 2 -> large element on right side
        if dg.l2mortars.orientations[m] == 1
          # L2 mortars in x-direction
          surface_flux_values[v, :, 1, large_element_id] .= large_surface_flux_values
        else
          # L2 mortars in y-direction
          surface_flux_values[v, :, 3, large_element_id] .= large_surface_flux_values
        end
      end
    end
  end
end


# Calculate and store fluxes across EC mortars
calc_mortar_flux!(dg::Dg2D, v::Val{:ec}) = calc_mortar_flux!(dg.elements.surface_flux_values, dg, v,
                                                             dg.ecmortars.neighbor_ids,
                                                             dg.ecmortars.u_lower,
                                                             dg.ecmortars.u_upper,
                                                             dg.ecmortars.u_large,
                                                             dg.ecmortars.orientations)
function calc_mortar_flux!(surface_flux_values, dg::Dg2D, ::Val{:ec},
                           neighbor_ids,
                           u_lower, u_upper, u_large,
                           orientations)
  # Type alias only for convenience
  A3d = MArray{Tuple{nvariables(dg), nnodes(dg), nnodes(dg)}, Float64}
  A1d = MArray{Tuple{nvariables(dg)}, Float64}

  # Pre-allocate data structures to speed up computation (thread-safe)
  fstar_upper_threaded = [A3d(undef) for _ in 1:Threads.nthreads()]
  fstar_lower_threaded = [A3d(undef) for _ in 1:Threads.nthreads()]
  fstarnode_upper_threaded = [A1d(undef) for _ in 1:Threads.nthreads()]
  fstarnode_lower_threaded = [A1d(undef) for _ in 1:Threads.nthreads()]

  # Store matrix references for convenience (notation: R -> large, L -> small)
  # Note: the same notation is used in the publications of Lucas Friedrich
  PR2L_upper = dg.mortar_forward_upper
  PR2L_lower = dg.mortar_forward_lower
  PL2R_upper = dg.ecmortar_reverse_upper
  PL2R_lower = dg.ecmortar_reverse_lower

  Threads.@threads for m in 1:dg.n_ecmortars
    large_element_id = dg.ecmortars.neighbor_ids[3, m]
    upper_element_id = dg.ecmortars.neighbor_ids[2, m]
    lower_element_id = dg.ecmortars.neighbor_ids[1, m]

    # Choose thread-specific pre-allocated container
    fstar_upper = fstar_upper_threaded[Threads.threadid()]
    fstar_lower = fstar_lower_threaded[Threads.threadid()]

    # Calculate fluxes
    if dg.ecmortars.large_sides[m] == 1 # -> small elements on right side, large element on left
      calc_fstar!(fstar_upper, u_large, u_upper, m, orientations, dg)
      calc_fstar!(fstar_lower, u_large, u_lower, m, orientations, dg)
    else # large_sides[m] == 2 -> small elements on left side, large element on right
      calc_fstar!(fstar_upper, u_upper, u_large, m, orientations, dg)
      calc_fstar!(fstar_lower, u_lower, u_large, m, orientations, dg)
    end

    # Transfer fluxes to elements
    if dg.ecmortars.large_sides[m] == 1 # -> small elements on right side, large element on left
      if dg.ecmortars.orientations[m] == 1
        # EC mortars in x-direction
        surface_flux_values[:, :, 2, large_element_id] .= 0.0
        surface_flux_values[:, :, 1, upper_element_id] .= 0.0
        surface_flux_values[:, :, 1, lower_element_id] .= 0.0
        for i in 1:nnodes(dg)
          for l in 1:nnodes(dg)
            for v in 1:nvariables(dg)
              surface_flux_values[v, i, 2, large_element_id] += (PL2R_upper[i, l] * fstar_upper[v, i, l] +
                                                                 PL2R_lower[i, l] * fstar_lower[v, i, l])
              surface_flux_values[v, i, 1, upper_element_id] +=  PR2L_upper[i, l] * fstar_upper[v, l, i]
              surface_flux_values[v, i, 1, lower_element_id] +=  PR2L_lower[i, l] * fstar_lower[v, l, i]
            end
          end
        end
      else
        # EC mortars in y-direction
        surface_flux_values[:, :, 4, large_element_id] .= 0.0
        surface_flux_values[:, :, 3, upper_element_id] .= 0.0
        surface_flux_values[:, :, 3, lower_element_id] .= 0.0
        for i in 1:nnodes(dg)
          for l in 1:nnodes(dg)
            for v in 1:nvariables(dg)
              surface_flux_values[v, i, 4, large_element_id] += (PL2R_upper[i, l] * fstar_upper[v, i, l] +
                                                                 PL2R_lower[i, l] * fstar_lower[v, i, l])
              surface_flux_values[v, i, 3, upper_element_id] +=  PR2L_upper[i, l] * fstar_upper[v, l, i]
              surface_flux_values[v, i, 3, lower_element_id] +=  PR2L_lower[i, l] * fstar_lower[v, l, i]
            end
          end
        end
      end
    else # large_sides[m] == 2 -> small elements on left side, large element on right
      if dg.ecmortars.orientations[m] == 1
        # EC mortars in x-direction
        surface_flux_values[:, :, 1, large_element_id] .= 0.0
        surface_flux_values[:, :, 2, upper_element_id] .= 0.0
        surface_flux_values[:, :, 2, lower_element_id] .= 0.0
        for i in 1:nnodes(dg)
          for l in 1:nnodes(dg)
            for v in 1:nvariables(dg)
              surface_flux_values[v, i, 1, large_element_id] += (PL2R_upper[i, l] * fstar_upper[v, l, i] +
                                                                 PL2R_lower[i, l] * fstar_lower[v, l, i])
              surface_flux_values[v, i, 2, upper_element_id] +=  PR2L_upper[i, l] * fstar_upper[v, i, l]
              surface_flux_values[v, i, 2, lower_element_id] +=  PR2L_lower[i, l] * fstar_lower[v, i, l]
            end
          end
        end
      else
        # EC mortars in y-direction
        surface_flux_values[:, :, 3, large_element_id] .= 0.0
        surface_flux_values[:, :, 4, upper_element_id] .= 0.0
        surface_flux_values[:, :, 4, lower_element_id] .= 0.0
        for i in 1:nnodes(dg)
          for l in 1:nnodes(dg)
            for v in 1:nvariables(dg)
              surface_flux_values[v, i, 3, large_element_id] += (PL2R_upper[i, l] * fstar_upper[v, l, i] +
                                                                 PL2R_lower[i, l] * fstar_lower[v, l, i])
              surface_flux_values[v, i, 4, upper_element_id] +=  PR2L_upper[i, l] * fstar_upper[v, i, l]
              surface_flux_values[v, i, 4, lower_element_id] +=  PR2L_lower[i, l] * fstar_lower[v, i, l]
            end
          end
        end
      end
    end
  end
end


# Calculate surface integrals and update u_t
calc_surface_integral!(dg::Dg2D) = calc_surface_integral!(dg.elements.u_t, dg.elements.surface_flux_values, dg)
function calc_surface_integral!(u_t, surface_flux_values, dg::Dg2D)
  @unpack lhat = dg

  Threads.@threads for element_id in 1:dg.n_elements
    for l in 1:nnodes(dg)
      for v in 1:nvariables(dg)
        # surface at -x
        u_t[v, 1,          l, element_id] -= surface_flux_values[v, l, 1, element_id] * lhat[1,          1]
        # surface at +x
        u_t[v, nnodes(dg), l, element_id] += surface_flux_values[v, l, 2, element_id] * lhat[nnodes(dg), 2]
        # surface at -y
        u_t[v, l, 1,          element_id] -= surface_flux_values[v, l, 3, element_id] * lhat[1,          1]
        # surface at +y
        u_t[v, l, nnodes(dg), element_id] += surface_flux_values[v, l, 4, element_id] * lhat[nnodes(dg), 2]
      end
    end
  end
end


# Apply Jacobian from mapping to reference element
function apply_jacobian!(dg::Dg2D)
  Threads.@threads for element_id in 1:dg.n_elements
    factor = -dg.elements.inverse_jacobian[element_id]
    for j in 1:nnodes(dg)
      for i in 1:nnodes(dg)
        for v in 1:nvariables(dg)
          dg.elements.u_t[v, i, j, element_id] *= factor
        end
      end
    end
  end
end


# Calculate source terms and apply them to u_t
function calc_sources!(dg::Dg2D, source_terms::Nothing, t)
  return nothing
end

function calc_sources!(dg::Dg2D, source_terms, t)
  Threads.@threads for element_id in 1:dg.n_elements
    source_terms(dg.elements.u_t, dg.elements.u,
                 dg.elements.node_coordinates, element_id, t, nnodes(dg), equations(dg))
  end
end


# Calculate stable time step size
function calc_dt(dg::Dg2D, cfl)
  min_dt = Inf
  for element_id in 1:dg.n_elements
    dt = calc_max_dt(dg.elements.u, element_id,
                     dg.elements.inverse_jacobian[element_id], cfl, equations(dg), dg)
    min_dt = min(min_dt, dt)
  end

  return min_dt
end

# Calculate blending factors used for shock capturing, or amr control
function calc_blending_factors!(alpha, alpha_pre_smooth, u,
                                alpha_max, alpha_min, do_smoothing,
                                indicator_variable, dg::Dg2D)
  # Calculate blending factor
  indicator_threaded = [zeros(1, nnodes(dg), nnodes(dg)) for idx in 1:Threads.nthreads()]
  modal_threaded     = [zeros(1, nnodes(dg), nnodes(dg)) for idx in 1:Threads.nthreads()]
  threshold = 0.5 * 10^(-1.8 * (nnodes(dg))^0.25)
  parameter_s = log((1 - 0.0001)/0.0001)

  Threads.@threads for element_id in 1:dg.n_elements
    indicator = indicator_threaded[Threads.threadid()]
    modal     = modal_threaded[Threads.threadid()]

    # Calculate indicator variables at Gauss-Lobatto nodes
    cons2indicator!(indicator, u, element_id, nnodes(dg), indicator_variable, equations(dg))

    # Convert to modal representation
    multiply_coordinatewise!(modal, indicator, dg.inverse_vandermonde_legendre)

    # Calculate total energies for all modes, without highest, without two highest
    total_energy = 0.0
    for j in 1:nnodes(dg), i in 1:nnodes(dg)
      total_energy += modal[1, i, j]^2
    end
    total_energy_clip1 = 0.0
    for j in 1:(nnodes(dg)-1), i in 1:(nnodes(dg)-1)
      total_energy_clip1 += modal[1, i, j]^2
    end
    total_energy_clip2 = 0.0
    for j in 1:(nnodes(dg)-2), i in 1:(nnodes(dg)-2)
      total_energy_clip2 += modal[1, i, j]^2
    end

    # Calculate energy in lower modes
    energy = max((total_energy - total_energy_clip1)/total_energy,
                 (total_energy_clip1 - total_energy_clip2)/total_energy_clip1)

    alpha[element_id] = 1 / (1 + exp(-parameter_s/threshold * (energy - threshold)))

    # Take care of the case close to pure DG
    if (alpha[element_id] < alpha_min)
      alpha[element_id] = 0.
    end

    # Take care of the case close to pure FV
    if (alpha[element_id] > 1-alpha_min)
      alpha[element_id] = 1.
    end

    # Clip the maximum amount of FV allowed
    alpha[element_id] = min(alpha_max, alpha[element_id])
  end

  if (do_smoothing)
    # Diffuse alpha values by setting each alpha to at least 50% of neighboring elements' alpha
    # Copy alpha values such that smoothing is indpedenent of the element access order
    alpha_pre_smooth .= alpha

    # Loop over interfaces
    for interface_id in 1:dg.n_interfaces
      # Get neighboring element ids
      left  = dg.interfaces.neighbor_ids[1, interface_id]
      right = dg.interfaces.neighbor_ids[2, interface_id]

      # Apply smoothing
      alpha[left]  = max(alpha_pre_smooth[left],  0.5 * alpha_pre_smooth[right], alpha[left])
      alpha[right] = max(alpha_pre_smooth[right], 0.5 * alpha_pre_smooth[left],  alpha[right])
    end

    # Loop over L2 mortars
    for l2mortar_id in 1:dg.n_l2mortars
      # Get neighboring element ids
      lower = dg.l2mortars.neighbor_ids[1, l2mortar_id]
      upper = dg.l2mortars.neighbor_ids[2, l2mortar_id]
      large = dg.l2mortars.neighbor_ids[3, l2mortar_id]

      # Apply smoothing
      alpha[lower] = max(alpha_pre_smooth[lower], 0.5 * alpha_pre_smooth[large], alpha[lower])
      alpha[upper] = max(alpha_pre_smooth[upper], 0.5 * alpha_pre_smooth[large], alpha[upper])
      alpha[large] = max(alpha_pre_smooth[large], 0.5 * alpha_pre_smooth[lower], alpha[large])
      alpha[large] = max(alpha_pre_smooth[large], 0.5 * alpha_pre_smooth[upper], alpha[large])
    end

    # Loop over EC mortars
    for ecmortar_id in 1:dg.n_ecmortars
      # Get neighboring element ids
      lower = dg.ecmortars.neighbor_ids[1, ecmortar_id]
      upper = dg.ecmortars.neighbor_ids[2, ecmortar_id]
      large = dg.ecmortars.neighbor_ids[3, ecmortar_id]

      # Apply smoothing
      alpha[lower] = max(alpha_pre_smooth[lower], 0.5 * alpha_pre_smooth[large], alpha[lower])
      alpha[upper] = max(alpha_pre_smooth[upper], 0.5 * alpha_pre_smooth[large], alpha[upper])
      alpha[large] = max(alpha_pre_smooth[large], 0.5 * alpha_pre_smooth[lower], alpha[large])
      alpha[large] = max(alpha_pre_smooth[large], 0.5 * alpha_pre_smooth[upper], alpha[large])
    end
  end
end


"""
    pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha, dg)

Given blending factors `alpha` and the solver `dg`, fill
`element_ids_dg` with the IDs of elements using a pure DG scheme and
`element_ids_dgfv` with the IDs of elements using a blended DG-FV scheme.
"""
function pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha, dg::Dg2D)
  empty!(element_ids_dg)
  empty!(element_ids_dgfv)

  for element_id in 1:dg.n_elements
    # Clip blending factor for values close to zero (-> pure DG)
    dg_only = isapprox(alpha[element_id], 0, atol=1e-12)
    if dg_only
      push!(element_ids_dg, element_id)
    else
      push!(element_ids_dgfv, element_id)
    end
  end
end
