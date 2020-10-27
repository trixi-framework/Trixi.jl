# Main DG data structure that contains all relevant data for the DG solver
mutable struct Dg1D{Eqn<:AbstractEquations, NVARS, POLYDEG,
                  SurfaceFlux, VolumeFlux, InitialCondition, SourceTerms, BoundaryConditions,
                  VolumeIntegralType, ShockIndicatorVariable,
                  VectorNnodes, MatrixNnodes, MatrixNnodes2,
                  InverseVandermondeLegendre, MortarMatrix,
                  VectorAnalysisNnodes, AnalysisVandermonde} <: AbstractDg{1, POLYDEG}
  equations::Eqn

  surface_flux_function::SurfaceFlux
  volume_flux_function::VolumeFlux

  initial_condition::InitialCondition
  source_terms::SourceTerms

  elements::ElementContainer1D{Float64, NVARS, POLYDEG}
  n_elements::Int

  interfaces::InterfaceContainer1D{Float64, NVARS, POLYDEG}
  n_interfaces::Int

  boundaries::BoundaryContainer1D{Float64, NVARS, POLYDEG}
  n_boundaries::Int
  n_boundaries_per_direction::SVector{2, Int}

  n_l2mortars::Int # TODO: Taal. Only needed for simulation summary output -> fix me when Taal is alive

  boundary_conditions::BoundaryConditions

  nodes::VectorNnodes
  weights::VectorNnodes
  inverse_weights::VectorNnodes
  inverse_vandermonde_legendre::InverseVandermondeLegendre
  lhat::MatrixNnodes2

  volume_integral_type::VolumeIntegralType
  dhat::MatrixNnodes
  dsplit::MatrixNnodes
  dsplit_transposed::MatrixNnodes

  amr_refine_right::MortarMatrix
  amr_refine_left::MortarMatrix
  amr_coarsen_right::MortarMatrix
  amr_coarsen_left::MortarMatrix


  analysis_nodes::VectorAnalysisNnodes
  analysis_weights::VectorAnalysisNnodes
  analysis_weights_volume::VectorAnalysisNnodes
  analysis_vandermonde::AnalysisVandermonde
  analysis_total_volume::Float64
  analysis_quantities::Vector{Symbol}
  save_analysis::Bool
  analysis_filename::String

  shock_indicator_variable::ShockIndicatorVariable
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
function Dg1D(equation::AbstractEquations{NDIMS, NVARS}, surface_flux_function, volume_flux_function, initial_condition, source_terms, mesh::TreeMesh{NDIMS}, POLYDEG) where {NDIMS, NVARS}
  # Get cells for which an element needs to be created (i.e., all leaf cells)
  leaf_cell_ids = leaf_cells(mesh.tree)

  # Initialize element container
  elements = init_elements(leaf_cell_ids, mesh, Float64, NVARS, POLYDEG)
  n_elements = nelements(elements)

  # Initialize interface container
  interfaces = init_interfaces(leaf_cell_ids, mesh, elements, Float64, NVARS, POLYDEG)
  n_interfaces = ninterfaces(interfaces)

  # Initialize boundaries
  boundaries, n_boundaries_per_direction = init_boundaries(leaf_cell_ids, mesh, elements, Float64, NVARS, POLYDEG)
  n_boundaries = nboundaries(boundaries)

  n_l2mortars = -1 # TODO: Taal. Only needed for simulation summary output -> fix me when Taal is alive

  # Sanity checks
  if isperiodic(mesh.tree)
    @assert n_interfaces == 1*n_elements ("For 1D and periodic domains, n_surf must be the same as 1*n_elem")
  end

  # Initialize boundary conditions
  boundary_conditions = init_boundary_conditions(n_boundaries_per_direction, mesh)

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
  dhat = calc_dhat(nodes, weights)
  dsplit = calc_dsplit(nodes, weights)
  dsplit_transposed = transpose(calc_dsplit(nodes, weights))

  # Initialize L2 mortar projection operators
  amr_refine_right  = calc_forward_upper(n_nodes)
  amr_refine_left   = calc_forward_lower(n_nodes)
  amr_coarsen_right = calc_reverse_upper(n_nodes, Val(:gauss))
  amr_coarsen_left  = calc_reverse_lower(n_nodes, Val(:gauss))

  # Initialize data structures for error analysis (by default, we use twice the
  # number of analysis nodes as the normal solution)
  analysis_polydeg = 2 * POLYDEG
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
                                   valid=["n/a", "gauss", "blast_wave"]))

  # Initialize storage for element variables
  element_variables = Dict{Symbol, Union{Vector{Float64}, Vector{Int}}}()

  # maximum and minimum alpha for shock capturing
  shock_alpha_max = parameter("shock_alpha_max", 0.5)
  shock_alpha_min = parameter("shock_alpha_min", 0.001)
  shock_alpha_smooth = parameter("shock_alpha_smooth", true)

  # variable used to compute the shock capturing indicator
  # "eval is evil"
  # This is a temporary hack until we have switched to a library based approach
  # with pure Julia code instead of parameter files.
  shock_indicator_variable = eval(Symbol(parameter("shock_indicator_variable", "density_pressure")))

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
  thread_cache = create_thread_cache_1d(NVARS, POLYDEG+1)

  # Store initial state integrals for conservation error calculation
  initial_state_integrals = Vector{Float64}()

  # Create actual DG solver instance
  dg = Dg1D(
      equation,
      surface_flux_function, volume_flux_function,
      initial_condition, source_terms,
      elements, n_elements,
      interfaces, n_interfaces,
      boundaries, n_boundaries, n_boundaries_per_direction,
      n_l2mortars,
      Tuple(boundary_conditions),
      SVector{POLYDEG+1}(nodes), SVector{POLYDEG+1}(weights), SVector{POLYDEG+1}(inverse_weights),
      inverse_vandermonde_legendre, SMatrix{POLYDEG+1,2}(lhat),
      volume_integral_type,
      SMatrix{POLYDEG+1,POLYDEG+1}(dhat), SMatrix{POLYDEG+1,POLYDEG+1}(dsplit), SMatrix{POLYDEG+1,POLYDEG+1}(dsplit_transposed),
      SMatrix{POLYDEG+1,POLYDEG+1}(amr_refine_right),   SMatrix{POLYDEG+1,POLYDEG+1}(amr_refine_left),
      SMatrix{POLYDEG+1,POLYDEG+1}(amr_coarsen_right), SMatrix{POLYDEG+1,POLYDEG+1}(amr_coarsen_left),
      SVector{analysis_polydeg+1}(analysis_nodes), SVector{analysis_polydeg+1}(analysis_weights), SVector{analysis_polydeg+1}(analysis_weights_volume),
      analysis_vandermonde, analysis_total_volume,
      analysis_quantities, save_analysis, analysis_filename,
      shock_indicator_variable, shock_alpha_max, shock_alpha_min, shock_alpha_smooth,
      amr_indicator, amr_alpha_max, amr_alpha_min, amr_alpha_smooth,
      element_variables, cache, thread_cache,
      initial_state_integrals)

  return dg
end


function create_thread_cache_1d(n_variables, n_nodes)
  # Type alias only for convenience
  A3d     = Array{Float64, 3}
  A2d     = Array{Float64, 2}
  A2dp1_x = Array{Float64, 2}

  MA1d    = MArray{Tuple{n_variables, n_nodes}, Float64}

  # Pre-allocate data structures to speed up computation (thread-safe)
  f1_threaded     = A3d[A3d(undef, n_variables, n_nodes, n_nodes) for _ in 1:Threads.nthreads()]
  fstar1_threaded = A2dp1_x[A2dp1_x(undef, n_variables, n_nodes+1) for _ in 1:Threads.nthreads()]

  indicator_threaded  = [A2d(undef, 1, n_nodes) for _ in 1:Threads.nthreads()]
  modal_threaded      = [A2d(undef, 1, n_nodes) for _ in 1:Threads.nthreads()]

  return (; f1_threaded,
            fstar1_threaded,
            indicator_threaded, modal_threaded)
end


# Count the number of interfaces that need to be created
function count_required_interfaces(mesh::TreeMesh{1}, cell_ids)
  count = 0

  # Iterate over all cells
  for cell_id in cell_ids
    for direction in 1:n_directions(mesh.tree)
      # Only count interfaces in positive direction to avoid double counting
      if direction == 1
        continue
      end

      # Skip if no neighbor exists
      if !has_any_neighbor(mesh.tree, cell_id, direction)
        continue
      end

      count += 1
    end
  end

  return count
end


# Count the number of boundaries that need to be created
function count_required_boundaries(mesh::TreeMesh{1}, cell_ids)
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


# Create element container, initialize element data, and return element container for further use
#
# nvars: number of variables
# polydeg: polynomial degree
function init_elements(cell_ids, mesh::TreeMesh{1}, RealT, nvars, polydeg)
  # Initialize container
  n_elements = length(cell_ids)
  elements = ElementContainer1D{RealT, nvars, polydeg}(n_elements)

  # Determine node locations
  n_nodes = polydeg + 1
  nodes, _ = gauss_lobatto_nodes_weights(n_nodes)

  init_elements!(elements, cell_ids, mesh, nodes)
  return elements
end

function init_elements!(elements, cell_ids, mesh::TreeMesh{1}, nodes)
  n_nodes = length(nodes)

  # Store cell ids
  elements.cell_ids .= cell_ids

  # Calculate inverse Jacobian and node coordinates
  for element_id in 1:nelements(elements)
    # Get cell id
    cell_id = cell_ids[element_id]

    # Get cell length
    dx = length_at_cell(mesh.tree, cell_id)

    # Calculate inverse Jacobian as 1/(h/2)
    elements.inverse_jacobian[element_id] = 2/dx

    # Calculate node coordinates
      for i in 1:n_nodes
        elements.node_coordinates[1, i, element_id] = (
            mesh.tree.coordinates[1, cell_id] + dx/2 * nodes[i])
      end
  end

  return elements
end


# Create interface container, initialize interface data, and return interface container for further use
#
# nvars: number of variables
# polydeg: polynomial degree
function init_interfaces(cell_ids, mesh::TreeMesh{1}, elements, RealT, nvars, polydeg)
  # Initialize container
  n_interfaces = count_required_interfaces(mesh, cell_ids)
  interfaces = InterfaceContainer1D{RealT, nvars, polydeg}(n_interfaces)

  # Connect elements with interfaces
  init_interfaces!(interfaces, elements, mesh)

  return interfaces
end


# Create boundaries container, initialize boundary data, and return boundaries container
#
# nvars: number of variables
# polydeg: polynomial degree
function init_boundaries(cell_ids, mesh::TreeMesh{1}, elements, RealT, nvars, polydeg)
  # Initialize container
  n_boundaries = count_required_boundaries(mesh, cell_ids)
  boundaries = BoundaryContainer1D{RealT, nvars, polydeg}(n_boundaries)

  # Connect elements with boundaries
  n_boundaries_per_direction = init_boundaries!(boundaries, elements, mesh)

  return boundaries, n_boundaries_per_direction
end


# Initialize connectivity between elements and interfaces
function init_interfaces!(interfaces, elements, mesh::TreeMesh{1})
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
      if direction == 1
        continue
      end

      # Skip if no neighbor exists and current cell is not small
      if !has_any_neighbor(mesh.tree, cell_id, direction)
        continue
      end

      count += 1

      if has_neighbor(mesh.tree, cell_id, direction)
        neighbor_cell_id = mesh.tree.neighbor_ids[direction, cell_id]
        if has_children(mesh.tree, neighbor_cell_id) # Cell has small neighbor
          interfaces.neighbor_ids[2, count] = c2e[mesh.tree.child_ids[1, neighbor_cell_id]]
        else # Cell has same refinement level neighbor
          interfaces.neighbor_ids[2, count] = c2e[neighbor_cell_id]
        end
      else # Cell is small and has large neighbor
        parent_id = mesh.tree.parent_ids[cell_id]
        neighbor_cell_id = mesh.tree.neighbor_ids[direction, parent_id]
        interfaces.neighbor_ids[2, count] = c2e[neighbor_cell_id]
      end

      interfaces.neighbor_ids[1, count] = element_id
      # Set orientation (x -> 1)
      interfaces.orientations[count] = 1
    end
  end

  @assert count == ninterfaces(interfaces) ("Actual interface count ($count) does not match " *
                                            "expectations $(ninterfaces(interfaces))")
end


# Initialize connectivity between elements and boundaries
function init_boundaries!(boundaries, elements, mesh::TreeMesh{1})
  # Reset boundaries count
  count = 0

  # Initialize boundary counts
  counts_per_direction = MVector(0, 0)

  # OBS! Iterate over directions first, then over elements, and count boundaries in each direction
  # Rationale: This way the boundaries are internally sorted by the directions -x, +x, -y etc.,
  #            obviating the need to store the boundary condition to be applied explicitly.
  # Loop over directions
  for direction in 1:n_directions(mesh.tree)
    # Iterate over all elements to find missing neighbors and to connect to boundaries
    for element_id in 1:nelements(elements)
      # Get cell id
      cell_id = elements.cell_ids[element_id]

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
      counts_per_direction[direction] += 1

      # Set neighbor element id
      boundaries.neighbor_ids[count] = element_id

      # Set neighbor side, which denotes the direction (1 -> negative, 2 -> positive) of the element
      if direction == 2
        boundaries.neighbor_sides[count] = 1
      else
        boundaries.neighbor_sides[count] = 2
      end

      # Set orientation (x -> 1)
      boundaries.orientations[count] = 1

      # Store node coordinates
      enc = elements.node_coordinates
      if direction == 1 # -x direction
        boundaries.node_coordinates[:, count] .= enc[:, 1,  element_id]
      elseif direction == 2 # +x direction
        boundaries.node_coordinates[:, count] .= enc[:, end, element_id]
      else
        error("should not happen")
      end
    end
  end

  @assert count == nboundaries(boundaries) ("Actual boundaries count ($count) does not match " *
                                            "expectations $(nboundaries(boundaries))")
  @assert sum(counts_per_direction) == count

  boundaries.n_boundaries_per_direction = SVector(counts_per_direction)

  return boundaries.n_boundaries_per_direction
end

function init_boundary_conditions(n_boundaries_per_direction, mesh::TreeMesh{1})
  # "eval is evil"
  # This is a temporary hack until we have switched to a library based approach
  # with pure Julia code instead of parameter files.
  bcs = parameter("boundary_conditions", ["nothing", "nothing"])
  if bcs isa AbstractArray
    boundary_conditions = eval_if_not_function.(bcs)
  else
    # This adds support for using a scalar boundary condition (like 'periodicity = "false"')
    boundary_conditions = eval_if_not_function.([bcs for _ in 1:n_directions(mesh.tree)])
  end

  # Sanity check about specifying boundary conditions
  for direction in 1:n_directions(mesh.tree)
    bc = boundary_conditions[direction]
    count = n_boundaries_per_direction[direction]
    if direction == 1
      dir = "-x"
    elseif direction == 2
      dir = "+x"
    end

    # All directions with boundaries should have a boundary condition
    if count > 0 && isnothing(bc)
      error("Found $(count) boundaries in the $(dir)-direction, but corresponding boundary " *
            "condition is '$(get_name(bc))'")
    end
  end

  return boundary_conditions
end


"""
    integrate(func, dg::Dg1D, args...; normalize=true)

Call function `func` for each DG node and integrate the result over the computational domain.

The function `func` is called as `func(i, j, element_id, dg, args...)` for each
volume node `(i, j)` and each `element_id`. Additional positional
arguments `args...` are passed along as well. If `normalize` is true, the result
is divided by the total volume of the computational domain.

# Examples
Calculate the integral of the time derivative of the entropy, i.e.,
∫(∂S/∂t)dΩ = ∫(∂S/∂u ⋅ ∂u/∂t)dΩ:
```julia
# Calculate integral of entropy time derivative
dsdu_ut = integrate(dg, dg.elements.u, dg.elements.u_t) do i, element_id, dg, u, u_t
  u_node   = get_node_vars(u,   dg, i, element_id)
  u_t_node = get_node_vars(u_t, dg, i, element_id)
  dot(cons2entropy(u_node, equations(dg)), u_t_node)
end
```
"""
function integrate(func, dg::Dg1D, args...; normalize=true)
  # Initialize integral with zeros of the right shape
  integral = zero(func(1, 1, dg, args...))

  # Use quadrature to numerically integrate over entire domain
  for element_id in 1:dg.n_elements
    jacobian_volume = inv(dg.elements.inverse_jacobian[element_id])^ndims(dg)
      for i in 1:nnodes(dg)
        integral += jacobian_volume * dg.weights[i] * func(i, element_id, dg, args...)
      end
  end

  # Normalize with total volume
  if normalize
    integral = integral/dg.analysis_total_volume
  end

  return integral
end


"""
    integrate(func, u, dg::Dg1D; normalize=true)
    integrate(u, dg::Dg1D; normalize=true)

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
function integrate(func, u, dg::Dg1D; normalize=true)
  func_wrapped = function(i, element_id, dg, u)
    u_local = get_node_vars(u, dg, i, element_id)
    return func(u_local)
  end
  return integrate(func_wrapped, dg, u; normalize=normalize)
end
integrate(u, dg::Dg1D; normalize=true) = integrate(identity, u, dg; normalize=normalize)


# Calculate L2/Linf error norms based on "exact solution"
function calc_error_norms(func, dg::Dg1D, t)
  # Gather necessary information
  equation = equations(dg)
  n_nodes_analysis = size(dg.analysis_vandermonde, 1)

  # pre-allocate buffers
  u = zeros(eltype(dg.elements.u),
            nvariables(dg), size(dg.analysis_vandermonde, 1))

  x = zeros(eltype(dg.elements.node_coordinates),
            1, size(dg.analysis_vandermonde, 1))

  # Set up data structures
  l2_error   = zero(func(get_node_vars(dg.elements.u, dg, 1, 1), equation))
  linf_error = zero(func(get_node_vars(dg.elements.u, dg, 1, 1), equation))

  # Iterate over all elements for error calculations
  for element_id in 1:dg.n_elements
    # Interpolate solution and node locations to analysis nodes
    multiply_dimensionwise!(u, dg.analysis_vandermonde, view(dg.elements.u,                :, :, element_id))
    multiply_dimensionwise!(x, dg.analysis_vandermonde, view(dg.elements.node_coordinates, :, :, element_id))

    # Calculate errors at each analysis node
    weights = dg.analysis_weights_volume
    jacobian_volume = inv(dg.elements.inverse_jacobian[element_id])^ndims(dg)
    for i in 1:n_nodes_analysis
      u_exact = dg.initial_condition(get_node_coords(x, dg, i), t, equation)
      diff = func(u_exact, equation) - func(get_node_vars(u, dg, i), equation)
      l2_error += diff.^2 * (weights[i] *  jacobian_volume)
      linf_error = @. max(linf_error, abs(diff))
    end
  end

  # For L2 error, divide by total volume
  l2_error = @. sqrt(l2_error / dg.analysis_total_volume)

  return l2_error, linf_error
end


# Integrate ∂S/∂u ⋅ ∂u/∂t over the entire domain
function calc_entropy_timederivative(dg::Dg1D, t)
  # Compute ut = rhs(u) with current solution u
  @notimeit timer() rhs!(dg, t)

  # Calculate ∫(∂S/∂u ⋅ ∂u/∂t)dΩ
  dsdu_ut = integrate(dg, dg.elements.u, dg.elements.u_t) do i, element_id, dg, u, u_t
    u_node   = get_node_vars(u,   dg, i, element_id)
    u_t_node = get_node_vars(u_t, dg, i, element_id)
    dot(cons2entropy(u_node, equations(dg)), u_t_node)
  end

  return dsdu_ut
end



"""
    analyze_solution(dg::Dg1D, mesh::TreeMesh, time, dt, step, runtime_absolute, runtime_relative)

Calculate error norms and other analysis quantities to analyze the solution
during a simulation, and return the L2 and Linf errors. `dg` and `mesh` are the
DG and the mesh instance, respectively. `time`, `dt`, and `step` refer to the
current simulation time, the last time step size, and the current time step
count. The run time (in seconds) is given in `runtime_absolute`, while the
performance index is specified in `runtime_relative`.

**Note:** Keep order of analysis quantities in sync with
          [`save_analysis_header`](@ref) when adding or changing quantities.
"""
function analyze_solution(dg::Dg1D, mesh::TreeMesh, time::Real, dt::Real, step::Integer,
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
  if parameter("amr_interval", 0)::Int > 0
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

  # Calculate L2/Linf errors, which are also returned by analyze_solution
  l2_error, linf_error = calc_error_norms(dg, time)

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
      @views res = maximum(abs, view(dg.elements.u_t, v, :, :))
      @printf("  % 10.8e", res)
      dg.save_analysis && @printf(f, "  % 10.8e", res)
    end
    println()
  end

  # L2/L∞ errors of the primitive variables
  if :l2_error_primitive in dg.analysis_quantities || :linf_error_primitive in dg.analysis_quantities
    l2_error_prim, linf_error_prim = calc_error_norms(cons2prim, dg, time)

    print(" Variable:    ")
    for v in 1:nvariables(equation)
      @printf("   %-14s", varnames_prim(equation)[v])
    end
    println()

    # L2 error
    if :l2_error_primitive in dg.analysis_quantities
      print(" L2 error prim.: ")
      for v in 1:nvariables(equation)
        @printf("%10.8e   ", l2_error_prim[v])
        dg.save_analysis && @printf(f, "  % 10.8e", l2_error_prim[v])
      end
      println()
    end

    # L∞ error
    if :linf_error_primitive in dg.analysis_quantities
      print(" Linf error pri.:")
      for v in 1:nvariables(equation)
        @printf("%10.8e   ", linf_error_prim[v])
        dg.save_analysis && @printf(f, "  % 10.8e", linf_error_prim[v])
      end
      println()
    end
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
    s = integrate(dg, dg.elements.u) do i, element_id, dg, u
      cons = get_node_vars(u, dg, i, element_id)
      return entropy(cons, equations(dg))
    end
    print(" ∑S:          ")
    @printf("  % 10.8e", s)
    dg.save_analysis && @printf(f, "  % 10.8e", s)
    println()
  end

  # Total energy
  if :energy_total in dg.analysis_quantities
    e_total = integrate(dg, dg.elements.u) do i, element_id, dg, u
      cons = get_node_vars(u, dg, i, element_id)
      return energy_total(cons, equations(dg))
    end
    print(" ∑e_total:    ")
    @printf("  % 10.8e", e_total)
    dg.save_analysis && @printf(f, "  % 10.8e", e_total)
    println()
  end

  # Kinetic energy
  if :energy_kinetic in dg.analysis_quantities
    e_kinetic = integrate(dg, dg.elements.u) do i, element_id, dg, u
      cons = get_node_vars(u, dg, i, element_id)
      return energy_kinetic(cons, equations(dg))
    end
    print(" ∑e_kinetic:  ")
    @printf("  % 10.8e", e_kinetic)
    dg.save_analysis && @printf(f, "  % 10.8e", e_kinetic)
    println()
  end

  # Internal energy
  if :energy_internal in dg.analysis_quantities
    e_internal = integrate(dg, dg.elements.u) do i, element_id, dg, u
      cons = get_node_vars(u, dg, i, element_id)
      return energy_internal(cons, equations(dg))
    end
    print(" ∑e_internal: ")
    @printf("  % 10.8e", e_internal)
    dg.save_analysis && @printf(f, "  % 10.8e", e_internal)
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
function save_analysis_header(filename, quantities, equation::AbstractEquations{1})
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
    if :l2_error_primitive in quantities
      for v in varnames_prim(equation)
        @printf(f, "   %-14s", "l2_" * v)
      end
    end
    if :linf_error_primitive in quantities
      for v in varnames_prim(equation)
        @printf(f, "   %-14s", "linf_" * v)
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
    println(f)
  end
end


# Call equation-specific initial conditions functions and apply to all elements
function set_initial_condition!(dg::Dg1D, time)
  equation = equations(dg)
  # make sure that the random number generator is reseted and the ICs are reproducible in the julia REPL/interactive mode
  seed!(0)
  for element_id in 1:dg.n_elements
      for i in 1:nnodes(dg)
        dg.elements.u[:, i, element_id] .= dg.initial_condition(
            dg.elements.node_coordinates[:, i, element_id], time, equation)
      end
  end
end


# Calculate time derivative
function rhs!(dg::Dg1D, t_stage)
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

  # Calculate surface integrals
  @timeit timer() "surface integral" calc_surface_integral!(dg)

  # Apply Jacobian from mapping to reference element
  @timeit timer() "Jacobian" apply_jacobian!(dg)

  # Calculate source terms
  @timeit timer() "source terms" calc_sources!(dg, dg.source_terms, t_stage)
end

# TODO: implement 1D!!!
# Apply positivity limiter of Zhang and Shu to nodal values elements.u
function apply_positivity_preserving_limiter!(dg::Dg1D)
end

# Calculate volume integral and update u_t
calc_volume_integral!(dg::Dg1D) = calc_volume_integral!(dg.elements.u_t, dg.volume_integral_type, dg)


# Calculate volume integral (DGSEM in weak form)
function calc_volume_integral!(u_t, ::Val{:weak_form}, dg::Dg1D)
  @unpack dhat = dg

  Threads.@threads for element_id in 1:dg.n_elements
    # Calculate volume integral
      for i in 1:nnodes(dg)
        u_node = get_node_vars(dg.elements.u, dg, i, element_id)

        flux1 = calcflux(u_node, 1, equations(dg))
        for l in 1:nnodes(dg)
          integral_contribution = dhat[l, i] * flux1
          add_to_node_vars!(u_t, integral_contribution, dg, l, element_id)
        end
      end
  end
end


# Calculate volume integral (DGSEM in split form)
@inline function calc_volume_integral!(u_t, volume_integral_type::Val{:split_form}, dg::Dg1D)
  calc_volume_integral!(u_t, volume_integral_type, dg.thread_cache, dg)
end


function calc_volume_integral!(u_t, ::Val{:split_form}, cache, dg::Dg1D)
  Threads.@threads for element_id in 1:dg.n_elements
    split_form_kernel!(u_t, element_id, cache, dg)
  end
end

@inline function split_form_kernel!(u_t, element_id, cache, dg::Dg1D, alpha=true)
  # true * [some floating point value] == [exactly the same floating point value]
  # This can get optimized away due to constant propagation.
  @unpack volume_flux_function, dsplit = dg

  # Calculate volume integral in one element
  for i in 1:nnodes(dg)
    u_node = get_node_vars(dg.elements.u, dg, i, element_id)

    # x direction
    # use consistency of the volume flux to make this evaluation cheaper
    flux = calcflux(u_node, 1, equations(dg))
    integral_contribution = alpha * dsplit[i, i] * flux
    add_to_node_vars!(u_t, integral_contribution, dg, i, element_id)
    # use symmetry of the volume flux for the remaining terms
    for ii in (i+1):nnodes(dg)
      u_node_ii = get_node_vars(dg.elements.u, dg, ii, element_id)
      flux = volume_flux_function(u_node, u_node_ii, 1, equations(dg))
      integral_contribution = alpha * dsplit[i, ii] * flux
      add_to_node_vars!(u_t, integral_contribution, dg, i, element_id)
      integral_contribution = alpha * dsplit[ii, i] * flux
      add_to_node_vars!(u_t, integral_contribution, dg, ii, element_id)
    end

  end
end


# Calculate volume integral (DGSEM in split form with shock capturing)
function calc_volume_integral!(u_t, ::Val{:shock_capturing}, dg::Dg1D)
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
                               element_ids_dg, element_ids_dgfv, thread_cache, dg::Dg1D)
  @unpack dsplit_transposed, inverse_weights = dg
  @unpack fstar1_threaded = thread_cache

  # Calculate blending factors α: u = u_DG * (1 - α) + u_FV * α
  @timeit timer() "blending factors" calc_blending_factors!(alpha, alpha_tmp, dg.elements.u,
    dg.shock_alpha_max,
    dg.shock_alpha_min,
    dg.shock_alpha_smooth,
    dg.shock_indicator_variable, thread_cache, dg)

  # Determine element ids for DG-only and blended DG-FV volume integral
  pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha, dg)

  # Loop over pure DG elements
  @timeit timer() "pure DG" Threads.@threads for element_id in element_ids_dg
    split_form_kernel!(u_t, element_id, thread_cache, dg)
  end

  # Loop over blended DG-FV elements
  @timeit timer() "blended DG-FV" Threads.@threads for element_id in element_ids_dgfv
    # Calculate DG volume integral contribution
    split_form_kernel!(u_t, element_id, thread_cache, dg, 1 - alpha[element_id])

    # Calculate FV two-point fluxes
    fstar1 = fstar1_threaded[Threads.threadid()]
    calcflux_fv!(fstar1, dg.elements.u, element_id, dg)

    # Calculate FV volume integral contribution
    for i in 1:nnodes(dg)
      for v in 1:nvariables(dg)
        u_t[v, i, element_id] += ( alpha[element_id] * (inverse_weights[i] * (fstar1[v, i+1] - fstar1[v, i]) ) )

      end
    end
  end
end


"""
    calcflux_fv!(fstar1, u_leftright, u, element_id, dg::Dg1D)

Calculate the finite volume fluxes inside the elements.

# Arguments
- `fstar1::AbstractArray{T} where T<:Real`:
- `dg::Dg1D`
- `u::AbstractArray{T} where T<:Real`
- `element_id::Integer`
"""
@inline function calcflux_fv!(fstar1, u, element_id, dg::Dg1D)
  @unpack surface_flux_function = dg

  fstar1[:, 1           ] .= zero(eltype(fstar1))
  fstar1[:, nnodes(dg)+1] .= zero(eltype(fstar1))

    for i in 2:nnodes(dg)
      u_ll = get_node_vars(u, dg, i-1, element_id)
      u_rr = get_node_vars(u, dg, i,   element_id)
      flux = surface_flux_function(u_ll, u_rr, 1, equations(dg)) # orientation 1: x direction
      set_node_vars!(fstar1, flux, dg, i)
    end
end


# Prolong solution to interfaces (for GL nodes: just a copy)
function prolong2interfaces!(dg::Dg1D)
  equation = equations(dg)

  Threads.@threads for s in 1:dg.n_interfaces
    left_element_id = dg.interfaces.neighbor_ids[1, s]
    right_element_id = dg.interfaces.neighbor_ids[2, s]
    # interface in x-direction
    for v in 1:nvariables(dg)
      dg.interfaces.u[1, v, s] = dg.elements.u[v, nnodes(dg), left_element_id]
      dg.interfaces.u[2, v, s] = dg.elements.u[v,          1, right_element_id]
    end
  end
end


# Prolong solution to boundaries (for GL nodes: just a copy)
function prolong2boundaries!(dg::Dg1D)
  equation = equations(dg)

  for b in 1:dg.n_boundaries
    element_id = dg.boundaries.neighbor_ids[b]
    if dg.boundaries.neighbor_sides[b] == 1 # Element in -x direction of boundary
      for v in 1:nvariables(dg)
        dg.boundaries.u[1, v, b] = dg.elements.u[v, nnodes(dg), element_id]
      end
    else # Element in +x direction of boundary
      for v in 1:nvariables(dg)
        dg.boundaries.u[2, v, b] = dg.elements.u[v, 1,          element_id]
      end
    end
  end
end


# Calculate and store the surface fluxes (standard Riemann and nonconservative parts) at an interface
# OBS! Regarding the nonconservative terms: 1) currently only needed for the MHD equations
#                                           2) not implemented for boundaries
calc_interface_flux!(dg::Dg1D) = calc_interface_flux!(dg.elements.surface_flux_values, dg)

function calc_interface_flux!(surface_flux_values,  dg::Dg1D)
  @unpack surface_flux_function = dg
  @unpack u, neighbor_ids, orientations = dg.interfaces

  Threads.@threads for s in 1:dg.n_interfaces
    # Get neighboring elements
    left_id  = neighbor_ids[1, s]
    right_id = neighbor_ids[2, s]

    # Determine interface direction with respect to elements:
    # orientation = 1: left -> 2, right -> 1
    left_direction  = 2 * orientations[s]
    right_direction = 2 * orientations[s] - 1

    # Call pointwise Riemann solver
    u_ll, u_rr = get_surface_node_vars(u, dg, s)
    flux = surface_flux_function(u_ll, u_rr, orientations[s], equations(dg))

    # Copy flux to left and right element storage
    for v in 1:nvariables(dg)
      surface_flux_values[v, left_direction, left_id]  = surface_flux_values[v, right_direction, right_id] = flux[v]
    end
  end
end


# Calculate and store boundary flux across domain boundaries
#NOTE: Do we need to dispatch on have_nonconservative_terms(dg.equations)?
calc_boundary_flux!(dg::Dg1D, time) = calc_boundary_flux!(dg.elements.surface_flux_values, dg, time)


function calc_boundary_flux!(surface_flux_values, dg::Dg1D, time)
  @unpack n_boundaries_per_direction, boundary_conditions = dg

  # Calculate indices
  lasts = accumulate(+, n_boundaries_per_direction)
  firsts = lasts - n_boundaries_per_direction .+ 1

  # Calc boundary fluxes in each direction
  calc_boundary_flux_by_direction!(surface_flux_values, dg, time,
                                   boundary_conditions[1], 1, firsts[1], lasts[1])
  calc_boundary_flux_by_direction!(surface_flux_values, dg, time,
                                   boundary_conditions[2], 2, firsts[2], lasts[2])
end

function calc_boundary_flux_by_direction!(surface_flux_values, dg::Dg1D, time, boundary_condition,
                                          direction, first_boundary_id, last_boundary_id)
  @unpack surface_flux_function = dg
  @unpack u, neighbor_ids, neighbor_sides, node_coordinates, orientations = dg.boundaries

  Threads.@threads for b in first_boundary_id:last_boundary_id
    # Get neighboring element
    neighbor_id = neighbor_ids[b]

    # Get boundary flux
    u_ll, u_rr = get_surface_node_vars(u, dg, b)
    if neighbor_sides[b] == 1 # Element is on the left, boundary on the right
      u_inner = u_ll
    else # Element is on the right, boundary on the left
      u_inner = u_rr
    end
    x = get_node_coords(node_coordinates, dg, b)
    flux = boundary_condition(u_inner, orientations[b], direction, x, time, surface_flux_function,
                                equations(dg))

    # Copy flux to left and right element storage
    for v in 1:nvariables(dg)
      surface_flux_values[v, direction, neighbor_id] = flux[v]
    end
  end
end



# Calculate surface integrals and update u_t
calc_surface_integral!(dg::Dg1D) = calc_surface_integral!(dg.elements.u_t, dg.elements.surface_flux_values, dg)
function calc_surface_integral!(u_t, surface_flux_values, dg::Dg1D)
  @unpack lhat = dg

  Threads.@threads for element_id in 1:dg.n_elements
      for v in 1:nvariables(dg)
        # surface at -x
        u_t[v, 1,          element_id] -= surface_flux_values[v, 1, element_id] * lhat[1,          1]
        # surface at +x
        u_t[v, nnodes(dg), element_id] += surface_flux_values[v, 2, element_id] * lhat[nnodes(dg), 2]
      end
  end
end


# Apply Jacobian from mapping to reference element
function apply_jacobian!(dg::Dg1D)
  Threads.@threads for element_id in 1:dg.n_elements
    factor = -dg.elements.inverse_jacobian[element_id]
      for i in 1:nnodes(dg)
        for v in 1:nvariables(dg)
          dg.elements.u_t[v, i, element_id] *= factor
        end
      end
  end
end


# Calculate source terms and apply them to u_t
function calc_sources!(dg::Dg1D, source_terms::Nothing, t)
  return nothing
end

function calc_sources!(dg::Dg1D, source_terms, t)
  Threads.@threads for element_id in 1:dg.n_elements
    source_terms(dg.elements.u_t, dg.elements.u,
                 dg.elements.node_coordinates, element_id, t, nnodes(dg), equations(dg))
  end
end


# Calculate stable time step size
function calc_dt(dg::Dg1D, cfl)
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
                                indicator_variable, thread_cache, dg::Dg1D)
  # temporary buffers
  @unpack indicator_threaded, modal_threaded = thread_cache
  # magic parameters
  threshold = 0.5 * 10^(-1.8 * (nnodes(dg))^0.25)
  parameter_s = log((1 - 0.0001)/0.0001)

  Threads.@threads for element_id in 1:dg.n_elements
    indicator  = indicator_threaded[Threads.threadid()]
    modal      = modal_threaded[Threads.threadid()]

    # Calculate indicator variables at Gauss-Lobatto nodes
    cons2indicator!(indicator, u, element_id, indicator_variable, dg)

    # Convert to modal representation
    multiply_dimensionwise!(modal, dg.inverse_vandermonde_legendre, indicator)

    # Calculate total energies for all modes, without highest, without two highest
    total_energy = 0.0
    for i in 1:nnodes(dg)
      total_energy += modal[1, i]^2
    end
    total_energy_clip1 = 0.0
    for i in 1:(nnodes(dg)-1)
      total_energy_clip1 += modal[1, i]^2
    end
    total_energy_clip2 = 0.0
    for  i in 1:(nnodes(dg)-2)
      total_energy_clip2 += modal[1, i]^2
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
  end
end

# Convert conservative variables to indicator variable for discontinuities (elementwise version)
@inline function cons2indicator!(indicator, u, element_id, indicator_variable, dg::Dg1D)
  eqs = equations(dg)

  for i in 1:nnodes(dg)
    u_node = get_node_vars(u, dg, i, element_id)
    indicator[1, i] = indicator_variable(u_node, eqs)
  end
end

"""
    pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha, dg)

Given blending factors `alpha` and the solver `dg`, fill
`element_ids_dg` with the IDs of elements using a pure DG scheme and
`element_ids_dgfv` with the IDs of elements using a blended DG-FV scheme.
"""
function pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha, dg::Dg1D)
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
