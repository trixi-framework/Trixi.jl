# Calculate time derivative
function rhs!(dg::Dg2D, t_stage, uses_mpi::Val{true})
  # Start to receive MPI data
  @timeit timer() "start MPI receive" start_mpi_receive!(dg)

  # Reset u_t
  @timeit timer() "reset ∂u/∂t" dg.elements.u_t .= 0

  # Prolong solution to MPI interfaces
  @timeit timer() "prolong2mpiinterfaces" prolong2mpiinterfaces!(dg)

  # Start to send MPI data
  @timeit timer() "start MPI send" start_mpi_send!(dg)

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

  # Finish to receive MPI data
  @timeit timer() "finish MPI receive" finish_mpi_receive!(dg)

  # Calculate MPI interface fluxes
  @timeit timer() "MPI interface flux" calc_mpi_interface_flux!(dg)

  # Calculate surface integrals
  @timeit timer() "surface integral" calc_surface_integral!(dg)

  # Apply Jacobian from mapping to reference element
  @timeit timer() "Jacobian" apply_jacobian!(dg)

  # Calculate source terms
  @timeit timer() "source terms" calc_sources!(dg, dg.source_terms, t_stage)

  # Finish to send MPI data
  @timeit timer() "finish MPI send" finish_mpi_send!(dg)
end


# Count the number of MPI interfaces that need to be created
function count_required_mpi_interfaces(mesh::TreeMesh2D, cell_ids)
  count = 0

  # Iterate over all cells
  for cell_id in cell_ids
    for direction in 1:n_directions(mesh.tree)
      # If no neighbor exists, current cell is small or at boundary and thus we need a mortar
      if !has_neighbor(mesh.tree, cell_id, direction)
        continue
      end

      # Skip if neighbor has children
      neighbor_cell_id = mesh.tree.neighbor_ids[direction, cell_id]
      if has_children(mesh.tree, neighbor_cell_id)
        continue
      end

      # Skip if neighbor is on this rank -> create regular interface instead
      if is_parallel() && is_own_cell(mesh.tree, neighbor_cell_id)
        continue
      end

      count += 1
    end
  end

  return count
end


# Create MPI interface container, initialize interface data, and return interface container for further use
function init_mpi_interfaces(cell_ids, mesh::TreeMesh2D, ::Val{NVARS}, ::Val{POLYDEG}, elements) where {NVARS, POLYDEG}
  # Initialize container
  n_mpi_interfaces = count_required_mpi_interfaces(mesh, cell_ids)
  mpi_interfaces = MpiInterfaceContainer2D{NVARS, POLYDEG}(n_mpi_interfaces)

  # Connect elements with interfaces
  init_mpi_interface_connectivity!(elements, mpi_interfaces, mesh)

  return mpi_interfaces
end


function start_mpi_receive!(dg::Dg2D)
  for (index, d) in enumerate(dg.mpi_neighbor_ranks)
    dg.mpi_recv_requests[index] = MPI.Irecv!(dg.mpi_recv_buffers[index], d, d, mpi_comm())
  end
end


# Initialize connectivity between elements and interfaces
function init_mpi_interface_connectivity!(elements, mpi_interfaces, mesh::TreeMesh2D)
  # Reset interface count
  count = 0

  # Iterate over all elements to find neighbors and to connect via mpi_interfaces
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

      # Skip if neighbor is on this MPI rank -> create regular interface instead
      if is_parallel() && is_own_cell(mesh.tree, neighbor_cell_id)
        continue
      end

      # Create interface between elements
      count += 1
      mpi_interfaces.local_element_ids[count] = element_id

      if direction in (2, 4) # element is "left" of interface, remote cell is "right" of interface
        mpi_interfaces.remote_sides[count] = 2
      else
        mpi_interfaces.remote_sides[count] = 1
      end

      # Set orientation (x -> 1, y -> 2)
      if direction in (1, 2) # x-direction
        mpi_interfaces.orientations[count] = 1
      else # y-direction
        mpi_interfaces.orientations[count] = 2
      end
    end
  end

  @assert count == nmpiinterfaces(mpi_interfaces) ("Actual interface count ($count) does not match "
                                                   * "expectations $(nmpiinterfaces(mpi_interfaces))")
end


# Initialize connectivity between MPI neighbor ranks
function init_mpi_neighbor_connectivity(elements, mpi_interfaces, mesh::TreeMesh2D)
  tree = mesh.tree

  # Determine neighbor ranks and sides for MPI interfaces
  neighbor_ranks = fill(-1, nmpiinterfaces(mpi_interfaces))
  # The global interface id is the smaller of the (globally unique) neighbor cell ids, multiplied by
  # number of directions (2 * ndims) plus direction minus one
  global_interface_ids = fill(-1, nmpiinterfaces(mpi_interfaces))
  for interface_id in 1:nmpiinterfaces(mpi_interfaces)
    orientation = mpi_interfaces.orientations[interface_id]
    remote_side = mpi_interfaces.remote_sides[interface_id]
    # Direction is from local cell to remote cell
    if orientation == 1 # MPI interface in x-direction
      if remote_side == 1 # remote cell on the "left" of MPI interface
        direction = 1
      else # remote cell on the "right" of MPI interface
        direction = 2
      end
    else # MPI interface in y-direction
      if remote_side == 1 # remote cell on the "left" of MPI interface
        direction = 3
      else # remote cell on the "right" of MPI interface
        direction = 4
      end
    end
    local_element_id = mpi_interfaces.local_element_ids[interface_id]
    local_cell_id = elements.cell_ids[local_element_id]
    remote_cell_id = tree.neighbor_ids[direction, local_cell_id]
    neighbor_ranks[interface_id] = tree.mpi_ranks[remote_cell_id]
    if local_cell_id < remote_cell_id
      global_interface_ids[interface_id] = 2 * ndims(tree) * local_cell_id + direction - 1
    else
      global_interface_ids[interface_id] = (2 * ndims(tree) * remote_cell_id +
                                            opposite_direction(direction) - 1)
    end
  end

  # Get sorted, unique neighbor ranks
  mpi_neighbor_ranks = unique(sort(neighbor_ranks))

  # Sort interfaces by global interface id
  p = sortperm(global_interface_ids)
  neighbor_ranks .= neighbor_ranks[p]
  interface_ids = collect(1:nmpiinterfaces(mpi_interfaces))[p]

  # For each neighbor rank, init connectivity data structures
  mpi_neighbor_interfaces = Vector{Vector{Int}}(undef, length(mpi_neighbor_ranks))
  for (index, d) in enumerate(mpi_neighbor_ranks)
    mpi_neighbor_interfaces[index] = interface_ids[findall(x->(x == d), neighbor_ranks)]
  end

  # Sanity check that we counted all interfaces exactly once
  @assert sum(length(v) for v in mpi_neighbor_interfaces) == nmpiinterfaces(mpi_interfaces)

  return mpi_neighbor_ranks, mpi_neighbor_interfaces
end


# Initialize MPI data structures
function init_mpi_data_structures(mpi_neighbor_interfaces, ::Val{NDIMS}, ::Val{NVARS},
                                  ::Val{POLYDEG}) where {NDIMS, NVARS, POLYDEG}
  data_size = NVARS * (POLYDEG + 1)^(NDIMS - 1)
  mpi_send_buffers = Vector{Vector{Float64}}(undef, length(mpi_neighbor_interfaces))
  mpi_recv_buffers = Vector{Vector{Float64}}(undef, length(mpi_neighbor_interfaces))
  for index in 1:length(mpi_neighbor_interfaces)
    mpi_send_buffers[index] = Vector{Float64}(undef, length(mpi_neighbor_interfaces[index]) * data_size)
    mpi_recv_buffers[index] = Vector{Float64}(undef, length(mpi_neighbor_interfaces[index]) * data_size)
  end

  mpi_send_requests = Vector{MPI.Request}(undef, length(mpi_neighbor_interfaces))
  mpi_recv_requests = Vector{MPI.Request}(undef, length(mpi_neighbor_interfaces))

  return mpi_send_buffers, mpi_recv_buffers, mpi_send_requests, mpi_recv_requests
end


function prolong2mpiinterfaces!(dg::Dg2D)
  equation = equations(dg)

  Threads.@threads for s in 1:dg.n_mpi_interfaces
    local_element_id = dg.mpi_interfaces.local_element_ids[s]
    if dg.mpi_interfaces.orientations[s] == 1 # interface in x-direction
      if dg.mpi_interfaces.remote_sides[s] == 1 # local element in positive direction
        for j in 1:nnodes(dg), v in 1:nvariables(dg)
          dg.mpi_interfaces.u[2, v, j, s] = dg.elements.u[v,          1, j, local_element_id]
        end
      else # local element in negative direction
        for j in 1:nnodes(dg), v in 1:nvariables(dg)
          dg.mpi_interfaces.u[1, v, j, s] = dg.elements.u[v, nnodes(dg), j, local_element_id]
        end
      end
    else # interface in y-direction
      if dg.mpi_interfaces.remote_sides[s] == 1 # local element in positive direction
        for i in 1:nnodes(dg), v in 1:nvariables(dg)
          dg.mpi_interfaces.u[2, v, i, s] = dg.elements.u[v, i,          1, local_element_id]
        end
      else # local element in negative direction
        for i in 1:nnodes(dg), v in 1:nvariables(dg)
          dg.mpi_interfaces.u[1, v, i, s] = dg.elements.u[v, i, nnodes(dg), local_element_id]
        end
      end
    end
  end
end


function start_mpi_send!(dg::Dg2D)
  data_size = nvariables(dg) * nnodes(dg)^(ndims(dg) - 1)

  for d in 1:length(dg.mpi_neighbor_ranks)
    send_buffer = dg.mpi_send_buffers[d]

    for (index, s) in enumerate(dg.mpi_neighbor_interfaces[d])
      first = (index - 1) * data_size + 1
      last =  (index - 1) * data_size + data_size

      if dg.mpi_interfaces.remote_sides[s] == 1 # local element in positive direction
        @views send_buffer[first:last] .= vec(dg.mpi_interfaces.u[2, :, :, s])
      else # local element in negative direction
        @views send_buffer[first:last] .= vec(dg.mpi_interfaces.u[1, :, :, s])
      end
    end
  end

  # Start sending
  for (index, d) in enumerate(dg.mpi_neighbor_ranks)
    dg.mpi_send_requests[index] = MPI.Isend(dg.mpi_send_buffers[index], d, mpi_rank(), mpi_comm())
  end
end


function finish_mpi_receive!(dg::Dg2D)
  data_size = nvariables(dg) * nnodes(dg)^(ndims(dg) - 1)

  # Start receiving and unpack received data until all communication is finished
  d, _ = MPI.Waitany!(dg.mpi_recv_requests)
  while d != 0
    recv_buffer = dg.mpi_recv_buffers[d]

    for (index, s) in enumerate(dg.mpi_neighbor_interfaces[d])
      first = (index - 1) * data_size + 1
      last =  (index - 1) * data_size + data_size

      if dg.mpi_interfaces.remote_sides[s] == 1 # local element in positive direction
        @views vec(dg.mpi_interfaces.u[1, :, :, s]) .= recv_buffer[first:last]
      else # local element in negative direction
        @views vec(dg.mpi_interfaces.u[2, :, :, s]) .= recv_buffer[first:last]
      end
    end

    d, _ = MPI.Waitany!(dg.mpi_recv_requests)
  end
end


# Calculate and store the surface fluxes (standard Riemann and nonconservative parts) at an MPI interface
# OBS! Regarding the nonconservative terms: 1) currently only needed for the MHD equations
#                                           2) not implemented for MPI
calc_mpi_interface_flux!(dg::Dg2D) = calc_mpi_interface_flux!(dg.elements.surface_flux_values,
                                                              have_nonconservative_terms(dg.equations),
                                                              dg)

function calc_mpi_interface_flux!(surface_flux_values, nonconservative_terms::Val{false}, dg::Dg2D)
  @unpack surface_flux_function = dg
  @unpack u, local_element_ids, orientations, remote_sides = dg.mpi_interfaces

  Threads.@threads for s in 1:dg.n_mpi_interfaces
    # Get local neighboring element
    element_id = local_element_ids[s]

    # Determine interface direction with respect to element:
    if orientations[s] == 1 # interface in x-direction
      if remote_sides[s] == 1 # local element in positive direction
        direction = 1
      else # local element in negative direction
        direction = 2
      end
    else # interface in y-direction
      if remote_sides[s] == 1 # local element in positive direction
        direction = 3
      else # local element in negative direction
        direction = 4
      end
    end

    for i in 1:nnodes(dg)
      # Call pointwise Riemann solver
      u_ll, u_rr = get_surface_node_vars(u, dg, i, s)
      flux = surface_flux_function(u_ll, u_rr, orientations[s], equations(dg))

      # Copy flux to local element storage
      for v in 1:nvariables(dg)
        surface_flux_values[v, i, direction, element_id] = flux[v]
      end
    end
  end
end


function finish_mpi_send!(dg::Dg2D)
  MPI.Waitall!(dg.mpi_send_requests)
end


function analyze_solution(dg::Dg2D, mesh::TreeMesh, time, dt, step, runtime_absolute,
                          runtime_relative, uses_mpi::Val{true}; solver_gravity=nothing)
  equation = equations(dg)

  # General information
  mpi_println()
  mpi_println("-"^80)
  mpi_println(" Simulation running '", get_name(equation), "' with POLYDEG = ", polydeg(dg))
  mpi_println("-"^80)
  mpi_println(" #timesteps:     " * @sprintf("% 14d", step) *
              "               " *
              " run time:       " * @sprintf("%10.8e s", runtime_absolute))
  mpi_println(" dt:             " * @sprintf("%10.8e", dt) *
              "               " *
              " PID:            " * @sprintf("%10.8e s", runtime_relative))
  mpi_println(" sim. time:      " * @sprintf("%10.8e", time) *
              "               " *
              " PID × #ranks:   " * @sprintf("%10.8e s", runtime_relative * n_mpi_ranks()))

  # Level information (only show for AMR)
  if parameter("amr_interval", 0)::Int > 0 && is_mpi_root()
    levels = Vector{Int}(undef, dg.n_elements)
    for element_id in 1:dg.n_elements
      levels[element_id] = mesh.tree.levels[dg.elements.cell_ids[element_id]]
    end
    min_level = minimum(levels)
    max_level = maximum(levels)

    mpi_println(" #elements:      " * @sprintf("% 14d", dg.n_elements))
    for level = max_level:-1:min_level+1
      mpi_println(" ├── level $level:    " * @sprintf("% 14d", count(x->x==level, levels)))
    end
    mpi_println(" └── level $min_level:    " * @sprintf("% 14d", count(x->x==min_level, levels)))
  end
  mpi_println()

  # Open file for appending and store time step and time information
  if dg.save_analysis && is_mpi_root()
    f = open(dg.analysis_filename, "a")
    @printf(f, "% 9d", step)
    @printf(f, "  %10.8e", time)
    @printf(f, "  %10.8e", dt)
  end

  # Calculate and print derived quantities (error norms, entropy etc.)
  # Variable names required for L2 error, Linf error, and conservation error
  if is_mpi_root()
    if any(q in dg.analysis_quantities for q in
          (:l2_error, :linf_error, :conservation_error, :residual))
      print(" Variable:    ")
      for v in 1:nvariables(equation)
        @printf("   %-14s", varnames_cons(equation)[v])
      end
      println()
    end
  end

  # Calculate L2/Linf errors, which are also returned by analyze_solution
  l2_error, linf_error = calc_error_norms(dg, time)

  if is_mpi_root()
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

    if is_mpi_root()
      print(" |∑U - ∑U₀|:  ")
      for v in 1:nvariables(equation)
        err = abs(state_integrals[v] - dg.initial_state_integrals[v])
        @printf("  % 10.8e", err)
        dg.save_analysis && @printf(f, "  % 10.8e", err)
      end
      println()
    end
  end

  # Residual (defined here as the vector maximum of the absolute values of the time derivatives)
  if :residual in dg.analysis_quantities
    mpi_print(" max(|Uₜ|):   ")
    for v in 1:nvariables(equation)
      # Calculate maximum absolute value of Uₜ
      res = maximum(abs, view(dg.elements.u_t, v, :, :, :))
      res = MPI.Reduce!(Ref(res), max, mpi_root(), mpi_comm())
      is_mpi_root() && @printf("  % 10.8e", res[])
      is_mpi_root() && dg.save_analysis && @printf(f, "  % 10.8e", res[])
    end
    mpi_println()
  end

  # L2/L∞ errors of the primitive variables
  if :l2_error_primitive in dg.analysis_quantities || :linf_error_primitive in dg.analysis_quantities
    l2_error_prim, linf_error_prim = calc_error_norms(cons2prim, dg, time)

    if is_mpi_root()
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
  end

  # Entropy time derivative
  if :dsdu_ut in dg.analysis_quantities
    dsdu_ut = calc_entropy_timederivative(dg, time)
    if is_mpi_root()
      print(" ∑∂S/∂U ⋅ Uₜ: ")
      @printf("  % 10.8e", dsdu_ut)
      dg.save_analysis && @printf(f, "  % 10.8e", dsdu_ut)
      println()
    end
  end

  # Entropy
  if :entropy in dg.analysis_quantities
    s = integrate(dg, dg.elements.u) do i, j, element_id, dg, u
      cons = get_node_vars(u, dg, i, j, element_id)
      return entropy(cons, equations(dg))
    end
    if is_mpi_root()
      print(" ∑S:          ")
      @printf("  % 10.8e", s)
      dg.save_analysis && @printf(f, "  % 10.8e", s)
      println()
    end
  end

  # Total energy
  if :energy_total in dg.analysis_quantities
    e_total = integrate(dg, dg.elements.u) do i, j, element_id, dg, u
      cons = get_node_vars(u, dg, i, j, element_id)
      return energy_total(cons, equations(dg))
    end
    if is_mpi_root()
      print(" ∑e_total:    ")
      @printf("  % 10.8e", e_total)
      dg.save_analysis && @printf(f, "  % 10.8e", e_total)
      println()
    end
  end

  # Kinetic energy
  if :energy_kinetic in dg.analysis_quantities
    e_kinetic = integrate(dg, dg.elements.u) do i, j, element_id, dg, u
      cons = get_node_vars(u, dg, i, j, element_id)
      return energy_kinetic(cons, equations(dg))
    end
    if is_mpi_root()
      print(" ∑e_kinetic:  ")
      @printf("  % 10.8e", e_kinetic)
      dg.save_analysis && @printf(f, "  % 10.8e", e_kinetic)
      println()
    end
  end

  # Internal energy
  if :energy_internal in dg.analysis_quantities
    e_internal = integrate(dg, dg.elements.u) do i, j, element_id, dg, u
      cons = get_node_vars(u, dg, i, j, element_id)
      return energy_internal(cons, equations(dg))
    end
    if is_mpi_root()
      print(" ∑e_internal: ")
      @printf("  % 10.8e", e_internal)
      dg.save_analysis && @printf(f, "  % 10.8e", e_internal)
      println()
    end
  end

  # Magnetic energy
  if :energy_magnetic in dg.analysis_quantities
    e_magnetic = integrate(dg, dg.elements.u) do i, j, element_id, dg, u
      cons = get_node_vars(u, dg, i, j, element_id)
      return energy_magnetic(cons, equations(dg))
    end
    if is_mpi_root()
      print(" ∑e_magnetic: ")
      @printf("  % 10.8e", e_magnetic)
      dg.save_analysis && @printf(f, "  % 10.8e", e_magnetic)
      println()
    end
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
    if is_mpi_root()
      print(" ∑e_pot:      ")
      @printf("  % 10.8e", e_potential)
      dg.save_analysis && @printf(f, "  % 10.8e", e_potential)
      println()
    end
  end

  # Solenoidal condition ∇ ⋅ B = 0
  if :l2_divb in dg.analysis_quantities || :linf_divb in dg.analysis_quantities
    l2_divb, linf_divb = calc_mhd_solenoid_condition(dg, time)
  end
  if is_mpi_root()
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
  end

  # Cross helicity
  if :cross_helicity in dg.analysis_quantities
    h_c = integrate(dg, dg.elements.u) do i, j, element_id, dg, u
      cons = get_node_vars(u, dg, i, j, element_id)
      return cross_helicity(cons, equations(dg))
    end
    if is_mpi_root()
      print(" ∑H_c:        ")
      @printf("  % 10.8e", h_c)
      dg.save_analysis && @printf(f, "  % 10.8e", h_c)
      println()
    end
  end

  if is_mpi_root()
    println("-"^80)
    println()

    # Add line break and close analysis file if it was opened
    if dg.save_analysis
      println(f)
      close(f)
    end
  end

  # Return errors for EOC analysis
  return l2_error, linf_error
end


# OBS! Global results are only calculated on MPI root
function calc_error_norms(func, dg::Dg2D, t, uses_mpi::Val{true})
  l2_error, linf_error = calc_error_norms(func, dg, t, Val(false))

  # Since the local L2 norm is already normalized and square-rooted, we need to undo this first
  global_l2_error = Vector(l2_error.^2 .* dg.analysis_total_volume)
  global_linf_error = Vector(linf_error)
  MPI.Reduce!(global_l2_error, +, mpi_root(), mpi_comm())
  MPI.Reduce!(global_linf_error, max, mpi_root(), mpi_comm())
  l2_error = convert(typeof(l2_error), global_l2_error)
  linf_error = convert(typeof(linf_error), global_linf_error)

  l2_error = @. sqrt(l2_error / dg.analysis_total_volume)

  return l2_error, linf_error
end


function calc_mhd_solenoid_condition(dg::Dg2D, t, mpi_parallel::Val{true})
  l2_divb, linf_divb = calc_mhd_solenoid_condition(func, dg, t, Val(false))

  # Since the local L2 norm is already normalized and square-rooted, we need to undo this first
  global_l2_divb = Vector(l2_divb.^2 .* dg.analysis_total_volume)
  global_linf_divb = Vector(linf_divb)
  MPI.Reduce!(global_l2_divb, +, mpi_root(), mpi_comm())
  MPI.Reduce!(global_linf_divb, max, mpi_root(), mpi_comm())
  l2_divb = convert(typeof(l2_divb), global_l2_divb)
  linf_divb = convert(typeof(linf_divb), global_linf_divb)

  l2_divb = @. sqrt(l2_divb / dg.analysis_total_volume)

  return l2_divb, linf_divb
end


# OBS! Global results are only calculated on MPI root
function integrate(func, dg::Dg2D, uses_mpi::Val{true}, args...; normalize=true)
  integral = integrate(func, dg, Val(false), args...; normalize=normalize)
  integral = MPI.Reduce!(Ref(integral), +, mpi_root(), mpi_comm())

  return is_mpi_root() ? integral[] : integral
end
