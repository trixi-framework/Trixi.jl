
function init_simulation_euler_gravity()
  # Print starup message
  print_startup_message()

  # Get number of dimensions
  ndims_ = parameter("ndims")::Int

  # Check if this is a restart from a previous result or a new simulation
  restart = parameter("restart", false)
  if restart
    error("restarting not yet supported for coupled Euler-gravity simulations")
  end

  # Initialize mesh
  begin
    print("Creating mesh... ")
    @timeit timer() "mesh creation" mesh = generate_mesh()
    mesh.current_filename = save_mesh_file(mesh)
    mesh.unsaved_changes = false
    println("done")
  end

  # Initialize system of equations
  print("Initializing system of equations... ")
  equations_name = parameter("equations")
  @assert equations_name == "euler_gravity" "This only works with 'euler_gravity' as equations type"
  equations_euler = make_equations("CompressibleEulerEquations", ndims_)
  equations_gravity = make_equations("HyperbolicDiffusionEquations", ndims_)
  println("done")

  # Initialize solver
  print("Initializing solver... ")
  solver_name = parameter("solver", valid=["dg"])
  solver = make_solver(solver_name, equations_euler, mesh)
  solver_gravity = make_solver(solver_name, equations_gravity, mesh,
                               surface_flux_function=flux_lax_friedrichs, volume_flux_function=flux_central)
  # `solver` = Euler solver -> this is to keep differences to original method to a minimum
  println("done")

  # Sanity checks
  # If DG volume integral type is weak form, volume flux type must be flux_central,
  # as everything else does not make sense
  if solver.volume_integral_type === Val(:weak_form) && solver.volume_flux_function !== flux_central
    error("using the weak formulation with a volume flux other than 'flux_central' does not make sense")
  end

  # Initialize solution
  amr_interval = parameter("amr_interval", 0)
  adapt_initial_conditions = parameter("adapt_initial_conditions", true)
  adapt_initial_conditions_only_refine = parameter("adapt_initial_conditions_only_refine", true)
  begin
    print("Applying initial conditions... ")
    t_start = parameter("t_start")
    time = t_start
    step = 0
    set_initial_conditions!(solver, time)
    set_initial_conditions!(solver_gravity, time)
    println("done")

    # If AMR is enabled, adapt mesh and re-apply ICs
    if amr_interval > 0 && adapt_initial_conditions
      @timeit timer() "initial condition AMR" has_changed = adapt!(mesh, solver, time,
          only_refine=adapt_initial_conditions_only_refine, passive_solvers=(solver_gravity,))

      # Iterate until mesh does not change anymore
      while has_changed
        set_initial_conditions!(solver, time)
        set_initial_conditions!(solver_gravity, time)
        @timeit timer() "initial condition AMR" has_changed = adapt!(mesh, solver, time,
            only_refine=adapt_initial_conditions_only_refine, passive_solvers=(solver_gravity,))
      end

      # Save mesh file
      mesh.current_filename = save_mesh_file(mesh)
      mesh.unsaved_changes = false
    end
  end
  t_end = parameter("t_end")

  # Init time integration
  time_integration_scheme = Symbol(parameter("time_integration_scheme", "timestep_carpenter_kennedy_erk54_2N!"))
  time_integration_function = eval(time_integration_scheme)

  # Print setup information
  solution_interval = parameter("solution_interval", 0)
  restart_interval = parameter("restart_interval", 0)
  polydeg = parameter("polydeg") # FIXME: This is currently the only DG-specific code in here
  n_steps_max = parameter("n_steps_max")
  cfl = parameter("cfl")
  sources = parameter("sources", "none")
  n_leaf_cells = count_leaf_cells(mesh.tree)
  min_level = minimum_level(mesh.tree)
  max_level = maximum_level(mesh.tree)
  domain_center = mesh.tree.center_level_0
  domain_length = mesh.tree.length_level_0
  min_dx = domain_length / 2^max_level
  max_dx = domain_length / 2^min_level
  s = ""
  s *= """| Simulation setup (Euler + Gravity)
          | ----------------
          | working directory:  $(pwd())
          | parameters file:    $(parameter("parameters_file"))
          | equations:          $(equations_name)
          | | Euler:
          | | | #variables:     $(nvariables(equations_euler))
          | | | variable names: $(join(varnames_cons(equations_euler), ", "))
          | | | sources:        $sources
          | | Gravity:
          | | | #variables:     $(nvariables(equations_gravity))
          | | | variable names: $(join(varnames_cons(equations_gravity), ", "))
          | | | sources:        $sources
          | restart:            $(restart ? "yes" : "no")
          """
  begin
    s *= "| initial conditions: $(get_name(solver.initial_conditions))\n"
    s *= "| t_start:            $t_start\n"
  end
  s *= """| t_end:              $t_end
          | AMR:                $(amr_interval > 0 ? "yes" : "no")
          """
  if amr_interval > 0
    s *= "| | AMR interval:     $amr_interval\n"
    s *= "| | adapt ICs:        $(adapt_initial_conditions ? "yes" : "no")\n"
  end
  s *= """| n_steps_max:        $n_steps_max
          | time integration:   $(get_name(time_integration_function))
          | restart interval:   $restart_interval
          | solution interval:  $solution_interval
          | #parallel threads:  $(Threads.nthreads())
          |
          | Solver
          | | solver:           $solver_name
          | | polydeg:          $polydeg
          | | CFL:              $cfl
          | | Euler solver:
          | | | volume integral:  $(get_name(solver.volume_integral_type))
          | | | volume flux:      $(get_name(solver.volume_flux_function))
          | | | surface flux:     $(get_name(solver.surface_flux_function))
          | | Gravity solver:
          | | | volume integral:  $(get_name(solver_gravity.volume_integral_type))
          | | | volume flux:      $(get_name(solver_gravity.volume_flux_function))
          | | | surface flux:     $(get_name(solver_gravity.surface_flux_function))
          | | #elements:        $(solver.n_elements)
          | | #interfaces:      $(solver.n_interfaces)
          | | #boundaries:      $(solver.n_boundaries)
          | | #l2mortars:       $(solver.n_l2mortars)
          | | #DOFs:            $(ndofs(solver)) + $(ndofs(solver_gravity))
          |
          | Mesh
          | | #cells:           $(length(mesh.tree))
          | | #leaf cells:      $n_leaf_cells
          | | minimum level:    $min_level
          | | maximum level:    $max_level
          | | domain center:    $(join(domain_center, ", "))
          | | domain length:    $domain_length
          | | minimum dx:       $min_dx
          | | maximum dx:       $max_dx
          """
  println()
  println(s)

  # Set up main loop
  save_final_solution = parameter("save_final_solution", true)
  save_final_restart = parameter("save_final_restart", true)
  analysis_interval = parameter("analysis_interval", 0)
  if analysis_interval > 0
    alive_interval = parameter("alive_interval", div(analysis_interval, 10))
  else
    alive_interval = 0
  end

  # Save initial conditions if desired
  if !restart && parameter("save_initial_solution", true)
    # we need to make sure, that derived quantities, such as e.g. blending
    # factor is already computed for the initial condition
    @notimeit timer() rhs!(solver, time)
    save_solution_file(solver, mesh, time, 0, step, "euler")

    @notimeit timer() rhs!(solver_gravity, time)
    save_solution_file(solver_gravity, mesh, time, 0, step, "gravity")
  end
  # Print initial solution analysis and initialize solution analysis
  if analysis_interval > 0
    if get_name(solver.initial_conditions) == "initial_conditions_eoc_test_coupled_euler_gravity"
      analyze_solution(solver, mesh, time, 0, step, 0, 0, solver_gravity=solver_gravity)
      # comment out for anything other than coupling convergence test
      println()
      analyze_solution(solver_gravity, mesh, time, 0, step, 0, 0)
    else
      analyze_solution(solver, mesh, time, 0, step, 0, 0, solver_gravity=solver_gravity)
    end
  end

  time_parameters = (time=time, step=step, t_end=t_end, cfl=cfl,
                    n_steps_max=n_steps_max,
                    save_final_solution=save_final_solution,
                    save_final_restart=save_final_restart,
                    analysis_interval=analysis_interval,
                    alive_interval=alive_interval,
                    solution_interval=solution_interval,
                    amr_interval=amr_interval,
                    restart_interval=restart_interval)
  return mesh, (solver, solver_gravity), time_parameters, time_integration_function
end


function run_simulation_euler_gravity(mesh, solvers, time_parameters, time_integration_function)
  @unpack time, step, t_end, cfl, n_steps_max,
          save_final_solution, save_final_restart,
          analysis_interval, alive_interval,
          solution_interval, amr_interval,
          restart_interval = time_parameters

  solver_euler, solver_gravity = solvers
  solver = solver_euler # To keep differences to `run_simulation` to a minimum

  loop_start_time = time_ns()
  analysis_start_time = time_ns()
  output_time = 0.0
  n_analysis_timesteps = 0

  # Declare variables to return error norms calculated in main loop
  # (needed such that the values are accessible outside of the main loop)
  local l2_error, linf_error

  # Start main loop (loop until final time step is reached)
  globals[:gravity_subcycles] = 0
  finalstep = false
  first_loop_iteration = true
  @timeit timer() "main loop" while !finalstep
    # Calculate time step size
    @timeit timer() "calc_dt" dt = calc_dt(solver, cfl)

    # Abort if time step size is NaN
    if isnan(dt)
      error("time step size `dt` is NaN")
    end

    # If the next iteration would push the simulation beyond the end time, set dt accordingly
    if time + dt > t_end
      dt = t_end - time
      finalstep = true
    end

    # Evolve solution by one time step
    @timeit timer() "timestep_euler_gravity!" begin
      timestep_euler_gravity!(solver_euler, solver_gravity, time, dt, time_parameters)
    end
    step += 1
    time += dt
    n_analysis_timesteps += 1

    # Check if we reached the maximum number of time steps
    if step == n_steps_max
      finalstep = true
    end

    # Analyze solution errors
    if analysis_interval > 0 && (step % analysis_interval == 0 || finalstep)
      # Calculate absolute and relative runtime
      runtime_absolute = (time_ns() - loop_start_time) / 10^9
      runtime_relative = ((time_ns() - analysis_start_time - output_time) / 10^9 /
                          (n_analysis_timesteps * ndofs(solver)))

      # Analyze solution
      if get_name(solver_euler.initial_conditions) == "initial_conditions_eoc_test_coupled_euler_gravity"
        l2_euler, linf_euler = @timeit timer() "analyze solution" analyze_solution(
            solver, mesh, time, dt, step, runtime_absolute, runtime_relative, solver_gravity=solver_gravity)
        # Pull gravity solver information from file
        timestep_gravity = eval(Symbol(parameter("time_integration_scheme_gravity")))
        cfl_gravity = parameter("cfl_gravity")::Float64
        rho0 = parameter("rho0")::Float64
        G = parameter("G")::Float64
        gravity_parameters = (; timestep_gravity, cfl_gravity, rho0, G)
        update_gravity!(solver_gravity, solver_euler.elements.u, gravity_parameters)
        l2_hypdiff, linf_hypdiff = @timeit timer() "analyze solution" analyze_solution(
            solver_gravity, mesh, time, dt, step, runtime_absolute, runtime_relative)
        l2_error   = vcat(l2_euler  , l2_hypdiff)
        linf_error = vcat(linf_euler, linf_hypdiff)
      else
        l2_error, linf_error = @timeit timer() "analyze solution" analyze_solution(
            solver, mesh, time, dt, step, runtime_absolute, runtime_relative, solver_gravity=solver_gravity)
      end

      # Reset time and counters
      analysis_start_time = time_ns()
      output_time = 0.0
      n_analysis_timesteps = 0
      if finalstep
        println("-"^80)
        println("Trixi simulation run finished.    Final time: $time    Time steps: $step")
        println("-"^80)
        println()
      end
    elseif alive_interval > 0 && step % alive_interval == 0
      runtime_absolute = (time_ns() - loop_start_time) / 10^9
      @printf("#t/s: %6d | dt: %.4e | Sim. time: %.4e | Run time: %.4e s\n",
              step, dt, time, runtime_absolute)
    end

    # Write solution file
    if solution_interval > 0 && (
        step % solution_interval == 0 || (finalstep && save_final_solution))
      output_start_time = time_ns()
      @timeit timer() "I/O" begin
        # Compute current AMR indicator values such that it can be written to
        # the solution file for the current number of elements
        if amr_interval > 0
          calc_amr_indicator(solver, mesh, time)
        end

        # If mesh has changed, write a new mesh file name
        if mesh.unsaved_changes
          mesh.current_filename = save_mesh_file(mesh, step)
          mesh.unsaved_changes = false
        end

        # Then write solution file
        save_solution_file(solver, mesh, time, dt, step, "euler")
        save_solution_file(solver_gravity, mesh, time, dt, step, "gravity")
      end
      output_time += time_ns() - output_start_time
    end

    # Perform adaptive mesh refinement
    if amr_interval > 0 && (step % amr_interval == 0) && !finalstep
      @timeit timer() "AMR" has_changed = adapt!(mesh, solver, time,
                                                 passive_solvers=(solver_gravity,))

      # Store if mesh has changed to write changed mesh file before next restart/solution output
      if has_changed
        mesh.unsaved_changes = true
      end
    end

    # The following call ensures that when doing memory allocation
    # measurements, the memory allocations for JIT compilation are discarded
    # (since virtually all relevant methods have already been called by now)
    if first_loop_iteration
      clear_malloc_data()
      first_loop_iteration = false
    end
  end

  # Print timer information
  print_timer(timer(), title="Trixi.jl", allocations=true, linechars=:ascii, compact=false)
  println()

  # Return error norms for EOC calculation
  println("Number of gravity subcycles: ", globals[:gravity_subcycles])
  if get_name(solver_euler.initial_conditions) == "initial_conditions_eoc_test_coupled_euler_gravity"
    return l2_error, linf_error, vcat(varnames_cons(solver.equations),
                                      varnames_cons(solver_gravity.equations))
  else
    return l2_error, linf_error, varnames_cons(solver.equations)
  end
end

