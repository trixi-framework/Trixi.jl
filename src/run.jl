using .Mesh: generate_mesh, load_mesh
using .Mesh.Trees: length, count_leaf_cells, minimum_level, maximum_level
using .Equations: make_equations, nvariables
using .Solvers: make_solver, set_initial_conditions, analyze_solution, calc_dt, ndofs,
                calc_amr_indicator, rhs!
using .TimeDisc: timestep!
using .Auxiliary: parse_commandline_arguments, parse_parameters_file,
                  parameter, timer, print_startup_message
using .Io: save_restart_file, save_solution_file, save_mesh_file, load_restart_file!
using .AMR: adapt!

using Printf: println, @printf
using TimerOutputs: @timeit, print_timer, reset_timer!, @notimeit
using Profile: clear_malloc_data


"""
    run(parameters_file=nothing; verbose=false, args=nothing)

Run a Trixi simulation with the parameters in `parameters_file`.

If `verbose` is `true`, additional output will be generated on the terminal
that may help with debugging.  If `args` is given, it should be an
`ARGS`-like array of strings that holds command line arguments, and will be
interpreted by the `parse_commandline_arguments` function. In this case, the values of
`parameters_file` and `verbose` are ignored.

# Examples
```julia
julia> Trixi.run("parameters.toml", verbose=true)
[...]
```
"""
function run(parameters_file=nothing; verbose=false, args=nothing)
  # Reset timer
  reset_timer!(timer())

  # Handle command line arguments
  @timeit timer() "parse command line" if !isnothing(args)
    # If args are given explicitly, parse command line arguments
    args = parse_commandline_arguments(args)
  else
    # Otherwise interpret keyword arguments as command line arguments
    args = Dict{String, Any}()
    if isnothing(parameters_file)
      error("missing 'parameters_file' argument")
    end
    args["parameters_file"] = parameters_file
    args["verbose"] = verbose
  end

  # Set global verbosity
  globals[:verbose] = args["verbose"]

  # Print starup message
  print_startup_message()

  # Parse parameters file
  @timeit timer() "read parameter file" parse_parameters_file(args["parameters_file"])

  # Check if this is a restart from a previous result or a new simulation
  restart = parameter("restart", false)
  if restart
    restart_filename = parameter("restart_filename")
  end

  # Initialize mesh
  if restart
    print("Loading mesh... ")
    @timeit timer() "mesh loading" mesh = load_mesh(restart_filename)
    println("done")
  else
    print("Creating mesh... ")
    @timeit timer() "mesh creation" mesh = generate_mesh()
    mesh.current_filename = save_mesh_file(mesh)
    mesh.unsaved_changes = false
    println("done")
  end

  # Initialize system of equations
  print("Initializing system of equations... ")
  equations_name = parameter("equations", valid=["linearscalaradvection", "euler", "mhd"])
  equations = make_equations(equations_name)
  println("done")

  # Initialize solver
  print("Initializing solver... ")
  solver_name = parameter("solver", valid=["dg"])
  solver = make_solver(solver_name, equations, mesh)
  println("done")

  # Sanity checks
  # If DG volume integral type is weak form, volume flux type must be central,
  # as everything else does not make sense
  if solver.volume_integral_type == :weak_form && equations.volume_flux_type != :central
    error("using the weak formulation with a volume flux other than 'central' does not make sense")
  end

  # Initialize solution
  amr_interval = parameter("amr_interval", 0)
  adapt_initial_conditions = parameter("adapt_initial_conditions", true)
  adapt_initial_conditions_only_refine = parameter("adapt_initial_conditions_only_refine", true)
  if restart
    print("Loading restart file...")
    time, step = load_restart_file!(solver, restart_filename)
    println("done")
  else
    print("Applying initial conditions... ")
    t_start = parameter("t_start")
    time = t_start
    step = 0
    set_initial_conditions(solver, time)
    println("done")

    # If AMR is enabled, adapt mesh and re-apply ICs
    if amr_interval > 0 && adapt_initial_conditions
      @timeit timer() "initial condition AMR" has_changed = adapt!(mesh, solver, time,
          only_refine=adapt_initial_conditions_only_refine)

      # Iterate until mesh does not change anymore
      while has_changed
        set_initial_conditions(solver, time)
        @timeit timer() "initial condition AMR" has_changed = adapt!(mesh, solver, time,
            only_refine=adapt_initial_conditions_only_refine)
      end

      # Save mesh file
      mesh.current_filename = save_mesh_file(mesh)
      mesh.unsaved_changes = false
    end
  end
  t_end = parameter("t_end")

  # Print setup information
  solution_interval = parameter("solution_interval", 0)
  restart_interval = parameter("restart_interval", 0)
  N = parameter("N") # FIXME: This is currently the only DG-specific code in here
  n_steps_max = parameter("n_steps_max")
  cfl = parameter("cfl")
  initial_conditions = parameter("initial_conditions")
  sources = parameter("sources", "none")
  n_leaf_cells = count_leaf_cells(mesh.tree)
  min_level = minimum_level(mesh.tree)
  max_level = maximum_level(mesh.tree)
  domain_center = mesh.tree.center_level_0
  domain_length = mesh.tree.length_level_0
  min_dx = domain_length / 2^max_level
  max_dx = domain_length / 2^min_level
  s = ""
  s *= """| Simulation setup
          | ----------------
          | working directory:  $(pwd())
          | parameters file:    $(args["parameters_file"])
          | equations:          $equations_name
          | | #variables:       $(nvariables(equations))
          | | variable names:   $(join(equations.varnames_cons, ", "))
          | sources:            $sources
          | restart:            $(restart ? "yes" : "no")
          """
  if restart
    s *= "| | restart timestep: $step\n"
    s *= "| | restart time:     $time\n"
  else
    s *= "| initial conditions: $initial_conditions\n"
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
          | restart interval:   $restart_interval
          | solution interval:  $solution_interval
          | #parallel threads:  $(Threads.nthreads())
          |
          | Solver
          | | solver:           $solver_name
          | | N:                $N
          | | CFL:              $cfl
          | | volume integral:  $(string(solver.volume_integral_type))
          | | volume flux:      $(string(equations.volume_flux_type))
          | | surface flux:     $(string(equations.surface_flux_type))
          | | #elements:        $(solver.n_elements)
          | | #surfaces:        $(solver.n_surfaces)
          | | #l2mortars:       $(solver.n_l2mortars)
          | | #DOFs:            $(ndofs(solver))
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
    save_solution_file(solver, mesh, time, 0, step)
  end

  # Print initial solution analysis and initialize solution analysis
  if analysis_interval > 0
    analyze_solution(solver, mesh, time, 0, step, 0, 0)
  end
  loop_start_time = time_ns()
  analysis_start_time = time_ns()
  output_time = 0.0
  n_analysis_timesteps = 0

  # Start main loop (loop until final time step is reached)
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
    timestep!(solver, time, dt)
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
      @timeit timer() "analyze solution" analyze_solution(
          solver, mesh, time, dt, step, runtime_absolute, runtime_relative)

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
        save_solution_file(solver, mesh, time, dt, step)
      end
      output_time += time_ns() - output_start_time
    end

    # Write restart file
    if restart_interval > 0 && (
        step % restart_interval == 0 || (finalstep && save_final_restart))
      output_start_time = time_ns()
      @timeit timer() "I/O" begin
        # If mesh has changed, write a new mesh file
        if mesh.unsaved_changes
          mesh.current_filename = save_mesh_file(mesh, step)
          mesh.unsaved_changes = false
        end

        # Then write restart file
        save_restart_file(solver, mesh, time, dt, step)
      end
      output_time += time_ns() - output_start_time
    end

    # Perform adaptive mesh refinement
    if amr_interval > 0 && (step % amr_interval == 0) && !finalstep
      @timeit timer() "AMR" has_changed = adapt!(mesh, solver, time)

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
  print_timer(timer(), title="trixi", allocations=true, linechars=:ascii, compact=false)
  println()
end
