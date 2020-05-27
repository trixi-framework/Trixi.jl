
"""
    run(parameters_file=nothing; verbose=false, args=nothing, refinement_level_increment=0)

Run a Trixi simulation with the parameters in `parameters_file`.

If `verbose` is `true`, additional output will be generated on the terminal
that may help with debugging.  If `args` is given, it should be an
`ARGS`-like array of strings that holds command line arguments, and will be
interpreted by the `parse_commandline_arguments` function. In this case, the values of
`parameters_file` and `verbose` are ignored. If a value for
`refinement_level_increment` is given, `initial_refinement_level` will be
 increased by this value before running the simulation (mostly used by EOC analysis).

# Examples
```julia
julia> Trixi.run("examples/parameters.toml", verbose=true)
[...]
```
"""
function run(parameters_file=nothing; verbose=false, args=nothing, refinement_level_increment=0)
  # Separate initialization and execution into two functions such that Julia can specialize
  # the code in `run_simulation` for the actual type of `solver` and `mesh`
  mesh, solver, time_parameters = init_simulation(
      parameters_file, verbose=verbose, args=args,
      refinement_level_increment=refinement_level_increment)
  run_simulation(mesh, solver, time_parameters)
end


function init_simulation(parameters_file; verbose=false, args=nothing, refinement_level_increment=0)
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

  # Start simulation with an increased initial refinement level if specified
  # for convergence analysis
  if refinement_level_increment != 0
    setparameter("initial_refinement_level",
      parameter("initial_refinement_level") + refinement_level_increment)
  end

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
  equations_name = parameter("equations", valid=["LinearScalarAdvection", "CompressibleEuler", "IdealMhd",
                                                 "HyperbolicDiffusion", "euler_gravity"])
  if equations_name == "euler_gravity"
    globals[:euler_gravity] = true
    equations_euler = make_equations("CompressibleEuler")
    # FIXME: Hack to set that the Euler equations have no source
    equations_euler.sources = "none"
    equations_gravity = make_equations("HyperbolicDiffusion")
  else
    globals[:euler_gravity] = false
    equations = make_equations(equations_name)
  end
  println("done")

  # Initialize solver
  print("Initializing solver... ")
  solver_name = parameter("solver", valid=["dg"])
  if globals[:euler_gravity]
    solver_euler = make_solver(solver_name, equations_euler, mesh)
    solver_gravity = make_solver(solver_name, equations_gravity, mesh)
  else
    solver = make_solver(solver_name, equations, mesh)
  end
  println("done")

  # Sanity checks
  # If DG volume integral type is weak form, volume flux type must be central_flux,
  # as everything else does not make sense
  if globals[:euler_gravity]
    if solver_euler.volume_integral_type == Val(:weak_form) && equations_euler.volume_flux != central_flux
      error("using the weak formulation with a volume flux other than 'central_flux' does not make sense")
    end
    if solver_gravity.volume_integral_type == Val(:weak_form) && equations_gravity.volume_flux != central_flux
      error("using the weak formulation with a volume flux other than 'central_flux' does not make sense")
    end
  else
    if solver.volume_integral_type == Val(:weak_form) && equations.volume_flux != central_flux
      error("using the weak formulation with a volume flux other than 'central_flux' does not make sense")
    end
  end

  # Initialize solution
  amr_interval = parameter("amr_interval", 0)
  adapt_initial_conditions = parameter("adapt_initial_conditions", true)
  adapt_initial_conditions_only_refine = parameter("adapt_initial_conditions_only_refine", true)
  if restart
    @assert !globals[:euler_gravity] "Not yet supported for coupled Euler-gravity simulations"
    print("Loading restart file...")
    time, step = load_restart_file!(solver, restart_filename)
    println("done")
  else
    print("Applying initial conditions... ")
    t_start = parameter("t_start")
    time = t_start
    step = 0
    if globals[:euler_gravity]
      set_initial_conditions(solver_euler, time)
      set_initial_conditions(solver_gravity, time)
    else
      set_initial_conditions(solver, time)
    end
    println("done")

    # If AMR is enabled, adapt mesh and re-apply ICs
    if amr_interval > 0 && adapt_initial_conditions
      @assert !globals[:euler_gravity] "Not yet supported for coupled Euler-gravity simulations"
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
  if globals[:euler_gravity]
    s *= """| Simulation setup (Euler + Gravity)
            | ----------------
            | working directory:  $(pwd())
            | parameters file:    $(args["parameters_file"])
            | equations:          $equations_name
            | | Euler:
            | | | #variables:     $(nvariables(equations_euler))
            | | | variable names: $(join(equations_euler.varnames_cons, ", "))
            | | | sources:        $(equations_euler.sources)
            | | Gravity:
            | | | #variables:     $(nvariables(equations_gravity))
            | | | variable names: $(join(equations_gravity.varnames_cons, ", "))
            | | | sources:        $(equations_gravity.sources)
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
            | | Euler solver:
            | | | volume integral:  $(strip_val(solver_euler.volume_integral_type))
            | | | volume flux:      $(string(equations_euler.volume_flux))
            | | | surface flux:     $(string(equations_euler.surface_flux))
            | | Gravity solver:
            | | | volume integral:  $(strip_val(solver_gravity.volume_integral_type))
            | | | volume flux:      $(string(equations_gravity.volume_flux))
            | | | surface flux:     $(string(equations_gravity.surface_flux))
            | | #elements:        $(solver_euler.n_elements)
            | | #surfaces:        $(solver_euler.n_surfaces)
            | | #boundaries:      $(solver_euler.n_boundaries)
            | | #l2mortars:       $(solver_euler.n_l2mortars)
            | | #DOFs:            $(ndofs(solver_euler)) + $(ndofs(solver_gravity))
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
  else
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
            | | volume integral:  $(strip_val(solver.volume_integral_type))
            | | volume flux:      $(string(equations.volume_flux))
            | | surface flux:     $(string(equations.surface_flux))
            | | #elements:        $(solver.n_elements)
            | | #surfaces:        $(solver.n_surfaces)
            | | #boundaries:      $(solver.n_boundaries)
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
  end
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
    if globals[:euler_gravity]
      @notimeit timer() rhs!(solver_euler, time)
      save_solution_file(solver_euler, mesh, time, 0, step)
#      save_solution_file(solver_euler, mesh, time, 0, step, "euler")
#      save_solution_file(solver_gravity, mesh, time, 0, step, "gravity")
    else
      @notimeit timer() rhs!(solver, time)
      save_solution_file(solver, mesh, time, 0, step)
    end
  end

  # Print initial solution analysis and initialize solution analysis
  if analysis_interval > 0
    if globals[:euler_gravity]
      analyze_solution(solver_euler, mesh, time, 0, step, 0, 0)
    else
      analyze_solution(solver, mesh, time, 0, step, 0, 0)
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
  if globals[:euler_gravity]
    return mesh, solver_euler, solver_gravity, time_parameters
  else
    return mesh, solver, time_parameters
  end
end


function run_simulation(mesh, solvers, time_parameters)
  @unpack time, step, t_end, cfl, n_steps_max,
          save_final_solution, save_final_restart,
          analysis_interval, alive_interval,
          solution_interval, amr_interval,
          restart_interval = time_parameters

  if isa(solvers, Tuple)
    solver_euler, solver_gravity = solvers
    solver = solver_euler
  else
    solver = solvers
  end

  loop_start_time = time_ns()
  analysis_start_time = time_ns()
  output_time = 0.0
  n_analysis_timesteps = 0

  # Declare variables to return error norms calculated in main loop
  # (needed such that the values are accessible outside of the main loop)
  local l2_error, linf_error

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
    if globals[:euler_gravity]
      timestep!(solver_euler, solver_gravity, time, dt)
    else
      timestep!(solver, time, dt)
    end
    step += 1
    time += dt
    n_analysis_timesteps += 1

    # Check if we reached the maximum number of time steps
    if step == n_steps_max
      finalstep = true
    end

    if !globals[:euler_gravity]
      # Check steady-state integration residual
      if solver.equations isa HyperbolicDiffusionEquations
        if maximum(abs.(solver.elements.u_t[1, :, :, :])) <= solver.equations.resid_tol
          println()
          println("-"^80)
          println("  Steady state tolerance of ",solver.equations.resid_tol," reached at time ",time)
          println("-"^80)
          println()
          finalstep = true
        end
      end
    end

    # Analyze solution errors
    if analysis_interval > 0 && (step % analysis_interval == 0 || finalstep)
      # Calculate absolute and relative runtime
      runtime_absolute = (time_ns() - loop_start_time) / 10^9
      runtime_relative = ((time_ns() - analysis_start_time - output_time) / 10^9 /
                          (n_analysis_timesteps * ndofs(solver)))

      # Analyze solution
      l2_error, linf_error = @timeit timer() "analyze solution" analyze_solution(
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
        if globals[:euler_gravity]
          save_solution_file(solver_euler, mesh, time, dt, step)
#          save_solution_file(solver_euler, mesh, time, dt, step, "euler")
#          save_solution_file(solver_gravity, mesh, time, dt, step, "gravity")
        else
          save_solution_file(solver, mesh, time, dt, step)
        end
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

  # Return error norms for EOC calculation
  return l2_error, linf_error, solver.equations.varnames_cons
end


"""
    convtest(parameters_file, iterations)

Run multiple Trixi simulations with the parameters in `parameters_file` and compute
the experimental order of convergence (EOC) in the ``L^2`` and ``L^\\infty`` norm.
The number of runs is specified by `iterations` and in each run the initial
refinement level will be increased by 1.
"""
function convtest(parameters_file, iterations)
  @assert(iterations > 1, "Number of iterations must be bigger than 1 for a convergence analysis")

  # Types of errors to be calcuated
  errors = Dict(:L2 => Float64[], :Linf => Float64[])

  # Declare variable to access variable names after for loop
  local variablenames

  # Run trixi and extract errors
  for i = 1:iterations
    println(string("Running convtest iteration ", i, "/", iterations))
    l2_error, linf_error, variablenames = run(parameters_file, refinement_level_increment = i - 1)

    # Collect errors as one vector to reshape later
    append!(errors[:L2], l2_error)
    append!(errors[:Linf], linf_error)
  end

  # Number of variables
  nvariables = length(variablenames)

  # Reshape errors to get a matrix where the i-th row represents the i-th iteration
  # and the j-th column represents the j-th variable
  errorsmatrix = Dict(kind => transpose(reshape(error, (nvariables, iterations))) for (kind, error) in errors)

  # Calculate EOCs where the columns represent the variables
  # As dx halves in every iteration the denominator needs to be log(1/2)
  eocs = Dict(kind => log.(error[2:end, :] ./ error[1:end-1, :]) ./ log(1 / 2) for (kind, error) in errorsmatrix)


  for (kind, error) in errorsmatrix
    println(kind)

    for v in variablenames
      @printf("%-20s", v)
    end
    println("")

    for k = 1:nvariables
      @printf("%-10s", "error")
      @printf("%-10s", "EOC")
    end
    println("")

    # Print errors for the first iteration
    for k = 1:nvariables
      @printf("%-10.2e", error[1, k])
      @printf("%-10s", "-")
    end
    println("")

    # For the following iterations print errors and EOCs
    for j = 2:iterations
      for k = 1:nvariables
        @printf("%-10.2e", error[j, k])
        @printf("%-10.2f", eocs[kind][j-1, k])
      end
      println("")
    end
    println("")

    # Print mean EOCs
    for k = 1:nvariables
      @printf("%-10s", "mean")
      @printf("%-10.2f", sum(eocs[kind][:, k]) ./ length(eocs[kind][:, k]))
    end
    println("")
    println("-"^80)
  end
end
