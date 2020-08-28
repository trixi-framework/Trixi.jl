
using LinearMaps: LinearMap


"""
    run(parameters_file=nothing; verbose=false, args=nothing, refinement_level_increment=0, parameters...)

Run a Trixi simulation with the parameters in `parameters_file`.
Parameters can be overriden by specifying them as keyword arguments (see examples).

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

Without changing the parameters file we can start a simulation with `polydeg = 1` and
`t_end = 0.5` as follows:
```julia
julia> Trixi.run("examples/parameters.toml", polydeg=1, t_end=0.5)
[...]
```
"""
function run(parameters_file=nothing; verbose=false, args=nothing, refinement_level_increment=0, parameters...)
  # Reset timer
  reset_timer!(timer())

  # Read command line or keyword arguments and parse parameters file
  init_parameters(parameters_file; verbose=verbose, args=args,
      refinement_level_increment=refinement_level_increment, parameters...)

  # Separate initialization and execution into two functions such that Julia can specialize
  # the code in `run_simulation` for the actual type of `solver` and `mesh`
  if parameter("equations") == "euler_gravity"
    globals[:euler_gravity] = true
    mesh, solver, time_parameters, time_integration_function = init_simulation_euler_gravity()
    run_simulation_euler_gravity(mesh, solver, time_parameters, time_integration_function)
  else
    globals[:euler_gravity] = false
    mesh, solver, time_parameters, time_integration_function = init_simulation()
    run_simulation(mesh, solver, time_parameters, time_integration_function)
  end
end


function init_parameters(parameters_file=nothing; verbose=false, args=nothing, refinement_level_increment=0, parameters...)
  # Read command line or keyword arguments
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

  # Parse parameters file
  @timeit timer() "read parameter file" parse_parameters_file(args["parameters_file"])

  # Override specified parameters
  for (parameter, value) in parameters
    setparameter(string(parameter), value)
  end

  # Start simulation with an increased initial refinement level if specified
  # for convergence analysis
  if refinement_level_increment != 0
    setparameter("initial_refinement_level",
      parameter("initial_refinement_level") + refinement_level_increment)
  end
end


function init_simulation()
  # Print starup message
  print_startup_message()

  # Get number of dimensions
  ndims_ = parameter("ndims")::Int

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
  equations_name = parameter("equations")
  equations = make_equations(equations_name, ndims_)
  println("done")

  # Initialize solver
  print("Initializing solver... ")
  solver_name = parameter("solver", valid=["dg"])
  solver = make_solver(solver_name, equations, mesh)
  println("done")

  # Sanity checks
  # If DG volume integral type is weak form, volume flux type must be flux_central,
  # as everything else does not make sense
  if solver.volume_integral_type === Val(:weak_form) && solver.volume_flux_function !== flux_central
    error("using the weak formulation with a volume flux other than 'flux_central' does not make sense")
  end

  if equations isa AbstractIdealGlmMhdEquations && solver.volume_integral_type === Val(:weak_form)
    error("The weak form is not implemented for $equations.")
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
    set_initial_conditions!(solver, time)
    println("done")

    # If AMR is enabled, adapt mesh and re-apply ICs
    if amr_interval > 0 && adapt_initial_conditions
      @timeit timer() "initial condition AMR" has_changed = adapt!(mesh, solver, time,
          only_refine=adapt_initial_conditions_only_refine)

      # Iterate until mesh does not change anymore
      while has_changed
        set_initial_conditions!(solver, time)
        @timeit timer() "initial condition AMR" has_changed = adapt!(mesh, solver, time,
            only_refine=adapt_initial_conditions_only_refine)
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
  s *= """| Simulation setup
          | ----------------
          | working directory:  $(pwd())
          | parameters file:    $(parameter("parameters_file"))
          | equations:          $(get_name(equations))
          | | #variables:       $(nvariables(equations))
          | | variable names:   $(join(varnames_cons(equations), ", "))
          | sources:            $sources
          | restart:            $(restart ? "yes" : "no")
          """
  if restart
    s *= "| | restart timestep: $step\n"
    s *= "| | restart time:     $time\n"
  else
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
          | | volume integral:  $(get_name(solver.volume_integral_type))
          | | volume flux:      $(get_name(solver.volume_flux_function))
          | | surface flux:     $(get_name(solver.surface_flux_function))
          | | #elements:        $(solver.n_elements)
          | | #interfaces:      $(solver.n_interfaces)
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

  time_parameters = (time=time, step=step, t_end=t_end, cfl=cfl,
                    n_steps_max=n_steps_max,
                    save_final_solution=save_final_solution,
                    save_final_restart=save_final_restart,
                    analysis_interval=analysis_interval,
                    alive_interval=alive_interval,
                    solution_interval=solution_interval,
                    amr_interval=amr_interval,
                    restart_interval=restart_interval)
  return mesh, solver, time_parameters, time_integration_function
end


function run_simulation(mesh, solver, time_parameters, time_integration_function)
  @unpack time, step, t_end, cfl, n_steps_max,
          save_final_solution, save_final_restart,
          analysis_interval, alive_interval,
          solution_interval, amr_interval,
          restart_interval = time_parameters

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
    time_integration_function(solver, time, dt)
    step += 1
    time += dt
    n_analysis_timesteps += 1

    # Check if we reached the maximum number of time steps
    if step == n_steps_max
      finalstep = true
    end

    # Check steady-state integration residual
    if solver.equations isa HyperbolicDiffusionEquations2D
      if maximum(abs, view(solver.elements.u_t, 1, :, :, :)) <= solver.equations.resid_tol
        println()
        println("-"^80)
        println("  Steady state tolerance of ",solver.equations.resid_tol," reached at time ",time)
        println("-"^80)
        println()
        finalstep = true
      end
    end
    if solver.equations isa HyperbolicDiffusionEquations3D
      if maximum(abs, view(solver.elements.u_t, 1, :, :, :, :)) <= solver.equations.resid_tol
        println()
        println("-"^80)
        println("  Steady state tolerance of ",solver.equations.resid_tol," reached at time ",time)
        println("-"^80)
        println()
        finalstep = true
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
  print_timer(timer(), title="Trixi.jl", allocations=true, linechars=:ascii, compact=false)
  println()

  # Return error norms for EOC calculation
  return l2_error, linf_error, varnames_cons(solver.equations)
end


"""
    convtest(parameters_file, iterations; parameters...)

Run multiple Trixi simulations with the parameters in `parameters_file` and compute
the experimental order of convergence (EOC) in the ``L^2`` and ``L^\\infty`` norm.
The number of runs is specified by `iterations` and in each run the initial
refinement level will be increased by 1. Parameters can be overriden by specifying them as
additional keyword arguments, which are passed to the respective call to `run`..
"""
function convtest(parameters_file, iterations; parameters...)
  @assert(iterations > 1, "Number of iterations must be bigger than 1 for a convergence analysis")

  # Types of errors to be calcuated
  errors = Dict(:L2 => Float64[], :Linf => Float64[])

  # Declare variable to access variable names after for loop
  local variablenames

  # Run trixi and extract errors
  for i = 1:iterations
    println(string("Running convtest iteration ", i, "/", iterations))
    l2_error, linf_error, variablenames = run(parameters_file; refinement_level_increment = i - 1,
                                              parameters...)

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


function compute_linear_structure(parameters_file=nothing, source_terms=nothing; verbose=false, args=nothing, refinement_level_increment=0, parameters...)
  # Read command line or keyword arguments and parse parameters file
  init_parameters(parameters_file; verbose=verbose, args=args,
      refinement_level_increment=refinement_level_increment, parameters...)
  globals[:euler_gravity] = false
  mesh, solver, time_parameters = init_simulation()

  equations(solver) isa Union{AbstractLinearScalarAdvectionEquation, AbstractHyperbolicDiffusionEquations} ||
    throw(ArgumentError("Only linear problems are supported."))

  # get the right hand side from the source terms
  solver.elements.u .= 0
  rhs!(solver, 0)
  b = vec(-solver.elements.u_t) |> copy

  # set the source terms to zero to extract the linear operator
  if solver isa Dg2D
    solver = Dg2D(solver.equations, solver.surface_flux_function, solver.volume_flux_function, solver.initial_conditions,
                  source_terms, mesh, polydeg(solver))
  elseif solver isa Dg3D
    solver = Dg3D(solver.equations, solver.surface_flux_function, solver.volume_flux_function, solver.initial_conditions,
                  source_terms, mesh, polydeg(solver))
  else
    error("not implemented")
  end
  A = LinearMap(length(solver.elements.u), ismutating=true) do dest,src
    vec(solver.elements.u) .= src
    rhs!(solver, 0)
    dest .= vec(solver.elements.u_t)
  end

  A, b
end


# Include source file with init and run methods for coupled Euler-gravity simulations
include("run_euler_gravity.jl")
