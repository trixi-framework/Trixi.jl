include("Jul1dge.jl")

using .Jul1dge
using .Jul1dge.Mesh: generate_mesh
using .Jul1dge.Mesh.Trees: size, count_leaf_cells, minimum_level, maximum_level
using .Jul1dge.Equations: make_equations, nvars
using .Jul1dge.Solvers: make_solver, setinitialconditions, analyze_solution, calcdt, ndofs
using .Jul1dge.TimeDisc: timestep!
using .Jul1dge.Auxiliary: parse_commandline_arguments, parse_parameters_file, parameter, timer
using .Jul1dge.Io: save_solution_file

using Printf: println, @printf
using TimerOutputs: @timeit, print_timer


function run()
  # Parse command line arguments
  args = parse_commandline_arguments()

  # Parse parameters file
  parse_parameters_file(args["parameters-file"])

  # Create mesh
  print("Creating mesh... ")
  @timeit timer() "mesh generation" mesh = generate_mesh()
  println("done")

  # Initialize system of equations
  print("Initializing system of equations... ")
  equations_name = parameter("equations", valid=["linearscalaradvection", "euler"])
  equations = make_equations(equations_name)
  println("done")

  # Initialize solver
  print("Initializing solver... ")
  solver_name = parameter("solver", valid=["dg"])
  solver = make_solver(solver_name, equations, mesh)
  println("done")

  # Apply initial condition
  print("Applying initial conditions... ")
  t_start = parameter("t_start")
  t_end = parameter("t_end")
  time = t_start
  setinitialconditions(solver, time)
  println("done")

  # Print setup information
  println()
  N = parameter("N") # FIXME: This is currently the only DG-specific code in here
  n_steps_max = parameter("n_steps_max")
  cfl = parameter("cfl")
  initialconditions = parameter("initialconditions")
  sources = parameter("sources", "none")
  ncells = size(mesh)
  n_leaf_cells = count_leaf_cells(mesh)
  min_level = minimum_level(mesh)
  max_level = maximum_level(mesh)
  domain_center = mesh.center_level_0
  domain_length = mesh.length_level_0
  min_dx = domain_length / 2^max_level
  max_dx = domain_length / 2^min_level
  s = """| Simulation setup
         | ----------------
         | N:                 $N
         | t_start:           $t_start
         | t_end:             $t_end
         | CFL:               $cfl
         | n_steps_max:       $n_steps_max
         | equations:         $equations_name
         | | #variables:      $(nvars(equations))
         | | variable names:  $(join(equations.varnames_cons, ", "))
         | initialconditions: $initialconditions
         | sources:           $sources
         | nelements:         $(solver.nelements)
         | #DOFs:             $(ndofs(solver))
         | #parallel threads: $(Threads.nthreads())
         |
         | Mesh
         | | #cells:          $ncells
         | | #leaf cells:     $n_leaf_cells
         | | minimum level:   $min_level
         | | maximum level:   $max_level
         | | domain center:   $(join(domain_center, ", "))
         | | domain length:   $domain_length
         | | minimum dx:      $min_dx
         | | maximum dx:      $max_dx
         """
  println(s)

  # Set up main loop
  step = 0
  finalstep = false
  solution_interval = parameter("solution_interval", 0)
  save_final_solution = parameter("save_final_solution", true)
  analysis_interval = parameter("analysis_interval", 0)
  if analysis_interval > 0
    alive_interval = parameter("alive_interval", div(analysis_interval, 10))
  else
    alive_interval = 0
  end

  # Save initial conditions if desired
  if parameter("save_initial_solutions", true)
    save_solution_file(solver, step)
  end

  # Print initial solution analysis and initialize solution analysis
  if analysis_interval > 0
    analyze_solution(solver, time, 0, step, 0, 0)
  end
  loop_start_time = time_ns()
  analysis_start_time = time_ns()
  output_time = 0.0
  n_analysis_timesteps = 0

  # Start main loop (loop until final time step is reached)
  @timeit timer() "main loop" while !finalstep
    @timeit timer() "calcdt" dt = calcdt(solver, cfl)

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
          solver, time, dt, step, runtime_absolute, runtime_relative)

      # Reset time and counters
      analysis_start_time = time_ns()
      output_time = 0.0
      n_analysis_timesteps = 0.0
      if finalstep
        println("-"^80)
        println("Jul1dge simulation run finished.    Final time: $time    Time steps: $step")
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
      @timeit timer() "I/O" save_solution_file(solver, step)
      output_time += time_ns() - output_start_time
    end
  end

  # Print timer information
  print_timer(timer(), title="jul1dge", allocations=true, linechars=:ascii, compact=false)
  println()
end

