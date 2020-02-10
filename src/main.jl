include("Jul1dge.jl")

using .Jul1dge
using .Jul1dge.Mesh: generate_mesh
using .Jul1dge.Mesh.Trees: size, count_leaf_nodes, minimum_level, maximum_level
using .Jul1dge.Equation: getsyseqn, nvars
using .Jul1dge.Solvers.DgMod: Dg, setinitialconditions, analyze_solution, calcdt
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

  # Retrieve repeatedly used parameters
  N = parameter("N")
  cfl = parameter("cfl")
  nstepsmax = parameter("nstepsmax")
  equations = parameter("syseqn")
  initialconditions = parameter("initialconditions")
  sources = parameter("sources", "none")
  t_start = parameter("t_start")
  t_end = parameter("t_end")

  # Create mesh
  print("Creating mesh... ")
  @timeit timer() "mesh generation" mesh = generate_mesh()
  println("done")

  # Initialize system of equations
  print("Initializing system of equations... ")
  if equations == "linearscalaradvection"
    advectionvelocity = parameter("advectionvelocity")
    syseqn = getsyseqn("linearscalaradvection", initialconditions, sources,
                       advectionvelocity)
  elseif equations == "euler"
    syseqn = getsyseqn("euler", initialconditions, sources)
  else
    error("unknown system of equations '$equations'")
  end
  println("done")

  # Initialize solver
  print("Initializing solver... ")
  dg = Dg(syseqn, mesh, N)
  println("done")

  # Apply initial condition
  print("Applying initial conditions... ")
  time = t_start
  setinitialconditions(dg, time)
  println("done")

  # Print setup information
  println()
  n_dofs_total = dg.ncells * (N + 1)^ndim
  n_nodes = size(mesh)
  n_leaf_nodes = count_leaf_nodes(mesh)
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
         | nstepsmax:         $nstepsmax
         | equation:          $equations
         | | #variables:      $(nvars(syseqn))
         | | variable names:  $(join(syseqn.varnames_cons, ", "))
         | initialconditions: $initialconditions
         | sources:           $sources
         | ncells:            $(dg.ncells)
         | #DOFs:             $n_dofs_total
         | #parallel threads: $(Threads.nthreads())
         |
         | Mesh
         | | #nodes:          $n_nodes
         | | #leaf nodes:     $n_leaf_nodes
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
    save_solution_file(dg, step)
  end

  # Print initial solution analysis and initialize solution analysis
  if analysis_interval > 0
    analyze_solution(dg, time, 0, step, 0, 0)
  end
  loop_start_time = time_ns()
  analysis_start_time = time_ns()
  output_time = 0.0
  n_analysis_timesteps = 0

  # Start main loop (loop until final time step is reached)
  @timeit timer() "main loop" while !finalstep
    @timeit timer() "calcdt" dt = calcdt(dg, cfl)

    # If the next iteration would push the simulation beyond the end time, set dt accordingly
    if time + dt > t_end
      dt = t_end - time
      finalstep = true
    end

    # Evolve solution by one time step
    timestep!(dg, time, dt)
    step += 1
    time += dt
    n_analysis_timesteps += 1

    # Check if we reached the maximum number of time steps
    if step == nstepsmax
      finalstep = true
    end

    # Analyze solution errors
    if analysis_interval > 0 && (step % analysis_interval == 0 || finalstep)
      # Calculate absolute and relative runtime
      runtime_absolute = (time_ns() - loop_start_time) / 10^9
      runtime_relative = ((time_ns() - analysis_start_time - output_time) / 10^9 /
                          (n_analysis_timesteps * n_dofs_total))

      # Analyze solution
      @timeit timer() "analyze solution" analyze_solution(
          dg, time, dt, step, runtime_absolute, runtime_relative)

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
      @timeit timer() "I/O" save_solution_file(dg, step)
      output_time += time_ns() - output_start_time
    end
  end

  # Print timer information
  print_timer(timer(), title="jul1dge", allocations=true, linechars=:ascii, compact=false)
  println()
end

