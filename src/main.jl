include("Jul1dge.jl")

using .Jul1dge
using .Jul1dge.MeshMod: Mesh
using .Jul1dge.Equation: getsyseqn
using .Jul1dge.DgMod: Dg, setinitialconditions, calc_error_norms, calcdt
using .Jul1dge.TimeDisc: timestep!
using .Jul1dge.Auxiliary: parse_commandline_arguments, parse_parameters_file, parameter, timer
using .Jul1dge.Io: save_solution_file

using Printf: println
using TimerOutputs: @timeit, print_timer


function run()
  # Parse command line arguments
  args = parse_commandline_arguments()

  # Parse parameters file
  parse_parameters_file(args["parameters-file"])

  # Retrieve repeatedly used parameters
  N = parameter("N")
  ncells = parameter("ncells")
  cfl = parameter("cfl")
  nstepsmax = parameter("nstepsmax")
  equations = parameter("syseqn")
  initialconditions = parameter("initialconditions")
  sources = parameter("sources", "none")
  t_start = parameter("t_start")
  t_end = parameter("t_end")

  # Create mesh
  print("Creating mesh... ")
  mesh = Mesh(parameter("x_start"), parameter("x_end"), ncells)
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
  t = t_start
  setinitialconditions(dg, t)
  println("done")

  # Print setup information
  println()
  s = """| Simulation setup
         | ----------------
         | N:                 $N
         | t_start:           $t_start
         | t_end:             $t_end
         | CFL:               $cfl
         | nstepsmax:         $nstepsmax
         | equation:          $equations
         | initialconditions: $initialconditions
         | sources:           $sources
         | ncells:            $ncells
         | #DOFs:             $(ncells * (N + 1)^ndim)
         """
  println(s)

  # Set up main loop
  step = 0
  finalstep = false
  solution_interval = parameter("solution_interval", 0)

  # Save initial conditions if desired
  if parameter("save_initial_solutions", true)
    save_solution_file(dg, step)
  end

  println("Step: #$step, t=$t")
  l2_error, linf_error = calc_error_norms(dg, t)
  println("--- variable:   $(syseqn.varnames)")
  println("--- L2 error:   $(l2_error)")
  println("--- Linf error: $(linf_error)")
  println()

  # Start main loop
  println("Starting main loop... ")
  @timeit timer() "main loop" while !finalstep
    @timeit timer() "calcdt" dt = calcdt(dg, cfl)

    if t + dt > t_end
      dt = t_end - t
      finalstep = true
    end

    timestep!(dg, t, dt)
    step += 1
    t += dt

    if step == nstepsmax
      finalstep = true
    end

    if step % 10 == 0 || finalstep
      println("Step: #$step, t=$t")
      l2_error, linf_error = calc_error_norms(dg, t)
      println("--- variable:   $(syseqn.varnames)")
      println("--- L2 error:   $(l2_error)")
      println("--- Linf error: $(linf_error)")
      println()
    end

    # Write solution file
    if solution_interval > 0 && (
        step % solution_interval == 0 || (finalstep && parameter("save_final_solution", true)))
      save_solution_file(dg, step)
    end
  end
  println("done")

  print_timer(timer(), title="jul1dge", allocations=true, linechars=:ascii, compact=false)
  println()
end

