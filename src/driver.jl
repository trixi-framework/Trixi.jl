#!/usr/bin/env julia

include("Jul1dge.jl")
using .Jul1dge
using .Jul1dge.MeshMod
using .Jul1dge.Equation
using .Jul1dge.DgMod
using .Jul1dge.TimeDisc
using .Jul1dge.Auxiliary
# using .Jul1dge.Io

using ArgParse
using Printf
using TimerOutputs

defaults = Dict(
# Computational domain
"x_start" => -10,
"x_end" => 10,

# Number of cells
"ncells" => 10,

# Polynomial degree
"N" => 3,

# Start and end time
"t_start" => 0.0,
"t_end" => 1.0,

# CFL number
"cfl" => 1.0,

# Maximum number of timesteps
"nstepsmax" => 10000,

# Advection velocity
"advectionvelocity" => 1.0,

# System of equations
"syseqn" => "linearscalaradvection",

# Initial conditions
"initialconditions" => "constant",

# Source terms
"sources" => "none"
)

function main()
  # Parse command line arguments
  args = parse_arguments()

  # Store repeatedly used values
  N = args["N"]
  ncells = args["ncells"]
  cfl = args["cfl"]
  nstepsmax = args["nstepsmax"]
  equations = args["syseqn"]
  initialconditions = args["initialconditions"]
  sources = args["sources"]
  x_start = defaults["x_start"]
  x_end = defaults["x_end"]
  t_start = defaults["t_start"]
  t_end = defaults["t_end"]

  # Create mesh
  print("Creating mesh... ")
  mesh = Mesh(x_start, x_end, ncells)
  println("done")

  # Initialize system of equations
  print("Initializing system of equations... ")
  if equations == "linearscalaradvection"
    syseqn = getsyseqn("linearscalaradvection", initialconditions, sources,
                       defaults["advectionvelocity"])
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
  # plot2file(dg, "initialconditions.pdf")
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
         | initialconditions: $initialconditions
         | ncells:            $ncells
         | #DOFs:             $(ncells * (N + 1)^ndim)
         """
  println(s)

  # Main loop
  println("Starting main loop... ")
  step = 0

  println("Step: #$step, t=$t")
  l2_error, linf_error = calc_error_norms(dg, t)
  println("--- variable:   $(syseqn.varnames)")
  println("--- L2 error:   $(l2_error)")
  println("--- Linf error: $(linf_error)")
  println()

  finalstep = false
  @timeit to "main loop" while !finalstep
    @timeit to "calcdt" dt = calcdt(dg, cfl)

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

    # plot2file(dg, @sprintf("solution_%04d.png", step))
  end
  println("done")
  # plot2file(dg, "solution.pdf")

  print_timer(to, title="jul1dge", allocations=true, linechars=:ascii, compact=false)
  println()
end


function parse_arguments()
  s = ArgParseSettings()
  @add_arg_table s begin
    "-N"
      help = "Polynomial degree"
      arg_type = Int
      default = defaults["N"]
    "--ncells"
      help = "Number of cells"
      arg_type = Int
      default = defaults["ncells"]
    "--nstepsmax"
      help = "Maximum number of time steps"
      arg_type = Int
      default = defaults["nstepsmax"]
    "--syseqn"
      help = "System of equations"
      arg_type = String
      default = defaults["syseqn"]
    "--initialconditions"
      help = "Initial conditions to be applied"
      arg_type = String
      default = defaults["initialconditions"]
    "--sources"
      help = "Source terms"
      arg_type = String
      default = defaults["sources"]
    "--cfl"
      help = "CFL number of time step calculation"
      arg_type = Float64
      default = defaults["cfl"]
  end

  return parse_args(s)
end


function interruptable_main()
  # Allow handling of interrupts by the user
  ccall(:jl_exit_on_sigint, Cvoid, (Cint,), 0)

  try
    main()
  catch e
    isa(e, InterruptException) || rethrow(e)
    println(stderr, "\nExecution interrupted by user (Ctrl-c)")
  end
end


if abspath(PROGRAM_FILE) == @__FILE__
  interruptable_main()
end
