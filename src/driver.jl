#!/usr/bin/env julia

include("Jul1dge.jl")
using .Jul1dge

using ArgParse
using Printf

defaults = Dict(
# Computational domain
"x_start" => -10,
"x_end" => 10,

# Number of cells
"ncells" => 80,

# Polynomial degree
"N" => 3,

# Start and end time
"t_start" => 0.0,
"t_end" => 30.0,

# CFL number
"cfl" => 1.0,

# Maximum number of timesteps
"nstepsmax" => 10000,

# Advection velocity
"advectionvelocity" => 1.0,

# Initial conditions
"initialconditions" => "constant"
)

function main()
  # Parse command line arguments
  args = parse_arguments()

  # Store repeatedly used values
  N = args["N"]
  ncells = args["ncells"]
  cfl = args["cfl"]
  nstepsmax = args["nstepsmax"]
  initialconditions = args["initialconditions"]
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
  # syseqn = getsyseqn("linearscalaradvection", defaults["advectionvelocity"])
  syseqn = getsyseqn("euler")
  println("done")

  # Initialize solver
  print("Initializing solver... ")
  dg = Dg(syseqn, mesh, N)
  println("done")

  # Apply initial condition
  print("Applying initial conditions... ")
  t = t_start
  setinitialconditions(dg, t, initialconditions)
  #plot2file(dg, "initialconditions.pdf")
  println("done")

  # Main loop
  println("Running main loop... ")
  step = 0
  finalstep = false
  while !finalstep
    dt = calcdt(dg, cfl)

    if t + dt > t_end
      dt = t_end - t
      finalstep = true
    end

    timestep!(dg, dt)
    step += 1
    t += dt

    if step == nstepsmax
      finalstep = true
    end

    if step % 10 == 0 || finalstep
      println("Step: #$step, t=$t")
    end

    plot2file(dg, @sprintf("solution_%04d.png", step))
  end
  println("done")
  # plot2file(dg, "solution.pdf")
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
    "--initialconditions"
      help = "Initial conditions to be applied"
      arg_type = String
      default = defaults["initialconditions"]
    "--cfl"
      help = "CFL number of time step calculation"
      arg_type = Float64
      default = defaults["cfl"]
  end

  return parse_args(s)
end


if abspath(PROGRAM_FILE) == @__FILE__
  main()
end
