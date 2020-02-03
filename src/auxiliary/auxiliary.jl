module Auxiliary

using ArgParse
using TimerOutputs
import Pkg
using Pkg.TOML

export timer
export parse_parameters_file
export parameter
export parse_commandline_arguments
export interruptable

const main_timer = TimerOutput()

const parameters = Dict()

timer() = main_timer

function parse_parameters_file(filename::AbstractString)
  parameters["default"] = Pkg.TOML.parsefile(filename)
end

function parameter(name::String, default=nothing; valid=nothing)
  if haskey(parameters["default"], name)
    value = parameters["default"][name]
  else
    if default == nothing
      error("requested paramter '$name' does not exist and no default value was provided")
    else
      value = default
    end
  end

  if valid != nothing
    if !(value in valid)
      error("'$value' is not a valid value for parameter '$name'")
    end
  end

  return value
end


function parse_commandline_arguments()
  s = ArgParseSettings()
  @add_arg_table s begin
    "--parameters-file", "-p"
      help = "Name of file with runtime parameters"
      arg_type = String
      default = "parameters.toml"
  end

  return parse_args(s)
end


"""
    interruptable(ex)

On Unix-like operating systems, gracefully handle user interrupts (SIGINT), also known as
Ctrl-c, while evaluation expression `ex`.
"""
macro interruptable(ex)
  @static Sys.isunix() && quote
    ccall(:jl_exit_on_sigint, Cvoid, (Cint,), 0)

    try
      # Try to run code
      $(esc(ex))
    catch e
      # Only catch interrupt exceptions and end with a nice message
      isa(e, InterruptException) || rethrow(e)
      println(stderr, "\nExecution interrupted by user (Ctrl-c)")
    end

    # Disable interrupt handling again
    ccall(:jl_exit_on_sigint, Cvoid, (Cint,), 1)
  end
end


end
