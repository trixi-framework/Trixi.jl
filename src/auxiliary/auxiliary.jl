module Auxiliary

include("containers.jl")

using ArgParse: ArgParseSettings, @add_arg_table, parse_args
using TimerOutputs: TimerOutput
using Pkg.TOML: parsefile

export timer
export parse_parameters_file
export parameter
export parse_commandline_arguments
export interruptable


# Store main timer for global timing of functions
const main_timer = TimerOutput()

# Always call timer() to hide implementation details
timer() = main_timer

# Initialize top-level parameters structure for program-wide parameters
const parameters = Dict()


# Parse parameters file into global dict
function parse_parameters_file(filename::AbstractString)
  parameters["default"] = parsefile(filename)
end


# Return parameter by name, optionally taking a default value and a range of valid values.
#
# If no default value is specified, the parameter is required and the program
# stops if the parameter was not found. The range of valid parameters is used
# to restrict parameters to sane values.
function parameter(name::String, default=nothing; valid=nothing)
  if haskey(parameters["default"], name)
    # If parameter exists, use its value
    value = parameters["default"][name]
  else
    # Otherwise check whether a default is given and abort if not
    if default == nothing
      error("requested paramter '$name' does not exist and no default value was provided")
    else
      value = default
    end
  end

  # If a range of valid values has been specified, check parameter value against it
  if valid != nothing
    if !(value in valid)
      error("'$value' is not a valid value for parameter '$name' (valid: $valid)")
    end
  end

  return value
end

# Return true if parameter exists.
parameter_exists(name::String) = haskey(parameters["default"], name)


# Parse command line arguments and return as dict
function parse_commandline_arguments(args=ARGS)
  s = ArgParseSettings()
  @add_arg_table s begin
    "--parameters-file", "-p"
      help = "Name of file with runtime parameters."
      arg_type = String
      dest_name = "parameters_file"
      default = "parameters.toml"
  end

  return parse_args(args, s)
end


# Allow an expression to be terminated gracefully by Ctrl-c.
#
# On Unix-like operating systems, gracefully handle user interrupts (SIGINT), also known as
# Ctrl-c, while evaluation expression `ex`.
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


# Print informative message at startup
function print_startup_message()
  s = """

    ████████╗██████╗ ██╗██╗  ██╗██╗
    ╚══██╔══╝██╔══██╗██║╚██╗██╔╝██║
       ██║   ██████╔╝██║ ╚███╔╝ ██║
       ██║   ██╔══██╗██║ ██╔██╗ ██║
       ██║   ██║  ██║██║██╔╝ ██╗██║
       ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝
    """
  println(s)
end


end
