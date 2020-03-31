module Auxiliary

using ArgParse: ArgParseSettings, @add_arg_table!, parse_args


# Handle command line arguments (if given) or interpret keyword arguments
function get_arguments(args; kwargs...)
  # Handle command line arguments
  if !isnothing(args)
    # If args are given explicitly, parse command line arguments
    args = parse_commandline_arguments(args)
  else
    # Otherwise interpret keyword arguments as command line arguments
    args = Dict{String, Any}()
    for (key, value) in kwargs
      args[string(key)] = value
    end

    # Clean up some of the arguments and provide defaults
    # FIXME: This is redundant to parse_commandline_arguments
    # If filename is a single string, convert it to array
    if !haskey(args, "filename")
      error("no input file was provided")
    end
    if isa(args["filename"], String)
      args["filename"] = [args["filename"]]
    end
    if !haskey(args, "format")
      args["format"] = "vtu"
    end
    if !haskey(args, "verbose")
      args["verbose"] = false
    end
    if !haskey(args, "hide_progress")
      args["hide_progress"] = false
    end
    if !haskey(args, "pvd")
      args["pvd"] = nothing
    end
    if !haskey(args, "output_directory")
      args["output_directory"] = "."
    end
    if !haskey(args, "nvisnodes")
      args["nvisnodes"] = nothing
    end
  end

  return args
end


# Parse command line arguments and return result
function parse_commandline_arguments(args=ARGS)
  # If anything is changed here, it should also be checked at the beginning of run()
  # FIXME: Refactor the code to avoid this redundancy
  s = ArgParseSettings()
  s.autofix_names = true
  @add_arg_table! s begin
    "filename"
      help = "Name of Trixi solution/restart/mesh file to convert to a VTK file."
      arg_type = String
      required = true
      nargs = '+'
    "--format", "-f"
      help = "Output format for solution/restart files. Can be 'vtu' or 'vti'."
      arg_type = String
      default = "vtu"
      range_tester = x -> x in ("vtu", "vti")
    "--verbose", "-v"
      help = "Enable verbose output to avoid despair over long plot times 😉"
      action = :store_true
    "--hide-progress"
      help = "Hide progress bar (will be hidden automatically if `--verbose` is given)"
      action = :store_true
    "--pvd"
      help = ("Use this filename to store PVD file (instead of auto-detecting name). Note that " *
              "only the name will be used (directory and file extension are ignored).")
      arg_type = String
    "--output-directory", "-o"
      help = "Output directory where generated images are stored"
      arg_type = String
      default = "."
    "--nvisnodes"
      help = ("Number of visualization nodes per element "
              * "(default: twice the number of DG nodes). "
              * "A value of zero uses the number of nodes in the DG elements.")
      arg_type = Int
      default = nothing
  end

  return parse_args(args, s)
end


####################################################################################################
# From auxiliary/auxiliary.jl
####################################################################################################
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


# Determine longest common prefix
function longest_common_prefix(strings::AbstractArray)
  # Return early if array is empty
  if isempty(strings)
    return ""
  end

  # Count length of common prefix, by ensuring that all strings are long enough
  # and then comparing the next character
  len = 0
  while all(length.(strings) .> len) && all(getindex.(strings, len+1) .== strings[1][len+1])
    len +=1
  end

  return strings[1][1:len]
end


end # module Auxiliary
