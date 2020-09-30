include("containers.jl")
include("math.jl")


# Enable debug timings `@timeit_debug timer() "name" stuff...`.
# This allows us to disable timings completely by executing
# `TimerOutputs.disable_debug_timings(Trixi)`
# and to enable them again by executing
# `TimerOutputs.enable_debug_timings(Trixi)`
timeit_debug_enabled() = true

# Store main timer for global timing of functions
const main_timer = TimerOutput()

# Always call timer() to hide implementation details
timer() = main_timer


"""
    PerformanceCounter

A `PerformanceCounter` be used to track the runtime performance of some calls.
Add a new runtime measurement via `put!(counter, runtime)` and get the averaged
runtime of all measurements added so far via `take!(counter)`, resetting the
`counter`.
"""
mutable struct PerformanceCounter
  ncalls_since_readout::Int
  runtime::Float64
end

PerformanceCounter() = PerformanceCounter(0, 0.0)

function Base.take!(counter::PerformanceCounter)
  time_per_call = counter.runtime / counter.ncalls_since_readout
  counter.ncalls_since_readout = 0
  counter.runtime = 0.0
  return time_per_call
end

function Base.put!(counter::PerformanceCounter, runtime::Real)
  counter.ncalls_since_readout += 1
  counter.runtime += runtime
end


# Initialize top-level parameters structure for program-wide parameters
const parameters = Dict{Symbol,Any}()


# Parse parameters file into global dict
function parse_parameters_file(filename)
  parameters[:default] = parsefile(filename)
  parameters[:default]["parameters_file"] = filename
end


# Return parameter by name, optionally taking a default value and a range of valid values.
#
# If no default value is specified, the parameter is required and the program
# stops if the parameter was not found. The range of valid parameters is used
# to restrict parameters to sane values.
function parameter(name, default=nothing; valid=nothing)
  if haskey(parameters[:default], name)
    # If parameter exists, use its value
    value = parameters[:default][name]
  else
    # Otherwise check whether a default is given and abort if not
    if default === nothing
      error("requested parameter '$name' does not exist and no default value was provided")
    else
      value = default
    end
  end

  # If a range of valid values has been specified, check parameter value against it
  if valid !== nothing
    if !(value in valid)
      error("'$value' is not a valid value for parameter '$name' (valid: $valid)")
    end
  end

  return value
end

"""
    examples_dir()

Return the directory where the example files provided with Trixi.jl are located. If Trixi is
installed as a regular package (with `]add Trixi`), these files are read-only and should *not* be
modified. To find out which files are available, use, e.g., `readdir`:

# Examples
```julia
julia> readdir(examples_dir())
4-element Array{String,1}:
 "2d"
 "3d"
 "README.md"
 "paper-self-gravitating-gas-dynamics"
```
"""
examples_dir() = joinpath(pathof(Trixi) |> dirname |> dirname, "examples")


"""
    get_examples()

Return a list of all example parameter files that are provided by Trixi.
"""
function get_examples()
  examples = String[]
  for (root, dirs, files) in walkdir(examples_dir())
    for f in files
      if endswith(f, ".toml")
        push!(examples, joinpath(root, f))
      end
    end
  end

  return examples
end


"""
    default_example()

Return the path to an example parameter file that can be used to quickly see Trixi in action.
"""
default_example() = joinpath(examples_dir(), "2d", "parameters.toml")


"""
    setparameter(name::String, value)

Set parameter with the specified `name` to the specified `value`.
"""
function setparameter(name::String, value)
  parameters[:default][name] = value
end

# Return true if parameter exists.
parameter_exists(name::String) = haskey(parameters[:default], name)


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


"""
    get_name(x)

Returns a name of `x` ready for pretty printing.
By default, return `string(y)` if `x isa Val{y}` and return `string(x)` otherwise.

# Examples

```jldoctest
julia> Trixi.get_name("test")
"test"

julia> Trixi.get_name(Val(:test))
"test"
```
"""
get_name(x) = string(x)
get_name(::Val{x}) where x = string(x)
