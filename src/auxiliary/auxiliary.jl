include("containers.jl")
include("math.jl")


# Store main timer for global timing of functions
const main_timer = TimerOutput()

# Always call timer() to hide implementation details
timer() = main_timer

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
      error("requested paramter '$name' does not exist and no default value was provided")
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
