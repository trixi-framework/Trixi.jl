module Auxiliary

using TimerOutputs
import Pkg
using Pkg.TOML

export to
export parse_parameters_file
export parameter

const to = TimerOutput()

const parameters = Dict()

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

end
