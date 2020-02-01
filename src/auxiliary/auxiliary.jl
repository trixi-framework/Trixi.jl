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

function parameter(name::String, default=nothing)
  if haskey(parameters["default"], name)
    return parameters["default"][name]
  else
    if default == nothing
      error("requested paramter '$name' does not exist and no default value was provided")
    else
      return default
    end
  end
end

end
