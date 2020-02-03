module Equation

using ..Jul1dge
using StaticArrays: SVector, MVector, MMatrix
import Base.show

export getsyseqn
export nvars
export AbstractSysEqn
export initialconditions
export sources
export calcflux
export riemann!
export maxdt

abstract type AbstractSysEqn{nvars_} end
nvars(s::AbstractSysEqn{nvars_}) where nvars_ = nvars_
name(s::AbstractSysEqn{nvars_}) where nvars_ = s.name
function Base.show(io::IO, s::AbstractSysEqn{nvars_}) where nvars_
  print("name = $(s.name), nvars = $nvars_, advectionvelocity = $(s.advectionvelocity)")
end

function getsyseqn(name::String, initialconditions::String, sources::String, args...)
  if name == "linearscalaradvection"
    return LinearScalarAdvection(initialconditions, sources, args...)
  elseif name == "euler"
    return Euler(initialconditions, sources, args...)
  else
    error("'$name' does not name a valid system of equations")
  end
end

include("linearscalaradvection.jl")
include("euler.jl")

end
