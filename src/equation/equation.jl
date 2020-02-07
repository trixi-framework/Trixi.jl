module Equation

using ..Jul1dge
using ..Auxiliary: parameter
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
export cons2prim


abstract type AbstractSysEqn{V} end
nvars(::Type{AbstractSysEqn{V}}) where V = V
nvars(::AbstractSysEqn{V}) where V = V
name(s::AbstractSysEqn) = s.name

function Base.show(io::IO, s::AbstractSysEqn)
  print("name = $(s.name), nvars = $(nvars(s))")
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
