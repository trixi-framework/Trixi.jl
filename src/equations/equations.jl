module Equations

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


# Base type from which all systems of equations types inherit from
abstract type AbstractSysEqn{V} end


# Retrieve number of variables from equation type
nvars(::Type{AbstractSysEqn{V}}) where V = V

# Retrieve number of variables from equation instance
nvars(::AbstractSysEqn{V}) where V = V


# Retrieve name of system of equations
name(s::AbstractSysEqn) = s.name


# Add method to show some information on system of equations
function Base.show(io::IO, s::AbstractSysEqn)
  print("name = $(s.name), nvars = $(nvars(s))")
end


# Create an instance of a system of equation type based on a given name
function getsyseqn(name::String)
  if name == "linearscalaradvection"
    return LinearScalarAdvection()
  elseif name == "euler"
    return Euler()
  else
    error("'$name' does not name a valid system of equations")
  end
end


# Include files with actual implementations for different systems of equations
include("linearscalaradvection.jl")
include("euler.jl")

end
