module SysEqnMod

using ..Jul1dge
using StaticArrays
import Base.show

export getsyseqn
export nvars
export AbstractSysEqn
export exactfunc

abstract type AbstractSysEqn{nvars_} end
nvars(s::AbstractSysEqn{nvars_}) where nvars_ = nvars_
name(s::AbstractSysEqn{nvars_}) where nvars_ = s.name
function Base.show(io::IO, s::AbstractSysEqn{nvars_}) where nvars_
  print("name = $(s.name), nvars = $nvars_, advectionvelocity = $(s.advectionvelocity)")
end

function getsyseqn(name::String, args...)
  if name == "linearscalaradvection"
    return LinearScalarAdvection(name, args...)
  else
    die("'$name' does not name a valid system of equations")
  end
end


####################################################################################################
# Linear scalar advection
####################################################################################################
struct LinearScalarAdvection <: AbstractSysEqn{1}
  name::String
  advectionvelocity::SVector{ndim}
end

function exactfunc(s::LinearScalarAdvection, x, t)
  return exp(-(x - s.advectionvelocity[1])^2)
end

end
