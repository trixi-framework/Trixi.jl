module DgMod

using ..Jul1dge
import ..SysEqnMod
using StaticArrays
using GaussQuadrature

export Dg
export setinitialconditions
export nvars
export syseqn
export polydeg

struct Dg{SysEqn <: SysEqnMod.AbstractSysEqn{nvars_} where nvars_, N, Np1}
  syseqn::SysEqn
  u::Array{Float64, 3}
  ut::Array{Float64, 3}
  ncells::Integer
  invjacobian::Array{Float64, 1}
  nodecoordinate::Array{Float64, 2}
  nodes::SVector{Np1}
  weights::SVector{Np1}
end

polydeg(dg::Dg{SysEqn, N}) where {SysEqn, N} = N
syseqn(dg::Dg{SysEqn, N}) where {SysEqn, N} = dg.syseqn
SysEqnMod.nvars(dg::Dg{SysEqn, N}) where {SysEqn, N} = SysEqnMod.nvars(syseqn(dg))

function Dg(s::SysEqnMod.AbstractSysEqn{nvars_}, mesh, N) where nvars_
  ncells = mesh.ncells
  u = zeros(Float64, nvars_, N + 1, ncells)
  ut = zeros(Float64, nvars_, N + 1, ncells)
  nodes, weights = legendre(N + 1, both)
  dg = Dg{typeof(s), N, N + 1}(s, u, ut, ncells, Array{Float64,1}(undef, ncells),
                        Array{Float64,2}(undef, N + 1, ncells), nodes, weights)

  for c in 1:ncells
    dx = mesh.length[c]
    dg.invjacobian[c] = 2/dx
    dg.nodecoordinate[:, c] = @. mesh.coordinate[c] + dx/2 * nodes[:]
  end

  return dg
end

function setinitialconditions(dg, t)
  for c = 1:dg.ncells
    for i = 1:(polydeg(dg) + 1)
      dg.u[:, i, c] .= exactfunc(syseqn(dg), dg.nodecoordinate[i, c], t)
    end
  end
end

function rhs(dg, mesh, t)
  # Reset ut
  dg.ut .= 0.0

  # Calculate volume integral
  volint(dg)
end

function volint(dg)
  for c = 1:dg.ncells
    for i = 1:polydeg(dg)
      
    end
  end
end

end
