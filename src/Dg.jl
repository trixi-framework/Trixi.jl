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
  dhat::SMatrix{Np1, Np1}
end


polydeg(dg::Dg{SysEqn, N}) where {SysEqn, N} = N
syseqn(dg::Dg{SysEqn, N}) where {SysEqn, N} = dg.syseqn
SysEqnMod.nvars(dg::Dg{SysEqn, N}) where {SysEqn, N} = SysEqnMod.nvars(syseqn(dg))


function Dg(s::SysEqnMod.AbstractSysEqn{nvars_}, mesh, N) where nvars_
  ncells = mesh.ncells
  u = zeros(Float64, nvars_, N + 1, ncells)
  ut = zeros(Float64, nvars_, N + 1, ncells)
  nodes, weights = legendre(N + 1, both)
  dhat = calcdhat(nodes, weights)
  dg = Dg{typeof(s), N, N + 1}(s, u, ut, ncells, Array{Float64,1}(undef, ncells),
                        Array{Float64,2}(undef, N + 1, ncells), nodes, weights, dhat)

  for c in 1:ncells
    dx = mesh.length[c]
    dg.invjacobian[c] = 2/dx
    dg.nodecoordinate[:, c] = @. mesh.coordinate[c] + dx/2 * nodes[:]
  end

  return dg
end


function calcdhat(nodes, weights)
  nnodes = length(nodes)
  dhat = polynomialderivativematrix(nodes)
  dhat = transpose(dhat)

  for n = 1:nnodes, j = 1:nnodes
    dhat[j, n] *= -weights[n] / weights[j]
  end

  return dhat
end


function polynomialderivativematrix(nodes)
  nnodes = length(nodes)
  d = zeros(nnodes, nnodes)
  wbary = barycentricweights(nodes)

  for i = 1:nnodes, j = 1:nnodes
    if j != i
      d[i, j] = wbary[j] / wbary[i] * 1 / (nodes[i] - nodes[j])
      d[i, i] -= d[i, j]
    end
  end

  return d
end


function barycentricweights(nodes)
  nnodes = length(nodes)
  weights = ones(nnodes)

  for j = 2:nnodes, k = 1:(j-1)
    weights[k] *= nodes[k] - nodes[j]
    weights[j] *= nodes[j] - nodes[k]
  end

  for j = 1:nnodes
    weights[j] = 1 / weights[j]
  end

  return weights
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
