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
  lhat::SMatrix{Np1, 2}
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
  lhat = zeros(N + 1, 2)
  lhat[:, 1] = calclhat(-1.0, nodes, weights)
  lhat[:, 2] = calclhat( 1.0, nodes, weights)
  dg = Dg{typeof(s), N, N + 1}(s, u, ut, ncells, Array{Float64,1}(undef, ncells),
                        Array{Float64,2}(undef, N + 1, ncells), nodes, weights, dhat, lhat)

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


function calclhat(x::Float64, nodes, weights)
  nnodes = length(nodes)
  wbary = barycentricweights(nodes)

  lhat = lagrangeinterpolatingpolynomials(x, nodes, wbary)

  for i = 1:nnodes
    lhat[i] /= weights[i]
  end

  return lhat
end


function lagrangeinterpolatingpolynomials(x::Float64, nodes, wbary)
  nnodes = length(nodes)
  polynomials = zeros(nnodes)

  for i = 1:nnodes
    if isapprox(x, nodes[i], rtol=eps(x))
      polynomials[i] = 1
      return polynomials
    end
  end

  for i = 1:nnodes
    polynomials[j] = wBary[j] / (x - nodes[j])
  end
  total = sum(polynomials)

  for i = 1:nnodes
    polynomials[j] /= total
  end

  return polynomials
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
