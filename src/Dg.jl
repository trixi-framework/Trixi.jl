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
export rhs!
export calcdt

struct Dg{SysEqn <: SysEqnMod.AbstractSysEqn{nvars_} where nvars_, N, Np1}
  syseqn::SysEqn
  u::Array{Float64, 3}
  ut::Array{Float64, 3}
  urk::Array{Float64, 3}
  ncells::Int
  invjacobian::Array{Float64, 1}
  nodecoordinate::Array{Float64, 2}
  surfaces::Array{Int, 2}

  usurf::Array{Float64, 3}
  fsurf::Array{Float64, 2}
  neighbors::Array{Int, 2}
  nsurfaces::Int

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
  urk = zeros(Float64, nvars_, N + 1, ncells)

  nsurfaces = ncells
  usurf = zeros(Float64, 2, nvars_, nsurfaces)
  fsurf = zeros(Float64, nvars_, nsurfaces)

  surfaces = zeros(Int, 2, ncells)
  neighbors = zeros(Int, 2, nsurfaces)
  # Order of cells, surfaces:
  # |---|---|---|
  # s c s c s c s
  # 1 1 2 2 3 3 1
  # Order of adjacent surfaces:
  # 1 --- 2
  # Order of adjacent cells:
  # 1  |  2
  for cell_id = 1:ncells
    surfaces[1, cell_id] = cell_id
    surfaces[2, cell_id] = cell_id + 1
  end
  surfaces[2, ncells] = 1
  for s = 1:nsurfaces
    neighbors[1, s] = s - 1
    neighbors[2, s] = s
  end
  neighbors[1, 1] = ncells

  nodes, weights = legendre(N + 1, both)
  dhat = calcdhat(nodes, weights)
  lhat = zeros(N + 1, 2)
  lhat[:, 1] = calclhat(-1.0, nodes, weights)
  lhat[:, 2] = calclhat( 1.0, nodes, weights)

  dg = Dg{typeof(s), N, N + 1}(s, u, ut, urk, ncells, Array{Float64,1}(undef, ncells),
                        Array{Float64,2}(undef, N + 1, ncells), surfaces, usurf, fsurf, neighbors,
                        nsurfaces, nodes, weights, dhat, lhat)

  for cell_id in 1:ncells
    dx = mesh.length[cell_id]
    dg.invjacobian[cell_id] = 2/dx
    dg.nodecoordinate[:, cell_id] = @. mesh.coordinate[cell_id] + dx/2 * nodes[:]
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


function setinitialconditions(dg, t, name::String)
  for cell_id = 1:dg.ncells
    for i = 1:(polydeg(dg) + 1)
      dg.u[:, i, cell_id] .= exactfunc(syseqn(dg), dg.nodecoordinate[i, cell_id], t, name)
    end
  end
end

function rhs!(dg)
  # Reset ut
  dg.ut .= 0.0

  # Calculate volume integral
  volint!(dg)

  # Prolong solution to surfaces
  prolong2surfaces!(dg)

  # Calculate surface fluxes
  surfflux!(dg)

  # Calculate surface integrals
  surfint!(dg)

  # Apply Jacobian from mapping to reference element
  applyjacobian!(dg)
end


function volint!(dg)
  N = polydeg(dg)
  nnodes = N + 1
  s = syseqn(dg)
  nvars_ = nvars(dg)

  for cell_id = 1:dg.ncells
    f::MMatrix{nvars_, nnodes} = calcflux(s, dg.u, cell_id, nnodes)
    for i = 1:nnodes
      for v = 1:nvars_
        for j = 1:nnodes
          dg.ut[v, i, cell_id] += dg.dhat[i, j] * f[v, j]
        end
      end
    end
  end
end


function prolong2surfaces!(dg)
  N = polydeg(dg)
  nnodes = N + 1
  s = syseqn(dg)
  nvars_ = nvars(dg)

  for s = 1:dg.nsurfaces
    left = dg.neighbors[1, s]
    right = dg.neighbors[2, s]
    for v = 1:nvars_
      dg.usurf[1, v, s] = dg.u[v, nnodes, left]
      dg.usurf[2, v, s] = dg.u[v, 1, right]
    end
  end
end


function surfflux!(dg)
  N = polydeg(dg)
  nnodes = N + 1
  s = syseqn(dg)

  for s = 1:dg.nsurfaces
    riemann!(dg.fsurf, dg.usurf, s, syseqn(dg), nnodes)
  end
end


function surfint!(dg)
  N = polydeg(dg)
  nnodes = N + 1
  nvars_ = nvars(dg)

  for cell_id = 1:dg.ncells
    left = dg.surfaces[1, cell_id]
    right = dg.surfaces[2, cell_id]

    for v = 1:nvars_
      dg.ut[v, 1,      cell_id] -= dg.fsurf[v, left ] * dg.lhat[1,      1]
      dg.ut[v, nnodes, cell_id] += dg.fsurf[v, right] * dg.lhat[nnodes, 2]
    end
  end
end


function applyjacobian!(dg)
  N = polydeg(dg)
  nnodes = N + 1
  nvars_ = nvars(dg)

  for cell_id = 1:dg.ncells
    for i = 1:nnodes
      for v = 1:nvars_
        dg.ut[v, i, cell_id] *= -dg.invjacobian[cell_id]
      end
    end
  end
end


function calcdt(dg, cfl)
  N = polydeg(dg)
  nnodes = N + 1

  mindt = Inf
  for cell_id = 1:dg.ncells
    dt = maxdt(syseqn(dg), dg.u, cell_id, nnodes, dg.invjacobian[cell_id], cfl)
    mindt = min(mindt, dt)
  end

  return mindt
end


end
