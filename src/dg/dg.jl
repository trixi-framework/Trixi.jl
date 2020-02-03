module DgMod

using ..Jul1dge
using ..Equation: AbstractSysEqn, initialconditions, calcflux, riemann!, sources, maxdt
import ..Equation: nvars # Import to allow method extension
using ..Auxiliary: timer
using StaticArrays: SVector, SMatrix, MMatrix
using GaussQuadrature: legendre, both
using TimerOutputs: @timeit

export Dg
export setinitialconditions
export nvars
export syseqn
export polydeg
export rhs!
export calcdt
export calc_error_norms

struct Dg{SysEqn <: AbstractSysEqn{nvars_} where nvars_, N, Np1, NAna, NAnap1}
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

  analysis_nodes::SVector{NAnap1}
  analysis_weights::SVector{NAnap1}
  analysis_weights_volume::SVector{NAnap1}
  analysis_vandermonde::SMatrix{NAnap1, Np1}
  analysis_total_volume::Float64
end


polydeg(dg::Dg{SysEqn, N}) where {SysEqn, N} = N
syseqn(dg::Dg{SysEqn, N}) where {SysEqn, N} = dg.syseqn
nvars(dg::Dg{SysEqn, N}) where {SysEqn, N} = nvars(syseqn(dg))


function Dg(s::AbstractSysEqn{nvars_}, mesh, N::Int) where nvars_
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

  NAna = 2 * (N + 1) - 1
  analysis_nodes, analysis_weights = legendre(NAna + 1, both)
  analysis_weights_volume = analysis_weights
  analysis_vandermonde = polynomialinterpolationmatrix(nodes, analysis_nodes)
  analysis_total_volume = sum(mesh.length.^ndim)

  dg = Dg{typeof(s), N, N + 1, NAna, NAna + 1}(
      s, u, ut, urk, ncells, Array{Float64,1}(undef, ncells),
      Array{Float64,2}(undef, N + 1, ncells), surfaces, usurf, fsurf,
      neighbors, nsurfaces, nodes, weights, dhat, lhat, analysis_nodes,
      analysis_weights, analysis_weights_volume, analysis_vandermonde,
      analysis_total_volume)

  for cell_id in 1:ncells
    dx = mesh.length[cell_id]
    dg.invjacobian[cell_id] = 2/dx
    dg.nodecoordinate[:, cell_id] = @. mesh.coordinate[cell_id] + dx/2 * nodes[:]
  end

  return dg
end


function calc_error_norms(dg::Dg{SysEqn, N}, t::Float64) where {SysEqn, N}
  s = syseqn(dg)
  nvars_ = nvars(s)
  nnodes_analysis = length(dg.analysis_nodes)

  l2_error = zeros(nvars_)
  linf_error = zeros(nvars_)
  u_exact = zeros(nvars_)

  for cell_id = 1:dg.ncells
    u = interpolate_nodes(dg.u[:, :, cell_id], dg.analysis_vandermonde, nvars_)
    x = interpolate_nodes(reshape(dg.nodecoordinate[:, cell_id], 1, :), dg.analysis_vandermonde, 1)
    jacobian = (1 / dg.invjacobian[cell_id])^ndim
    for i = 1:nnodes_analysis
      u_exact = initialconditions(s, x[i], t)
      diff = similar(u_exact)
      @. diff = u_exact - u[:, i]
      @. l2_error += diff^2 * dg.analysis_weights_volume[i] * jacobian
      @. linf_error = max(linf_error, abs(diff))
    end
  end

  @. l2_error = sqrt(l2_error / dg.analysis_total_volume)

  return l2_error, linf_error
end


function interpolate_nodes(data_in::AbstractArray{T, 2},
                           vandermonde::AbstractArray{T, 2}, nvars_::Integer) where T
  nnodes_out = size(vandermonde, 1)
  nnodes_in = size(vandermonde, 2)
  data_out = zeros(eltype(data_in), nvars_, nnodes_out)

  for i = 1:nnodes_out
    for j = 1:nnodes_in
      for v = 1:nvars_
        data_out[v, i] += vandermonde[i, j] * data_in[v, j]
      end
    end
  end

  return data_out
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


function polynomialinterpolationmatrix(nodes_in, nodes_out)
  nnodes_in = length(nodes_in)
  nnodes_out = length(nodes_out)
  wbary_in = barycentricweights(nodes_in)
  vdm = zeros(nnodes_out, nnodes_in)

  for k = 1:nnodes_out
    match = false
    for j = 1:nnodes_in
      if isapprox(nodes_out[k], nodes_in[j], rtol=eps())
        match = true
        vdm[k, j] = 1
      end
    end

    if match == false
      s = 0.0
      for j = 1:nnodes_in
        t = wbary_in[j] / (nodes_out[k] - nodes_in[j])
        vdm[k, j] = t
        s += t
      end
      for j = 1:nnodes_in
        vdm[k, j] = vdm[k, j] / s
      end
    end
  end

  return vdm
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
    polynomials[i] = wbary[i] / (x - nodes[i])
  end
  total = sum(polynomials)

  for i = 1:nnodes
    polynomials[i] /= total
  end

  return polynomials
end


function setinitialconditions(dg, t)
  s = syseqn(dg)

  for cell_id = 1:dg.ncells
    for i = 1:(polydeg(dg) + 1)
      dg.u[:, i, cell_id] .= initialconditions(s, dg.nodecoordinate[i, cell_id], t)
    end
  end
end

function rhs!(dg, t_stage)
  # Reset ut
  @timeit timer() "reset ut" dg.ut .= 0.0

  # Calculate volume integral
  @timeit timer() "volint" volint!(dg)

  # Prolong solution to surfaces
  @timeit timer() "prolong2surfaces" prolong2surfaces!(dg)

  # Calculate surface fluxes
  @timeit timer() "surfflux!" surfflux!(dg)

  # Calculate surface integrals
  @timeit timer() "surfint!" surfint!(dg)

  # Apply Jacobian from mapping to reference element
  @timeit timer() "applyjacobian" applyjacobian!(dg)

  # Calculate source terms
  @timeit timer() "calcsources" calcsources!(dg, t_stage)
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


function calcsources!(dg, t)
  s = syseqn(dg)
  if s.sources == "none"
    return
  end

  N = polydeg(dg)
  nnodes = N + 1
  nvars_ = nvars(dg)

  for cell_id = 1:dg.ncells
    sources(syseqn(dg), dg.ut, dg.u, dg.nodecoordinate, cell_id, t, nnodes)
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
