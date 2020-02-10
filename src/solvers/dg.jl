module DgSolver

include("interpolation.jl")

using ...Jul1dge
using ..Solvers # Use everything to allow method extension via "function <parent_module>.<method>"
using ...Equations: AbstractEquation, initialconditions, calcflux, riemann!, sources, maxdt
import ...Equations: nvars # Import to allow method extension
using ...Auxiliary: timer
using ...Mesh.Trees: Tree, leaf_nodes, length_at_node
using .Interpolation: interpolate_nodes, calcdhat,
                      polynomialinterpolationmatrix, calclhat, gausslobatto
using StaticArrays: SVector, SMatrix, MMatrix
using TimerOutputs: @timeit
using Printf: @sprintf, @printf

export Dg
export setinitialconditions
export nvars
export equations
export polydeg
export rhs!
export calcdt
export calc_error_norms
export analyze_solution


# Main DG data structure that contains all relevant data for the DG solver
struct Dg{Eqn <: AbstractEquation{nvars_} where nvars_, N, Np1, NAna, NAnap1} <: AbstractSolver
  equations::Eqn
  u::Array{Float64, 3}
  ut::Array{Float64, 3}
  urk::Array{Float64, 3}
  flux::Array{Float64, 3}
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


# Return polynomial degree for a DG solver
polydeg(::Dg{Eqn, N}) where {Eqn, N} = N


# Return system of equations instance for a DG solver
Solvers.equations(dg::Dg) = dg.equations


# Return number of variables for the system of equations in use
nvars(dg::Dg) = nvars(equations(dg))


# Convenience constructor to create DG solver instance
function Dg(s::AbstractEquation{nvars_}, mesh::Tree, N::Int) where nvars_
  # Determine number of cells
  leaf_node_ids = leaf_nodes(mesh)
  ncells = length(leaf_node_ids)

  # Initialize data structures
  u = zeros(Float64, nvars_, N + 1, ncells)
  ut = zeros(Float64, nvars_, N + 1, ncells)
  urk = zeros(Float64, nvars_, N + 1, ncells)
  flux = zeros(Float64, nvars_, N + 1, ncells)

  nsurfaces = ncells
  usurf = zeros(Float64, 2, nvars_, nsurfaces)
  fsurf = zeros(Float64, nvars_, nsurfaces)

  surfaces = zeros(Int, 2, ncells)
  neighbors = zeros(Int, 2, nsurfaces)

  # Create surfaces between elements
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


  # Initialize interpolation data structures
  nodes, weights = gausslobatto(N + 1)
  dhat = calcdhat(nodes, weights)
  lhat = zeros(N + 1, 2)
  lhat[:, 1] = calclhat(-1.0, nodes, weights)
  lhat[:, 2] = calclhat( 1.0, nodes, weights)

  # Initialize data structures for error analysis (by default, we use twice the
  # number of analysis nodes as the normal solution)
  NAna = 2 * (N + 1) - 1
  analysis_nodes, analysis_weights = gausslobatto(NAna + 1)
  analysis_weights_volume = analysis_weights
  analysis_vandermonde = polynomialinterpolationmatrix(nodes, analysis_nodes)
  analysis_total_volume = sum(mesh.length_level_0.^ndim)

  # Create actual DG solver instance
  dg = Dg{typeof(s), N, N + 1, NAna, NAna + 1}(
      s, u, ut, urk, flux, ncells, Array{Float64,1}(undef, ncells),
      Array{Float64,2}(undef, N + 1, ncells), surfaces, usurf, fsurf,
      neighbors, nsurfaces, nodes, weights, dhat, lhat, analysis_nodes,
      analysis_weights, analysis_weights_volume, analysis_vandermonde,
      analysis_total_volume)

  # Calculate inverse Jacobian and node coordinates
  for cell_id in 1:ncells
    node_id = leaf_node_ids[cell_id]
    dx = length_at_node(mesh, node_id)
    dg.invjacobian[cell_id] = 2/dx
    dg.nodecoordinate[:, cell_id] = @. mesh.coordinates[1, node_id] + dx/2 * nodes[:]
  end

  return dg
end


# Calculate L2/Linf error norms based on "exact solution"
function calc_error_norms(dg::Dg, t::Float64)
  # Gather necessary information
  s = equations(dg)
  nvars_ = nvars(s)
  nnodes_analysis = length(dg.analysis_nodes)

  # Set up data structures
  l2_error = zeros(nvars_)
  linf_error = zeros(nvars_)
  u_exact = zeros(nvars_)

  # Iterate over all cells for error calculations
  for cell_id = 1:dg.ncells
    # Interpolate solution and node locations to analysis nodes
    u = interpolate_nodes(dg.u[:, :, cell_id], dg.analysis_vandermonde, nvars_)
    x = interpolate_nodes(reshape(dg.nodecoordinate[:, cell_id], 1, :), dg.analysis_vandermonde, 1)

    # Calculate errors at each analysis node
    jacobian = (1 / dg.invjacobian[cell_id])^ndim
    for i = 1:nnodes_analysis
      u_exact = initialconditions(s, x[i], t)
      diff = similar(u_exact)
      @. diff = u_exact - u[:, i]
      @. l2_error += diff^2 * dg.analysis_weights_volume[i] * jacobian
      @. linf_error = max(linf_error, abs(diff))
    end
  end

  # For L2 error, divide by total volume
  @. l2_error = sqrt(l2_error / dg.analysis_total_volume)

  return l2_error, linf_error
end


# Calculate error norms and print information for user
function Solvers.analyze_solution(dg::Dg{Eqn, N}, t::Real, dt::Real,
                                  step::Integer, runtime_absolute::Real,
  runtime_relative::Real) where {Eqn, N}
  s = equations(dg)
  nvars_ = nvars(s)

  l2_error, linf_error = calc_error_norms(dg, t)

  println()
  println("-"^80)
  println(" Simulation running '$(s.name)' with N = $N")
  println("-"^80)
  println(" #timesteps:    " * @sprintf("% 14d", step))
  println(" dt:            " * @sprintf("%10.8e", dt))
  println(" run time:      " * @sprintf("%10.8e s", runtime_absolute))
  println(" Time/DOF/step: " * @sprintf("%10.8e s", runtime_relative))
  print(" Variable:    ")
  for v in 1:nvars_
    @printf("  %-14s", s.varnames_cons[v])
  end
  println()
  print(" L2 error:    ")
  for v in 1:nvars_
    @printf("  %10.8e", l2_error[v])
  end
  println()
  print(" Linf error:  ")
  for v in 1:nvars_
    @printf("  %10.8e", linf_error[v])
  end
  println()
  println()
end


# Call equation-specific initial conditions functions and apply to all cells
function Solvers.setinitialconditions(dg::Dg, t)
  s = equations(dg)

  for cell_id = 1:dg.ncells
    for i = 1:(polydeg(dg) + 1)
      dg.u[:, i, cell_id] .= initialconditions(s, dg.nodecoordinate[i, cell_id], t)
    end
  end
end


# Calculate time derivative
function Solvers.rhs!(dg::Dg, t_stage)
  # Reset ut
  @timeit timer() "reset ut" dg.ut .= 0.0

  # Calculate volume flux
  @timeit timer() "volflux" volflux!(dg)

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


# Calculate and store volume fluxes
function volflux!(dg)
  N = polydeg(dg)
  nnodes = N + 1
  s = equations(dg)

  @inbounds Threads.@threads for cell_id = 1:dg.ncells
    dg.flux[:, :, cell_id] = calcflux(s, dg.u, cell_id, nnodes)
  end
end


# Calculate volume integral and update u_t
function volint!(dg)
  N = polydeg(dg)
  nnodes = N + 1
  nvars_ = nvars(dg)

  @inbounds Threads.@threads for cell_id = 1:dg.ncells
    for i = 1:nnodes
      for v = 1:nvars_
        for j = 1:nnodes
          dg.ut[v, i, cell_id] += dg.dhat[i, j] * dg.flux[v, j, cell_id]
        end
      end
    end
  end
end


# Prolong solution to surfaces (for GL nodes: just a copy)
function prolong2surfaces!(dg)
  N = polydeg(dg)
  nnodes = N + 1
  s = equations(dg)
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


# Calculate and store fluxes across surfaces
function surfflux!(dg)
  N = polydeg(dg)
  nnodes = N + 1
  s = equations(dg)

  @inbounds Threads.@threads for s = 1:dg.nsurfaces
    riemann!(dg.fsurf, dg.usurf, s, equations(dg), nnodes)
  end
end


# Calculate surface integrals and update u_t
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


# Apply Jacobian from mapping to reference element
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


# Calculate source terms and apply them to u_t
function calcsources!(dg::Dg, t)
  s = equations(dg)
  if s.sources == "none"
    return
  end

  N = polydeg(dg)
  nnodes = N + 1
  nvars_ = nvars(dg)

  for cell_id = 1:dg.ncells
    sources(equations(dg), dg.ut, dg.u, dg.nodecoordinate, cell_id, t, nnodes)
  end
end


# Calculate stable time step size
function Solvers.calcdt(dg::Dg, cfl)
  N = polydeg(dg)
  nnodes = N + 1

  mindt = Inf
  for cell_id = 1:dg.ncells
    dt = maxdt(equations(dg), dg.u, cell_id, nnodes, dg.invjacobian[cell_id], cfl)
    mindt = min(mindt, dt)
  end

  return mindt
end

end # module
