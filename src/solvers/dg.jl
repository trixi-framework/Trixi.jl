module DgSolver

include("interpolation.jl")

using ...Jul1dge
using ..Solvers # Use everything to allow method extension via "function <parent_module>.<method>"
using ...Equations: AbstractEquation, initial_conditions, calcflux!, riemann!, sources, calc_max_dt
import ...Equations: nvariables # Import to allow method extension
using ...Auxiliary: timer
using ...Mesh: TreeMesh
using ...Mesh.Trees: leaf_cells, length_at_cell
using .Interpolation: interpolate_nodes, calc_dhat,
                      polynomial_interpolation_matrix, calc_lhat, gauss_lobatto_nodes_weights
using StaticArrays: SVector, SMatrix, MMatrix
using TimerOutputs: @timeit
using Printf: @sprintf, @printf

export Dg
export set_initial_conditions
export nvariables
export equations
export polydeg
export rhs!
export calc_dt
export calc_error_norms
export analyze_solution


# Main DG data structure that contains all relevant data for the DG solver
struct Dg{Eqn <: AbstractEquation{V} where V, N, Np1, NAna, NAnap1} <: AbstractSolver
  equations::Eqn
  u::Array{Float64, 3}
  u_t::Array{Float64, 3}
  u_rungekutta::Array{Float64, 3}
  flux::Array{Float64, 3}
  n_elements::Int
  inverse_jacobian::Array{Float64, 1}
  node_coordinates::Array{Float64, 2}
  surface_ids::Array{Int, 2}

  u_surfaces::Array{Float64, 3}
  flux_surfaces::Array{Float64, 2}
  neighbor_ids::Array{Int, 2}
  n_surfaces::Int

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


# Convenience constructor to create DG solver instance
function Dg(equation::AbstractEquation{V}, mesh::TreeMesh, N::Int) where V
  # Determine number of elements
  leaf_cell_ids = leaf_cells(mesh.tree)
  n_elements = length(leaf_cell_ids)

  # Initialize data structures
  u = zeros(Float64, V, N + 1, n_elements)
  u_t = zeros(Float64, V, N + 1, n_elements)
  u_rungekutta = zeros(Float64, V, N + 1, n_elements)
  flux = zeros(Float64, V, N + 1, n_elements)

  n_surfaces = n_elements
  u_surfaces = zeros(Float64, 2, V, n_surfaces)
  flux_surfaces = zeros(Float64, V, n_surfaces)

  surface_ids = zeros(Int, 2, n_elements)
  neighbor_ids = zeros(Int, 2, n_surfaces)

  # Create surfaces between elements
  # Order of elements, surfaces:
  # |---|---|---|
  # s c s c s c s
  # 1 1 2 2 3 3 1
  # Order of adjacent surfaces:
  # 1 --- 2
  # Order of adjacent elements:
  # 1  |  2
  for element_id = 1:n_elements
    surface_ids[1, element_id] = element_id
    surface_ids[2, element_id] = element_id + 1
  end
  surface_ids[2, n_elements] = 1
  for s = 1:n_surfaces
    neighbor_ids[1, s] = s - 1
    neighbor_ids[2, s] = s
  end
  neighbor_ids[1, 1] = n_elements


  # Initialize interpolation data structures
  nodes, weights = gauss_lobatto_nodes_weights(N + 1)
  dhat = calc_dhat(nodes, weights)
  lhat = zeros(N + 1, 2)
  lhat[:, 1] = calc_lhat(-1.0, nodes, weights)
  lhat[:, 2] = calc_lhat( 1.0, nodes, weights)

  # Initialize data structures for error analysis (by default, we use twice the
  # number of analysis nodes as the normal solution)
  NAna = 2 * (N + 1) - 1
  analysis_nodes, analysis_weights = gauss_lobatto_nodes_weights(NAna + 1)
  analysis_weights_volume = analysis_weights
  analysis_vandermonde = polynomial_interpolation_matrix(nodes, analysis_nodes)
  analysis_total_volume = sum(mesh.tree.length_level_0.^ndim)

  # Create actual DG solver instance
  dg = Dg{typeof(equation), N, N + 1, NAna, NAna + 1}(
      equation, u, u_t, u_rungekutta, flux, n_elements, Array{Float64,1}(undef, n_elements),
      Array{Float64,2}(undef, N + 1, n_elements), surface_ids, u_surfaces, flux_surfaces,
      neighbor_ids, n_surfaces, nodes, weights, dhat, lhat, analysis_nodes,
      analysis_weights, analysis_weights_volume, analysis_vandermonde,
      analysis_total_volume)

  # Calculate inverse Jacobian and node coordinates
  for element_id in 1:n_elements
    cell_id = leaf_cell_ids[element_id]
    dx = length_at_cell(mesh.tree, cell_id)
    dg.inverse_jacobian[element_id] = 2/dx
    dg.node_coordinates[:, element_id] = @. mesh.tree.coordinates[1, cell_id] + dx/2 * nodes[:]
  end

  return dg
end


# Return polynomial degree for a DG solver
@inline polydeg(::Dg{Eqn, N}) where {Eqn, N} = N


# Return number of nodes in one direction
@inline nnodes(::Dg{Eqn, N}) where {Eqn, N} = N + 1


# Return system of equations instance for a DG solver
@inline Solvers.equations(dg::Dg) = dg.equations


# Return number of variables for the system of equations in use
@inline nvariables(dg::Dg) = nvariables(equations(dg))


# Return number of degrees of freedom
@inline Solvers.ndofs(dg::Dg) = dg.n_elements * (polydeg(dg) + 1)^ndim


# Calculate L2/Linf error norms based on "exact solution"
function calc_error_norms(dg::Dg, t::Float64)
  # Gather necessary information
  equation = equations(dg)
  n_nodes_analysis = length(dg.analysis_nodes)

  # Set up data structures
  l2_error = zeros(nvariables(equation))
  linf_error = zeros(nvariables(equation))
  u_exact = zeros(nvariables(equation))

  # Iterate over all elements for error calculations
  for element_id = 1:dg.n_elements
    # Interpolate solution and node locations to analysis nodes
    u = interpolate_nodes(dg.u[:, :, element_id], dg.analysis_vandermonde, nvariables(equation))
    x = interpolate_nodes(reshape(dg.node_coordinates[:, element_id], 1, :),
                          dg.analysis_vandermonde, 1)

    # Calculate errors at each analysis node
    jacobian = (1 / dg.inverse_jacobian[element_id])^ndim
    for i = 1:n_nodes_analysis
      u_exact = initial_conditions(equation, x[i], t)
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
function Solvers.analyze_solution(dg::Dg{Eqn, N}, time::Real, dt::Real,
                                  step::Integer, runtime_absolute::Real,
  runtime_relative::Real) where {Eqn, N}
  equation = equations(dg)

  l2_error, linf_error = calc_error_norms(dg, time)

  println()
  println("-"^80)
  println(" Simulation running '$(equation.name)' with N = $N")
  println("-"^80)
  println(" #timesteps:    " * @sprintf("% 14d", step))
  println(" dt:            " * @sprintf("%10.8e", dt))
  println(" sim. time:     " * @sprintf("%10.8e", time))
  println(" run time:      " * @sprintf("%10.8e s", runtime_absolute))
  println(" Time/DOF/step: " * @sprintf("%10.8e s", runtime_relative))
  print(" Variable:    ")
  for v in 1:nvariables(equation)
    @printf("  %-14s", equation.varnames_cons[v])
  end
  println()
  print(" L2 error:    ")
  for v in 1:nvariables(equation)
    @printf("  %10.8e", l2_error[v])
  end
  println()
  print(" Linf error:  ")
  for v in 1:nvariables(equation)
    @printf("  %10.8e", linf_error[v])
  end
  println()
  println()
end


# Call equation-specific initial conditions functions and apply to all elements
function Solvers.set_initial_conditions(dg::Dg, time::Float64)
  equation = equations(dg)

  for element_id = 1:dg.n_elements
    for i = 1:(polydeg(dg) + 1)
      dg.u[:, i, element_id] .= initial_conditions(equation, dg.node_coordinates[i, element_id],
                                                   time)
    end
  end
end


# Calculate time derivative
function Solvers.rhs!(dg::Dg, t_stage)
  # Reset u_t
  @timeit timer() "reset ∂u/∂t" dg.u_t .= 0.0

  # Calculate volume flux
  @timeit timer() "volume flux" calc_volume_flux!(dg)

  # Calculate volume integral
  @timeit timer() "volume integral" calc_volume_integral!(dg, dg.u_t, dg.dhat, dg.flux)

  # Prolong solution to surfaces
  @timeit timer() "prolong2surfaces" prolong2surfaces!(dg)

  # Calculate surface fluxes
  @timeit timer() "surface flux" calc_surfaces_flux!(dg.flux_surfaces, dg.u_surfaces, dg)

  # Calculate surface integrals
  @timeit timer() "surface integral" calc_surface_integral!(dg, dg.u_t, dg.flux_surfaces, dg.lhat,
                                                            dg.surface_ids)

  # Apply Jacobian from mapping to reference element
  @timeit timer() "Jacobian" apply_jacobian!(dg)

  # Calculate source terms
  @timeit timer() "source terms" calc_sources!(dg, t_stage)
end


# Calculate and store volume fluxes
function calc_volume_flux!(dg)
  N = polydeg(dg)
  equation = equations(dg)

  @inbounds Threads.@threads for element_id = 1:dg.n_elements
     @views calcflux!(dg.flux[:, :, element_id], equation, dg.u, element_id, nnodes(dg))
  end
end


# Calculate volume integral and update u_t
function calc_volume_integral!(dg, u_t::Array{Float64, 3}, dhat::SMatrix, flux::Array{Float64, 3})
  @inbounds Threads.@threads for element_id = 1:dg.n_elements
    for i = 1:nnodes(dg)
      for v = 1:nvariables(dg)
        for j = 1:nnodes(dg)
          u_t[v, i, element_id] += dhat[i, j] * flux[v, j, element_id]
        end
      end
    end
  end
end


# Prolong solution to surfaces (for GL nodes: just a copy)
function prolong2surfaces!(dg)
  equation = equations(dg)

  for s = 1:dg.n_surfaces
    left = dg.neighbor_ids[1, s]
    right = dg.neighbor_ids[2, s]
    for v = 1:nvariables(dg)
      dg.u_surfaces[1, v, s] = dg.u[v, nnodes(dg), left]
      dg.u_surfaces[2, v, s] = dg.u[v, 1, right]
    end
  end
end


# Calculate and store fluxes across surfaces
function calc_surfaces_flux!(flux_surfaces::Array{Float64, 2}, u_surfaces::Array{Float64, 3}, dg)
  @inbounds Threads.@threads for surface_id = 1:dg.n_surfaces
    riemann!(dg.flux_surfaces, dg.u_surfaces, surface_id, equations(dg), nnodes(dg))
  end
end


# Calculate surface integrals and update u_t
function calc_surface_integral!(dg, u_t::Array{Float64, 3}, flux_surfaces::Array{Float64, 2},
                                lhat::SMatrix, surface_ids::Array{Int, 2})
  for element_id = 1:dg.n_elements
    left = surface_ids[1, element_id]
    right = surface_ids[2, element_id]

    for v = 1:nvariables(dg)
      u_t[v, 1,          element_id] -= flux_surfaces[v, left ] * lhat[1,          1]
      u_t[v, nnodes(dg), element_id] += flux_surfaces[v, right] * lhat[nnodes(dg), 2]
    end
  end
end


# Apply Jacobian from mapping to reference element
function apply_jacobian!(dg)
  for element_id = 1:dg.n_elements
    for i = 1:nnodes(dg)
      for v = 1:nvariables(dg)
        dg.u_t[v, i, element_id] *= -dg.inverse_jacobian[element_id]
      end
    end
  end
end


# Calculate source terms and apply them to u_t
function calc_sources!(dg::Dg, t)
  equation = equations(dg)
  if equation.sources == "none"
    return
  end

  for element_id = 1:dg.n_elements
    sources(equations(dg), dg.u_t, dg.u, dg.node_coordinates, element_id, t, nnodes(dg))
  end
end


# Calculate stable time step size
function Solvers.calc_dt(dg::Dg, cfl)
  min_dt = Inf
  for element_id = 1:dg.n_elements
    dt = calc_max_dt(equations(dg), dg.u, element_id, nnodes(dg),
                     dg.inverse_jacobian[element_id], cfl)
    min_dt = min(min_dt, dt)
  end

  return min_dt
end

end # module
