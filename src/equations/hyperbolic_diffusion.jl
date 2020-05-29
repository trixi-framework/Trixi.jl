
@doc raw"""
    HyperbolicDiffusionEquations

The linear hyperbolic diffusion equations in two space dimensions.
A description of this system can be found in Sec. 2.5 of the book "I Do Like CFD, Too: Vol 1".
""" #TODO: DOI or something similar
struct HyperbolicDiffusionEquations <: AbstractEquation{3}
  Lr::Float64
  Tr::Float64
  nu::Float64
  resid_tol::Float64
end

function HyperbolicDiffusionEquations()
  # diffusion coefficient
  nu = parameter("nu", 1.0)
  # relaxation length scale
  Lr = parameter("Lr", 1.0/(2.0*pi))
  # relaxation time
  Tr = Lr*Lr/nu
  # stopping tolerance for the pseudotime "steady-state"
  resid_tol = parameter("resid_tol", 1e-12)
  HyperbolicDiffusionEquations(Lr, Tr, nu, resid_tol)
end


get_name(::HyperbolicDiffusionEquations) = "HyperbolicDiffusionEquations"
varnames_cons(::HyperbolicDiffusionEquations) = @SVector ["phi", "p", "q"]
varnames_prim(::HyperbolicDiffusionEquations) = @SVector ["phi", "p", "q"]


# Set initial conditions at physical location `x` for pseudo-time `t`
function initial_conditions_poisson_periodic(equation::HyperbolicDiffusionEquations, x, t)
  # elliptic equation: -νΔϕ = f
  # depending on initial constant state, c, for phi this converges to the solution ϕ + c
  if iszero(t)
    phi = 0.0
    p   = 0.0
    q   = 0.0
  else
    phi = sin(2.0*pi*x[1])*sin(2.0*pi*x[2])
    p   = 2*pi*cos(2.0*pi*x[1])*sin(2.0*pi*x[2])
    q   = 2*pi*sin(2.0*pi*x[1])*cos(2.0*pi*x[2])
  end
  return @SVector [phi, p, q]
end

function initial_conditions_poisson_nonperiodic(equation::HyperbolicDiffusionEquations, x, t)
  # elliptic equation: -νΔϕ = f
  if t == 0.0
    phi = 1.0
    p   = 1.0
    q   = 1.0
  else
    phi = 2.0*cos(pi*x[1])*sin(2.0*pi*x[2]) + 2.0 # ϕ
    p   = -2.0*pi*sin(pi*x[1])*sin(2.0*pi*x[2])   # ϕ_x
    q   = 4.0*pi*cos(pi*x[1])*cos(2.0*pi*x[2])    # ϕ_y
  end
  return @SVector [phi, p, q]
end

function initial_conditions_harmonic_nonperiodic(equation::HyperbolicDiffusionEquations, x, t)
  # elliptic equation: -νΔϕ = f
  if t == 0.0
    phi = 1.0
    p   = 1.0
    q   = 1.0
  else
    C   = 1.0/sinh(pi)
    phi = C*(sinh(pi*x[1])*sin(pi*x[2]) + sinh(pi*x[2])*sin(pi*x[1]))
    p   = C*pi*(cosh(pi*x[1])*sin(pi*x[2]) + sinh(pi*x[2])*cos(pi*x[1]))
    q   = C*pi*(sinh(pi*x[1])*cos(pi*x[2]) + cosh(pi*x[2])*sin(pi*x[1]))
  end
  return @SVector [phi, p, q]
end


# Apply source terms
function source_terms_poisson_periodic(equation::HyperbolicDiffusionEquations, ut, u, x, element_id, t, n_nodes)
  # elliptic equation: -νΔϕ = f
  # analytical solution: phi = sin(2πx)*sin(2πy) and f = -8νπ^2 sin(2πx)*sin(2πy)
  inv_Tr = inv(equation.Tr)
  C = -8.0*equation.nu*pi*pi

  for j in 1:n_nodes
    for i in 1:n_nodes
      x1 = x[1, i, j, element_id]
      x2 = x[2, i, j, element_id]
      tmp1 = sin(2.0*pi*x1)
      tmp2 = sin(2.0*pi*x2)
      ut[1, i, j, element_id] -= C*tmp1*tmp2
      ut[2, i, j, element_id] -= inv_Tr * u[2, i, j, element_id]
      ut[3, i, j, element_id] -= inv_Tr * u[3, i, j, element_id]
    end
  end

  return nothing
end

function source_terms_poisson_nonperiodic(equation::HyperbolicDiffusionEquations, ut, u, x, element_id, t, n_nodes)
  # elliptic equation: -νΔϕ = f
  # analytical solution: ϕ = 2cos(πx)sin(2πy) + 2 and f = 10π^2cos(πx)sin(2πy)
  inv_Tr = inv(equation.Tr)

  for j in 1:n_nodes
    for i in 1:n_nodes
      x1 = x[1, i, j, element_id]
      x2 = x[2, i, j, element_id]
      ut[1, i, j, element_id] += 10 * pi^2 * cos(pi*x1) * sin(2.0*pi*x2)
      ut[2, i, j, element_id] -= inv_Tr * u[2, i, j, element_id]
      ut[3, i, j, element_id] -= inv_Tr * u[3, i, j, element_id]
    end
  end

  return nothing
end

function source_terms_harmonic(equation::HyperbolicDiffusionEquations, ut, u, x, element_id, t, n_nodes)
  # harmonic solution ϕ = (sinh(πx)sin(πy) + sinh(πy)sin(πx))/sinh(π), so f = 0
  inv_Tr = inv(equation.Tr)

  for j in 1:n_nodes
    for i in 1:n_nodes
      ut[2, i, j, element_id] -= inv_Tr * u[2, i, j, element_id]
      ut[3, i, j, element_id] -= inv_Tr * u[3, i, j, element_id]
    end
  end

  return nothing
end


# Calculate 2D flux (element version)
@inline function calcflux!(f1::AbstractArray{Float64},
                           f2::AbstractArray{Float64},
                           equation::HyperbolicDiffusionEquations,
                           u::AbstractArray{Float64}, element_id::Int,
                           n_nodes::Int)
  for j = 1:n_nodes
    for i = 1:n_nodes
      phi = u[1, i, j, element_id]
      p   = u[2, i, j, element_id]
      q   = u[3, i, j, element_id]
      @views calcflux!(f1[:, i, j], f2[:, i, j], equation, phi, p, q)
    end
  end
end


# Calculate 2D flux (pointwise version)
@inline function calcflux!(f1::AbstractArray{Float64},
                           f2::AbstractArray{Float64},
                           equation::HyperbolicDiffusionEquations,
                           phi::Float64, p::Float64, q::Float64)
  f1[1]  = -equation.nu*p
  f1[2]  = -phi/equation.Tr
  f1[3]  = 0.0

  f2[1]  = -equation.nu*q
  f2[2]  = 0.0
  f2[3]  = -phi/equation.Tr

  return nothing
end


# Calculate 2D two-point flux (element version)
@inline function calcflux_twopoint!(f1, f2, f1_diag, f2_diag,
                                    volume_flux, equation::HyperbolicDiffusionEquations, u, element_id, n_nodes)
  # Calculate regular volume fluxes
  calcflux!(f1_diag, f2_diag, equation, u, element_id, n_nodes)

  for j = 1:n_nodes
    for i = 1:n_nodes
      # Set diagonal entries (= regular volume fluxes due to consistency)
      for v in 1:nvariables(equation)
        f1[v, i, i, j] = f1_diag[v, i, j]
        f2[v, j, i, j] = f2_diag[v, i, j]
      end

      # Flux in x-direction
      for l = i + 1:n_nodes
        flux = volume_flux(equation, 1, # 1-> x-direction
                           u[1, i, j, element_id], u[2, i, j, element_id], u[3, i, j, element_id],
                           u[1, l, j, element_id], u[2, l, j, element_id], u[3, l, j, element_id])
        for v in 1:nvariables(equation)
          f1[v, i, l, j] = f1[v, l, i, j] = flux[v]
        end
      end

      # Flux in y-direction
      for l = j + 1:n_nodes
        flux = volume_flux(equation, 2, # 2 -> y-direction
                           u[1, i, j, element_id], u[2, i, j, element_id], u[3, i, j, element_id],
                           u[1, i, l, element_id], u[2, i, l, element_id], u[3, i, l, element_id])
        for v in 1:nvariables(equation)
          f2[v, j, i, l] = f2[v, l, i, j] = flux[v]
        end
      end
    end
  end
end


# Central two-point flux (identical to weak form volume integral, except for floating point errors)
function flux_central(equation::HyperbolicDiffusionEquations, orientation,
                      phi_ll, p_ll, q_ll,
                      phi_rr, p_rr, q_rr)
  # Calculate regular 1D fluxes
  f_ll = MVector{3, Float64}(undef)
  f_rr = MVector{3, Float64}(undef)
  calcflux1D!(f_ll, equation, phi_ll, p_ll, q_ll, orientation)
  calcflux1D!(f_rr, equation, phi_rr, p_rr, q_rr, orientation)

  # Average regular fluxes
  @. 0.5 * (f_ll + f_rr)
end


# Calculate 1D flux in for a single point
@inline function calcflux1D!(f, equation::HyperbolicDiffusionEquations,
                             phi, p, q, orientation)
  if orientation == 1
    f[1]  = -equation.nu*p
    f[2]  = -phi/equation.Tr
    f[3]  = 0.0
  else
    f[1]  = -equation.nu*q
    f[2]  = 0.0
    f[3]  = -phi/equation.Tr
  end

  return nothing
end


# Calculate flux across interface with different states on both sides (EC mortar version)
# - `destination::AbstractArray{T,3} where T<:Real`:
#   The array of surface flux values (updated inplace).
# - `surface_flux`:
#   The surface flux as a function.
# - `u_surfaces_left::AbstractArray{T,3} where T<:Real``
# - `u_surfaces_right::AbstractArray{T,3} where T<:Real``
# - `surface_id::Integer`
# - `equation::AbstractEquations`
# - `n_nodes::Integer`
# - `orientations::Vector{T} where T<:Integer`
# See equations.jl
function riemann!(destination, surface_flux, u_surfaces_left, u_surfaces_right, surface_id,
                  equation::HyperbolicDiffusionEquations, n_nodes, orientations)
  # Call pointwise Riemann solver
  # i -> left, j -> right
  for j = 1:n_nodes
    for i = 1:n_nodes
      flux = surface_flux(equation, orientations[surface_id],
                          u_surfaces_left[1, i, surface_id],
                          u_surfaces_left[2, i, surface_id],
                          u_surfaces_left[3, i, surface_id],
                          u_surfaces_right[1, j, surface_id],
                          u_surfaces_right[2, j, surface_id],
                          u_surfaces_right[3, j, surface_id])

      # Copy flux back to actual flux array
      for v in 1:nvariables(equation)
        destination[v, i, j] = flux[v]
      end
    end
  end
end

# Calculate flux across interface with different states on both sides (surface version)
# - `destination::AbstractArray{T,2} where T<:Real`:
#   The array of surface flux values (updated inplace).
# - `surface_flux`:
#   The surface flux as a function.
# - `u_surfaces::AbstractArray{T,4} where T<:Real``
# - `surface_id::Integer`
# - `equation::AbstractEquations`
# - `n_nodes::Integer`
# - `orientations::Vector{T} where T<:Integer`
# See equations.jl
function riemann!(destination, surface_flux, u_surfaces, surface_id,
                  equation::HyperbolicDiffusionEquations, n_nodes, orientations)
  # Call pointwise Riemann solver
  for i = 1:n_nodes
    flux = surface_flux(equation, orientations[surface_id],
                        u_surfaces[1, 1, i, surface_id],
                        u_surfaces[1, 2, i, surface_id],
                        u_surfaces[1, 3, i, surface_id],
                        u_surfaces[2, 1, i, surface_id],
                        u_surfaces[2, 2, i, surface_id],
                        u_surfaces[2, 3, i, surface_id])

    # Copy flux back to actual flux array
    for v in 1:nvariables(equation)
      destination[v, i] = flux[v]
    end
  end
end


function flux_lax_friedrichs(equation::HyperbolicDiffusionEquations, orientation,
                             phi_ll, p_ll, q_ll,
                             phi_rr, p_rr, q_rr,)
  # Obtain left and right fluxes
  f_ll = zeros(MVector{3})
  f_rr = zeros(MVector{3})
  calcflux1D!(f_ll, equation, phi_ll, p_ll, q_ll, orientation)
  calcflux1D!(f_rr, equation, phi_rr, p_rr, q_rr, orientation)

  λ_max = sqrt(equation.nu/equation.Tr)
  f1 = 1/2 * (f_ll[1] + f_rr[1]) - 1/2 * λ_max * (phi_rr - phi_ll)
  f2 = 1/2 * (f_ll[2] + f_rr[2]) - 1/2 * λ_max * (p_rr   - p_ll)
  f3 = 1/2 * (f_ll[3] + f_rr[3]) - 1/2 * λ_max * (q_rr   - q_ll)

  return (f1, f2, f3)
end


function flux_upwind(equation::HyperbolicDiffusionEquations, orientation,
                     phi_ll, p_ll, q_ll,
                     phi_rr, p_rr, q_rr,)
  # Obtain left and right fluxes
  f_ll = zeros(MVector{3})
  f_rr = zeros(MVector{3})
  calcflux1D!(f_ll, equation, phi_ll, p_ll, q_ll, orientation)
  calcflux1D!(f_rr, equation, phi_rr, p_rr, q_rr, orientation)

  # this is an optimized version of the application of the upwind dissipation matrix:
  #   dissipation = 0.5*R_n*|Λ|*inv(R_n)[[u]]
  λ_max = sqrt(equation.nu/equation.Tr)
  f1 = 1/2 * (f_ll[1] + f_rr[1]) - 1/2 * λ_max * (phi_rr - phi_ll)
  if orientation == 1 # x-direction
    f2 = 1/2 * (f_ll[2] + f_rr[2]) - 1/2 * λ_max * (p_rr - p_ll)
    f3 = 1/2 * (f_ll[3] + f_rr[3])
  else # y-direction
    f2 = 1/2 * (f_ll[2] + f_rr[2])
    f3 = 1/2 * (f_ll[3] + f_rr[3]) - 1/2 * λ_max * (q_rr - q_ll)
  end

  return (f1, f2, f3)
end


# Determine maximum stable time step based on polynomial degree and CFL number
function calc_max_dt(equation::HyperbolicDiffusionEquations, u::Array{Float64, 4},
                     element_id::Int, n_nodes::Int,
                     invjacobian::Float64, cfl::Float64)
  dt = cfl * 2 / (invjacobian * sqrt(equation.nu/equation.Tr)) / n_nodes

  return dt
end

# Convert conservative variables to primitive
function cons2prim(equation::HyperbolicDiffusionEquations, cons::Array{Float64, 4})
  return cons
end

# Convert conservative variables to entropy found in I Do Like CFD, Too, Vol. 1
function cons2entropy(equation::HyperbolicDiffusionEquations,
                      cons::Array{Float64, 4}, n_nodes::Int,
                      n_elements::Int)
  entropy = similar(cons)
  @. entropy[1, :, :, :] = cons[1, :, :, :]
  @. entropy[2, :, :, :] = equation.Lr*equation.Lr*cons[2, :, :, :]
  @. entropy[3, :, :, :] = equation.Lr*equation.Lr*cons[3, :, :, :]

  return entropy
end
