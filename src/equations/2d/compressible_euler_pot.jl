
@doc raw"""
    CompressibleEulerPotEquations2D

The compressible Euler equations for an ideal gas in two space dimensions.
"""
struct CompressibleEulerPotEquations2D <: AbstractCompressibleEulerEquations{2, 4}
  c_p::Float64
  c_v::Float64
  R_d::Float64
  kappa::Float64
  gamma::Float64
  _grav::Float64
  p0::Float64
  a::Float64
end

function CompressibleEulerPotEquations2D()
  c_p = parameter("c_p",1004)
  c_v = parameter("c_v",717)
  R_d = parameter("R_d",c_p-c_v)
  kappa = parameter("kappa",(c_p-c_v)/c_p)
  gamma = parameter("gamma", c_p/c_v)
  _grav = parameter("_grav",9.81)
  p0 = parameter("p0",1.e5)
  a = parameter("a",360.e0)


  CompressibleEulerPotEquations2D(c_p,c_v,R_d,kappa,gamma,_grav,p0,a)
end


get_name(::CompressibleEulerPotEquations2D) = "CompressibleEulerPotEquations2D"
varnames_cons(::CompressibleEulerPotEquations2D) = @SVector ["rho", "rho_v1", "rho_v2", "rho_pot"]
varnames_prim(::CompressibleEulerPotEquations2D) = @SVector ["rho", "v1", "v2", "pot"]


"""
Warm bubble test from paper:
Wicker, L. J., and W. C. Skamarock, 1998: A time-splitting scheme
for the elastic equations incorporating second-order Runge–Kutta
time differencing. Mon. Wea. Rev., 126, 1992–1999.
"""

function initial_conditions_warm_bubble(x, t, equation::CompressibleEulerPotEquations2D)

  xc = 0
  zc = 2000
  r = sqrt((x[1] - xc)^2 + (x[2] - zc)^2)
  rc = 2000
  θ_ref = 300
  Δθ = 0

  if r <= rc
     Δθ = 2 * cospi(0.5*r/rc)^2
  end

  #Perturbed state:
  θ = θ_ref + Δθ # potential temperature
  π_exner = 1 - equation._grav / (equation.c_p * θ) * x[2] # exner pressure
  ρ = equation.p0 / (equation.R_d * θ) * (π_exner)^(equation.c_v / equation.R_d) # density

  v1 = 20
  v2 = 0
  ρ_v1 = ρ * v1
  ρ_v2 = ρ * v2
  ρ_θ = ρ * θ
  return @SVector [ρ, ρ_v1, ρ_v2, ρ_θ]
end

# Apply source terms
function source_terms_warm_bubble(ut, u, x, element_id, t, n_nodes, equation::CompressibleEulerPotEquations2D)
  for j in 1:n_nodes, i in 1:n_nodes
    x1 = x[1, i, j, element_id]
    x2 = x[2, i, j, element_id]
    ut[3, i, j, element_id] +=  -equation._grav * u[1, i, j, element_id]
  end
  return nothing
end

# Calculate 1D flux for a single point
@inline function calcflux(u, orientation, equation::CompressibleEulerPotEquations2D)
  rho, rho_v1, rho_v2, rho_θ = u
  v1 = rho_v1/rho
  v2 = rho_v2/rho
  θ = rho_θ/rho
  p = (equation.R_d * rho_θ / equation.p0^equation.kappa)^(1 / (1 - equation.kappa))
  if orientation == 1
    f1 = rho_v1
    f2 = rho_v1 * v1 + p
    f3 = rho_v1 * v2
    f4 = rho_v1 * θ
  else
    f1 = rho_v2
    f2 = rho_v2 * v1
    f3 = rho_v2 * v2 + p
    f4 = rho_v2 * θ
  end
  return SVector(f1, f2, f3, f4)
end


"""
    function flux_shima_etal(u_ll, u_rr, orientation, equation::CompressibleEulerPotEquations2D)

This flux is is a modification of the original kinetic energy preserving two-point flux by
Kuya, Totani and Kawai (2018)
  Kinetic energy and entropy preserving schemes for compressible flows
  by split convective forms
  [DOI: 10.1016/j.jcp.2018.08.058](https://doi.org/10.1016/j.jcp.2018.08.058)
The modification is in the energy flux to guarantee pressure equilibrium and was developed by
Nao Shima, Yuichi Kuya, Yoshiharu Tamaki, Soshi Kawai (JCP 2020)
  Preventing spurious pressure oscillations in split convective form discretizations for
  compressible flows
"""
@inline function flux_shima_etal(u_ll, u_rr, orientation, equation::CompressibleEulerPotEquations2D)
  # Unpack left and right state
  rho_ll, rho_v1_ll, rho_v2_ll, rho_θ_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_θ_rr = u_rr

  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  θ_ll = rho_θ_ll / rho_ll
  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  θ_rr = rho_θ_rr / rho_rr

  # Average each factor of products in flux
  rho_avg = 1/2 * (rho_ll + rho_rr)
  v1_avg  = 1/2 * ( v1_ll +  v1_rr)
  v2_avg  = 1/2 * ( v2_ll +  v2_rr)
  θ_avg  = 1/2 * ( θ_ll +  θ_rr)

  # Calculate fluxes depending on orientation
  if orientation == 1
    f1 = rho_avg * v1_avg
    f2 = rho_avg * v1_avg * v1_avg + p_avg
    f3 = rho_avg * v1_avg * v2_avg
    f4 = rho_avg * v1_avg * θ_avg
  else
    f1 = rho_avg * v2_avg
    f2 = rho_avg * v2_avg * v1_avg
    f3 = rho_avg * v2_avg * v2_avg + p_avg
    f4 = rho_avg * v2_avg * θ_avg
  end

  return SVector(f1, f2, f3, f4)
end

"""
function flux_lmars(u_ll, u_rr, orientation, equation::CompressibleEulerPotEquations2D)

Chen, X., N. Andronova, B. Van Leer, J. E. Penner, J. P. Boyd, C. Jablonowski, and S. Lin, 2013: 
A Control-Volume Model of the Compressible Euler Equations with a Vertical Lagrangian Coordinate. 
Mon. Wea. Rev., 141, 2526–2544, https://doi.org/10.1175/MWR-D-12-00129.1.

"""

function flux_lmars(u_ll, u_rr, orientation, equation::CompressibleEulerPotEquations2D)
  # Calculate primitive variables and speed of sound
  rho_ll, rho_v1_ll, rho_v2_ll, rho_θ_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_θ_rr = u_rr

  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  θ_ll = rho_θ_ll / rho_ll
  p_ll = (equation.R_d * rho_θ_ll / equation.p0^equation.kappa)^(1 / (1 - equation.kappa))
  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  θ_rr = rho_θ_rr / rho_rr
  p_rr = (equation.R_d * rho_θ_rr / equation.p0^equation.kappa)^(1 / (1 - equation.kappa))


  rhoM = 0.5 * (rho_ll + rho_rr)
  if orientation == 1 # x-direction
    pM = 0.5 * (p_ll + p_rr) - 0.5 * rhoM * equation.a * (v1_rr - v1_ll) 
    vM = 0.5 * (v1_ll + v1_rr) - 1 / (2 * rhoM * equation.a) * (p_rr - p_ll) 
    if vM >= 0
      f = u_ll  * vM + pM * SVector(0, 1, 0, 0)
    else
      f = u_rr  * vM + pM * SVector(0, 1, 0, 0)
    end  
  else # y-direction
    pM = 0.5 * (p_ll + p_rr) - 0.5 * rhoM * equation.a * (v2_rr - v2_ll) 
    vM = 0.5 * (v2_ll + v2_rr) - 1 / (2 * rhoM * equation.a) * (p_rr - p_ll) 
    if vM >= 0
      f = u_ll * vM + pM * SVector(0, 0, 1, 0)
    else
      f = u_rr * vM + pM * SVector(0, 0, 1, 0)
    end  
  end
  return f
end


# Determine maximum stable time step based on polynomial degree and CFL number
function calc_max_dt(u, element_id, invjacobian, cfl,
                     equation::CompressibleEulerPotEquations2D, dg)
  λ_max = 0.0
  for j in 1:nnodes(dg), i in 1:nnodes(dg)
    rho, rho_v1, rho_v2, rho_θ = get_node_vars(u, dg, i, j, element_id)
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v_mag = sqrt(v1^2 + v2^2)
    p = (equation.R_d * rho_θ / equation.p0^equation.kappa)^(1 / (1 - equation.kappa))
    c = sqrt(equation.gamma * p / rho)
    λ_max = max(λ_max, v_mag + c)
  end

  dt = cfl * 2 / (nnodes(dg) * invjacobian * λ_max)

  return dt
end


# Convert conservative variables to primitive
function cons2prim(cons, equation::CompressibleEulerPotEquations2D)
  prim = similar(cons)
  @. prim[1, :, :, :] = cons[1, :, :, :]
  @. prim[2, :, :, :] = cons[2, :, :, :] / cons[1, :, :, :]
  @. prim[3, :, :, :] = cons[3, :, :, :] / cons[1, :, :, :]
  @. prim[4, :, :, :] = cons[4, :, :, :] / cons[1, :, :, :]
  return prim
end

# Convert conservative variables to potential
function cons2pot(cons, equation::CompressibleEulerPotEquations2D)
  pot = similar(cons)
  @. pot[1, :, :, :] = cons[1, :, :, :]
  @. pot[2, :, :, :] = cons[2, :, :, :] / cons[1, :, :, :]
  @. pot[3, :, :, :] = cons[3, :, :, :] / cons[1, :, :, :]
  @. pot[4, :, :, :] = cons[4, :, :, :] / cons[1, :, :, :]
  return pot
end

# Convert conservative variables to entropy
function cons2entropy(cons, n_nodes, n_elements, equation::CompressibleEulerPotEquations2D)
  entropy = similar(cons)
  v = zeros(2,n_nodes,n_nodes,n_elements)
  v_square = zeros(n_nodes,n_nodes,n_elements)
  p = zeros(n_nodes,n_nodes,n_elements)
  s = zeros(n_nodes,n_nodes,n_elements)
  rho_p = zeros(n_nodes,n_nodes,n_elements)

  @. v[1, :, :, :] = cons[2, :, :, :] / cons[1, :, :, :]
  @. v[2, :, :, :] = cons[3, :, :, :] / cons[1, :, :, :]
  @. v_square[ :, :, :] = v[1, :, :, :]*v[1, :, :, :]+v[2, :, :, :]*v[2, :, :, :]
  @. p[ :, :, :] = (equation.R_d * cons[4, :, :, :] / equation.p0^equation.kappa)^(1 / (1 - equation.kappa))
  @. s[ :, :, :] = log(p[:, :, :]) - equation.gamma*log(cons[1, :, :, :])
  @. rho_p[ :, :, :] = cons[1, :, :, :] / p[ :, :, :]

  @. entropy[1, :, :, :] = (equation.gamma - s[:,:,:])/(equation.gamma-1) -
                           0.5*rho_p[:,:,:]*v_square[:,:,:]
  @. entropy[2, :, :, :] = rho_p[:,:,:]*v[1,:,:,:]
  @. entropy[3, :, :, :] = rho_p[:,:,:]*v[2,:,:,:]
  @. entropy[4, :, :, :] = -rho_p[:,:,:]

  return entropy
end


# Convert primitive to conservative variables
function prim2cons(prim, equation::CompressibleEulerPotEquations2D)
  cons = similar(prim)
  cons[1] = prim[1]
  cons[2] = prim[2] * prim[1]
  cons[3] = prim[3] * prim[1]
  cons[4] = prim[4] * prim[1]
  return cons
end


# Convert conservative variables to indicator variable for discontinuities (elementwise version)
@inline function cons2indicator!(indicator, cons, element_id, n_nodes, indicator_variable,
                                 equation::CompressibleEulerPotEquations2D)
  for j in 1:n_nodes
    for i in 1:n_nodes
      indicator[1, i, j] = cons2indicator(cons[1, i, j, element_id], cons[2, i, j, element_id],
                                          cons[3, i, j, element_id], cons[4, i, j, element_id],
                                          indicator_variable, equation)
    end
  end
end


# Convert conservative variables to indicator variable for discontinuities (pointwise version)
@inline function cons2indicator(rho, rho_v1, rho_v2, rho_θ, ::Val{:density},
                                equation::CompressibleEulerPotEquations2D)
  # Indicator variable is rho
  return rho
end


# Convert conservative variables to indicator variable for discontinuities (pointwise version)
@inline function cons2indicator(rho, rho_v1, rho_v2, rho_θ, ::Val{:density_pressure},
                                equation::CompressibleEulerPotEquations2D)
  # Calculate pressure
  p = (equation.R_d * rho_θ / equation.p0^equation.kappa)^(1 / (1 - equation.kappa))

  # Indicator variable is rho * p
  return rho * p
end


# Convert conservative variables to indicator variable for discontinuities (pointwise version)
@inline function cons2indicator(rho, rho_v1, rho_v2, rho_θ, ::Val{:pressure},
                                equation::CompressibleEulerPotEquations2D)
  # Indicator variable is p
  return (equation.R_d * rho_θ / equation.p0^equation.kappa)^(1 / (1 - equation.kappa))
end


# Calculate thermodynamic entropy for a conservative state `cons`
@inline function entropy_thermodynamic(cons, equation::CompressibleEulerPotEquations2D)
  # Pressure
  p = (equation.R_d * cons[4] / equation.p0^equation.kappa)^(1 / (1 - equation.kappa))

  # Thermodynamic entropy
  s = log(p) - equation.gamma*log(cons[1])

  return s
end

# Calculate potential temperature for a conservative state `cons`
@inline function pottemp_thermodynamic(cons, equation::CompressibleEulerPotEquations2D)

  return cons[4] / cons[1]
end


# Calculate mathematical entropy for a conservative state `cons`
@inline function entropy_math(cons, equation::CompressibleEulerPotEquations2D)
  # Mathematical entropy
  S = -entropy_thermodynamic(cons, equation) * cons[1] / (equation.gamma - 1)

  return S
end


# Default entropy is the mathematical entropy
@inline entropy(cons, equation::CompressibleEulerPotEquations2D) = entropy_math(cons, equation)


# Calculate total energy for a conservative state `cons`
@inline energy_total(cons, ::CompressibleEulerPotEquations2D) = cons[4]


# Calculate kinetic energy for a conservative state `cons`
@inline function energy_kinetic(cons, equation::CompressibleEulerPotEquations2D)
  return 0.5 * (cons[2]^2 + cons[3]^2)/cons[1]
end


# Calculate internal energy for a conservative state `cons`
@inline function energy_internal(cons, equation::CompressibleEulerPotEquations2D)
  return energy_total(cons, equation) - energy_kinetic(cons, equation)
end
