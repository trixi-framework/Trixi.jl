
@doc raw"""
    CompressibleEulerMulticomponentEquations2D

!!! warning "Experimental code"
    This system of equations is experimental and can change any time.

Multicomponent version of the compressible Euler equations for an ideal gas in two space dimensions.
"""
struct CompressibleEulerMulticomponentEquations2D{RealT<:Real} <: AbstractCompressibleEulerMulticomponentEquations{2, 5}
  gamma1          ::RealT 
  gamma2          ::RealT
  gas_constant1   ::RealT
  gas_constant2   ::RealT
  cv1             ::RealT
  cv2             ::RealT
  cp1             ::RealT
  cp2             ::RealT
end

function CompressibleEulerMulticomponentEquations2D()

  # Set ratio of specific heat of the gas per species 
  # (It holds: gamma = cp/cv) (Can be used for consistency check!)
  gamma1  = 1.4
  gamma2  = 1.4
  
  # Set specific gas constant with respect to the molar mass per species 
  # (It holds: gas_constant = R/M, with R = molar gas constant, M = molar mass) 
  # (For a calorically and thermally perfect gas Mayer's relation holds: gas_constant = cp - cv) (Can be used for consistency check!)
  gas_constant1     = 0.4
  gas_constant2     = 0.4

  # Set specific heat for a constant volume per species
  cv1    = 1.0
  cv2    = 1.0

  # Set specific heat for a constant pressure per species
  cp1    = 1.4
  cp2    = 1.4

  CompressibleEulerMulticomponentEquations2D(gamma1, gamma2, gas_constant1, gas_constant2, cv1, cv2, cp1, cp2)
end


function CompressibleEulerMulticomponentEquations2D(gamma1, gamma2, gas_constant1, gas_constant2)
  
  # Set specific heat for a constant volume per species
  cv1    = gas_constant1 / (gamma1 - 1.0)
  cv2    = gas_constant2 / (gamma2 - 1.0)

  # Set specific heat for a constant pressure per species
  cp1    = gas_constant1 + cv1 
  cp2    = gas_constant2 + cv2

  CompressibleEulerMulticomponentEquations2D(gamma1, gamma2, gas_constant1, gas_constant2, cv1, cv2, cp1, cp2)
end


get_name(::CompressibleEulerMulticomponentEquations2D) = "CompressibleEulerMulticomponentEquations2D"
varnames(::typeof(cons2cons), ::CompressibleEulerMulticomponentEquations2D) = @SVector ["rho1", "rho2", "rho_v1", "rho_v2", "rho_e"]
varnames(::typeof(cons2prim), ::CompressibleEulerMulticomponentEquations2D) = @SVector ["rho1", "rho2", "v1", "v2", "p"]

# Set initial conditions at physical location `x` for time `t`

"""
    initial_condition_convergence_test(x, t, equations::CompressibleEulerMulticomponentEquations2D)

A smooth initial condition used for convergence tests in combination with
[`source_terms_convergence_test`](@ref)
(and [`boundary_condition_convergence_test`](@ref) in non-periodic domains).
"""
function initial_condition_convergence_test(x, t, equations::CompressibleEulerMulticomponentEquations2D)
  c       = 2
  A       = 0.1
  L       = 2
  f       = 1/L
  omega   = 2 * pi * f
  ini     = c + A * sin(omega * (x[1] + x[2] - t))

  v1      = 1.0
  v2      = 1.0

  rho     = ini
  rho1    = 0.2 * rho
  rho2    = 0.8 * rho  
  rho_v1  = rho * v1
  rho_v2  = rho * v2
  rho_e   = rho^2

  return @SVector [rho1, rho2, rho_v1, rho_v2, rho_e]
end

"""
    source_terms_convergence_test(u, x, t, equations::CompressibleEulerMulticomponentEquations2D)

Source terms used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref)
(and [`boundary_condition_convergence_test`](@ref) in non-periodic domains).
"""
@inline function source_terms_convergence_test(u, x, t, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack cv1, cv2, gamma1, gamma2 = equations
  # Same settings as in `initial_condition`
  c       = 2
  A       = 0.1
  L       = 2
  f       = 1/L
  omega   = 2 * pi * f

  rho1    = u[1]
  rho2    = u[2]

  gamma   = (rho1*cv1*gamma1 + rho2*cv2*gamma2) / (rho1*cv1 + rho2*cv2)

  x1, x2  = x
  si, co  = sincos((x1 + x2 - t)*omega)
  tmp1    = co * A * omega
  tmp2    = si * A
  tmp3    = gamma - 1
  tmp4    = (2*c - 1)*tmp3
  tmp5    = (2*tmp2*gamma - 2*tmp2 + tmp4 + 1)*tmp1
  tmp6    = tmp2 + c

  du1     = 0.2 * tmp1
  du2     = 0.8 * tmp1
  du3     = tmp5
  du4     = tmp5
  du5     = 2*((tmp6 - 1.0)*tmp3 + tmp6*gamma)*tmp1

  # Original terms (without performanc enhancements)
  # du1 = cos((x1 + x2 - t)*ω)*A*ω
  # du2 = (2*sin((x1 + x2 - t)*ω)*A*γ - 2*sin((x1 + x2 - t)*ω)*A +
  #                             2*c*γ - 2*c - γ + 2)*cos((x1 + x2 - t)*ω)*A*ω
  # du3 = (2*sin((x1 + x2 - t)*ω)*A*γ - 2*sin((x1 + x2 - t)*ω)*A +
  #                             2*c*γ - 2*c - γ + 2)*cos((x1 + x2 - t)*ω)*A*ω
  # du3 = 2*((c - 1 + sin((x1 + x2 - t)*ω)*A)*(γ - 1) +
  #                             (sin((x1 + x2 - t)*ω)*A + c)*γ)*cos((x1 + x2 - t)*ω)*A*ω

  return SVector(du1, du2, du3, du4, du5)
end

"""
    boundary_condition_convergence_test(u_inner, orientation, direction, x, t,
                                        surface_flux_function,
                                        equations::CompressibleEulerMulticomponentEquations2D)

Boundary conditions used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref) and [`source_terms_convergence_test`](@ref).
"""
function boundary_condition_convergence_test(u_inner, orientation, direction, x, t,
                                              surface_flux_function,
                                              equations::CompressibleEulerMulticomponentEquations2D)
  u_boundary = initial_condition_convergence_test(x, t, equations)

  # Calculate boundary flux
  if direction in (2, 4) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
  end

  return flux
end


"""
    initial_condition_shock_bubble(x, t, equations::CompressibleEulerMulticomponentEquations2D)

A shock-bubble testcase for multicomponent Euler equations
- Ayoub Gouasmi, Karthik Duraisamy, Scott Murman
  Formulation of Entropy-Stable schemes for the multicomponent compressible Euler equations
  [arXiv: 1904.00972](https://arxiv.org/abs/1904.00972)
"""
function initial_condition_shock_bubble(x, t, equations::CompressibleEulerMulticomponentEquations2D)
  # bubble test case, see Agertz et al. https://arxiv.org/pdf/1904.00972
  # other reference: https://www.researchgate.net/profile/Pep_Mulet/publication/222675930_A_flux-split_algorithm_applied_to_conservative_models_for_multicomponent_compressible_flows/links/568da54508aeaa1481ae7af0.pdf
  # typical domain is rectangular, we change it to a square, as Trixi can only do squares
  @unpack gas_constant1, gas_constant2 = equations

  # Positivity Preserving Parameter, can be set to zero if scheme is positivity preserving
  delta   = 0.03

  # Region I 
  rho1_1  = delta
  rho2_1  = 1.225 * gas_constant1/gas_constant2 - delta
  v1_1    = 0.0
  v2_1    = 0.0
  p_1     = 101325

  # Region II 
  rho1_2  = 1.225-delta
  rho2_2  = delta
  v1_2    = 0.0
  v2_2    = 0.0
  p_2     = 101325

  # Region III 
  rho1_3  = 1.6861 - delta
  rho2_3  = delta
  v1_3    = -113.5243
  v2_3    = 0.0
  p_3     = 159060

  # Set up Region I & II:
  inicenter = SVector(0.0, 0.0)
  x_norm = x[1] - inicenter[1]
  y_norm = x[2] - inicenter[2]
  r = sqrt(x_norm^2 + y_norm^2)  

  if (x[1] > 0.50)
    # Set up Region III
    rho1    = rho1_3
    rho2    = rho2_3
    v1      = v1_3 
    v2      = v2_3
    p       = p_3
  elseif (r < 0.25)
    # Set up Region I
    rho1    = rho1_1
    rho2    = rho2_1
    v1      = v1_1
    v2      = v2_1 
    p       = p_1
  else
    # Set up Region II
    rho1    = rho1_2
    rho2    = rho2_2
    v1      = v1_2
    v2      = v2_2
    p       = p_2
  end 

  return prim2cons(SVector(rho1, rho2, v1, v2, p), equations)
end


"""
    initial_condition_weak_blast_wave(x, t, equations::CompressibleEulerMulticomponentEquations2D)

A for multicomponent adapted weak blast wave taken from
- Sebastian Hennemann, Gregor J. Gassner (2020)
  A provably entropy stable subcell shock capturing approach for high order split form DG
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
function initial_condition_weak_blast_wave(x, t, equations::CompressibleEulerMulticomponentEquations2D)
  # From Hennemann & Gassner JCP paper 2020 (Sec. 6.3)
  # Set up polar coordinates
  inicenter         = SVector(0.0, 0.0)
  x_norm            = x[1] - inicenter[1]
  y_norm            = x[2] - inicenter[2]
  r                 = sqrt(x_norm^2 + y_norm^2)
  phi               = atan(y_norm, x_norm)
  sin_phi, cos_phi  = sincos(phi)

  # Calculate primitive variables.    !!! NOTE: EC flux does not transfer mass across an interface separating two different species
  rho1              = r > 0.5 ? 0.2*1.0 : 0.2*1.1691
  rho2              = r > 0.5 ? 0.8*1.0 : 0.8*1.1691
  rho               = rho1 + rho2
  v1                = r > 0.5 ? 0.0 : 0.1882 * cos_phi
  v2                = r > 0.5 ? 0.0 : 0.1882 * sin_phi
  p                 = r > 0.5 ? 1.0 : 1.245

  return prim2cons(SVector(rho1, rho2, v1, v2, p), equations)
end


# Calculate 1D flux for a single point
@inline function calcflux(u, orientation, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack cv1, cv2, gamma1, gamma2 = equations
  rho1, rho2, rho_v1, rho_v2, rho_e = u

  rho   = rho1 + rho2
  v1    = rho_v1/rho
  v2    = rho_v2/rho
  gamma = (rho1*cv1*gamma1 + rho2*cv2*gamma2) / (rho1*cv1 + rho2*cv2)
  p     = (gamma - 1) * (rho_e - 0.5 * rho * (v1^2 + v2^2))
  #p     = pressure(u, equations)

  if orientation == 1
    f1  = rho1 * v1
    f2  = rho2 * v1
    f3  = rho_v1 * v1 + p
    f4  = rho_v2 * v1
    f5  = (rho_e + p) * v1
  else
    f1  = rho1 * v2
    f2  = rho2 * v2
    f3  = rho_v1 * v2
    f4  = rho_v2 * v2 + p
    f5  = (rho_e + p) * v2
  end
  return SVector(f1, f2, f3, f4, f5)
end


"""
    flux_chandrashekar(u_ll, u_rr, orientation, equations::CompressibleEulerMulticomponentEquations2D)

Entropy conserving two-point flux by
- Ayoub Gouasmi, Karthik Duraisamy (2020)
  "Formulation of Entropy-Stable schemes for the multicomponent compressible Euler equations""
  arXiv:1904.00972v3 [math.NA] 4 Feb 2020
"""
@inline function flux_chandrashekar(u_ll, u_rr, orientation, equations::CompressibleEulerMulticomponentEquations2D)
  # Unpack left and right state
  @unpack gas_constant1, gas_constant2, cv1, cv2 = equations
  rho1_ll, rho2_ll, rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  rho1_rr, rho2_rr, rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr

  # helpful primitive variables
  rho_ll      = rho1_ll + rho2_ll
  rho_rr      = rho1_rr + rho2_rr
  v1_ll       = rho_v1_ll/rho_ll
  v2_ll       = rho_v2_ll/rho_ll
  v1_rr       = rho_v1_rr/rho_rr
  v2_rr       = rho_v2_rr/rho_rr

  # helpful mean values
  rho1_mean   = ln_mean(rho1_ll,rho1_rr)
  rho2_mean   = ln_mean(rho2_ll,rho2_rr) 
  rho1_avg    = 0.5 * (rho1_ll + rho1_rr)
  rho2_avg    = 0.5 * (rho2_ll + rho2_rr)
  v1_avg      = 0.5 * (v1_ll + v1_rr)
  v2_avg      = 0.5 * (v2_ll + v2_rr)
  v1_square   = 0.5 * (v1_ll^2 + v1_rr^2)
  v2_square   = 0.5 * (v2_ll^2 + v2_rr^2)
  v_sum       = v1_avg + v2_avg

  # multicomponent specific values
  enth        = rho1_avg * gas_constant1 + rho2_avg * gas_constant2
  T_ll        = (rho_e_ll - 0.5 * rho_ll * (v1_ll^2 + v2_ll^2)) / (rho1_ll * cv1 + rho2_ll * cv2)
  T_rr        = (rho_e_rr - 0.5 * rho_rr * (v1_rr^2 + v2_rr^2)) / (rho1_rr * cv1 + rho2_rr * cv2)
  T           = 0.5 * (1.0/T_ll + 1.0/T_rr)
  T_log       = ln_mean(1.0/T_ll, 1.0/T_rr)

  # Calculate fluxes depending on orientation
  if orientation == 1
    f1 = rho1_mean * v1_avg
    f2 = rho2_mean * v1_avg
    f3 = (f1 + f2) * v1_avg + enth/T 
    f4 = (f1 + f2) * v2_avg
    f5 = (f1 * cv1 + f2 * cv2)/T_log - 0.5 * (v1_square + v2_square) * (f1 + f2) + v1_avg * f3 + v2_avg * f4
  else
    f1 = rho1_mean * v2_avg
    f2 = rho2_mean * v2_avg
    f3 = (f1 + f2) * v1_avg 
    f4 = (f1 + f2) * v2_avg + enth/T
    f5 = (f1 * cv1 + f2 * cv2)/T_log - 0.5 * (v1_square + v2_square) * (f1 + f2) + v1_avg * f3 + v2_avg * f4
  end

  return SVector(f1, f2, f3, f4, f5)
end

"""
    flux_chandrashekar_stable(u_ll, u_rr, orientation, equations::CompressibleEulerMulticomponentEquations2D)

Entropy-Stable two-point flux by
- Ayoub Gouasmi, Karthik Duraisamy (2020)
  "Formulation of Entropy-Stable schemes for the multicomponent compressible Euler equations""
  arXiv:1904.00972v3 [math.NA] 4 Feb 2020
"""
@inline function flux_chandrashekar_stable(u_ll, u_rr, orientation, equations::CompressibleEulerMulticomponentEquations2D)
  # Unpack left and right state
  @unpack gas_constant1, gas_constant2, cv1, cv2, gamma1, gamma2 = equations
  rho1_ll, rho2_ll, rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  rho1_rr, rho2_rr, rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr

  # helpful primitive variables
  rho_ll = rho1_ll + rho2_ll
  rho_rr = rho1_rr + rho2_rr
  v1_ll = rho_v1_ll/rho_ll
  v2_ll = rho_v2_ll/rho_ll
  v1_rr = rho_v1_rr/rho_rr
  v2_rr = rho_v2_rr/rho_rr

  # needed mean values
  rho1_mean = ln_mean(rho1_ll,rho1_rr)
  rho2_mean = ln_mean(rho2_ll,rho2_rr) 
  rho1_avg  = 0.5 * (rho1_ll + rho1_rr)
  rho2_avg  = 0.5 * (rho2_ll + rho2_rr)
  rho_avg   = 0.5 * (rho_ll + rho_rr)
  v1_avg    = 0.5 * (v1_ll + v1_rr)
  v2_avg    = 0.5 * (v2_ll + v2_rr)
  v1_square = 0.5 * (v1_ll^2 + v1_rr^2)
  v2_square = 0.5 * (v2_ll^2 + v2_rr^2)
  v_ll_sq   = 0.5 * (v1_ll^2 + v2_ll^2)
  v_rr_sq   = 0.5 * (v1_rr^2 + v2_rr^2)
  v_sum     = v1_avg + v2_avg

  # multicomponent specific values
  enth      = rho1_avg * gas_constant1 + rho2_avg * gas_constant2
  T_ll      = (rho_e_ll - 0.5 * rho_ll * (v1_ll^2 + v2_ll^2)) / (rho1_ll * cv1 + rho2_ll * cv2)
  T_rr      = (rho_e_rr - 0.5 * rho_rr * (v1_rr^2 + v2_rr^2)) / (rho1_rr * cv1 + rho2_rr * cv2)
  T         = 0.5 * (T_ll + T_rr)
  T_log     = ln_mean(1.0/T_ll, 1.0/T_rr)

  gamma     = (rho1_avg*cv1*gamma1 + rho2_avg*cv2*gamma2) / (rho1_avg*cv1 + rho2_avg*cv2)

  # Calculate dissipation operator to be ES 
  R_x       = zeros(Float64, (5, 5))
  R_y       = zeros(Float64, (5, 5))
  Lambda_x  = zeros(Float64, (5, 5))
  Lambda_y  = zeros(Float64, (5, 5))
  Tau       = zeros(Float64, (5, 5))
  W         = zeros(Float64, (5))
  Dv_x      = zeros(Float64, (5))
  Dv_y      = zeros(Float64, (5))

  # Help variables
  e1    = equations.cv1 * T
  e2    = equations.cv2 * T
  r     = (rho1_avg / rho_avg) * gas_constant1 + (rho2_avg / rho_avg) * gas_constant2 
  a     = sqrt(gamma * T * r) 
  k     = 0.5 * (v1_square + v2_square) 
  h1    = e1 + gas_constant1 * T 
  h2    = e2 + gas_constant2 * T 
  h     = (rho1_avg / rho_avg) * h1 + (rho2_avg / rho_avg) * h2 
  d1    = h1 - gamma * e1 
  d2    = h2 - gamma * e2 
  ht    = h + k
  tauh  = sqrt(rho_avg/(gamma * r))

  # -------------------------------- #

  R_x[1,1]  = 1.0 
  R_x[1,2]  = 0.0 
  R_x[1,3]  = 0.0 
  R_x[1,4]  = rho1_avg / rho_avg 
  R_x[1,5]  = R_x[1,4]

  R_x[2,1]  = 0.0 
  R_x[2,2]  = 1.0 
  R_x[2,3]  = 0.0 
  R_x[2,4]  = rho2_avg / rho_avg 
  R_x[2,5]  = R_x[2,4]

  R_x[3,1]  = v1_avg 
  R_x[3,2]  = v1_avg 
  R_x[3,3]  = 0.0 
  R_x[3,4]  = v1_avg + a 
  R_x[3,5]  = v1_avg - a 

  R_x[4,1]  = v2_avg 
  R_x[4,2]  = v2_avg 
  R_x[4,3]  = a 
  R_x[4,4]  = v2_avg 
  R_x[4,5]  = v2_avg 

  R_x[5,1]  = k - (d1/(gamma - 1.0)) 
  R_x[5,2]  = k - (d2/(gamma - 1.0)) 
  R_x[5,3]  = a * v2_avg 
  R_x[5,4]  = ht + a * v1_avg 
  R_x[5,5]  = ht - a * v1_avg 

  # -------------------------------- #

  R_y[1,1]  = 1.0 
  R_y[1,2]  = 0.0 
  R_y[1,3]  = 0.0 
  R_y[1,4]  = rho1_avg / rho_avg 
  R_y[1,5]  = R_y[1,4] 

  R_y[2,1]  = 0.0 
  R_y[2,2]  = 1.0 
  R_y[2,3]  = 0.0 
  R_y[2,4]  = rho2_avg / rho_avg
  R_y[2,5]  = R_y[2,4] 

  R_y[3,1]  = v1_avg 
  R_y[3,2]  = v1_avg 
  R_y[3,3]  = -1.0 * a 
  R_y[3,4]  = v1_avg 
  R_y[3,5]  = v1_avg 

  R_y[4,1]  = v2_avg 
  R_y[4,2]  = v2_avg 
  R_y[4,3]  = 0.0 
  R_y[4,4]  = v2_avg + a 
  R_y[4,5]  = v2_avg - a 

  R_y[5,1]  = k - (d1/(gamma - 1)) 
  R_y[5,2]  = k - (d2/(gamma - 1)) 
  R_y[5,3]  = -1.0 * a * v1_avg 
  R_y[5,4]  = ht + a * v2_avg 
  R_y[5,5]  = ht - a * v2_avg 

  # -------------------------------- #

  Lambda_x[1,1] = abs(v1_avg) 
  Lambda_x[2,2] = abs(v1_avg) 
  Lambda_x[3,3] = abs(v1_avg) 
  Lambda_x[4,4] = abs(v1_avg + a) 
  Lambda_x[5,5] = abs(v1_avg - a) 

  # -------------------------------- #

  Lambda_y[1,1] = abs(v2_avg)
  Lambda_y[2,2] = abs(v2_avg) 
  Lambda_y[3,3] = abs(v2_avg) 
  Lambda_y[4,4] = abs(v2_avg + a) 
  Lambda_y[5,5] = abs(v2_avg - a) 

  # -------------------------------- #

  Tau[1,1]  = tauh * (-1.0 * sqrt((rho1_avg/rho_avg) * (rho2_avg/rho_avg)) * sqrt(gamma * (gas_constant2 / gas_constant1)))
  Tau[1,2]  = tauh * ((rho1_avg / rho_avg) * sqrt(gamma - 1.0)) 
  Tau[1,3]  = 0.0 
  Tau[1,4]  = 0.0 
  Tau[1,5]  = 0.0 

  Tau[2,1]  = tauh * (sqrt((rho1_avg/rho_avg) * (rho2_avg/rho_avg)) * sqrt(gamma * (gas_constant1 / gas_constant2)))
  Tau[2,2]  = tauh * ((rho2_avg / rho_avg) * sqrt(gamma - 1.0)) 
  Tau[2,3]  = 0.0 
  Tau[2,4]  = 0.0 
  Tau[2,5]  = 0.0 

  Tau[3,1]  = 0.0
  Tau[3,2]  = 0.0
  Tau[3,3]  = tauh / a
  Tau[3,4]  = 0.0
  Tau[3,5]  = 0.0

  Tau[4,1]  = 0.0 
  Tau[4,2]  = 0.0 
  Tau[4,3]  = 0.0 
  Tau[4,4]  = tauh / sqrt(2) 
  Tau[4,5]  = 0.0 

  Tau[5,1]  = 0.0 
  Tau[5,2]  = 0.0 
  Tau[5,3]  = 0.0 
  Tau[5,4]  = 0.0 
  Tau[5,5]  = tauh / sqrt(2) 

  # -------------------------------- #

  w_ll  = cons2entropy(u_ll, equations)
  w_rr  = cons2entropy(u_rr, equations)

  W  = w_rr - w_ll

  # -------------------------------- #

  Dv_x = 0.5 * (R_x * Tau) * Lambda_x * transpose(R_x * Tau) * W 
  Dv_y = 0.5 * (R_y * Tau) * Lambda_y * transpose(R_y * Tau) * W

  # -------------------------------- #

  # Calculate fluxes depending on orientation
  if orientation == 1
    f1 = (rho1_mean * v1_avg) - Dv_x[1]
    f2 = (rho2_mean * v1_avg) - Dv_x[2]
    f3 = ((f1 + f2) * v1_avg + enth * T) - Dv_x[3]
    f4 = ((f1 + f2) * v2_avg) - Dv_x[4]
    f5 = ((f1 * cv1 + f2 * cv2)/T_log - 0.5 * (v1_square + v2_square) * (f1 + f2) + v1_avg * f3 + v2_avg * f4) - Dv_x[5]
  else
    f1 = (rho1_mean * v2_avg) - Dv_y[1]
    f2 = (rho2_mean * v2_avg) - Dv_y[2]
    f3 = ((f1 + f2) * v1_avg) - Dv_y[3] 
    f4 = ((f1 + f2) * v2_avg + enth * T) - Dv_y[4]
    f5 = ((f1 * cv1 + f2 * cv2)/T_log - 0.5 * (v1_square + v2_square) * (f1 + f2) + v1_avg * f3 + v2_avg * f4) - Dv_y[5]
  end

  return SVector(f1, f2, f3, f4, f5)
end


function flux_lax_friedrichs(u_ll, u_rr, orientation, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack cv1, cv2, gamma1, gamma2 = equations
  # Calculate primitive variables and speed of sound
  rho1_ll, rho2_ll, rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  rho1_rr, rho2_rr, rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr

  rho_ll = rho1_ll + rho2_ll
  rho_rr = rho1_rr + rho2_rr

  gamma_ll = (rho1_ll*cv1*gamma1 + rho2_ll*cv2*gamma2)/(rho1_ll*cv1 + rho2_ll*cv2)
  gamma_rr = (rho1_rr*cv1*gamma1 + rho2_rr*cv2*gamma2)/(rho1_rr*cv1 + rho2_rr*cv2)

  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v_mag_ll = sqrt(v1_ll^2 + v2_ll^2)
  p_ll = (gamma_ll - 1) * (rho_e_ll - 1/2 * rho_ll * v_mag_ll^2)
  c_ll = sqrt(gamma_ll * p_ll / rho_ll)
  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  v_mag_rr = sqrt(v1_rr^2 + v2_rr^2)
  p_rr = (gamma_rr - 1) * (rho_e_rr - 1/2 * rho_rr * v_mag_rr^2)
  c_rr = sqrt(gamma_rr * p_rr / rho_rr)

  # Obtain left and right fluxes
  f_ll = calcflux(u_ll, orientation, equations)
  f_rr = calcflux(u_rr, orientation, equations)

  λ_max = max(v_mag_ll, v_mag_rr) + max(c_ll, c_rr)
  f1 = 1/2 * (f_ll[1] + f_rr[1]) - 1/2 * λ_max * (rho1_rr   - rho1_ll)
  f2 = 1/2 * (f_ll[2] + f_rr[2]) - 1/2 * λ_max * (rho2_rr   - rho2_ll)
  f3 = 1/2 * (f_ll[3] + f_rr[3]) - 1/2 * λ_max * (rho_v1_rr - rho_v1_ll)
  f4 = 1/2 * (f_ll[4] + f_rr[4]) - 1/2 * λ_max * (rho_v2_rr - rho_v2_ll)
  f5 = 1/2 * (f_ll[5] + f_rr[5]) - 1/2 * λ_max * (rho_e_rr  - rho_e_ll)

  return SVector(f1, f2, f3, f4, f5)
end


@inline function max_abs_speeds(u, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack cv1, cv2, gamma1, gamma2 = equations
  rho1, rho2, rho_v1, rho_v2, rho_e = u

  rho   = rho1 + rho2
  v1    = rho_v1 / rho
  v2    = rho_v2 / rho

  gamma = (rho1*cv1*gamma1 + rho2*cv2*gamma2)/(rho1*cv1 + rho2*cv2)
  p     = (gamma - 1) * (rho_e - 1/2 * rho * (v1^2 + v2^2))
  c     = sqrt(gamma * p / rho)

  return abs(v1) + c, abs(v2) + c
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack cv1, cv2, gamma1, gamma2 = equations
  rho1, rho2, rho_v1, rho_v2, rho_e = u

  rho   = rho1 + rho2
  v1    = rho_v1 / rho
  v2    = rho_v2 / rho
  gamma = (rho1*cv1*gamma1 + rho2*cv2*gamma2)/(rho1*cv1 + rho2*cv2)
  p     = (gamma - 1) * (rho_e - 0.5 * rho * (v1^2 + v2^2))

  return SVector(rho1, rho2, v1, v2, p)
end


# Convert conservative variables to entropy
@inline function cons2entropy(u, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack cv1, cv2, gamma1, gamma2, gas_constant1, gas_constant2 = equations
  rho1, rho2, rho_v1, rho_v2, rho_e = u

  rho       = rho1 + rho2
  v1        = rho_v1 / rho
  v2        = rho_v2 / rho
  v_square  = v1^2 + v2^2
  gamma     = (rho1*cv1*gamma1 + rho2*cv2*gamma2)/(rho1*cv1 + rho2*cv2)
  p         = (gamma - 1) * (rho_e - 0.5 * rho * v_square)
  s         = log(p) - gamma*log(rho)
  rho_p     = rho / p

  # Multicomponent stuff
  T         = (rho_e - 0.5 * rho * v_square) / (rho1 * cv1 + rho2 * cv2)
  s1        = cv1 * log(T) - gas_constant1 * log(rho1)
  s2        = cv2 * log(T) - gas_constant2 * log(rho2)

  # Entropy variables
  w1        = -s1 + gas_constant1 + cv1 - (v_square / (2*T)) # + e01/T (bec. e01 = 0 for compressible euler) (DONT confuse it with e1 which is e1 = cv * T)
  w2        = -s2 + gas_constant2 + cv2 - (v_square / (2*T)) # + e02/T (bec. e02 = 0 for compressible euler)
  w3        = (v1)/T
  w4        = (v2)/T
  w5        = (-1.0)/T 

  return SVector(w1, w2, w3, w4, w5)
end


# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack cv1, cv2, gamma1, gamma2 = equations
  rho1, rho2, v1, v2, p = prim
  rho     = rho1 + rho2

  gamma  = (rho1*cv1*gamma1 + rho2*cv2*gamma2)/(rho1*cv1 + rho2*cv2)
  rho_v1  = rho * v1
  rho_v2  = rho * v2
  rho_e   = p/(gamma-1) + 0.5 * (rho_v1 * v1 + rho_v2 * v2)

  return SVector(rho1, rho2, rho_v1, rho_v2, rho_e)
end


@inline function density_pressure(u, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack cv1, cv2, gamma1, gamma2 = equations
 rho1, rho2, rho_v1, rho_v2, rho_e = u
 
 rho          = rho1 + rho2
 gamma        = (rho1*cv1*gamma1 + rho2*cv2*gamma2)/(rho1*cv1 + rho2*cv2)
 rho_times_p  = (gamma - 1) * (rho * rho_e - 0.5 * (rho_v1^2 + rho_v2^2))
 return rho_times_p
end

