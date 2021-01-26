
@doc raw"""
    CompressibleEulerMulticomponentEquations2D

!!! warning "Experimental code"
    This system of equations is experimental and can change any time.

Multicomponent version of the compressible Euler equations for an ideal gas in two space dimensions.
"""

struct CompressibleEulerMulticomponentEquations2D{NCOMP, NVARS, RealT<:Real} <: AbstractCompressibleEulerMulticomponentEquations{2, NVARS}
  gamma             ::SVector{NCOMP, RealT} 
  gas_constant      ::SVector{NCOMP, RealT}
  cv                ::SVector{NCOMP, RealT}
  cp                ::SVector{NCOMP, RealT}
  CompressibleEulerMulticomponentEquations2D{NCOMP, NVARS, RealT}(gamma, gas_constant) where {NCOMP, NVARS, RealT<:Real} = new(gamma, gas_constant, gas_constant./(gamma.-1), gas_constant + gas_constant./(gamma .-1))
end


function CompressibleEulerMulticomponentEquations2D(; gamma, gas_constant)

  length(gamma) == length(gas_constant) || throw(DimensionMismatch("gamma and gas_constant should have the same length"))
  NVARS = length(gamma) + 3
  NCOMP = length(gamma)

  return CompressibleEulerMulticomponentEquations2D{NCOMP, NVARS, Float64}(gamma, gas_constant)
end


get_name(::CompressibleEulerMulticomponentEquations2D) = "CompressibleEulerMulticomponentEquations2D"


function varnames(::typeof(cons2cons), equations::CompressibleEulerMulticomponentEquations2D)
  @unpack gamma = equations

  NCOMP = length(gamma)
  NVARS = length(gamma) + 3
  
  cons  = ["rho_v1", "rho_v2", "rho_e"]
  
  for i in 1:NCOMP
    add   = [string("rho",i)]
    append!(cons, add)
  end

  result  = SVector{NVARS}(cons)

  return result

end

                                                                                
function varnames(::typeof(cons2prim), equations::CompressibleEulerMulticomponentEquations2D)
  @unpack gamma = equations

  NCOMP = length(gamma)
  NVARS = length(gamma) + 3
  
  prim  = ["v1", "v2", "p"]
  
  for i in 1:NCOMP
    add   = [string("rho",i)]
    append!(prim, add)
  end

  result  = SVector{NVARS}(prim)

  return result

end


 # Set initial conditions at physical location `x` for time `t`

"""
    initial_condition_convergence_test(x, t, equations::CompressibleEulerMulticomponentEquations2D)

A smooth initial condition used for convergence tests in combination with
[`source_terms_convergence_test`](@ref)
(and [`boundary_condition_convergence_test`](@ref) in non-periodic domains).
"""
function initial_condition_convergence_test(x, t, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack cv, gamma = equations
  prim    = zeros(length(gamma)+3)
  c       = 2
  A       = 0.1
  L       = 2
  f       = 1/L
  omega   = 2 * pi * f
  ini     = c + A * sin(omega * (x[1] + x[2] - t))

  v1      = 1.0
  v2      = 1.0

  rho     = ini
  # Multiple Components
  for i in 1:length(gamma)
    factor      = 2^(i-1) * (1-2)/(1-2^length(gamma)) # divides 1 by NCOMP, each fraction is double the previous fraction.
    prim[i+3]   = factor * rho
  end

  prim[1] = rho * v1
  prim[2] = rho * v2
  prim[3] = rho^2

  result  = SVector{length(gamma)+3}(prim)

  return result 
end

"""
    source_terms_convergence_test(u, x, t, equations::CompressibleEulerMulticomponentEquations2D)

Source terms used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref)
(and [`boundary_condition_convergence_test`](@ref) in non-periodic domains).
"""
@inline function source_terms_convergence_test(u, x, t, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack cv, gamma = equations
  du    = zeros(length(gamma)+3)
  # Same settings as in `initial_condition`
  c       = 2
  A       = 0.1
  L       = 2
  f       = 1/L
  omega   = 2 * pi * f

  gammas  = getgamma(u, equations)

  x1, x2  = x
  si, co  = sincos((x1 + x2 - t)*omega)
  tmp1    = co * A * omega
  tmp2    = si * A
  tmp3    = gammas - 1
  tmp4    = (2*c - 1)*tmp3
  tmp5    = (2*tmp2*gammas - 2*tmp2 + tmp4 + 1)*tmp1
  tmp6    = tmp2 + c

  for i in 1:length(gamma)
    factor  = 2^(i-1) * (1-2)/(1-2^length(gamma)) # divides 1 by NCOMP, each fraction is double the previous fraction.
    du[i+3] = factor * tmp1
  end

  du[1] = tmp5
  du[2] = tmp5
  du[3] = 2*((tmp6 - 1.0)*tmp3 + tmp6*gammas)*tmp1

  result  = SVector{length(gamma)+3}(du)

  return result 
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
  @unpack gas_constant = equations

  # Positivity Preserving Parameter, can be set to zero if scheme is positivity preserving
  delta   = 0.03

  # Region I
  rho1_1  = delta
  rho2_1  = 1.225 * gas_constant[1]/gas_constant[2] - delta
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

  return prim2cons(SVector(v1, v2, p, rho1, rho2), equations)
end


function initial_condition_shock_bubble_3comp(x, t, equations::CompressibleEulerMulticomponentEquations2D)
  # bubble test case, see Agertz et al. https://arxiv.org/pdf/1904.00972
  # other reference: https://www.researchgate.net/profile/Pep_Mulet/publication/222675930_A_flux-split_algorithm_applied_to_conservative_models_for_multicomponent_compressible_flows/links/568da54508aeaa1481ae7af0.pdf
  # adapted to 3 component testcase
  # typical domain is rectangular, we change it to a square, as Trixi can only do squares
  
  @unpack gas_constant = equations

  # Positivity Preserving Parameter, can be set to zero if scheme is positivity preserving
  delta   = 0.05

  # Region Ia
  rho1_1a   = delta
  rho2_1a   = 1.225 * gas_constant[1]/gas_constant[2] - delta
  rho3_1a   = delta
  v1_1a     = 0.0
  v2_1a     = 0.0
  p_1a      = 101325

  # Region Ib
  rho1_1b   = delta
  rho2_1b   = delta
  rho3_1b   = 1.225 * gas_constant[1]/gas_constant[3] - delta
  v1_1b     = 0.0
  v2_1b     = 0.0
  p_1b      = 101325

  # Region II 
  rho1_2    = 1.225-delta
  rho2_2    = delta
  rho3_2    = delta
  v1_2      = 0.0
  v2_2      = 0.0
  p_2       = 101325

  # Region III 
  rho1_3    = 1.6861 - delta
  rho2_3    = delta
  rho3_3    = delta
  v1_3      = -113.5243
  v2_3      = 0.0
  p_3       = 159060

  # Set up Region I & II:
  inicenter_a = SVector(0.0, 1.8)
  x_norm_a    = x[1] - inicenter_a[1]
  y_norm_a    = x[2] - inicenter_a[2]
  r_a         = sqrt(x_norm_a^2 + y_norm_a^2)  

  inicenter_b = SVector(0.0, -1.8)
  x_norm_b    = x[1] - inicenter_b[1]
  y_norm_b    = x[2] - inicenter_b[2]
  r_b         = sqrt(x_norm_b^2 + y_norm_b^2)  


  if (x[1] > 0.50)
    # Set up Region III
    rho1    = rho1_3
    rho2    = rho2_3
    rho3    = rho3_3
    v1      = v1_3 
    v2      = v2_3
    p       = p_3
  elseif (r_a < 0.25)
    # Set up Region I
    rho1    = rho1_1a
    rho2    = rho2_1a
    rho3    = rho3_1a
    v1      = v1_1a
    v2      = v2_1a
    p       = p_1a
  elseif (r_b < 0.25)
    rho1    = rho1_1b
    rho2    = rho2_1b
    rho3    = rho3_1b
    v1      = v1_1b
    v2      = v2_1b
    p       = p_1b
  else
    # Set up Region II
    rho1    = rho1_2
    rho2    = rho2_2
    rho3    = rho3_2
    v1      = v1_2
    v2      = v2_2
    p       = p_2
  end 

  return prim2cons(SVector(v1, v2, p, rho1, rho2, rho3), equations)
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
  @unpack gamma, cv, gas_constant, cp = equations
  prim              = zeros(length(gamma)+3)
  # Set up polar coordinates
  inicenter         = SVector(0.0, 0.0)
  x_norm            = x[1] - inicenter[1]
  y_norm            = x[2] - inicenter[2]
  r                 = sqrt(x_norm^2 + y_norm^2)
  phi               = atan(y_norm, x_norm)
  sin_phi, cos_phi  = sincos(phi)

  # Multiple Components
  for i in 1:length(gamma)
    factor      = 2^(i-1) * (1-2)/(1-2^length(gamma)) # divides 1 by NCOMP, each fraction is double the previous fraction.
    prim[i+3]   = r > 0.5 ? factor*1.0 : factor*1.1691
  end

  v1                = r > 0.5 ? 0.0 : 0.1882 * cos_phi
  v2                = r > 0.5 ? 0.0 : 0.1882 * sin_phi
  p                 = r > 0.5 ? 1.0 : 1.245

  prim[1:3]         = [v1, v2, p]

  result            = SVector{length(gamma)+3}(prim)
  
  return prim2cons(result, equations)
end


# Calculate 1D flux for a single point
@inline function calcflux(u, orientation, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack gamma, cv = equations
  rho_v1, rho_v2, rho_e  = u
  f         = zeros(length(gamma)+3)
  rho       = 0

  for i = 1:length(gamma)
    rho += u[i+3]
  end

  v1    = rho_v1/rho
  v2    = rho_v2/rho
  gammas= getgamma(u, equations)
  p     = (gammas - 1) * (rho_e - 0.5 * rho * (v1^2 + v2^2))

  if orientation == 1
    for i = 1:length(gamma)
      f[i+3] = u[i+3] * v1
    end
    f1  = rho_v1 * v1 + p
    f2  = rho_v2 * v1
    f3  = (rho_e + p) * v1
  else
    for i = 1:length(gamma)
      f[i+3] = u[i+3] * v2
    end
    f1  = rho_v1 * v2
    f2  = rho_v2 * v2 + p
    f3  = (rho_e + p) * v2
  end

  f[1:3]  = [f1, f2, f3]

  result  = SVector{length(gamma)+3}(f)

  return result
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
  @unpack gamma, gas_constant, cv = equations
  rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr
  rho_ll      = 0
  rho_rr      = 0
  rhok_mean   = SVector{length(gamma),Float64}(ln_mean(u_ll[i+3], u_rr[i+3]) for i = 1:length(gamma))
  rhok_avg    = SVector{length(gamma),Float64}(0.5 * (u_ll[i+3] + u_rr[i+3]) for i = 1:length(gamma))

  # Iterating over all partial densities
  for i = 1:length(gamma)
    rho_ll        += u_ll[i+3]
    rho_rr        += u_rr[i+3]
  end 

  # extract velocities
  v1_ll       = rho_v1_ll/rho_ll
  v2_ll       = rho_v2_ll/rho_ll
  v1_rr       = rho_v1_rr/rho_rr
  v2_rr       = rho_v2_rr/rho_rr
  v1_avg      = 0.5 * (v1_ll + v1_rr)
  v2_avg      = 0.5 * (v2_ll + v2_rr)
  v1_square   = 0.5 * (v1_ll^2 + v1_rr^2)
  v2_square   = 0.5 * (v2_ll^2 + v2_rr^2)
  v_sum       = v1_avg + v2_avg

  enth      = 0
  help1_ll  = 0
  help1_rr  = 0

  for i = 1:length(gamma)
    enth      += rhok_avg[i] * gas_constant[i] 
    help1_ll  += u_ll[i+3] * cv[i] 
    help1_rr  += u_rr[i+3] * cv[i]
  end

  T_ll        = (rho_e_ll - 0.5 * rho_ll * (v1_ll^2 + v2_ll^2)) / help1_ll 
  T_rr        = (rho_e_rr - 0.5 * rho_rr * (v1_rr^2 + v2_rr^2)) / help1_rr 
  T           = 0.5 * (1.0/T_ll + 1.0/T_rr)
  T_log       = ln_mean(1.0/T_ll, 1.0/T_rr)

  # Calculate fluxes depending on orientation
  help1       = 0
  help2       = 0
  if orientation == 1
    f_rho       = SVector{length(gamma),Float64}(rhok_mean[i]*v1_avg for i = 1:length(gamma))
    for i = 1:length(gamma)
      help1     += f_rho[i] * cv[i] 
      help2     += f_rho[i]
    end
    f1 = (help2) * v1_avg + enth/T 
    f2 = (help2) * v2_avg
    f3 = (help1)/T_log - 0.5 * (v1_square + v2_square) * (help2) + v1_avg * f1 + v2_avg * f2
  else
    f_rho       = SVector{length(gamma),Float64}(rhok_mean[i]*v2_avg for i = 1:length(gamma))
    for i = 1:length(gamma)
      help1     += f_rho[i] * cv[i] 
      help2     += f_rho[i]
    end
    f1 = (help2) * v1_avg 
    f2 = (help2) * v2_avg + enth/T
    f3 = (help1)/T_log - 0.5 * (v1_square + v2_square) * (help2) + v1_avg * f1 + v2_avg * f2
  end
  f_else  = SVector{3, Float64}(f1, f2, f3)

  result  = vcat(f_else, f_rho)
  return result
end


function flux_lax_friedrichs(u_ll, u_rr, orientation, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack cv, gamma = equations
  # Calculate primitive variables and speed of sound
  rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr
  rho_ll = 0
  rho_rr = 0

  for i = 1:length(gamma)
    rho_ll += u_ll[i+3]
    rho_rr += u_rr[i+3]
  end 

  gamma_ll = getgamma(u_ll, equations)
  gamma_rr = getgamma(u_rr, equations)

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
  
  f  = SVector{length(gamma)+3,Float64}(1/2 * (f_ll[i] + f_rr[i]) - 1/2 * λ_max * (u_rr[i] - u_ll[i]) for i = 1:length(gamma)+3)

  return f
end


@inline function max_abs_speeds(u, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack cv, gamma = equations
  rho_v1, rho_v2, rho_e = u
  rho   = 0

  for i = 1:length(gamma)
    rho += u[i+3]
  end  

  v1    = rho_v1 / rho
  v2    = rho_v2 / rho

  gammas= getgamma(u, equations)
  p     = (gammas - 1) * (rho_e - 1/2 * rho * (v1^2 + v2^2))
  c     = sqrt(gammas * p / rho)

  return abs(v1) + c, abs(v2) + c
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack cv, gamma = equations
  rho_v1, rho_v2, rho_e = u
  prim  = zeros(length(gamma)+3)
  rho   = 0

  for i = 1:length(gamma)
    rho       += u[i+3]
    prim[i+3] = u[i+3]
  end 

  v1    = rho_v1 / rho
  v2    = rho_v2 / rho
  gammas= getgamma(u, equations)
  p     = (gammas - 1) * (rho_e - 0.5 * rho * (v1^2 + v2^2))
  prim[1:3] = [v1, v2, p]

  result    = SVector{length(gamma)+3}(prim)

  return result
end


# Convert conservative variables to entropy
@inline function cons2entropy(u, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack cv, gamma, gas_constant = equations
  rho_v1, rho_v2, rho_e = u
  rho       = 0

  entrop  = zeros(length(gamma)+3)

  for i = 1:length(gamma)
    rho += u[i+3]
  end 

  v1        = rho_v1 / rho
  v2        = rho_v2 / rho
  v_square  = v1^2 + v2^2
  gammas    = getgamma(u, equations)
  p         = (gammas - 1) * (rho_e - 0.5 * rho * v_square)
  s         = log(p) - gammas*log(rho)
  rho_p     = rho / p

  # Multicomponent stuff
  help1 = 0
  for i = 1:length(gamma)
    help1 += u[i+3] * cv[i]
  end

  T         = (rho_e - 0.5 * rho * v_square) / (help1)

  for i = 1:length(gamma)
    entrop[i+3] = -1.0 * (cv[i] * log(T) - gas_constant[i] * log(u[i+3])) + gas_constant[i] + cv[i] - (v_square / (2*T))
  end
  w1        = (v1)/T
  w2        = (v2)/T
  w3        = (-1.0)/T 

  entrop[1:3] = [w1, w2, w3]

  result      = SVector{length(gamma)+3}(entrop)

  return result
end


# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack cv, gamma = equations
  v1, v2, p = prim
  rho = 0
  cons = zeros(length(gamma)+3)
  
  for i = 1:length(gamma)
    rho         += prim[i+3]
    cons[i+3]   = prim[i+3]
  end

  gammas  = getgamma(prim, equations)

  rho_v1  = rho * v1
  rho_v2  = rho * v2
  rho_e   = p/(gammas-1) + 0.5 * (rho_v1 * v1 + rho_v2 * v2)

  cons[1:3] = [rho_v1, rho_v2, rho_e] 

  result    = SVector{length(gamma)+3}(cons)

  return result
end

@inline function getgamma(u, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack cv, gamma = equations

  help1 = 0.0
  help2 = 0.0
  for i in 1:length(gamma)
    help1 += u[i+3] * cv[i] * gamma[i]
    help2 += u[i+3] * cv[i]
  end

  gammas = help1/help2

  return gammas
end


@inline function density_pressure(u, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack cv, gamma = equations
  rho_v1, rho_v2, rho_e = u
  rho = 0
  
  for i = 1:length(gamma)
    rho += u[i+3]
  end

  gammas       = getgamma(u, equations)
  rho_times_p  = (gammas - 1) * (rho * rho_e - 0.5 * (rho_v1^2 + rho_v2^2))

  return rho_times_p
end


@inline function density(u, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack gamma = equations
  rho = 0

  for i = 1:length(gamma)
    rho += u[i+3]
  end

  return rho
 end

