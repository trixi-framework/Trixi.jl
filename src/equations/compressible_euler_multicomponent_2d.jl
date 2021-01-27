
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
  CompressibleEulerMulticomponentEquations2D{NCOMP, NVARS, RealT}(gamma, gas_constant) where {NCOMP, NVARS, RealT<:Real} = new(gamma, gas_constant, gas_constant./(gamma.-1), gas_constant .+ gas_constant./(gamma .-1))
end


function CompressibleEulerMulticomponentEquations2D(; gamma, gas_constant)

  _gamma        = promote(gamma...)
  _gas_constant = promote(gas_constant...)
  T             = promote_type(eltype(_gamma), eltype(_gas_constant))

  length(_gamma) == length(_gas_constant) || throw(DimensionMismatch("gamma and gas_constant should have the same length"))
  NVARS = length(_gamma) + 3
  NCOMP = length(_gamma)

  return CompressibleEulerMulticomponentEquations2D{NCOMP, NVARS, T}(map(T, _gamma), map(T, _gas_constant))
end


get_name(::CompressibleEulerMulticomponentEquations2D) = "CompressibleEulerMulticomponentEquations2D"


function varnames(::typeof(cons2cons), equations::CompressibleEulerMulticomponentEquations2D{NCOMP, NVARS}) where {NCOMP, NVARS}
  
  cons  = ("rho_v1", "rho_v2", "rho_e")
  rhos  = ntuple(n -> "rho" * string(n), Val(NCOMP))
  return SVector{NVARS}(cons..., rhos...)
end

                                                                                
function varnames(::typeof(cons2prim), equations::CompressibleEulerMulticomponentEquations2D{NCOMP, NVARS}) where {NCOMP, NVARS}

  prim  = ("v1", "v2", "p")
  rhos  = ntuple(n -> "rho" * string(n), Val(NCOMP))  
  return SVector{NVARS}(prim..., rhos...)
end


 # Set initial conditions at physical location `x` for time `t`

"""
    initial_condition_convergence_test(x, t, equations::CompressibleEulerMulticomponentEquations2D)

A smooth initial condition used for convergence tests in combination with
[`source_terms_convergence_test`](@ref)
(and [`boundary_condition_convergence_test`](@ref) in non-periodic domains).
"""
function initial_condition_convergence_test(x, t, equations::CompressibleEulerMulticomponentEquations2D{NCOMP, NVARS}) where {NCOMP, NVARS}
  @unpack cv, gamma = equations
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
  prim_rho  = SVector{NCOMP, eltype(gamma)}(2^(i-1) * (1-2)/(1-2^NCOMP) * rho for i = 1:NCOMP)

  prim1 = rho * v1
  prim2 = rho * v2
  prim3 = rho^2

  prim_else = SVector{3, eltype(gamma)}(prim1, prim2, prim3)

  return vcat(prim_else, prim_rho)
end

"""
    source_terms_convergence_test(u, x, t, equations::CompressibleEulerMulticomponentEquations2D)

Source terms used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref)
(and [`boundary_condition_convergence_test`](@ref) in non-periodic domains).
"""
@inline function source_terms_convergence_test(u, x, t, equations::CompressibleEulerMulticomponentEquations2D{NCOMP, NVARS}) where {NCOMP, NVARS}
  @unpack gamma = equations
  # Same settings as in `initial_condition`
  c       = 2
  A       = 0.1
  L       = 2
  f       = 1/L
  omega   = 2 * pi * f

  gammas  = totalgamma(u, equations)

  x1, x2  = x
  si, co  = sincos((x1 + x2 - t)*omega)
  tmp1    = co * A * omega
  tmp2    = si * A
  tmp3    = gammas - 1
  tmp4    = (2*c - 1)*tmp3
  tmp5    = (2*tmp2*gammas - 2*tmp2 + tmp4 + 1)*tmp1
  tmp6    = tmp2 + c

  du_rho  = SVector{NCOMP, eltype(gamma)}(2^(i-1) * (1-2)/(1-2^NCOMP) * tmp1 for i = 1:NCOMP)

  du1 = tmp5
  du2 = tmp5
  du3 = 2*((tmp6 - 1.0)*tmp3 + tmp6*gammas)*tmp1

  du_else  = SVector{3, eltype(gamma)}(du1, du2, du3)

  return vcat(du_else, du_rho)
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
    initial_condition_shock_bubble(x, t, equations::CompressibleEulerMulticomponentEquations2D{2})

A shock-bubble testcase for multicomponent Euler equations
- Ayoub Gouasmi, Karthik Duraisamy, Scott Murman
  Formulation of Entropy-Stable schemes for the multicomponent compressible Euler equations
  [arXiv: 1904.00972](https://arxiv.org/abs/1904.00972)
"""
function initial_condition_shock_bubble(x, t, equations::CompressibleEulerMulticomponentEquations2D{2})
  # bubble test case, see Gouasmi et al. https://arxiv.org/pdf/1904.00972
  # other reference: https://www.researchgate.net/profile/Pep_Mulet/publication/222675930_A_flux-split_algorithm_applied_to_conservative_models_for_multicomponent_compressible_flows/links/568da54508aeaa1481ae7af0.pdf
  # typical domain is rectangular, we change it to a square, as Trixi can only do squares
  @unpack gas_constant = equations

  # Positivity Preserving Parameter, can be set to zero if scheme is positivity preserving
  delta   = 0.03

  # Region I
  rho1_1  = delta
  rho2_1  = 1.225 * gas_constant[1]/gas_constant[2] - delta
  v1_1    = zero(delta)
  v2_1    = zero(delta)
  p_1     = 101325

  # Region II 
  rho1_2  = 1.225-delta
  rho2_2  = delta
  v1_2    = zero(delta)
  v2_2    = zero(delta)
  p_2     = 101325

  # Region III 
  rho1_3  = 1.6861 - delta
  rho2_3  = delta
  v1_3    = -113.5243
  v2_3    = zero(delta)
  p_3     = 159060

  # Set up Region I & II:
  inicenter = SVector(zero(delta), zero(delta))
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

"""
    initial_condition_shock_bubble(x, t, equations::CompressibleEulerMulticomponentEquations2D{3})

Adaption of the shock-bubble testcase for multicomponent Euler equations to 3 components
- Ayoub Gouasmi, Karthik Duraisamy, Scott Murman
  Formulation of Entropy-Stable schemes for the multicomponent compressible Euler equations
  [arXiv: 1904.00972](https://arxiv.org/abs/1904.00972)
"""
function initial_condition_shock_bubble(x, t, equations::CompressibleEulerMulticomponentEquations2D{3})
  # bubble test case, see Gouasmi et al. https://arxiv.org/pdf/1904.00972
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
  v1_1a     = zero(delta)
  v2_1a     = zero(delta)
  p_1a      = 101325

  # Region Ib
  rho1_1b   = delta
  rho2_1b   = delta
  rho3_1b   = 1.225 * gas_constant[1]/gas_constant[3] - delta
  v1_1b     = zero(delta)
  v2_1b     = zero(delta)
  p_1b      = 101325

  # Region II 
  rho1_2    = 1.225-delta
  rho2_2    = delta
  rho3_2    = delta
  v1_2      = zero(delta)
  v2_2      = zero(delta)
  p_2       = 101325

  # Region III 
  rho1_3    = 1.6861 - delta
  rho2_3    = delta
  rho3_3    = delta
  v1_3      = -113.5243
  v2_3      = zero(delta)
  p_3       = 159060

  # Set up Region I & II:
  inicenter_a = SVector(zero(delta), 1.8)
  x_norm_a    = x[1] - inicenter_a[1]
  y_norm_a    = x[2] - inicenter_a[2]
  r_a         = sqrt(x_norm_a^2 + y_norm_a^2)  

  inicenter_b = SVector(zero(delta), -1.8)
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
function initial_condition_weak_blast_wave(x, t, equations::CompressibleEulerMulticomponentEquations2D{NCOMP, NVARS}) where {NCOMP, NVARS}
  # From Hennemann & Gassner JCP paper 2020 (Sec. 6.3)
  @unpack gamma = equations
  # Set up polar coordinates
  inicenter         = SVector(0.0, 0.0)
  x_norm            = x[1] - inicenter[1]
  y_norm            = x[2] - inicenter[2]
  r                 = sqrt(x_norm^2 + y_norm^2)
  phi               = atan(y_norm, x_norm)
  sin_phi, cos_phi  = sincos(phi)

  prim_rho          = SVector{NCOMP, eltype(gamma)}(r > 0.5 ? 2^(i-1) * (1-2)/(1-2^NCOMP)*1.0 : 2^(i-1) * (1-2)/(1-2^NCOMP)*1.1691 for i = 1:NCOMP)

  v1                = r > 0.5 ? 0.0 : 0.1882 * cos_phi
  v2                = r > 0.5 ? 0.0 : 0.1882 * sin_phi
  p                 = r > 0.5 ? 1.0 : 1.245

  prim_else         = SVector{3, eltype(gamma)}(v1, v2, p)
  
  return prim2cons(vcat(prim_else, prim_rho),equations)
end


# Calculate 1D flux for a single point
@inline function calcflux(u, orientation, equations::CompressibleEulerMulticomponentEquations2D{NCOMP, NVARS}) where {NCOMP, NVARS}
  @unpack gamma = equations
  rho_v1, rho_v2, rho_e  = u

  rho = density(u, equations)

  v1    = rho_v1/rho
  v2    = rho_v2/rho
  gammas= totalgamma(u, equations)
  p     = (gammas - 1) * (rho_e - 0.5 * rho * (v1^2 + v2^2))

  if orientation == 1
    f_rho       = SVector{NCOMP, eltype(gamma)}(u[i+3]*v1 for i = 1:NCOMP)
    f1  = rho_v1 * v1 + p
    f2  = rho_v2 * v1
    f3  = (rho_e + p) * v1
  else
    f_rho       = SVector{NCOMP, eltype(gamma)}(u[i+3]*v2 for i = 1:NCOMP)
    f1  = rho_v1 * v2
    f2  = rho_v2 * v2 + p
    f3  = (rho_e + p) * v2
  end

  f_else  = SVector{3, eltype(gamma)}(f1, f2, f3)

  return vcat(f_else, f_rho) 
end


"""
    flux_chandrashekar(u_ll, u_rr, orientation, equations::CompressibleEulerMulticomponentEquations2D)

Entropy conserving two-point flux by
- Ayoub Gouasmi, Karthik Duraisamy (2020)
  "Formulation of Entropy-Stable schemes for the multicomponent compressible Euler equations""
  arXiv:1904.00972v3 [math.NA] 4 Feb 2020
"""
@inline function flux_chandrashekar(u_ll, u_rr, orientation, equations::CompressibleEulerMulticomponentEquations2D{NCOMP, NVARS}) where {NCOMP, NVARS}
  # Unpack left and right state
  @unpack gamma, gas_constant, cv = equations
  rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr
  rhok_mean   = SVector{NCOMP, eltype(gamma)}(ln_mean(u_ll[i+3], u_rr[i+3]) for i = 1:NCOMP)
  rhok_avg    = SVector{NCOMP, eltype(gamma)}(0.5 * (u_ll[i+3] + u_rr[i+3]) for i = 1:NCOMP)

  # Iterating over all partial densities
  rho_ll      = density(u_ll, equations)
  rho_rr      = density(u_rr, equations)

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

  enth      = zero(v_sum)
  help1_ll  = zero(v1_ll)
  help1_rr  = zero(v1_rr)

  for i = 1:NCOMP
    enth      += rhok_avg[i] * gas_constant[i] 
    help1_ll  += u_ll[i+3] * cv[i] 
    help1_rr  += u_rr[i+3] * cv[i]
  end

  T_ll        = (rho_e_ll - 0.5 * rho_ll * (v1_ll^2 + v2_ll^2)) / help1_ll 
  T_rr        = (rho_e_rr - 0.5 * rho_rr * (v1_rr^2 + v2_rr^2)) / help1_rr 
  T           = 0.5 * (1.0/T_ll + 1.0/T_rr)
  T_log       = ln_mean(1.0/T_ll, 1.0/T_rr)

  # Calculate fluxes depending on orientation
  help1       = zero(T_ll)
  help2       = zero(T_rr)
  if orientation == 1
    f_rho       = SVector{NCOMP, eltype(gamma)}(rhok_mean[i]*v1_avg for i = 1:NCOMP)
    for i = 1:NCOMP
      help1     += f_rho[i] * cv[i] 
      help2     += f_rho[i]
    end
    f1 = (help2) * v1_avg + enth/T 
    f2 = (help2) * v2_avg
    f3 = (help1)/T_log - 0.5 * (v1_square + v2_square) * (help2) + v1_avg * f1 + v2_avg * f2
  else
    f_rho       = SVector{NCOMP, eltype(gamma)}(rhok_mean[i]*v2_avg for i = 1:NCOMP)
    for i = 1:NCOMP
      help1     += f_rho[i] * cv[i] 
      help2     += f_rho[i]
    end
    f1 = (help2) * v1_avg 
    f2 = (help2) * v2_avg + enth/T
    f3 = (help1)/T_log - 0.5 * (v1_square + v2_square) * (help2) + v1_avg * f1 + v2_avg * f2
  end
  f_else  = SVector{3, eltype(gamma)}(f1, f2, f3)

  return vcat(f_else, f_rho)
end


function flux_lax_friedrichs(u_ll, u_rr, orientation, equations::CompressibleEulerMulticomponentEquations2D{NCOMP, NVARS}) where {NCOMP, NVARS}
  @unpack gamma = equations
  # Calculate primitive variables and speed of sound
  rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr

  rho_ll   = density(u_ll, equations)
  rho_rr   = density(u_rr, equations)
  gamma_ll = totalgamma(u_ll, equations)
  gamma_rr = totalgamma(u_rr, equations)

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
  
  f  = SVector{NVARS, eltype(gamma)}(1/2 * (f_ll[i] + f_rr[i]) - 1/2 * λ_max * (u_rr[i] - u_ll[i]) for i = 1:NVARS)

  return f
end


@inline function max_abs_speeds(u, equations::CompressibleEulerMulticomponentEquations2D)
  rho_v1, rho_v2, rho_e = u

  rho   = density(u, equations)
  v1    = rho_v1 / rho
  v2    = rho_v2 / rho

  gammas= totalgamma(u, equations)
  p     = (gammas - 1) * (rho_e - 1/2 * rho * (v1^2 + v2^2))
  c     = sqrt(gammas * p / rho)

  return abs(v1) + c, abs(v2) + c
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::CompressibleEulerMulticomponentEquations2D{NCOMP, NVARS}) where {NCOMP, NVARS}
  @unpack cv, gamma = equations
  rho_v1, rho_v2, rho_e = u

  prim_rho = SVector{NCOMP, eltype(gamma)}(u[i+3] for i = 1:NCOMP)

  rho   = density(u, equations)
  v1    = rho_v1 / rho
  v2    = rho_v2 / rho
  gammas= totalgamma(u, equations)
  p     = (gammas - 1) * (rho_e - 0.5 * rho * (v1^2 + v2^2))
  prim_else =  SVector{3, eltype(gamma)}(v1, v2, p)

  return vcat(prim_else, prim_rho)
end


# Convert conservative variables to entropy
@inline function cons2entropy(u, equations::CompressibleEulerMulticomponentEquations2D{NCOMP, NVARS}) where {NCOMP, NVARS}
  @unpack cv, gamma, gas_constant = equations
  rho_v1, rho_v2, rho_e = u

  rho       = density(u, equations)

  v1        = rho_v1 / rho
  v2        = rho_v2 / rho
  v_square  = v1^2 + v2^2
  gammas    = totalgamma(u, equations)
  p         = (gammas - 1) * (rho_e - 0.5 * rho * v_square)
  s         = log(p) - gammas*log(rho)
  rho_p     = rho / p

  # Multicomponent stuff
  help1 = zero(v1)

  for i = 1:NCOMP
    help1 += u[i+3] * cv[i]
  end

  T         = (rho_e - 0.5 * rho * v_square) / (help1)

  entrop_rho  = SVector{NCOMP, eltype(gamma)}( -1.0 * (cv[i] * log(T) - gas_constant[i] * log(u[i+3])) + gas_constant[i] + cv[i] - (v_square / (2*T)) for i = 1:NCOMP)

  w1        = (v1)/T
  w2        = (v2)/T
  w3        = (-1.0)/T 

  entrop_else = SVector{3, eltype(gamma)}(w1, w2, w3)

  return vcat(entrop_else, entrop_rho)
end


# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::CompressibleEulerMulticomponentEquations2D{NCOMP, NVARS}) where {NCOMP, NVARS}
  @unpack cv, gamma = equations
  v1, v2, p = prim
  
  cons_rho  = SVector{NCOMP, eltype(gamma)}(prim[i+3] for i = 1:NCOMP)

  rho     = density(prim, equations)
  gammas  = totalgamma(prim, equations)

  rho_v1  = rho * v1
  rho_v2  = rho * v2
  rho_e   = p/(gammas-1) + 0.5 * (rho_v1 * v1 + rho_v2 * v2)

  cons_else = SVector{3, eltype(gamma)}(rho_v1, rho_v2, rho_e)

  return vcat(cons_else, cons_rho)
end

@inline function totalgamma(u, equations::CompressibleEulerMulticomponentEquations2D{NCOMP, NVARS}) where {NCOMP, NVARS}
  @unpack cv, gamma = equations

  help1 = zero(u[1])
  help2 = zero(u[1])

  for i in 1:NCOMP
    help1 += u[i+3] * cv[i] * gamma[i]
    help2 += u[i+3] * cv[i]
  end

  return help1/help2
end


@inline function density_pressure(u, equations::CompressibleEulerMulticomponentEquations2D)
  rho_v1, rho_v2, rho_e = u
  
  rho          = density(u, equations)
  gammas       = totalgamma(u, equations)
  rho_times_p  = (gammas - 1) * (rho * rho_e - 0.5 * (rho_v1^2 + rho_v2^2))

  return rho_times_p
end


@inline function density(u, equations::CompressibleEulerMulticomponentEquations2D{NCOMP, NVARS}) where {NCOMP, NVARS}
  rho = zero(u[1])

  for i = 1:NCOMP
    rho += u[i+3]
  end

  return rho
 end

