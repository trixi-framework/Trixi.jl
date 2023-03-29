# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


@doc raw"""
    IdealGlmMhdMultiIonEquations1D

The ideal compressible multi-ion GLM-MHD equations in one space dimension.

* Until now, actually without GLM
"""
mutable struct IdealGlmMhdMultiIonEquations1D{NVARS, NCOMP, RealT<:Real} <: AbstractIdealGlmMhdMultiIonEquations{1, NVARS, NCOMP}
  gammas            ::SVector{NCOMP, RealT} # Heat capacity rations
  charge_to_mass    ::SVector{NCOMP, RealT} # Charge to mass ratios

  function IdealGlmMhdMultiIonEquations1D{NVARS, NCOMP, RealT}(gammas       ::SVector{NCOMP, RealT},
                                                                     charge_to_mass::SVector{NCOMP, RealT}) where {NVARS, NCOMP, RealT<:Real}

    NCOMP >= 1 || throw(DimensionMismatch("`gammas` and `charge_to_mass` have to be filled with at least one value"))

    new(gammas, charge_to_mass)
  end
end

function IdealGlmMhdMultiIonEquations1D(; gammas, charge_to_mass)

  _gammas         = promote(gammas...)
  _charge_to_mass = promote(charge_to_mass...)
  RealT           = promote_type(eltype(_gammas), eltype(_charge_to_mass))

  NVARS = length(_gammas) * 5 + 3
  NCOMP = length(_gammas)

  __gammas        = SVector(map(RealT, _gammas))
  __charge_to_mass = SVector(map(RealT, _charge_to_mass))

  return IdealGlmMhdMultiIonEquations1D{NVARS, NCOMP, RealT}(__gammas, __charge_to_mass)
end

@inline Base.real(::IdealGlmMhdMultiIonEquations1D{NVARS, NCOMP, RealT}) where {NVARS, NCOMP, RealT} = RealT

have_nonconservative_terms(::IdealGlmMhdMultiIonEquations1D) = False() #TODO: Change to True() after testing fluxes

function varnames(::typeof(cons2cons), equations::IdealGlmMhdMultiIonEquations1D)

  cons  = ("B1", "B2", "B3")
  for i in eachcomponent(equations)
    cons = (cons..., tuple("rho" * string(i),"rho_v1" * string(i), "rho_v2" * string(i), "rho_v3" * string(i), "rho_e" * string(i))...)
  end
  
  return cons
end

function varnames(::typeof(cons2prim), equations::IdealGlmMhdMultiIonEquations1D)

  prim  = ("B1", "B2", "B3")
  for i in eachcomponent(equations)
    prim = (prim..., tuple("rho" * string(i),"v1" * string(i), "v2" * string(i), "v3" * string(i), "p" * string(i))...)
  end
  
  return prim
end


# """
#     initial_condition_convergence_test(x, t, equations::IdealGlmMhdMultiIonEquations1D)

# An Alfvén wave as smooth initial condition used for convergence tests.
# """
# function initial_condition_convergence_test(x, t, equations::IdealGlmMhdMultiIonEquations1D)
#   # smooth Alfvén wave test from Derigs et al. FLASH (2016)
#   # domain must be set to [0, 1], γ = 5/3

#   rho = 1.0
#   prim_rho  = SVector{ncomponents(equations), real(equations)}(2^(i-1) * (1-2)/(1-2^ncomponents(equations)) * rho for i in eachcomponent(equations))
#   v1 = 0.0
#   si, co = sincos(2 * pi * x[1])
#   v2 = 0.1 * si
#   v3 = 0.1 * co
#   p = 0.1
#   B1 = 1.0
#   B2 = v2
#   B3 = v3
#   prim_other = SVector{7, real(equations)}(v1, v2, v3, p, B1, B2, B3)
#   return prim2cons(vcat(prim_other, prim_rho), equations)
# end


"""
    initial_condition_weak_blast_wave(x, t, equations::IdealGlmMhdMultiIonEquations1D)

A weak blast wave adapted from
- Sebastian Hennemann, Gregor J. Gassner (2020)
  A provably entropy stable subcell shock capturing approach for high order split form DG
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
function initial_condition_weak_blast_wave(x, t, equations::IdealGlmMhdMultiIonEquations1D)
  # Adapted MHD version of the weak blast wave from Hennemann & Gassner JCP paper 2020 (Sec. 6.3)
  # Same discontinuity in the velocities but with magnetic fields
  # Set up polar coordinates
  inicenter = (0)
  x_norm = x[1] - inicenter[1]
  r = sqrt(x_norm^2)
  phi = atan(x_norm)

  # Calculate primitive variables
  rho = zero(real(equations))
  if r > 0.5
    rho = 1.0
  else
    rho = 1.1691
  end
  v1 = r > 0.5 ? 0.0 : 0.1882 * cos(phi)
  p = r > 0.5 ? 1.0 : 1.245

  prim = (1.0, 1.0, 1.0)
  for i in eachcomponent(equations)
    prim = (prim..., 2^(i-1) * (1-2)/(1-2^ncomponents(equations)) * rho, v1, 0.0, 0.0, p)
  end

  return prim2cons(SVector{nvariables(equations), real(equations)}(prim), equations)
end


# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer, equations::IdealGlmMhdMultiIonEquations1D)
  B1, B2, B3, _ = u
  
  total_electron_charge, v1_plus, v2_plus, v3_plus, vk1_plus, vk2_plus, vk3_plus = auxiliary_variables(u, equations)

  f_B1 = 0.0
  f_B2 = v1_plus * B2 - v2_plus * B1
  f_B3 = v1_plus * B3 - v3_plus * B1

  f = (f_B1, f_B2, f_B3)

  mag_en = 0.5*(B1^2 + B2^2 + B3^2)

  for k in eachcomponent(equations)
    rho, rho_v1, rho_v2, rho_v3, rho_e = get_component(k, u, equations)
    v1 = rho_v1/rho
    v2 = rho_v2/rho
    v3 = rho_v3/rho
    kin_en = 0.5 * rho * (v1^2 + v2^2 + v3^2)
    
    gamma = equations.gammas[k]
    p = (gamma - 1) * (rho_e - kin_en - mag_en)

    f1 = rho_v1
    f2 = rho_v1*v1 + p #+ mag_en - B1^2
    f3 = rho_v1*v2 #- B1*B2
    f4 = rho_v1*v3 #- B1*B3
    f5 = (kin_en + gamma*p/(gamma - 1))*v1 + + 2 * mag_en * vk1_plus[k] - B1*(vk1_plus[k] * B1 + vk2_plus[k] * B2 + vk3_plus[k] * B3)

    f = (f..., f1, f2, f3, f4, f5)
  end

  return SVector{nvariables(equations), real(equations)}(f)
end

"""
Total non-conservative two-point flux
"""
@inline function flux_nonconservative_all(u_ll, u_rr, orientation::Integer,
  equations::ShallowWaterEquations1D)

  # Compute Powell (only needed for non-constant B1)

  # Compute term 2
  
  # Compute term 3

  return f
end

# """
#     flux_derigs_etal(u_ll, u_rr, orientation, equations::IdealGlmMhdEquations1D)

# Entropy conserving two-point flux adapted by
# - Derigs et al. (2018)
#   Ideal GLM-MHD: About the entropy consistent nine-wave magnetic field
#   divergence diminishing ideal magnetohydrodynamics equations for multi-ion
#   [DOI: 10.1016/j.jcp.2018.03.002](https://doi.org/10.1016/j.jcp.2018.03.002)
# """
# function flux_derigs_etal(u_ll, u_rr, orientation::Integer, equations::IdealGlmMhdMultiIonEquations1D)
#   # Unpack left and right states to get velocities, pressure, and inverse temperature (called beta)
#   B1_ll, B2_ll, B3_ll, _ = u_ll
#   B1_rr, B2_rr, B3_rr, _ = u_rr
#   @unpack gammas = equations

#   rho_ll = density(u_ll, equations)
#   rho_rr = density(u_rr, equations)

#   gamma_ll = totalgamma(u_ll, equations)
#   gamma_rr = totalgamma(u_rr, equations)

#   rhok_mean   = SVector{ncomponents(equations), real(equations)}(ln_mean(u_ll[i+7], u_rr[i+7]) for i in eachcomponent(equations))
#   rhok_avg    = SVector{ncomponents(equations), real(equations)}(0.5 * (u_ll[i+7] + u_rr[i+7]) for i in eachcomponent(equations))

#   v1_ll = rho_v1_ll/rho_ll
#   v2_ll = rho_v2_ll/rho_ll
#   v3_ll = rho_v3_ll/rho_ll
#   v1_rr = rho_v1_rr/rho_rr
#   v2_rr = rho_v2_rr/rho_rr
#   v3_rr = rho_v3_rr/rho_rr
#   vel_norm_ll = v1_ll^2 + v2_ll^2 + v3_ll^2
#   vel_norm_rr = v1_rr^2 + v2_rr^2 + v3_rr^2
#   mag_norm_ll = B1_ll^2 + B2_ll^2 + B3_ll^2
#   mag_norm_rr = B1_rr^2 + B2_rr^2 + B3_rr^2
#   # for convenience store v⋅B
#   vel_dot_mag_ll = v1_ll*B1_ll + v2_ll*B2_ll + v3_ll*B3_ll
#   vel_dot_mag_rr = v1_rr*B1_rr + v2_rr*B2_rr + v3_rr*B3_rr

#   # Compute the necessary mean values needed for either direction
#   v1_avg = 0.5*(v1_ll+v1_rr)
#   v2_avg = 0.5*(v2_ll+v2_rr)
#   v3_avg = 0.5*(v3_ll+v3_rr)
#   v_sum  = v1_avg + v2_avg + v3_avg
#   B1_avg = 0.5*(B1_ll+B1_rr)
#   B2_avg = 0.5*(B2_ll+B2_rr)
#   B3_avg = 0.5*(B3_ll+B3_rr)
#   vel_norm_avg = 0.5*(vel_norm_ll+vel_norm_rr)
#   mag_norm_avg = 0.5*(mag_norm_ll+mag_norm_rr)
#   vel_dot_mag_avg = 0.5*(vel_dot_mag_ll+vel_dot_mag_rr)

#   enth      = zero(v_sum)
#   help1_ll  = zero(v1_ll)
#   help1_rr  = zero(v1_rr)

#   for i in eachcomponent(equations)
#     enth      += rhok_avg[i] * gas_constants[i]
#     help1_ll  += u_ll[i+7] * cv[i]
#     help1_rr  += u_rr[i+7] * cv[i]
#   end

#   T_ll        = (rho_e_ll - 0.5*rho_ll * (vel_norm_ll) - 0.5*mag_norm_ll) / help1_ll
#   T_rr        = (rho_e_rr - 0.5*rho_rr * (vel_norm_rr) - 0.5*mag_norm_rr) / help1_rr
#   T           = 0.5 * (1.0/T_ll + 1.0/T_rr)
#   T_log       = ln_mean(1.0/T_ll, 1.0/T_rr)

#   # Calculate fluxes depending on orientation with specific direction averages
#   help1       = zero(T_ll)
#   help2       = zero(T_rr)

#   f_rho       = SVector{ncomponents(equations), real(equations)}(rhok_mean[i]*v1_avg for i in eachcomponent(equations))
#   for i in eachcomponent(equations)
#     help1     += f_rho[i] * cv[i]
#     help2     += f_rho[i]
#   end
#   f1 = help2 * v1_avg + enth/T + 0.5 * mag_norm_avg - B1_avg*B1_avg
#   f2 = help2 * v2_avg - B1_avg*B2_avg
#   f3 = help2 * v3_avg - B1_avg*B3_avg
#   f5 = 0.0
#   f6 = v1_avg*B2_avg - v2_avg*B1_avg
#   f7 = v1_avg*B3_avg - v3_avg*B1_avg

#   # total energy flux is complicated and involves the previous eight components
#   v1_mag_avg = 0.5*(v1_ll*mag_norm_ll + v1_rr*mag_norm_rr)

#   f4 = (help1/T_log) - 0.5 * (vel_norm_avg) * (help2) + f1 * v1_avg + f2 * v2_avg + f3 * v3_avg +
#         f5 * B1_avg + f6 * B2_avg + f7 * B3_avg - 0.5*v1_mag_avg +
#         B1_avg * vel_dot_mag_avg


#   f_other  = SVector{7, real(equations)}(f1, f2, f3, f4, f5, f6, f7)

#   return vcat(f_other, f_rho)
# end

"""
# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
  !!!ATTENTION: This routine is provisory. TODO: Update with the right max_abs_speed
"""
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::IdealGlmMhdMultiIonEquations1D)
  # Calculate fast magnetoacoustic wave speeds
  # left
  cf_ll = calc_fast_wavespeed(u_ll, orientation, equations)
  # right
  cf_rr = calc_fast_wavespeed(u_rr, orientation, equations)

  # Calculate velocities (ignore orientation since it is always "1" in 1D)
  v_ll = zero(u_ll[1])
  v_rr = zero(u_rr[1])
  for k in eachcomponent(equations)
    rho, rho_v1, _ = get_component(k, u_ll, equations)
    v_ll = max(v_ll, abs(rho_v1 / rho))
    rho, rho_v1, _ = get_component(k, u_rr, equations)
    v_rr = max(v_rr, abs(rho_v1 / rho))
  end

  λ_max = max(abs(v_ll), abs(v_rr)) + max(cf_ll, cf_rr)
end


@inline function max_abs_speeds(u, equations::IdealGlmMhdMultiIonEquations1D)
  
  v1 = zero(u[1])
  for k in eachcomponent(equations)
    rho, rho_v1, _ = get_component(k, u, equations)
    v1 = max(v1, abs(rho_v1 / rho))
  end

  cf_x_direction = calc_fast_wavespeed(u, 1, equations)

  return (abs(v1) + cf_x_direction, )
end


"""
Convert conservative variables to primitive
"""
function cons2prim(u, equations::IdealGlmMhdMultiIonEquations1D)
  @unpack gammas = equations
  B1, B2, B3, _ = u

  prim = (B1, B2, B3)
  for k in eachcomponent(equations)
    rho, rho_v1, rho_v2, rho_v3, rho_e = get_component(k, u, equations)
    srho = 1 / rho
    v1 = srho * rho_v1
    v2 = srho * rho_v2
    v3 = srho * rho_v3

    p = (gammas[k] - 1) * (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3
                                 + B1 * B1 + B2 * B2 + B3 * B3))
    prim = (prim..., rho, v1, v2, v3, p)
  end

  return SVector{nvariables(equations), real(equations)}(prim)
end

"""
Convert conservative variables to entropy
"""
@inline function cons2entropy(u, equations::IdealGlmMhdMultiIonEquations1D)
  @unpack gammas = equations
  B1, B2, B3, _ = u

  prim = cons2prim(u, equations)
  entropy = ()
  rho_p_plus = zero(u[1])
  for k in eachcomponent(equations)
    rho, v1, v2, v3, p = get_component(k, prim, equations)
    s = log(p) - gammas[k] * log(rho)
    rho_p = rho / p
    w1 = (gammas[k] - s) / (gammas[k] - 1) - 0.5 * rho_p * (v1^2 + v2^2 + v3^2)
    w2 = rho_p * v1
    w3 = rho_p * v2
    w4 = rho_p * v3
    w5 = -rho_p
    rho_p_plus += rho_p
    entropy = (entropy..., w1, w2, w3, w4, w5)
  end

  # Additional non-conservative variables
  w6 = rho_p_plus * B1
  w7 = rho_p_plus * B2
  w8 = rho_p_plus * B3
  entropy = (w6, w7, w8, entropy...)
  
  return SVector{nvariables(equations), real(equations)}(entropy)
end


"""
Convert primitive to conservative variables
"""
@inline function prim2cons(prim, equations::IdealGlmMhdMultiIonEquations1D)
  @unpack gammas = equations
  B1, B2, B3, _ = prim

  cons = (B1, B2, B3)
  for k in eachcomponent(equations)
    rho, v1, v2, v3, p = get_component(k, prim, equations)
    rho_v1 = rho * v1
    rho_v2 = rho * v2
    rho_v3 = rho * v3

    rho_e = p/(gammas[k] - 1.0) + 0.5 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3) +
                                  0.5 * (B1^2 + B2^2 + B3^2)
    cons = (cons..., rho, rho_v1, rho_v2, rho_v3, rho_e)
  end

  return SVector{nvariables(equations), real(equations)}(cons)
end

"""
Compute the fastest wave speed for ideal MHD equations: c_f, the fast magnetoacoustic eigenvalue
  !!! ATTENTION: This routine is provisory.. Change once the fastest wave speed is known!!
"""
@inline function calc_fast_wavespeed(cons, direction, equations::IdealGlmMhdMultiIonEquations1D)
  B1, B2, B3, _ = cons

  c_f = zero(cons[1])
  for k in eachcomponent(equations)
    rho, rho_v1, rho_v2, rho_v3, rho_e = get_component(k, cons, equations)
    
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    v_mag = sqrt(v1^2 + v2^2 + v3^2)
    gamma = equations.gammas[k]
    p = (gamma - 1)*(rho_e - 0.5*rho*v_mag^2 - 0.5*(B1^2 + B2^2 + B3^2))
    a_square = gamma * p / rho
    sqrt_rho = sqrt(rho)

    b1 = B1 / sqrt_rho
    b2 = B2 / sqrt_rho
    b3 = B3 / sqrt_rho
    b_square = b1^2 + b2^2 + b3^2

    c_f = max(c_f, sqrt(0.5*(a_square + b_square) + 0.5*sqrt((a_square + b_square)^2 - 4.0*a_square*b1^2)))
  end

  return c_f
end

"""
Routine to compute the auxiliary variables:
* total_electron_charge
* v*_plus: Charge-averaged velocity
* vk*_plus: Contribution of each species to the charge-averaged velocity
"""
@inline function auxiliary_variables(u, equations::IdealGlmMhdMultiIonEquations1D)

  total_electron_charge = zero(u[1])
  
  vk1_plus = zeros(typeof(u[1]), ncomponents(equations))
  vk2_plus = zeros(typeof(u[1]), ncomponents(equations))
  vk3_plus = zeros(typeof(u[1]), ncomponents(equations))

  for k in eachcomponent(equations)
    rho_k = u[(k-1)*5+4]
    rho_v1_k = u[(k-1)*5+5]
    rho_v2_k = u[(k-1)*5+6]
    rho_v3_k = u[(k-1)*5+7]
    total_electron_charge += rho_k * equations.charge_to_mass[k]
    vk1_plus[k] = rho_v1_k * equations.charge_to_mass[k]
    vk2_plus[k] = rho_v2_k * equations.charge_to_mass[k]
    vk3_plus[k] = rho_v3_k * equations.charge_to_mass[k]
  end
  vk1_plus ./= total_electron_charge
  vk2_plus ./= total_electron_charge
  vk3_plus ./= total_electron_charge
  v1_plus = sum(vk1_plus)
  v2_plus = sum(vk2_plus)
  v3_plus = sum(vk3_plus)

  return total_electron_charge, v1_plus, v2_plus, v3_plus, SVector{ncomponents(equations), real(equations)}(vk1_plus),
                                                           SVector{ncomponents(equations), real(equations)}(vk2_plus),
                                                           SVector{ncomponents(equations), real(equations)}(vk3_plus)
end

"""
Get the flow variables of component k
"""
@inline function get_component(k, u, equations::IdealGlmMhdMultiIonEquations1D)
  return SVector{5, real(equations)}(u[(k-1)*5+4:(k-1)*5+8])
end

end # @muladd
