
@doc raw"""
    IdealGlmMhdMulticomponentEquations1D

The ideal compressible multicomponent GLM-MHD equations in one space dimension.
"""
mutable struct IdealGlmMhdMulticomponentEquations1D{NVARS, NCOMP, RealT<:Real} <: AbstractIdealGlmMhdMulticomponentEquations{1, NVARS, NCOMP}
  gammas            ::SVector{NCOMP, RealT}
  gas_constants     ::SVector{NCOMP, RealT}
  cv                ::SVector{NCOMP, RealT}
  cp                ::SVector{NCOMP, RealT}

  function IdealGlmMhdMulticomponentEquations1D{NVARS, NCOMP, RealT}(gammas       ::SVector{NCOMP, RealT},
                                                                     gas_constants::SVector{NCOMP, RealT}) where {NVARS, NCOMP, RealT<:Real}

    NCOMP >= 1 || throw(DimensionMismatch("`gammas` and `gas_constants` have to be filled with at least one value"))

    cv = gas_constants ./ (gammas .- 1)
    cp = gas_constants + gas_constants ./ (gammas .- 1)

    new(gammas, gas_constants, cv, cp)
  end
end

function IdealGlmMhdMulticomponentEquations1D(; gammas, gas_constants)

  _gammas        = promote(gammas...)
  _gas_constants = promote(gas_constants...)
  RealT          = promote_type(eltype(_gammas), eltype(_gas_constants))

  NVARS = length(_gammas) + 7
  NCOMP = length(_gammas)

  __gammas        = SVector(map(RealT, _gammas))
  __gas_constants = SVector(map(RealT, _gas_constants))

  return IdealGlmMhdMulticomponentEquations1D{NVARS, NCOMP, RealT}(__gammas, __gas_constants)
end

@inline Base.real(::IdealGlmMhdMulticomponentEquations1D{NVARS, NCOMP, RealT}) where {NVARS, NCOMP, RealT} = RealT

have_nonconservative_terms(::IdealGlmMhdMulticomponentEquations1D) = Val(false)

function varnames(::typeof(cons2cons), equations::IdealGlmMhdMulticomponentEquations1D)

  cons  = ("rho_v1", "rho_v2", "rho_v3", "rho_e", "B1", "B2", "B3")
  rhos  = ntuple(n -> "rho" * string(n), Val(ncomponents(equations)))
  return (cons..., rhos...)
end

function varnames(::typeof(cons2prim), equations::IdealGlmMhdMulticomponentEquations1D)

  prim  = ("v1", "v2", "v3", "p", "B1", "B2", "B3")
  rhos  = ntuple(n -> "rho" * string(n), Val(ncomponents(equations)))
  return (prim..., rhos...)
end


"""
    initial_condition_convergence_test(x, t, equations::IdealGlmMhdMulticomponentEquations1D)

An Alfvén wave as smooth initial condition used for convergence tests.
"""
function initial_condition_convergence_test(x, t, equations::IdealGlmMhdMulticomponentEquations1D)
  # smooth Alfvén wave test from Derigs et al. FLASH (2016)
  # domain must be set to [0, 1], γ = 5/3

  rho = 1.0
  prim_rho  = SVector{ncomponents(equations), real(equations)}(2^(i-1) * (1-2)/(1-2^ncomponents(equations)) * rho for i in eachcomponent(equations))
  v1 = 0.0
  si, co = sincos(2 * pi * x[1])
  v2 = 0.1 * si
  v3 = 0.1 * co
  p = 0.1
  B1 = 1.0
  B2 = v2
  B3 = v3
  prim_other = SVector{7, real(equations)}(v1, v2, v3, p, B1, B2, B3)
  return prim2cons(vcat(prim_other, prim_rho), equations)
end


"""
    initial_condition_briowu_shock_tube(x, t, equations::IdealGlmMhdMulticomponentEquations1D)

Compound shock tube test case for one dimensional ideal MHD equations. It is bascially an
MHD extension of the Sod shock tube. Taken from Section V of the article
- Brio and Wu (1988)
  An Upwind Differencing Scheme for the Equations of Ideal Magnetohydrodynamics
  [DOI: 10.1016/0021-9991(88)90120-9](https://doi.org/10.1016/0021-9991(88)90120-9)
"""
function initial_condition_briowu_shock_tube(x, t, equations::IdealGlmMhdMulticomponentEquations1D)
  # domain must be set to [0, 1], γ = 2, final time = 0.12
  if x[1] < 0.5
    rho       = 1.0
    prim_rho  = SVector{ncomponents(equations), real(equations)}(2^(i-1) * (1-2)/(1-2^ncomponents(equations)) * rho for i in eachcomponent(equations))
  else
    rho       = 0.125
    prim_rho  = SVector{ncomponents(equations), real(equations)}(2^(i-1) * (1-2)/(1-2^ncomponents(equations)) * rho for i in eachcomponent(equations))
  end
  v1 = 0.0
  v2 = 0.0
  v3 = 0.0
  p = x[1] < 0.5 ? 1.0 : 0.1
  B1 = 0.75
  B2 = x[1] < 0.5 ? 1.0 : -1.0
  B3 = 0.0

  prim_other = SVector{7, real(equations)}(v1, v2, v3, p, B1, B2, B3)
  return prim2cons(vcat(prim_other, prim_rho), equations)
end


# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer, equations::IdealGlmMhdMulticomponentEquations1D)
  rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3 = u

  rho = density(u, equations)

  v1 = rho_v1/rho
  v2 = rho_v2/rho
  v3 = rho_v3/rho
  kin_en = 0.5 * rho * (v1^2 + v2^2 + v3^2)
  mag_en = 0.5*(B1^2 + B2^2 + B3^2)
  gamma = totalgamma(u, equations)
  p = (gamma - 1) * (rho_e - kin_en - mag_en)


  f_rho = densities(u, v1, equations)
  f1 = rho_v1*v1 + p + mag_en - B1^2
  f2 = rho_v1*v2 - B1*B2
  f3 = rho_v1*v3 - B1*B3
  f4 = (kin_en + gamma*p/(gamma - 1) + 2*mag_en)*v1 - B1*(v1*B1 + v2*B2 + v3*B3)
  f5 = 0.0
  f6 = v1*B2 - v2*B1
  f7 = v1*B3 - v3*B1


  f_other  = SVector{7, real(equations)}(f1, f2, f3, f4, f5, f6, f7)

  return vcat(f_other, f_rho)
end


"""
    flux_derigs_etal(u_ll, u_rr, orientation, equations::IdealGlmMhdEquations1D)

Entropy conserving two-point flux adapted by
- Derigs et al. (2018)
  Ideal GLM-MHD: About the entropy consistent nine-wave magnetic field
  divergence diminishing ideal magnetohydrodynamics equations for multicomponent
  [DOI: 10.1016/j.jcp.2018.03.002](https://doi.org/10.1016/j.jcp.2018.03.002)
"""
function flux_derigs_etal(u_ll, u_rr, orientation::Integer, equations::IdealGlmMhdMulticomponentEquations1D)
  # Unpack left and right states to get velocities, pressure, and inverse temperature (called beta)
  rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll = u_ll
  rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr = u_rr
  @unpack gammas, gas_constants, cv = equations

  rho_ll = density(u_ll, equations)
  rho_rr = density(u_rr, equations)

  gamma_ll = totalgamma(u_ll, equations)
  gamma_rr = totalgamma(u_rr, equations)

  rhok_mean   = SVector{ncomponents(equations), real(equations)}(ln_mean(u_ll[i+7], u_rr[i+7]) for i in eachcomponent(equations))
  rhok_avg    = SVector{ncomponents(equations), real(equations)}(0.5 * (u_ll[i+7] + u_rr[i+7]) for i in eachcomponent(equations))

  v1_ll = rho_v1_ll/rho_ll
  v2_ll = rho_v2_ll/rho_ll
  v3_ll = rho_v3_ll/rho_ll
  v1_rr = rho_v1_rr/rho_rr
  v2_rr = rho_v2_rr/rho_rr
  v3_rr = rho_v3_rr/rho_rr
  vel_norm_ll = v1_ll^2 + v2_ll^2 + v3_ll^2
  vel_norm_rr = v1_rr^2 + v2_rr^2 + v3_rr^2
  mag_norm_ll = B1_ll^2 + B2_ll^2 + B3_ll^2
  mag_norm_rr = B1_rr^2 + B2_rr^2 + B3_rr^2
  # for convenience store v⋅B
  vel_dot_mag_ll = v1_ll*B1_ll + v2_ll*B2_ll + v3_ll*B3_ll
  vel_dot_mag_rr = v1_rr*B1_rr + v2_rr*B2_rr + v3_rr*B3_rr

  # Compute the necessary mean values needed for either direction
  v1_avg = 0.5*(v1_ll+v1_rr)
  v2_avg = 0.5*(v2_ll+v2_rr)
  v3_avg = 0.5*(v3_ll+v3_rr)
  v_sum  = v1_avg + v2_avg + v3_avg
  B1_avg = 0.5*(B1_ll+B1_rr)
  B2_avg = 0.5*(B2_ll+B2_rr)
  B3_avg = 0.5*(B3_ll+B3_rr)
  vel_norm_avg = 0.5*(vel_norm_ll+vel_norm_rr)
  mag_norm_avg = 0.5*(mag_norm_ll+mag_norm_rr)
  vel_dot_mag_avg = 0.5*(vel_dot_mag_ll+vel_dot_mag_rr)

  enth      = zero(v_sum)
  help1_ll  = zero(v1_ll)
  help1_rr  = zero(v1_rr)

  for i in eachcomponent(equations)
    enth      += rhok_avg[i] * gas_constants[i]
    help1_ll  += u_ll[i+7] * cv[i]
    help1_rr  += u_rr[i+7] * cv[i]
  end

  T_ll        = (rho_e_ll - 0.5*rho_ll * (vel_norm_ll) - 0.5*mag_norm_ll) / help1_ll
  T_rr        = (rho_e_rr - 0.5*rho_rr * (vel_norm_rr) - 0.5*mag_norm_rr) / help1_rr
  T           = 0.5 * (1.0/T_ll + 1.0/T_rr)
  T_log       = ln_mean(1.0/T_ll, 1.0/T_rr)

  # Calculate fluxes depending on orientation with specific direction averages
  help1       = zero(T_ll)
  help2       = zero(T_rr)

  f_rho       = SVector{ncomponents(equations), real(equations)}(rhok_mean[i]*v1_avg for i in eachcomponent(equations))
  for i in eachcomponent(equations)
    help1     += f_rho[i] * cv[i]
    help2     += f_rho[i]
  end
  f1 = help2 * v1_avg + enth/T + 0.5 * mag_norm_avg - B1_avg*B1_avg
  f2 = help2 * v2_avg - B1_avg*B2_avg
  f3 = help2 * v3_avg - B1_avg*B3_avg
  f5 = 0.0
  f6 = v1_avg*B2_avg - v2_avg*B1_avg
  f7 = v1_avg*B3_avg - v3_avg*B1_avg

  # total energy flux is complicated and involves the previous eight components
  v1_mag_avg = 0.5*(v1_ll*mag_norm_ll + v1_rr*mag_norm_rr)

  f4 = (help1/T_log) - 0.5 * (vel_norm_avg) * (help2) + f1 * v1_avg + f2 * v2_avg + f3 * v3_avg +
        f5 * B1_avg + f6 * B2_avg + f7 * B3_avg - 0.5*v1_mag_avg +
        B1_avg * vel_dot_mag_avg


  f_other  = SVector{7, real(equations)}(f1, f2, f3, f4, f5, f6, f7)

  return vcat(f_other, f_rho)
end


# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::IdealGlmMhdMulticomponentEquations1D)
  rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll = u_ll
  rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr = u_rr

  rho_ll   = density(u_ll, equations)
  rho_rr   = density(u_rr, equations)
  gamma_ll = totalgamma(u_ll, equations)
  gamma_rr = totalgamma(u_rr, equations)

  # Calculate velocities and fast magnetoacoustic wave speeds
  # left
  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v3_ll = rho_v3_ll / rho_ll
  v_mag_ll = sqrt(v1_ll^2 + v2_ll^2 + v3_ll^2)
  cf_ll = calc_fast_wavespeed(u_ll, orientation, equations)
  # right
  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  v3_rr = rho_v3_rr / rho_rr
  v_mag_rr = sqrt(v1_rr^2 + v2_rr^2 + v3_rr^2)
  cf_rr = calc_fast_wavespeed(u_rr, orientation, equations)

  λ_max = max(v_mag_ll, v_mag_rr) + max(cf_ll, cf_rr)
end


@inline function max_abs_speeds(u, equations::IdealGlmMhdMulticomponentEquations1D)
  rho_v1, _ = u

  rho = density(u, equations)

  v1 = rho_v1 / rho

  cf_x_direction = calc_fast_wavespeed(u, 1, equations)

  return (abs(v1) + cf_x_direction, )
end


# Convert conservative variables to primitive
function cons2prim(u, equations::IdealGlmMhdMulticomponentEquations1D)
  rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3 = u

  prim_rho = SVector{ncomponents(equations), real(equations)}(u[i+7] for i in eachcomponent(equations))
  rho = density(u, equations)

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  v3 = rho_v3 / rho

  gamma = totalgamma(u, equations)

  p = (gamma - 1) * (rho_e - 0.5*rho*(v1^2 + v2^2 + v3^2) - 0.5*(B1^2 + B2^2 + B3^2))
  prim_other =  SVector{7, real(equations)}(v1, v2, v3, p, B1, B2, B3)
  return vcat(prim_other, prim_rho)
end

# Convert conservative variables to entropy
@inline function cons2entropy(u, equations::IdealGlmMhdMulticomponentEquations1D)
  rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3 = u
  @unpack cv, gammas, gas_constants = equations

  rho = density(u, equations)

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  v3 = rho_v3 / rho
  v_square = v1^2 + v2^2 + v3^2
  gamma = totalgamma(u, equations)
  p = (gamma - 1) * (rho_e - 0.5*rho*v_square - 0.5*(B1^2 + B2^2 + B3^2))
  s = log(p) - gamma*log(rho)
  rho_p = rho / p

  # Multicomponent stuff
  help1 = zero(v1)

  for i in eachcomponent(equations)
    help1 += u[i+7] * cv[i]
  end

  T         = (rho_e - 0.5 * rho * v_square - 0.5*(B1^2 + B2^2 + B3^2)) / (help1)

  entrop_rho  = SVector{ncomponents(equations), real(equations)}( -1.0 * (cv[i] * log(T) - gas_constants[i] * log(u[i+7])) + gas_constants[i] + cv[i] - (v_square / (2*T)) for i in eachcomponent(equations))

  w1 = v1 / T
  w2 = v2 / T
  w3 = v3 / T
  w4 = -1.0 / T
  w5 = B1 / T
  w6 = B2 / T
  w7 = B3 / T

  entrop_other = SVector{7, real(equations)}(w1, w2, w3, w4, w5, w6, w7)

  return vcat(entrop_other, entrop_rho)
end


# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::IdealGlmMhdMulticomponentEquations1D)
  v1, v2, v3, p, B1, B2, B3 = prim

  cons_rho = SVector{ncomponents(equations), real(equations)}(prim[i+7] for i in eachcomponent(equations))
  rho = density(prim, equations)

  rho_v1 = rho * v1
  rho_v2 = rho * v2
  rho_v3 = rho * v3

  gamma = totalgamma(prim, equations)
  rho_e = p/(gamma-1) + 0.5 * (rho_v1*v1 + rho_v2*v2 + rho_v3*v3) +
                                 0.5 * (B1^2 + B2^2 + B3^2)

  cons_other = SVector{7, real(equations)}(rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3)

  return vcat(cons_other, cons_rho)
end


@inline function density_pressure(u, equations::IdealGlmMhdMulticomponentEquations1D)
  rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3 = u
  rho = density(u, equations)
  gamma = totalgamma(u, equations)
  p = (gamma - 1)*(rho_e - 0.5 * (rho_v1^2 + rho_v2^2 + rho_v3^2) / rho
                                   - 0.5 * (B1^2 + B2^2 + B3^2)
                                   )
  return rho * p
end


# Compute the fastest wave speed for ideal MHD equations: c_f, the fast magnetoacoustic eigenvalue
@inline function calc_fast_wavespeed(cons, direction, equations::IdealGlmMhdMulticomponentEquations1D)
  rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3 = cons
  rho = density(cons, equations)
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  v3 = rho_v3 / rho
  v_mag = sqrt(v1^2 + v2^2 + v3^2)
  gamma = totalgamma(cons, equations)
  p = (gamma - 1)*(rho_e - 0.5*rho*v_mag^2 - 0.5*(B1^2 + B2^2 + B3^2))
  a_square = gamma * p / rho
  sqrt_rho = sqrt(rho)
  b1 = B1 / sqrt_rho
  b2 = B2 / sqrt_rho
  b3 = B3 / sqrt_rho
  b_square = b1^2 + b2^2 + b3^2

  c_f = sqrt(0.5*(a_square + b_square) + 0.5*sqrt((a_square + b_square)^2 - 4.0*a_square*b1^2))

  return c_f
end


@inline function density(u, equations::IdealGlmMhdMulticomponentEquations1D)
  rho = zero(u[1])

  for i in eachcomponent(equations)
    rho += u[i+7]
  end

  return rho
 end


 @inline function totalgamma(u, equations::IdealGlmMhdMulticomponentEquations1D)
  @unpack cv, gammas = equations

  help1 = zero(u[1])
  help2 = zero(u[1])

  for i in eachcomponent(equations)
    help1 += u[i+7] * cv[i] * gammas[i]
    help2 += u[i+7] * cv[i]
  end

  return help1/help2
end


@inline function densities(u, v, equations::IdealGlmMhdMulticomponentEquations1D)

  return SVector{ncomponents(equations), real(equations)}(u[i+7]*v for i in eachcomponent(equations))
 end
