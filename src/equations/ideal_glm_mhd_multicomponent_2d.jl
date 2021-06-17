
@doc raw"""
    IdealGlmMhdMulticomponentEquations2D

The ideal compressible multicomponent GLM-MHD equations in two space dimensions.
"""
mutable struct IdealGlmMhdMulticomponentEquations2D{NVARS, NCOMP, RealT<:Real} <: AbstractIdealGlmMhdMulticomponentEquations{2, NVARS, NCOMP}
  gammas            ::SVector{NCOMP, RealT}
  gas_constants     ::SVector{NCOMP, RealT}
  cv                ::SVector{NCOMP, RealT}
  cp                ::SVector{NCOMP, RealT}
  c_h               ::RealT # GLM cleaning speed

  function IdealGlmMhdMulticomponentEquations2D{NVARS, NCOMP, RealT}(gammas       ::SVector{NCOMP, RealT},
                                                                     gas_constants::SVector{NCOMP, RealT}) where {NVARS, NCOMP, RealT<:Real}

    NCOMP >= 1 || throw(DimensionMismatch("`gammas` and `gas_constants` have to be filled with at least one value"))

    cv = gas_constants ./ (gammas .- 1)
    cp = gas_constants + gas_constants ./ (gammas .- 1)
    c_h = convert(eltype(gammas), NaN)

    new(gammas, gas_constants, cv, cp, c_h)
  end
end

function IdealGlmMhdMulticomponentEquations2D(; gammas, gas_constants)

  _gammas        = promote(gammas...)
  _gas_constants = promote(gas_constants...)
  RealT          = promote_type(eltype(_gammas), eltype(_gas_constants))

  NVARS = length(_gammas) + 8
  NCOMP = length(_gammas)

  __gammas        = SVector(map(RealT, _gammas))
  __gas_constants = SVector(map(RealT, _gas_constants))

  return IdealGlmMhdMulticomponentEquations2D{NVARS, NCOMP, RealT}(__gammas, __gas_constants)
end

@inline Base.real(::IdealGlmMhdMulticomponentEquations2D{NVARS, NCOMP, RealT}) where {NVARS, NCOMP, RealT} = RealT

have_nonconservative_terms(::IdealGlmMhdMulticomponentEquations2D) = Val(true)

function varnames(::typeof(cons2cons), equations::IdealGlmMhdMulticomponentEquations2D)

  cons  = ("rho_v1", "rho_v2", "rho_v3", "rho_e", "B1", "B2", "B3", "psi")
  rhos  = ntuple(n -> "rho" * string(n), Val(ncomponents(equations)))
  return (cons..., rhos...)
end

function varnames(::typeof(cons2prim), equations::IdealGlmMhdMulticomponentEquations2D)

  prim  = ("v1", "v2", "v3", "p", "B1", "B2", "B3", "psi")
  rhos  = ntuple(n -> "rho" * string(n), Val(ncomponents(equations)))
  return (prim..., rhos...)
end

default_analysis_integrals(::IdealGlmMhdMulticomponentEquations2D)  = (entropy_timederivative, Val(:l2_divb), Val(:linf_divb))


"""
    initial_condition_convergence_test(x, t, equations::IdealGlmMhdMulticomponentEquations2D)

An Alfvén wave as smooth initial condition used for convergence tests.
"""
function initial_condition_convergence_test(x, t, equations::IdealGlmMhdMulticomponentEquations2D)
  # smooth Alfvén wave test from Derigs et al. FLASH (2016)
  # domain must be set to [0, 1/cos(α)] x [0, 1/sin(α)], γ = 5/3
  alpha = 0.25*pi
  x_perp = x[1]*cos(alpha) + x[2]*sin(alpha)
  B_perp = 0.1*sin(2.0*pi*x_perp)
  rho = 1
  prim_rho  = SVector{ncomponents(equations), real(equations)}(2^(i-1) * (1-2)/(1-2^ncomponents(equations)) * rho for i in eachcomponent(equations))
  v1 = -B_perp*sin(alpha)
  v2 =  B_perp*cos(alpha)
  v3 = 0.1*cos(2.0*pi*x_perp)
  p = 0.1
  B1 = cos(alpha) + v1
  B2 = sin(alpha) + v2
  B3 = v3
  psi = 0.0
  prim_other         = SVector{8, real(equations)}(v1, v2, v3, p, B1, B2, B3, psi)
  return prim2cons(vcat(prim_other, prim_rho), equations)
end


"""
    initial_condition_weak_blast_wave(x, t, equations::IdealGlmMhdMulticomponentEquations2D)

A weak blast wave adapted from
- Sebastian Hennemann, Gregor J. Gassner (2020)
  A provably entropy stable subcell shock capturing approach for high order split form DG
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
function initial_condition_weak_blast_wave(x, t, equations::IdealGlmMhdMulticomponentEquations2D)
  # Adapted MHD version of the weak blast wave from Hennemann & Gassner JCP paper 2020 (Sec. 6.3)
  # Same discontinuity in the velocities but with magnetic fields
  # Set up polar coordinates
  inicenter         = SVector(0.0, 0.0)
  x_norm            = x[1] - inicenter[1]
  y_norm            = x[2] - inicenter[2]
  r                 = sqrt(x_norm^2 + y_norm^2)
  phi               = atan(y_norm, x_norm)
  sin_phi, cos_phi  = sincos(phi)

  prim_rho          = SVector{ncomponents(equations), real(equations)}(r > 0.5 ? 2^(i-1) * (1-2)/(1-2^ncomponents(equations))*1.0 : 2^(i-1) * (1-2)/(1-2^ncomponents(equations))*1.1691 for i in eachcomponent(equations))

  v1                = r > 0.5 ? 0.0 : 0.1882 * cos_phi
  v2                = r > 0.5 ? 0.0 : 0.1882 * sin_phi
  p                 = r > 0.5 ? 1.0 : 1.245

  prim_other         = SVector{8, real(equations)}(v1, v2, 0.0, p, 1.0, 1.0, 1.0, 0.0)

  return prim2cons(vcat(prim_other, prim_rho),equations)
end


"""
    initial_condition_rotor(x, t, equations::IdealGlmMhdMulticomponentEquations2D)

The classical MHD rotor test case adapted to twocomponent. Here, the setup is taken from
- Dominik Derigs, Gregor J. Gassner, Stefanie Walch & Andrew R. Winters (2018)
  Entropy Stable Finite Volume Approximations for Ideal Magnetohydrodynamics
  [doi: 10.1365/s13291-018-0178-9](https://doi.org/10.1365/s13291-018-0178-9)
"""
function initial_condition_rotor(x, t, equations::IdealGlmMhdMulticomponentEquations2D)
  # setup taken from Derigs et al. DMV article (2018)
  # domain must be [0, 1] x [0, 1], γ = 1.4
  dx = x[1] - 0.5
  dy = x[2] - 0.5
  r = sqrt(dx^2 + dy^2)
  f = (0.115 - r)/0.015
  if r <= 0.1
    rho1 = 10.0
    rho2 = 5.0
    v1 = -20.0*dy
    v2 = 20.0*dx
  elseif r >= 0.115
    rho1 = 1.0
    rho2 = 0.5
    v1 = 0.0
    v2 = 0.0
  else
    rho1 = 1.0 + 9.0*f
    rho2 = 0.5 + 4.5*f
    v1 = -20.0*f*dy
    v2 = 20.0*f*dx
  end
  v3 = 0.0
  p = 1.0
  B1 = 5.0/sqrt(4.0*pi)
  B2 = 0.0
  B3 = 0.0
  psi = 0.0
  return prim2cons(SVector(v1, v2, v3, p, B1, B2, B3, psi, rho1, rho2), equations)
end


# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer, equations::IdealGlmMhdMulticomponentEquations2D)
  rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = u
  @unpack c_h = equations

  rho = density(u, equations)

  v1 = rho_v1/rho
  v2 = rho_v2/rho
  v3 = rho_v3/rho
  kin_en = 0.5 * rho * (v1^2 + v2^2 + v3^2)
  mag_en = 0.5*(B1^2 + B2^2 + B3^2)
  gamma = totalgamma(u, equations)
  p = (gamma - 1) * (rho_e - kin_en - mag_en - 0.5*psi^2)

  if orientation == 1
    f_rho = densities(u, v1, equations)
    f1 = rho_v1*v1 + p + mag_en - B1^2
    f2 = rho_v1*v2 - B1*B2
    f3 = rho_v1*v3 - B1*B3
    f4 = (kin_en + gamma*p/(gamma - 1) + 2*mag_en)*v1 - B1*(v1*B1 + v2*B2 + v3*B3) + c_h*psi*B1
    f5 = c_h*psi
    f6 = v1*B2 - v2*B1
    f7 = v1*B3 - v3*B1
    f8 = c_h*B1
  else # orientation == 2
    f_rho = densities(u, v2, equations)
    f1 = rho_v2*v1 - B1*B2
    f2 = rho_v2*v2 + p + mag_en - B2^2
    f3 = rho_v2*v3 - B2*B3
    f4 = (kin_en + gamma*p/(gamma - 1) + 2*mag_en)*v2 - B2*(v1*B1 + v2*B2 + v3*B3) + c_h*psi*B2
    f5 = v2*B1 - v1*B2
    f6 = c_h*psi
    f7 = v2*B3 - v3*B2
    f8 = c_h*B2
  end

  f_other  = SVector{8, real(equations)}(f1, f2, f3, f4, f5, f6, f7, f8)

  return vcat(f_other, f_rho)
end


# Calculate the nonconservative terms from Powell and Galilean invariance
# OBS! This is scaled by 1/2 becuase it will cancel later with the factor of 2 in dsplit_transposed
@inline function calcflux_twopoint_nonconservative!(f1, f2, u, element,
                                                    equations::IdealGlmMhdMulticomponentEquations2D,
                                                    dg, cache)
  for j in eachnode(dg), i in eachnode(dg)
    rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = get_node_vars(u, equations, dg, i, j, element)

    rho = density(u, equations)

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho

    # Powell nonconservative term: Φ^Pow = (0, B_1, B_2, B_3, v⋅B, v_1, v_2, v_3, 0)
    prim_rho = SVector{ncomponents(equations), real(equations)}(0 for i in eachcomponent(equations))
    phi_pow_1 = 0.5 * SVector(B1, B2, B3, v1*B1 + v2*B2 + v3*B3, v1, v2, v3, 0)
    phi_pow = vcat(phi_pow_1, prim_rho)

    # Galilean nonconservative term: Φ^Gal_{1,2} = (0, 0, 0, 0, ψ v_{1,2}, 0, 0, 0, v_{1,2})
    # x-direction
    phi_gal_x_1 = 0.5 * SVector(0, 0, 0, v1*psi, 0, 0, 0, v1)
    phi_gal_x = vcat(phi_gal_x_1, prim_rho)

    # y-direction
    phi_gal_y_1 = 0.5 * SVector(0, 0, 0, v2*psi, 0, 0, 0, v2)
    phi_gal_y = vcat(phi_gal_y_1, prim_rho)

    # add both nonconservative terms into the volume
    for l in eachnode(dg)
      _, _, _, _, B1, _, _, psi = get_node_vars(u, equations, dg, l, j, element)
      for v in eachvariable(equations)
        f1[v, l, i, j] += phi_pow[v] * B1 + phi_gal_x[v] * psi
      end
      _, _, _, _, _, B2, _, psi = get_node_vars(u, equations, dg, i, l, element)
      for v in eachvariable(equations)
        f2[v, l, i, j] += phi_pow[v] * B2 + phi_gal_y[v] * psi
      end
    end
  end

  return nothing
end


"""
    flux_derigs_etal(u_ll, u_rr, orientation, equations::IdealGlmMhdEquations2D)

Entropy conserving two-point flux adapted by
- Derigs et al. (2018)
  Ideal GLM-MHD: About the entropy consistent nine-wave magnetic field
  divergence diminishing ideal magnetohydrodynamics equations for multicomponent
  [DOI: 10.1016/j.jcp.2018.03.002](https://doi.org/10.1016/j.jcp.2018.03.002)
"""
function flux_derigs_etal(u_ll, u_rr, orientation::Integer, equations::IdealGlmMhdMulticomponentEquations2D)
  # Unpack left and right states to get velocities, pressure, and inverse temperature (called beta)
  rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
  rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr
  @unpack gammas, gas_constants, cv, c_h = equations

  rho_ll = density(u_ll, equations)
  rho_rr = density(u_rr, equations)

  gamma_ll = totalgamma(u_ll, equations)
  gamma_rr = totalgamma(u_rr, equations)

  rhok_mean   = SVector{ncomponents(equations), real(equations)}(ln_mean(u_ll[i+8], u_rr[i+8]) for i in eachcomponent(equations))
  rhok_avg    = SVector{ncomponents(equations), real(equations)}(0.5 * (u_ll[i+8] + u_rr[i+8]) for i in eachcomponent(equations))

  v1_ll = rho_v1_ll/rho_ll
  v2_ll = rho_v2_ll/rho_ll
  v3_ll = rho_v3_ll/rho_ll
  v1_rr = rho_v1_rr/rho_rr
  v2_rr = rho_v2_rr/rho_rr
  v3_rr = rho_v3_rr/rho_rr
  v1_sq = 0.5 * (v1_ll^2 + v1_rr^2)
  v2_sq = 0.5 * (v2_ll^2 + v2_rr^2)
  v3_sq = 0.5 * (v3_ll^2 + v3_rr^2)
  v_sq = v1_sq + v2_sq + v3_sq
  B1_sq = 0.5 * (B1_ll^2 + B1_rr^2)
  B2_sq = 0.5 * (B2_ll^2 + B2_rr^2)
  B3_sq = 0.5 * (B3_ll^2 + B3_rr^2)
  B_sq = B1_sq + B2_sq + B3_sq
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
  psi_avg = 0.5*(psi_ll+psi_rr)
  vel_norm_avg = 0.5*(vel_norm_ll+vel_norm_rr)
  mag_norm_avg = 0.5*(mag_norm_ll+mag_norm_rr)
  vel_dot_mag_avg = 0.5*(vel_dot_mag_ll+vel_dot_mag_rr)

  enth      = zero(v_sum)
  help1_ll  = zero(v1_ll)
  help1_rr  = zero(v1_rr)

  for i in eachcomponent(equations)
    enth      += rhok_avg[i] * gas_constants[i]
    help1_ll  += u_ll[i+8] * cv[i]
    help1_rr  += u_rr[i+8] * cv[i]
  end

  T_ll        = (rho_e_ll - 0.5*rho_ll * (vel_norm_ll) - 0.5*mag_norm_ll - 0.5*psi_ll^2) / help1_ll
  T_rr        = (rho_e_rr - 0.5*rho_rr * (vel_norm_rr) - 0.5*mag_norm_rr - 0.5*psi_rr^2) / help1_rr
  T           = 0.5 * (1.0/T_ll + 1.0/T_rr)
  T_log       = ln_mean(1.0/T_ll, 1.0/T_rr)

  # Calculate fluxes depending on orientation with specific direction averages
  help1       = zero(T_ll)
  help2       = zero(T_rr)
  if orientation == 1
    f_rho       = SVector{ncomponents(equations), real(equations)}(rhok_mean[i]*v1_avg for i in eachcomponent(equations))
    for i in eachcomponent(equations)
      help1     += f_rho[i] * cv[i]
      help2     += f_rho[i]
    end
    f1 = help2 * v1_avg + enth/T + 0.5 * mag_norm_avg - B1_avg*B1_avg
    f2 = help2 * v2_avg - B1_avg*B2_avg
    f3 = help2 * v3_avg - B1_avg*B3_avg
    f5 = c_h*psi_avg
    f6 = v1_avg*B2_avg - v2_avg*B1_avg
    f7 = v1_avg*B3_avg - v3_avg*B1_avg
    f8 = c_h*B1_avg
    # total energy flux is complicated and involves the previous eight components
    psi_B1_avg = 0.5*(B1_ll*psi_ll + B1_rr*psi_rr)
    v1_mag_avg = 0.5*(v1_ll*mag_norm_ll + v1_rr*mag_norm_rr)

    f4 = (help1/T_log) - 0.5 * (vel_norm_avg) * (help2) + f1 * v1_avg + f2 * v2_avg + f3 * v3_avg +
          f5 * B1_avg + f6 * B2_avg + f7 * B3_avg + f8 * psi_avg - 0.5*v1_mag_avg +
          B1_avg * vel_dot_mag_avg - c_h * psi_B1_avg

  else
    f_rho       = SVector{ncomponents(equations), real(equations)}(rhok_mean[i]*v2_avg for i in eachcomponent(equations))
    for i in eachcomponent(equations)
      help1     += f_rho[i] * cv[i]
      help2     += f_rho[i]
    end
    f1 = help2 * v1_avg - B1_avg*B2_avg
    f2 = help2 * v2_avg + enth/T + 0.5 * mag_norm_avg - B2_avg*B2_avg
    f3 = help2 * v3_avg - B2_avg*B3_avg
    f5 = v2_avg*B1_avg - v1_avg*B2_avg
    f6 = c_h*psi_avg
    f7 = v2_avg*B3_avg - v3_avg*B2_avg
    f8 = c_h*B2_avg

    # total energy flux is complicated and involves the previous eight components
    psi_B2_avg = 0.5*(B2_ll*psi_ll + B2_rr*psi_rr)
    v2_mag_avg = 0.5*(v2_ll*mag_norm_ll + v2_rr*mag_norm_rr)

    f4 = (help1/T_log) - 0.5 * (vel_norm_avg) * (help2) + f1 * v1_avg + f2 * v2_avg + f3 * v3_avg +
          f5 * B1_avg + f6 * B2_avg + f7 * B3_avg + f8 * psi_avg - 0.5*v2_mag_avg +
          B2_avg * vel_dot_mag_avg - c_h * psi_B2_avg

  end

  f_other  = SVector{8, real(equations)}(f1, f2, f3, f4, f5, f6, f7, f8)

  return vcat(f_other, f_rho)
end


# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::IdealGlmMhdMulticomponentEquations2D)
  rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
  rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

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


"""
    noncons_interface_flux(u_left, u_right, orientation, mode, equations::IdealGlmMhdEquations2D)

Strong form of non-conservative flux on a surface (Powell and GLM terms)
```math
phi^L 1/2 (B^L + B^R)_{normal} - phi^L B^L+{normal} = phi^L 1/2 (B^R - B^L)_{normal}
```
!!! note
    The non-conservative interface flux depends on the discretization. Following "modes" are available:
    * `:weak`: 'weak' formulation of split DG already includes the contribution
      ``-1/2 (phi^L B^L_{normal})`` so this mode only adds ``1/2 (phi^L B^R_{normal})``,
      analogously for the Galilean nonconservative term
    * `:whole`: This mode adds the whole non-conservative term: phi^L 1/2 (B^R-B^L)
    * `:inner`: This mode adds the split-form DG volume integral contribution: This is equivalent to
      ``(2)-(1) - 1/2 (phi^L B^L)``
!!! warning
    This is non-unique along an interface! The normal direction is super important.
"""
@inline function noncons_interface_flux(u_left, u_right, orientation, mode, equations::IdealGlmMhdMulticomponentEquations2D)
  rho_v1_ll, rho_v2_ll, rho_v3_ll, _, B1_ll, B2_ll, B3_ll, psi_ll = u_left
  _, _, _, _, B1_rr, B2_rr, _, psi_rr = u_right

  # extract velocites from the left
  rho_ll = density(u_left, equations)

  v1_ll  = rho_v1_ll / rho_ll
  v2_ll  = rho_v2_ll / rho_ll
  v3_ll  = rho_v3_ll / rho_ll
  v_dot_B_ll = v1_ll*B1_ll + v2_ll*B2_ll + v3_ll*B3_ll
  # extract magnetic field variable from the right and set the normal velocity
  # Note, both depend upon the orientation and need psi_rr
  if mode==:weak
    if orientation == 1 # x-direction
      v_normal = v1_ll
      B_normal = B1_rr
    else # y-direction
      v_normal = v2_ll
      B_normal = B2_rr
    end
    psi_norm = psi_rr
  elseif mode==:whole
    if orientation == 1 # x-direction
      v_normal = v1_ll
      B_normal = B1_rr - B1_ll
    else # y-direction
      v_normal = v2_ll
      B_normal = B2_rr - B2_ll
    end
    psi_norm = psi_rr - psi_ll
  else #mode==:inner
    if orientation == 1 # x-direction
      v_normal = v1_ll
      B_normal =-B1_ll
    else # y-direction
      v_normal = v2_ll
      B_normal =-B2_ll
    end
    psi_norm =-psi_ll
  end

  # compute the nonconservative flux: Powell (with B_normal) and Galilean (with v_normal)
  noncons1 = 0.5 * B_normal * B1_ll
  noncons2 = 0.5 * B_normal * B2_ll
  noncons3 = 0.5 * B_normal * B3_ll
  noncons4 = 0.5 * B_normal * v_dot_B_ll + 0.5 * v_normal * psi_ll * psi_norm
  noncons5 = 0.5 * B_normal * v1_ll
  noncons6 = 0.5 * B_normal * v2_ll
  noncons7 = 0.5 * B_normal * v3_ll
  noncons8 = 0.5 * v_normal * psi_norm

  noncons_rho = densities(u_left, 0.0, equations)
  noncons_other = SVector{8, real(equations)}(noncons1, noncons2, noncons3, noncons4, noncons5, noncons6, noncons7, noncons8)

  return vcat(noncons_other, noncons_rho)
end


@inline function max_abs_speeds(u, equations::IdealGlmMhdMulticomponentEquations2D)
  rho_v1, rho_v2, rho_v3, _ = u

  rho = density(u, equations)

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  v3 = rho_v3 / rho

  cf_x_direction = calc_fast_wavespeed(u, 1, equations)
  cf_y_direction = calc_fast_wavespeed(u, 2, equations)

  return (abs(v1) + cf_x_direction, abs(v2) + cf_y_direction, )
end


@inline function density_pressure(u, equations::IdealGlmMhdMulticomponentEquations2D)
  rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = u
  rho = density(u, equations)
  gamma = totalgamma(u, equations)
  p = (gamma - 1)*(rho_e - 0.5 * (rho_v1^2 + rho_v2^2 + rho_v3^2) / rho
                                   - 0.5 * (B1^2 + B2^2 + B3^2)
                                   - 0.5 * psi^2)
  return rho * p
end


# Convert conservative variables to primitive
function cons2prim(u, equations::IdealGlmMhdMulticomponentEquations2D)
  rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = u

  prim_rho = SVector{ncomponents(equations), real(equations)}(u[i+8] for i in eachcomponent(equations))
  rho = density(u, equations)

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  v3 = rho_v3 / rho

  gamma = totalgamma(u, equations)

  p = (gamma - 1) * (rho_e - 0.5*rho*(v1^2 + v2^2 + v3^2) - 0.5*(B1^2 + B2^2 + B3^2) - 0.5*psi^2)
  prim_other =  SVector{8, real(equations)}(v1, v2, v3, p, B1, B2, B3, psi)
  return vcat(prim_other, prim_rho)
end

# Convert conservative variables to entropy
@inline function cons2entropy(u, equations::IdealGlmMhdMulticomponentEquations2D)
  rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = u
  @unpack cv, gammas, gas_constants = equations

  rho = density(u, equations)

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  v3 = rho_v3 / rho
  v_square = v1^2 + v2^2 + v3^2
  gamma = totalgamma(u, equations)
  p = (gamma - 1) * (rho_e - 0.5*rho*v_square - 0.5*(B1^2 + B2^2 + B3^2) - 0.5*psi^2)
  s = log(p) - gamma*log(rho)
  rho_p = rho / p

  # Multicomponent stuff
  help1 = zero(v1)

  for i in eachcomponent(equations)
    help1 += u[i+8] * cv[i]
  end

  T         = (rho_e - 0.5 * rho * v_square - 0.5*(B1^2 + B2^2 + B3^2) - 0.5*psi^2) / (help1)

  entrop_rho  = SVector{ncomponents(equations), real(equations)}( -1.0 * (cv[i] * log(T) - gas_constants[i] * log(u[i+8])) + gas_constants[i] + cv[i] - (v_square / (2*T)) for i in eachcomponent(equations))

  w1 = v1 / T
  w2 = v2 / T
  w3 = v3 / T
  w4 = -1.0 / T
  w5 = B1 / T
  w6 = B2 / T
  w7 = B3 / T
  w8 = psi / T

  entrop_other = SVector{8, real(equations)}(w1, w2, w3, w4, w5, w6, w7, w8)

  return vcat(entrop_other, entrop_rho)
end


# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::IdealGlmMhdMulticomponentEquations2D)
  v1, v2, v3, p, B1, B2, B3, psi = prim

  cons_rho = SVector{ncomponents(equations), real(equations)}(prim[i+8] for i in eachcomponent(equations))
  rho = density(prim, equations)

  rho_v1 = rho * v1
  rho_v2 = rho * v2
  rho_v3 = rho * v3

  gamma = totalgamma(prim, equations)
  rho_e = p/(gamma-1) + 0.5 * (rho_v1*v1 + rho_v2*v2 + rho_v3*v3) +
                                 0.5 * (B1^2 + B2^2 + B3^2) + 0.5 * psi^2

  cons_other = SVector{8, real(equations)}(rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi)

  return vcat(cons_other, cons_rho)
end


# Compute the fastest wave speed for ideal MHD equations: c_f, the fast magnetoacoustic eigenvalue
@inline function calc_fast_wavespeed(cons, direction, equations::IdealGlmMhdMulticomponentEquations2D)
  rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = cons
  rho = density(cons, equations)
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  v3 = rho_v3 / rho
  v_mag = sqrt(v1^2 + v2^2 + v3^2)
  gamma = totalgamma(cons, equations)
  p = (gamma - 1)*(rho_e - 0.5*rho*v_mag^2 - 0.5*(B1^2 + B2^2 + B3^2) - 0.5*psi^2)
  a_square = gamma * p / rho
  sqrt_rho = sqrt(rho)
  b1 = B1 / sqrt_rho
  b2 = B2 / sqrt_rho
  b3 = B3 / sqrt_rho
  b_square = b1^2 + b2^2 + b3^2
  if direction == 1 # x-direction
    c_f = sqrt(0.5*(a_square + b_square) + 0.5*sqrt((a_square + b_square)^2 - 4.0*a_square*b1^2))
  else
    c_f = sqrt(0.5*(a_square + b_square) + 0.5*sqrt((a_square + b_square)^2 - 4.0*a_square*b2^2))
  end
  return c_f
end


@inline function density(u, equations::IdealGlmMhdMulticomponentEquations2D)
  rho = zero(u[1])

  for i in eachcomponent(equations)
    rho += u[i+8]
  end

  return rho
 end


 @inline function totalgamma(u, equations::IdealGlmMhdMulticomponentEquations2D)
  @unpack cv, gammas = equations

  help1 = zero(u[1])
  help2 = zero(u[1])

  for i in eachcomponent(equations)
    help1 += u[i+8] * cv[i] * gammas[i]
    help2 += u[i+8] * cv[i]
  end

  return help1/help2
end


@inline function densities(u, v, equations::IdealGlmMhdMulticomponentEquations2D)

  return SVector{ncomponents(equations), real(equations)}(u[i+8]*v for i in eachcomponent(equations))
 end
