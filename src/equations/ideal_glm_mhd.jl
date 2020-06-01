
@doc raw"""
    IdealGlmMhdEquations

The ideal compressible GLM-MHD equations in two space dimensions.
"""
mutable struct IdealGlmMhdEquations <: AbstractEquation{9}
  gamma::Float64
  c_h::Float64 # GLM cleaning speed
end

function IdealGlmMhdEquations()
  gamma = parameter("gamma", 1.4)
  c_h = 0.0   # GLM cleaning wave speed
  IdealGlmMhdEquations(gamma, c_h)
end


get_name(::IdealGlmMhdEquations) = "IdealGlmMhdEquations"
have_nonconservative_terms(::IdealGlmMhdEquations) = Val(true)
varnames_cons(::IdealGlmMhdEquations) = @SVector ["rho", "rho_v1", "rho_v2", "rho_v3", "rho_e", "B1", "B2", "B3", "psi"]
varnames_prim(::IdealGlmMhdEquations) = @SVector ["rho", "v1", "v2", "v3", "p", "B1", "B2", "B3", "psi"]
default_analysis_quantities(::IdealGlmMhdEquations) = (:l2_error, :linf_error, :dsdu_ut,
                                                       :l2_divb, :linf_divb)


# Set initial conditions at physical location `x` for time `t`
function initial_conditions_constant(equation::IdealGlmMhdEquations, x, t)
  rho = 1.0
  rho_v1 = 0.1
  rho_v2 = -0.2
  rho_v3 = -0.5
  rho_e = 50.0
  B1 = 3.0
  B2 = -1.2
  B3 = 0.5
  psi = 0.0
  return @SVector [rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi]
end

function initial_conditions_convergence_test(equation::IdealGlmMhdEquations, x, t)
  # smooth Alfvén wave test from Derigs et al. FLASH (2016)
  # domain must be set to [0, 1/cos(α)] x [0, 1/sin(α)], γ = 5/3
  alpha = 0.25*pi
  x_perp = x[1]*cos(alpha) + x[2]*sin(alpha)
  B_perp = 0.1*sin(2.0*pi*x_perp)
  rho = 1.0
  v1 = -B_perp*sin(alpha)
  v2 = B_perp*cos(alpha)
  v3 = 0.1*cos(2.0*pi*x_perp)
  p = 0.1
  B1 = cos(alpha) + v1
  B2 = sin(alpha) + v2
  B3 = v3
  psi = 0.0
  return prim2cons(equation, @SVector [rho, v1, v2, v3, p, B1, B2, B3, psi])
end

function initial_conditions_orszag_tang(equation::IdealGlmMhdEquations, x, t)
  # setup taken from Derigs et al. DMV article (2018)
  # domain must be [0, 1] x [0, 1], γ = 5/3
  rho = 1.0
  v1 = -sin(2.0*pi*x[2])
  v2 = sin(2.0*pi*x[1])
  v3 = 0.0
  p = 1.0/equation.gamma
  B1 = -sin(2.0*pi*x[2])/equation.gamma
  B2 = sin(4.0*pi*x[1])/equation.gamma
  B3 = 0.0
  psi = 0.0
  return prim2cons(equation, @SVector [rho, v1, v2, v3, p, B1, B2, B3, psi])
end

function initial_conditions_rotor(equation::IdealGlmMhdEquations, x, t)
  # setup taken from Derigs et al. DMV article (2018)
  # domain must be [0, 1] x [0, 1], γ = 1.4
  dx = x[1] - 0.5
  dy = x[2] - 0.5
  r = sqrt(dx^2 + dy^2)
  f = (0.115 - r)/0.015
  if r <= 0.1
    rho = 10.0
    v1 = -20.0*dy
    v2 = 20.0*dx
  elseif r >= 0.115
    rho = 1.0
    v1 = 0.0
    v2 = 0.0
  else
    rho = 1.0 + 9.0*f
    v1 = -20.0*f*dy
    v2 = 20.0*f*dx
  end
  v3 = 0.0
  p = 1.0
  B1 = 5.0/sqrt(4.0*pi)
  B2 = 0.0
  B3 = 0.0
  psi = 0.0
  return prim2cons(equation, @SVector [rho, v1, v2, v3, p, B1, B2, B3, psi])
end

function initial_conditions_mhd_blast(equation::IdealGlmMhdEquations, x, t)
  # setup taken from Derigs et al. DMV article (2018)
  # domain must be [-0.5, 0.5] x [-0.5, 0.5], γ = 1.4
  r = sqrt(x[1]^2 + x[2]^2)
  f = (0.1 - r)/0.01
  if r <= 0.09
    p = 1000.0
  elseif r >= 0.1
    p = 0.1
  else
    p = 0.1 + 999.9*f
  end
  rho = 1.0
  v1 = 0.0
  v2 = 0.0
  v3 = 0.0
  B1 = 100.0/sqrt(4.0*pi)
  B2 = 0.0
  B3 = 0.0
  psi = 0.0
  return prim2cons(equation, @SVector [rho, v1, v2, v3, p, B1, B2, B3, psi])
end

function initial_conditions_ec_test(equation::IdealGlmMhdEquations, x, t)
  # Adapted MHD version of the weak blast wave from Hennemann & Gassner JCP paper 2020 (Sec. 6.3)
  # Same discontinuity in the velocities but with magnetic fields
  # Set up polar coordinates
  inicenter = [0, 0]
  x_norm = x[1] - inicenter[1]
  y_norm = x[2] - inicenter[2]
  r = sqrt(x_norm^2 + y_norm^2)
  phi = atan(y_norm, x_norm)

  # Calculate primitive variables
  rho = r > 0.5 ? 1.0 : 1.1691
  v1 = r > 0.5 ? 0.0 : 0.1882 * cos(phi)
  v2 = r > 0.5 ? 0.0 : 0.1882 * sin(phi)
  p = r > 0.5 ? 1.0 : 1.245

  return prim2cons(equation, @SVector [rho, v1, v2, 0.0, p, 1.0, 1.0, 1.0, 0.0])
end


# Pre-defined source terms should be implemented as
# function source_terms_WHATEVER(equation::IdealGlmMhdEquations, ut, u, x, element_id, t, n_nodes)


# Calculate 1D flux in for a single point
@inline function calcflux(equation::IdealGlmMhdEquations, orientation, u)
  rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = u
  v1 = rho_v1/rho
  v2 = rho_v2/rho
  v3 = rho_v3/rho
  mag_en = 0.5*(B1^2 + B2^2 + B3^2)
  p = (equation.gamma - 1) * (rho_e - 0.5*rho*(v1^2 + v2^2 + v3^2) - mag_en - 0.5*psi^2)
  if orientation == 1
    f1 = rho_v1
    f2 = rho_v1*v1 + p + mag_en - B1^2
    f3 = rho_v1*v2 - B1*B2
    f4 = rho_v1*v3 - B1*B3
    f5 = (rho_e + p + mag_en)*v1 - B1*(v1*B1 + v2*B2 + v3*B3) + equation.c_h*psi*B1
    f6 = equation.c_h*psi
    f7 = v1*B2 - v2*B1
    f8 = v1*B3 - v3*B1
    f9 = equation.c_h*B1
  else
    f1 = rho_v2
    f2 = rho_v2*v1 - B1*B2
    f3 = rho_v2*v2 + p + mag_en - B2^2
    f4 = rho_v2*v3 - B2*B3
    f5 = (rho_e + p + mag_en)*v2 - B2*(v1*B1 + v2*B2 + v3*B3) + equation.c_h*psi*B2
    f6 = v2*B1 - v1*B2
    f7 = equation.c_h*psi
    f8 = v2*B3 - v3*B2
    f9 = equation.c_h*B2
  end

  return SVector(f1, f2, f3, f4, f5, f6, f7, f8, f9)
end


# Calculate the nonconservative terms from Powell and Galilean invariance
# OBS! This is scaled by 1/2 becuase it will cancel later with the factor of 2 in dsplit_transposed
@inline function calcflux_twopoint_nonconservative!(f1, f2, dg, equation::IdealGlmMhdEquations, u, element_id)
  phi_pow   = zeros(MVector{9})
  phi_gal_x = zeros(MVector{9})
  phi_gal_y = zeros(MVector{9})
  for j in 1:nnodes(dg)
    for i in 1:nnodes(dg)
      v1 = u[2,i,j,element_id] / u[1,i,j,element_id]
      v2 = u[3,i,j,element_id] / u[1,i,j,element_id]
      v3 = u[4,i,j,element_id] / u[1,i,j,element_id]
      # Powell nonconservative term: Φ^Pow = (0, B_1, B_2, B_3, v⋅B, v_1, v_2, v_3, 0)
      phi_pow[2] = 0.5*u[6,i,j,element_id]
      phi_pow[3] = 0.5*u[7,i,j,element_id]
      phi_pow[4] = 0.5*u[8,i,j,element_id]
      phi_pow[5] = 0.5*(v1*u[6,i,j,element_id] + v2*u[7,i,j,element_id] + v3*u[8,i,j,element_id])
      phi_pow[6] = 0.5*v1
      phi_pow[7] = 0.5*v2
      phi_pow[8] = 0.5*v3
      # Galilean nonconservative term: Φ^Gal_{1,2} = (0, 0, 0, 0, ψ v_{1,2}, 0, 0, 0, v_{1,2})
      # x-direction
      phi_gal_x[5] = 0.5*v1*u[9,i,j,element_id]
      phi_gal_x[9] = 0.5*v1
      # y-direction
      phi_gal_y[5] = 0.5*v2*u[9,i,j,element_id]
      phi_gal_y[9] = 0.5*v2
      # add both nonconservative terms into the volume
      for l in 1:nnodes(dg)
        f1[:,l,i,j] += phi_pow * u[6,l,j,element_id] + phi_gal_x * u[9,l,j,element_id]
        f2[:,l,i,j] += phi_pow * u[7,i,l,element_id] + phi_gal_y * u[9,i,l,element_id]
      end
    end
  end
end


"""
    flux_derigs_etal(equation::IdealGlmMhdEquations, orientation, u_ll, u_rr)

Entropy conserving two-point flux by Derigs et al. (2018)
  Ideal GLM-MHD: About the entropy consistent nine-wave magnetic field
  divergence diminishing ideal magnetohydrodynamics equations
[DOI: 10.1016/j.jcp.2018.03.002](https://doi.org/10.1016/j.jcp.2018.03.002)
"""
function flux_derigs_etal(equation::IdealGlmMhdEquations, orientation, u_ll, u_rr)
  # Unpack left and right states to get velocities, pressure, and inverse temperature (called beta)
  rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

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
  p_ll = (equation.gamma - 1)*(rho_e_ll - 0.5*rho_ll*vel_norm_ll - 0.5*mag_norm_ll - 0.5*psi_ll^2)
  p_rr = (equation.gamma - 1)*(rho_e_rr - 0.5*rho_rr*vel_norm_rr - 0.5*mag_norm_rr - 0.5*psi_rr^2)
  beta_ll = 0.5*rho_ll/p_ll
  beta_rr = 0.5*rho_rr/p_rr
  # for convenience store v⋅B
  vel_dot_mag_ll = v1_ll*B1_ll + v2_ll*B2_ll + v3_ll*B3_ll
  vel_dot_mag_rr = v1_rr*B1_rr + v2_rr*B2_rr + v3_rr*B3_rr

  # Compute the necessary mean values needed for either direction
  rho_avg  = 0.5*(rho_ll+rho_rr)
  rho_mean = ln_mean(rho_ll,rho_rr)
  beta_mean = ln_mean(beta_ll,beta_rr)
  beta_avg = 0.5*(beta_ll+beta_rr)
  v1_avg = 0.5*(v1_ll+v1_rr)
  v2_avg = 0.5*(v2_ll+v2_rr)
  v3_avg = 0.5*(v3_ll+v3_rr)
  p_mean = 0.5*rho_avg/beta_avg
  B1_avg = 0.5*(B1_ll+B1_rr)
  B2_avg = 0.5*(B2_ll+B2_rr)
  B3_avg = 0.5*(B3_ll+B3_rr)
  psi_avg = 0.5*(psi_ll+psi_rr)
  vel_norm_avg = 0.5*(vel_norm_ll+vel_norm_rr)
  mag_norm_avg = 0.5*(mag_norm_ll+mag_norm_rr)
  vel_dot_mag_avg = 0.5*(vel_dot_mag_ll+vel_dot_mag_rr)

  # Calculate fluxes depending on orientation with specific direction averages
  if orientation == 1
    f1 = rho_mean*v1_avg
    f2 = f1*v1_avg + p_mean + 0.5*mag_norm_avg - B1_avg*B1_avg
    f3 = f1*v2_avg - B1_avg*B2_avg
    f4 = f1*v3_avg - B1_avg*B3_avg
    f6 = equation.c_h*psi_avg
    f7 = v1_avg*B2_avg - v2_avg*B1_avg
    f8 = v1_avg*B3_avg - v3_avg*B1_avg
    f9 = equation.c_h*B1_avg
    # total energy flux is complicated and involves the previous eight components
    psi_B1_avg = 0.5*(B1_ll*psi_ll + B1_rr*psi_rr)
    v1_mag_avg = 0.5*(v1_ll*mag_norm_ll + v1_rr*mag_norm_rr)
    f5 = (f1*0.5*(1/(equation.gamma-1)/beta_mean - vel_norm_avg) + f2*v1_avg + f3*v2_avg +
          f4*v3_avg + f6*B1_avg + f7*B2_avg + f8*B3_avg + f9*psi_avg - 0.5*v1_mag_avg +
          B1_avg*vel_dot_mag_avg - equation.c_h*psi_B1_avg)
  else
    f1 = rho_mean*v2_avg
    f2 = f1*v1_avg - B1_avg*B2_avg
    f3 = f1*v2_avg + p_mean + 0.5*mag_norm_avg - B2_avg*B2_avg
    f4 = f1*v3_avg - B2_avg*B3_avg
    f6 = v2_avg*B1_avg - v1_avg*B2_avg
    f7 = equation.c_h*psi_avg
    f8 = v2_avg*B3_avg - v3_avg*B2_avg
    f9 = equation.c_h*B2_avg
    # total energy flux is complicated and involves the previous eight components
    psi_B2_avg = 0.5*(B2_ll*psi_ll + B2_rr*psi_rr)
    v2_mag_avg = 0.5*(v2_ll*mag_norm_ll + v2_rr*mag_norm_rr)
    f5 = (f1*0.5*(1/(equation.gamma-1)/beta_mean - vel_norm_avg) + f2*v1_avg + f3*v2_avg +
          f4*v3_avg + f6*B1_avg + f7*B2_avg + f8*B3_avg + f9*psi_avg - 0.5*v2_mag_avg +
          B2_avg*vel_dot_mag_avg - equation.c_h*psi_B2_avg)
  end

  return SVector(f1, f2, f3, f4, f5, f6, f7, f8, f9)
end


function flux_lax_friedrichs(equation::IdealGlmMhdEquations, orientation, u_ll, u_rr)
  rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

  # Calculate velocities and fast magnetoacoustic wave speeds
  # left
  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v3_ll = rho_v3_ll / rho_ll
  v_mag_ll = sqrt(v1_ll^2 + v2_ll^2 + v3_ll^2)
  cf_ll = calc_fast_wavespeed(equation, orientation, u_ll)
  # right
  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  v3_rr = rho_v3_rr / rho_rr
  v_mag_rr = sqrt(v1_rr^2 + v2_rr^2 + v3_rr^2)
  cf_rr = calc_fast_wavespeed(equation, orientation, u_rr)

  # Obtain left and right fluxes
  f_ll = calcflux(equation, orientation, u_ll)
  f_rr = calcflux(equation, orientation, u_rr)

  λ_max = max(v_mag_ll, v_mag_rr) + max(cf_ll, cf_rr)
  f1 = 1/2 * (f_ll[1] + f_rr[1]) - 1/2 * λ_max * (rho_rr    - rho_ll)
  f2 = 1/2 * (f_ll[2] + f_rr[2]) - 1/2 * λ_max * (rho_v1_rr - rho_v1_ll)
  f3 = 1/2 * (f_ll[3] + f_rr[3]) - 1/2 * λ_max * (rho_v2_rr - rho_v2_ll)
  f4 = 1/2 * (f_ll[4] + f_rr[4]) - 1/2 * λ_max * (rho_v3_rr - rho_v3_ll)
  f5 = 1/2 * (f_ll[5] + f_rr[5]) - 1/2 * λ_max * (rho_e_rr  - rho_e_ll)
  f6 = 1/2 * (f_ll[6] + f_rr[6]) - 1/2 * λ_max * (B1_rr     - B1_ll)
  f7 = 1/2 * (f_ll[7] + f_rr[7]) - 1/2 * λ_max * (B2_rr     - B2_ll)
  f8 = 1/2 * (f_ll[8] + f_rr[8]) - 1/2 * λ_max * (B3_rr     - B3_ll)
  f9 = 1/2 * (f_ll[9] + f_rr[9]) - 1/2 * λ_max * (psi_rr    - psi_ll)

  return SVector(f1, f2, f3, f4, f5, f6, f7, f8, f9)
end


# strong form of nonconservative flux on a side, e.g., the Powell term
#     phi^L 1/2 (B^L+B^R) normal - phi^L B^L normal = phi^L 1/2 (B^R-B^L) normal
# OBS! 1) "weak" formulation of split DG already includes the contribution -1/2(phi^L B^L normal)
#         so this routine only adds 1/2(phi^L B^R nvec)
#         analogously for the Galilean nonconservative term
#      2) this is non-unique along a surface! normal direction is super important
function noncons_surface_flux!(noncons_flux::AbstractArray{Float64},
                               u_left::AbstractArray{Float64},
                               u_right::AbstractArray{Float64},
                               surface_id::Int, equation::IdealGlmMhdEquations, n_nodes::Int,
                               orientations::Vector{Int})
  for i in 1:n_nodes
    # extract necessary variable from the left
    v1_ll  = u_left[2,i,surface_id]/u_left[1,i,surface_id]
    v2_ll  = u_left[3,i,surface_id]/u_left[1,i,surface_id]
    v3_ll  = u_left[4,i,surface_id]/u_left[1,i,surface_id]
    B1_ll  = u_left[6,i,surface_id]
    B2_ll  = u_left[7,i,surface_id]
    B3_ll  = u_left[8,i,surface_id]
    psi_ll = u_left[9,i,surface_id]
    v_dot_B_ll = v1_ll*B1_ll + v2_ll*B2_ll + v3_ll*B3_ll
    # extract necessary magnetic field variable from the right and normal velocity
    # both depend upon the orientation
    if orientations[surface_id] == 1
      v_normal = v1_ll
      B_normal = u_right[6,i,surface_id]
      psi_rr   = u_right[9,i,surface_id]
    else
      v_normal = v2_ll
      B_normal = u_right[7,i,surface_id]
      psi_rr   = u_right[9,i,surface_id]
    end
    # compute the nonconservative flux: Powell (with B_normal) and Galilean (with v_normal)
    noncons_flux[1,i] = 0.0
    noncons_flux[2,i] = 0.5*B_normal*B1_ll
    noncons_flux[3,i] = 0.5*B_normal*B2_ll
    noncons_flux[4,i] = 0.5*B_normal*B3_ll
    noncons_flux[5,i] = 0.5*B_normal*v_dot_B_ll + 0.5*v_normal*psi_ll*psi_rr
    noncons_flux[6,i] = 0.5*B_normal*v1_ll
    noncons_flux[7,i] = 0.5*B_normal*v2_ll
    noncons_flux[8,i] = 0.5*B_normal*v3_ll
    noncons_flux[9,i] = 0.5*v_normal*psi_rr
  end
end


# 1) Determine maximum stable time step based on polynomial degree and CFL number
# 2) Update the GLM cleaning wave speed c_h to be the largest value of the fast
#    magnetoacoustic over the entire domain (note this routine is called in a loop
#    over all elements in dg.jl)
function calc_max_dt(equation::IdealGlmMhdEquations, u::Array{Float64, 4},
                     element_id::Int, n_nodes::Int,
                     invjacobian::Float64, cfl::Float64)
  λ_max = 0.0
  equation.c_h = 0.0
  for j = 1:n_nodes
    for i = 1:n_nodes
      v1 = u[2, i, j, element_id]/u[1, i, j, element_id]
      v2 = u[3, i, j, element_id]/u[1, i, j, element_id]
      v3 = u[4, i, j, element_id]/u[1, i, j, element_id]
      v_mag = sqrt(v1^2 + v2^2 + v3^2)
      cf_x_direction = calc_fast_wavespeed(equation, 1, u[:, i, j, element_id])
      cf_y_direction = calc_fast_wavespeed(equation, 2, u[:, i, j, element_id])
      cf_max = max(cf_x_direction,cf_y_direction)
      equation.c_h = max(equation.c_h,cf_max) # GLM cleaning speed = c_f
      λ_max = max(λ_max, v_mag + cf_max)
    end
  end

  dt = cfl * 2 / (invjacobian * λ_max) / n_nodes

  return dt
end


# Convert conservative variables to primitive
function cons2prim(equation::IdealGlmMhdEquations, cons::Array{Float64, 4})
  prim = similar(cons)
  @. prim[1, :, :, :] = cons[1, :, :, :]
  @. prim[2, :, :, :] = cons[2, :, :, :] / cons[1, :, :, :]
  @. prim[3, :, :, :] = cons[3, :, :, :] / cons[1, :, :, :]
  @. prim[4, :, :, :] = cons[4, :, :, :] / cons[1, :, :, :]
  @. prim[5, :, :, :] = ((equation.gamma - 1)
                         *(cons[5, :, :, :] - 0.5 * (cons[2, :, :, :] * prim[2, :, :, :] +
                                                     cons[3, :, :, :] * prim[3, :, :, :] +
                                                     cons[4, :, :, :] * prim[4, :, :, :])
                                            - 0.5 * (cons[6, :, :, :] * cons[6, :, :, :] +
                                                     cons[7, :, :, :] * cons[7, :, :, :] +
                                                     cons[8, :, :, :] * cons[8, :, :, :])
                                            - 0.5 * cons[9, :, :, :] * cons[9, :, :, :]))
  @. prim[6, :, :, :] = cons[6, :, :, :]
  @. prim[7, :, :, :] = cons[7, :, :, :]
  @. prim[8, :, :, :] = cons[8, :, :, :]
  @. prim[9, :, :, :] = cons[9, :, :, :]
  return prim
end


# Convert conservative variables to entropy
function cons2entropy(equation::IdealGlmMhdEquations, cons::Array{Float64, 4}, n_nodes::Int, n_elements::Int)
  entropy = similar(cons)
  v = zeros(3,n_nodes,n_nodes,n_elements)
  B = zeros(3,n_nodes,n_nodes,n_elements)
  v_square = zeros(n_nodes,n_nodes,n_elements)
  p = zeros(n_nodes,n_nodes,n_elements)
  s = zeros(n_nodes,n_nodes,n_elements)
  rho_p = zeros(n_nodes,n_nodes,n_elements)
  # velocities
  @. v[1, :, :, :] = cons[2, :, :, :] / cons[1, :, :, :]
  @. v[2, :, :, :] = cons[3, :, :, :] / cons[1, :, :, :]
  @. v[3, :, :, :] = cons[4, :, :, :] / cons[1, :, :, :]
  # magnetic fields
  @. B[1, :, :, :] = cons[6, :, :, :]
  @. B[2, :, :, :] = cons[7, :, :, :]
  @. B[3, :, :, :] = cons[8, :, :, :]
  # kinetic energy, pressure, entropy
  @. v_square[ :, :, :] = v[1, :, :, :]*v[1, :, :, :] + v[2, :, :, :]*v[2, :, :, :] +
                          v[3, :, :, :]*v[3, :, :, :]
  @. p[ :, :, :] = ((equation.gamma - 1)*(cons[5, :, :, :] - 0.5*cons[1, :, :, :]*v_square[:,:,:] -
                    0.5*(B[1, :, :, :]*B[1, :, :, :] + B[2, :, :, :]*B[2, :, :, :] +
                         B[3, :, :, :]*B[3, :, :, :]) - 0.5*cons[9, :, :, :]*cons[9, :, :, :]))
  @. s[ :, :, :] = log(p[:, :, :]) - equation.gamma*log(cons[1, :, :, :])
  @. rho_p[ :, :, :] = cons[1, :, :, :] / p[ :, :, :]

  @. entropy[1, :, :, :] = (equation.gamma - s[:,:,:])/(equation.gamma-1) -
                           0.5*rho_p[:,:,:]*v_square[:,:,:]
  @. entropy[2, :, :, :] = rho_p[:,:,:]*v[1,:,:,:]
  @. entropy[3, :, :, :] = rho_p[:,:,:]*v[2,:,:,:]
  @. entropy[4, :, :, :] = rho_p[:,:,:]*v[3,:,:,:]
  @. entropy[5, :, :, :] = -rho_p[:,:,:]
  @. entropy[6, :, :, :] = rho_p[:,:,:]*B[1,:,:,:]
  @. entropy[7, :, :, :] = rho_p[:,:,:]*B[2,:,:,:]
  @. entropy[8, :, :, :] = rho_p[:,:,:]*B[3,:,:,:]
  @. entropy[9, :, :, :] = rho_p[:,:,:]*cons[9,:,:,:]

  return entropy
end

# Convert primitive to conservative variables
function prim2cons(equation::IdealGlmMhdEquations, prim::AbstractArray{Float64})
  cons = similar(prim)
  cons[1] = prim[1]
  cons[2] = prim[2] * prim[1]
  cons[3] = prim[3] * prim[1]
  cons[4] = prim[4] * prim[1]
  cons[5] = prim[5]/(equation.gamma-1)+0.5*(cons[2]*prim[2] + cons[3]*prim[3] + cons[4]*prim[4])+
            0.5*(prim[6]*prim[6] + prim[7]*prim[7] + prim[8]*prim[8] + 0.5*prim[9]*prim[9])
  cons[6] = prim[6]
  cons[7] = prim[7]
  cons[8] = prim[8]
  cons[9] = prim[9]
  return cons
end


# Convert conservative variables to indicator variable for discontinuities (elementwise version)
@inline function cons2indicator!(indicator::AbstractArray{Float64}, equation::IdealGlmMhdEquations,
                                 cons::AbstractArray{Float64},
                                 element_id::Int, n_nodes::Int, indicator_variable)
  for j in 1:n_nodes
    for i in 1:n_nodes
      indicator[1, i, j] = cons2indicator(equation,
                                          cons[1, i, j, element_id], cons[2, i, j, element_id],
                                          cons[3, i, j, element_id], cons[4, i, j, element_id],
                                          cons[5, i, j, element_id], cons[6, i, j, element_id],
                                          cons[7, i, j, element_id], cons[8, i, j, element_id],
                                          cons[9, i, j, element_id], indicator_variable)
    end
  end
end



# Convert conservative variables to indicator variable for discontinuities (pointwise version)
@inline function cons2indicator(equation::IdealGlmMhdEquations, rho, rho_v1, rho_v2, rho_v3, rho_e,
                                B1, B2, B3, psi, ::Val{:density})
  # Indicator variable is rho
  return rho
end



# Convert conservative variables to indicator variable for discontinuities (pointwise version)
@inline function cons2indicator(equation::IdealGlmMhdEquations, rho, rho_v1, rho_v2, rho_v3, rho_e,
                                B1, B2, B3, psi, ::Val{:pressure})
  v1 = rho_v1/rho
  v2 = rho_v2/rho
  v3 = rho_v3/rho
  # Indicator variable is p
  p = (equation.gamma - 1)*(rho_e - 0.5*rho*(v1^2 + v2^2 + v3^2)
                                  - 0.5*(B1^2 + B2^2 + B3^2)
                                  - 0.5*psi^2)
  return p
end


# Convert conservative variables to indicator variable for discontinuities (pointwise version)
@inline function cons2indicator(equation::IdealGlmMhdEquations, rho, rho_v1, rho_v2, rho_v3, rho_e,
                                B1, B2, B3, psi, ::Val{:density_pressure})
  v1 = rho_v1/rho
  v2 = rho_v2/rho
  v3 = rho_v3/rho
  # Calculate pressure
  p = (equation.gamma - 1)*(rho_e - 0.5*rho*(v1^2 + v2^2 + v3^2)
                                  - 0.5*(B1^2 + B2^2 + B3^2)
                                  - 0.5*psi^2)
  # Indicator variable is rho * p
  return rho * p
end


# Compute the fastest wave speed for ideal MHD equations: c_f, the fast magnetoacoustic eigenvalue
@inline function calc_fast_wavespeed(equation::IdealGlmMhdEquations, direction, cons)
  rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = cons
  v1 = rho_v1/rho
  v2 = rho_v2/rho
  v3 = rho_v3/rho
  v_mag = sqrt(v1^2 + v2^2 + v3^2)
  p = (equation.gamma - 1)*(rho_e - 0.5*rho*v_mag^2 - 0.5*(B1^2 + B2^2 + B3^2) - 0.5*psi^2)
  a_square = equation.gamma * p / rho
  b1 = B1/sqrt(rho)
  b2 = B2/sqrt(rho)
  b3 = B3/sqrt(rho)
  b_square = b1^2 + b2^2 + b3^2
  if direction == 1 # x-direction
    c_f = sqrt(0.5*(a_square + b_square) + 0.5*sqrt((a_square + b_square)^2 - 4.0*a_square*b1^2))
  else
    c_f = sqrt(0.5*(a_square + b_square) + 0.5*sqrt((a_square + b_square)^2 - 4.0*a_square*b2^2))
  end
  return c_f
end


# Calculate thermodynamic entropy for a conservative state `cons`
@inline function entropy_thermodynamic(cons, equation::IdealGlmMhdEquations)
  # Pressure
  p = (equation.gamma - 1) * (cons[5] - 1/2 * (cons[2]^2 + cons[3]^2 + cons[4]^2) / cons[1]
                                      - 1/2 * (cons[6]^2 + cons[7]^2 + cons[8]^2)
                                      - 1/2 * cons[9]^2)

  # Thermodynamic entropy
  s = log(p) - equation.gamma*log(cons[1])

  return s
end


# Calculate mathematical entropy for a conservative state `cons`
@inline function entropy_math(cons, equation::IdealGlmMhdEquations)
  S = -entropy_thermodynamic(cons, equation) * cons[1] / (equation.gamma - 1)

  return S
end


# Default entropy is the mathematical entropy
@inline entropy(cons, equation::IdealGlmMhdEquations) = entropy_math(cons, equation)


# Calculate total energy for a conservative state `cons`
@inline energy_total(cons, ::IdealGlmMhdEquations) = cons[5]


# Calculate kinetic energy for a conservative state `cons`
@inline function energy_kinetic(cons, equation::IdealGlmMhdEquations)
  return 0.5 * (cons[2]^2 + cons[3]^2 + cons[4]^2)/cons[1]
end


# Calculate the magnetic energy for a conservative state `cons'.
#  OBS! For non-dinmensional form of the ideal MHD magnetic pressure ≡ magnetic energy
@inline function energy_magnetic(cons, ::IdealGlmMhdEquations)
  return 0.5 * (cons[6]^2 + cons[7]^2 + cons[8]^2)
end


# Calculate internal energy for a conservative state `cons`
@inline function energy_internal(cons, equation::IdealGlmMhdEquations)
  return (energy_total(cons, equation)
          - energy_kinetic(cons, equation)
          - energy_magnetic(cons, equation)
          - cons[9]^2 / 2)
end


# Calcluate the cross helicity (\vec{v}⋅\vec{B}) for a conservative state `cons'
@inline function cross_helicity(cons, ::IdealGlmMhdEquations)
  return (cons[2]*cons[6] + cons[3]*cons[7] + cons[4]*cons[8]) / cons[1]
end
