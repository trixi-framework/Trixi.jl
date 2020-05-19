module MhdEquations

using ...Trixi
using ..Equations # Use everything to allow method extension via "function Equations.<method>"
using ...Auxiliary: parameter
using ...Auxiliary.Math: ln_mean
using StaticArrays: @SVector, SVector, MVector, MMatrix, MArray

# Export all symbols that should be available from Equations
export Mhd
export initial_conditions
export sources
export calcflux!
export riemann!
export noncons_surface_flux!
export calc_max_dt
export cons2prim
export cons2entropy
export cons2indicator
export cons2indicator!


# Main data structure for system of equations "Mhd"
mutable struct Mhd{SurfaceFlux} <: AbstractEquation{9}
  name::String
  initial_conditions::String
  sources::String
  varnames_cons::SVector{9, String}
  varnames_prim::SVector{9, String}
  gamma::Float64
  c_h::Float64 # GLM cleaning speed
  surface_flux::SurfaceFlux
  volume_flux_type::Symbol
  have_nonconservative_terms::Bool
end

function Mhd()
  name = "mhd"
  initial_conditions = parameter("initial_conditions")
  sources = parameter("sources", "none")
  varnames_cons = @SVector ["rho", "rho_v1", "rho_v2", "rho_v3", "rho_e", "B1", "B2", "B3", "psi"]
  varnames_prim = @SVector ["rho", "v1", "v2", "v3", "p", "B1", "B2", "B3", "psi"]
  gamma = parameter("gamma", 1.4)
  c_h = 0.0   # GLM cleaning wave speed
  surface_flux_type = Symbol(parameter("surface_flux", "lax_friedrichs_flux",
                                       valid=["lax_friedrichs_flux", "central_flux", "derigs_et_al_flux"]))
  surface_flux = eval(surface_flux_type)
  volume_flux_type = Symbol(parameter("volume_flux_type", "central_flux",
                                      valid=["central_flux", "derigs_et_al_flux"]))
  have_nonconservative_terms = true
  Mhd(name, initial_conditions, sources, varnames_cons, varnames_prim, gamma, c_h,
      surface_flux, volume_flux_type, have_nonconservative_terms)
end


# Set initial conditions at physical location `x` for time `t`
function Equations.initial_conditions(equation::Mhd, x::AbstractArray{Float64}, t::Real)
  name = equation.initial_conditions
  if name == "constant"
    rho = 1.0
    rho_v1 = 0.1
    rho_v2 = -0.2
    rho_v3 = -0.5
    rho_e = 50.0
    B1 = 3.0
    B2 = -1.2
    B3 = 0.5
    psi = 0.0
    return [rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi]
  elseif name == "convergence_test"
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
    return prim2cons(equation, [rho, v1, v2, v3, p, B1, B2, B3, psi])
  elseif name == "orszag_tang"
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
    return prim2cons(equation, [rho, v1, v2, v3, p, B1, B2, B3, psi])
  elseif name == "rotor"
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
    return prim2cons(equation, [rho, v1, v2, v3, p, B1, B2, B3, psi])
  elseif name == "mhd_blast"
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
    return prim2cons(equation, [rho, v1, v2, v3, p, B1, B2, B3, psi])
  elseif name == "ec_test"
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

    return prim2cons(equation, [rho, v1, v2, 0.0, p, 1.0, 1.0, 1.0, 0.0])
  else
    error("Unknown initial condition '$name'")
  end

end


# Apply source terms
function Equations.sources(equation::Mhd, ut, u, x, element_id, t, n_nodes)
  name = equation.sources
  error("Unknown source term '$name'")
end


# Calculate 2D flux (element version)
@inline function Equations.calcflux!(f1::AbstractArray{Float64},
                                     f2::AbstractArray{Float64},
                                     equation::Mhd,
                                     u::AbstractArray{Float64}, element_id::Int,
                                     n_nodes::Int)
  for j = 1:n_nodes
    for i = 1:n_nodes
      rho    = u[1, i, j, element_id]
      rho_v1 = u[2, i, j, element_id]
      rho_v2 = u[3, i, j, element_id]
      rho_v3 = u[4, i, j, element_id]
      rho_e  = u[5, i, j, element_id]
      B1     = u[6, i, j, element_id]
      B2     = u[7, i, j, element_id]
      B3     = u[8, i, j, element_id]
      psi    = u[9, i, j, element_id]
      @views calcflux!(f1[:, i, j], f2[:, i, j], equation, rho, rho_v1, rho_v2, rho_v3, rho_e,
                       B1, B2, B3, psi)
    end
  end
end


# Calculate 2D flux (pointwise version)
@inline function Equations.calcflux!(f1::AbstractArray{Float64},
                                     f2::AbstractArray{Float64},
                                     equation::Mhd,
                                     rho::Float64, rho_v1::Float64,
                                     rho_v2::Float64, rho_v3::Float64,
                                     rho_e::Float64, B1::Float64,
                                     B2::Float64, B3::Float64, psi::Float64)
  v1 = rho_v1/rho
  v2 = rho_v2/rho
  v3 = rho_v3/rho
  mag_en = 0.5*(B1^2 + B2^2 + B3^2)
  v_dot_B = v1*B1 + v2*B2 + v3*B3
  p = (equation.gamma - 1)*(rho_e - 0.5*rho*(v1^2 + v2^2 + v3^2) - mag_en - 0.5*psi^2)

  f1[1] = rho_v1
  f1[2] = rho_v1*v1 + p + mag_en - B1^2
  f1[3] = rho_v1*v2 - B1*B2
  f1[4] = rho_v1*v3 - B1*B3
  f1[5] = (rho_e + p + mag_en)*v1 - B1*v_dot_B + equation.c_h*psi*B1
  f1[6] = equation.c_h*psi
  f1[7] = v1*B2 - v2*B1
  f1[8] = v1*B3 - v3*B1
  f1[9] = equation.c_h*B1

  f2[1] = rho_v2
  f2[2] = rho_v2*v1 - B1*B2
  f2[3] = rho_v2*v2 + p + mag_en - B2^2
  f2[4] = rho_v2*v3 - B2*B3
  f2[5] = (rho_e + p + mag_en)*v2 - B2*v_dot_B + equation.c_h*psi*B2
  f2[6] = v2*B1 - v1*B2
  f2[7] = equation.c_h*psi
  f2[8] = v2*B3 - v3*B2
  f2[9] = equation.c_h*B2

  return nothing
end


# Calculate 2D two-point flux (decide which volume flux type to use)
@inline function Equations.calcflux_twopoint!(f1::AbstractArray{Float64},
                                              f2::AbstractArray{Float64},
                                              f1_diag::AbstractArray{Float64},
                                              f2_diag::AbstractArray{Float64},
                                              equation::Mhd,
                                              u::AbstractArray{Float64},
                                              element_id::Int, n_nodes::Int)
  calcflux_twopoint!(f1, f2, f1_diag, f2_diag, Val(equation.volume_flux_type),
                     equation, u, element_id, n_nodes)
end


# Calculate 2D two-point flux (element version)
@inline function Equations.calcflux_twopoint!(f1::AbstractArray{Float64},
                                              f2::AbstractArray{Float64},
                                              f1_diag::AbstractArray{Float64},
                                              f2_diag::AbstractArray{Float64},
                                              twopoint_flux_type::Val,
                                              equation::Mhd,
                                              u::AbstractArray{Float64},
                                              element_id::Int, n_nodes::Int)
  # Calculate regular volume fluxes
  calcflux!(f1_diag, f2_diag, equation, u, element_id, n_nodes)
  # split form advective fluxes
  for j = 1:n_nodes
    for i = 1:n_nodes
      # Set diagonal entries (= regular volume fluxes due to consistency)
      for v in 1:nvariables(equation)
        f1[v, i, i, j] = f1_diag[v, i, j]
        f2[v, j, i, j] = f2_diag[v, i, j]
      end

      # Flux in x-direction
      for l = i + 1:n_nodes
        @views symmetric_twopoint_flux!(f1[:, l, i, j], twopoint_flux_type,
                                        equation, 1, # 1-> x-direction
                                        u[1, i, j, element_id], u[2, i, j, element_id],
                                        u[3, i, j, element_id], u[4, i, j, element_id],
                                        u[5, i, j, element_id], u[6, i, j, element_id],
                                        u[7, i, j, element_id], u[8, i, j, element_id],
                                        u[9, i, j, element_id],
                                        u[1, l, j, element_id], u[2, l, j, element_id],
                                        u[3, l, j, element_id], u[4, l, j, element_id],
                                        u[5, l, j, element_id], u[6, l, j, element_id],
                                        u[7, l, j, element_id], u[8, l, j, element_id],
                                        u[9, l, j, element_id])
        for v in 1:nvariables(equation)
          f1[v, i, l, j] = f1[v, l, i, j]
        end
      end

      # Flux in y-direction
      for l = j + 1:n_nodes
        @views symmetric_twopoint_flux!(f2[:, l, i, j], twopoint_flux_type,
                                        equation, 2, # 2 -> y-direction
                                        u[1, i, j, element_id], u[2, i, j, element_id],
                                        u[3, i, j, element_id], u[4, i, j, element_id],
                                        u[5, i, j, element_id], u[6, i, j, element_id],
                                        u[7, i, j, element_id], u[8, i, j, element_id],
                                        u[9, i, j, element_id],
                                        u[1, i, l, element_id], u[2, i, l, element_id],
                                        u[3, i, l, element_id], u[4, i, l, element_id],
                                        u[5, i, l, element_id], u[6, i, l, element_id],
                                        u[7, i, l, element_id], u[8, i, l, element_id],
                                        u[9, i, l, element_id])
        for v in 1:nvariables(equation)
          f2[v, j, i, l] = f2[v, l, i, j]
        end
      end
    end
  end
  # add in nonconservative terms; see GLM paper from Derigs et al. 2018
  @views add_noncons_flux!(f1,f2,u[:,:,:,element_id],n_nodes)
end

# Calculate the nonconservative terms from Powell and Galilean invariance
# OBS! This is scaled by 1/2 becuase it will cancel later with the factor of 2 in dsplit_transposed
@inline function add_noncons_flux!(f::AbstractArray{Float64,4}, # x-direction advective flux
                                   g::AbstractArray{Float64,4}, # y-direction advective flux
                                   u::AbstractArray{Float64,3}, # solution on an element
                                   n_nodes::Int)
  phi_pow   = zeros(MVector{9})
  phi_gal_x = zeros(MVector{9})
  phi_gal_y = zeros(MVector{9})
  for j in 1:n_nodes
    for i in 1:n_nodes
      v1 = u[2,i,j]/u[1,i,j]
      v2 = u[3,i,j]/u[1,i,j]
      v3 = u[4,i,j]/u[1,i,j]
      # Powell nonconservative term: Φ^Pow = (0, B_1, B_2, B_3, v⋅B, v_1, v_2, v_3, 0)
      phi_pow[2] = 0.5*u[6,i,j]
      phi_pow[3] = 0.5*u[7,i,j]
      phi_pow[4] = 0.5*u[8,i,j]
      phi_pow[5] = 0.5*(v1*u[6,i,j] + v2*u[7,i,j] + v3*u[8,i,j])
      phi_pow[6] = 0.5*v1
      phi_pow[7] = 0.5*v2
      phi_pow[8] = 0.5*v3
      # Galilean nonconservative term: Φ^Gal_{1,2} = (0, 0, 0, 0, ψ v_{1,2}, 0, 0, 0, v_{1,2})
      # x-direction
      phi_gal_x[5] = 0.5*v1*u[9,i,j]
      phi_gal_x[9] = 0.5*v1
      # y-direction
      phi_gal_y[5] = 0.5*v2*u[9,i,j]
      phi_gal_y[9] = 0.5*v2
      # add both nonconservative terms into the volume
      for l in 1:n_nodes
        f[:,l,i,j] += phi_pow * u[6,l,j] + phi_gal_x * u[9,l,j]
        g[:,l,i,j] += phi_pow * u[7,i,l] + phi_gal_y * u[9,i,l]
      end
    end
  end
end


# Central two-point flux (identical to weak form volume integral, except for floating point errors)
@inline function symmetric_twopoint_flux!(f, ::Val{:central_flux},
                                          equation::Mhd, orientation,
                                          rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll,
                                          B1_ll, B2_ll, B3_ll, psi_ll,
                                          rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr,
                                          B1_rr, B2_rr, B3_rr, psi_rr)
  flux = central_flux(equation, orientation,
                      rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll,
                      B1_ll, B2_ll, B3_ll, psi_ll,
                      rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr,
                      B1_rr, B2_rr, B3_rr, psi_rr)

  for i in 1:9
    f[i] = flux[i]
  end

  return nothing
end

function central_flux(equation::Mhd, orientation,
                      rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll,
                      B1_ll, B2_ll, B3_ll, psi_ll,
                      rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr,
                      B1_rr, B2_rr, B3_rr, psi_rr)
  # Calculate regular 1D fluxes
  f_ll = MVector{9, Float64}(undef)
  f_rr = MVector{9, Float64}(undef)
  calcflux1D!(f_ll, equation, rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll,
              B1_ll, B2_ll, B3_ll, psi_ll, orientation)
  calcflux1D!(f_rr, equation, rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr,
              B1_rr, B2_rr, B3_rr, psi_rr, orientation)
  # Average state of the advective fluxes
  @. 0.5 * (f_ll + f_rr)
end

# Entropy conserving two-point flux for GLM-MHD equations by Derigs et al. 2018; equation (4.43)
@inline function symmetric_twopoint_flux!(f, ::Val{:derigs_et_al_flux},
                                          equation::Mhd, orientation,
                                          rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll,
                                          B1_ll, B2_ll, B3_ll, psi_ll,
                                          rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr,
                                          B1_rr, B2_rr, B3_rr, psi_rr)
  flux = derigs_et_al_flux(equation, orientation,
                           rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll,
                           B1_ll, B2_ll, B3_ll, psi_ll,
                           rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr,
                           B1_rr, B2_rr, B3_rr, psi_rr)

  for i in 1:9
    f[i] = flux[i]
  end

  return nothing
end

"""
    derigs_et_al_flux(equation::Mhd, orientation,
                      rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll,
                      B1_ll, B2_ll, B3_ll, psi_ll,
                      rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr,
                      B1_rr, B2_rr, B3_rr, psi_rr)

Entropy conserving two-point flux by Derigs et al. (2018)
  Ideal GLM-MHD: About the entropy consistent nine-wave magnetic field
  divergence diminishing ideal magnetohydrodynamics equations
[DOI: 10.1016/j.jcp.2018.03.002](https://doi.org/10.1016/j.jcp.2018.03.002)
"""
function derigs_et_al_flux(equation::Mhd, orientation,
                           rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll,
                           B1_ll, B2_ll, B3_ll, psi_ll,
                           rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr,
                           B1_rr, B2_rr, B3_rr, psi_rr)
  # Unpack left and right states to get velocities, pressure, and inverse temperature (called beta)
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

  return (f1, f2, f3, f4, f5, f6, f7, f8, f9)
end


# Calculate 1D flux in for a single point
@inline function calcflux1D!(f::AbstractArray{Float64}, equation::Mhd, rho::Float64,
                             rho_v1::Float64, rho_v2::Float64, rho_v3::Float64, rho_e::Float64,
                             B1::Float64, B2::Float64, B3::Float64, psi::Float64, orientation::Int)
  v1 = rho_v1/rho
  v2 = rho_v2/rho
  v3 = rho_v3/rho
  mag_en = 0.5*(B1^2 + B2^2 + B3^2)
  p = (equation.gamma - 1) * (rho_e - 0.5*rho*(v1^2 + v2^2 + v3^2) - mag_en - 0.5*psi^2)
  if orientation == 1
    f[1]  = rho_v1
    f[2]  = rho_v1*v1 + p + mag_en - B1^2
    f[3]  = rho_v1*v2 - B1*B2
    f[4]  = rho_v1*v3 - B1*B3
    f[5]  = (rho_e + p + mag_en)*v1 - B1*(v1*B1 + v2*B2 + v3*B3) + equation.c_h*psi*B1
    f[6]  = equation.c_h*psi
    f[7]  = v1*B2 - v2*B1
    f[8]  = v1*B3 - v3*B1
    f[9]  = equation.c_h*B1
  else
    f[1]  = rho_v2
    f[2]  = rho_v2*v1 - B1*B2
    f[3]  = rho_v2*v2 + p + mag_en - B2^2
    f[4]  = rho_v2*v3 - B2*B3
    f[5]  = (rho_e + p + mag_en)*v2 - B2*(v1*B1 + v2*B2 + v3*B3) + equation.c_h*psi*B2
    f[6]  = v2*B1 - v1*B2
    f[7]  = equation.c_h*psi
    f[8]  = v2*B3 - v3*B2
    f[9]  = equation.c_h*B2
  end
end


# Calculate flux across interface with different states on both sides (EC mortar version)
function Equations.riemann!(surface_flux::AbstractArray{Float64, 3},
                            fstarnode::AbstractVector{Float64},
                            u_surfaces_left::AbstractArray{Float64, 3},
                            u_surfaces_right::AbstractArray{Float64, 3},
                            surface_id::Int,
                            equation::Mhd, n_nodes::Int,
                            orientations::Vector{Int})
  # Call pointwise Riemann solver
  # i -> left, j -> right
  for j = 1:n_nodes
    for i = 1:n_nodes
      # Store flux in pre-allocated `fstarnode` to avoid allocations in loop
      riemann!(fstarnode,
               u_surfaces_left[1, i, surface_id],
               u_surfaces_left[2, i, surface_id],
               u_surfaces_left[3, i, surface_id],
               u_surfaces_left[4, i, surface_id],
               u_surfaces_left[5, i, surface_id],
               u_surfaces_left[6, i, surface_id],
               u_surfaces_left[7, i, surface_id],
               u_surfaces_left[8, i, surface_id],
               u_surfaces_left[9, i, surface_id],
               u_surfaces_right[1, j, surface_id],
               u_surfaces_right[2, j, surface_id],
               u_surfaces_right[3, j, surface_id],
               u_surfaces_right[4, j, surface_id],
               u_surfaces_right[5, j, surface_id],
               u_surfaces_right[6, j, surface_id],
               u_surfaces_right[7, j, surface_id],
               u_surfaces_right[8, j, surface_id],
               u_surfaces_right[9, j, surface_id],
               equation, orientations[surface_id])

      # Copy flux back to actual flux array
      for v in 1:nvariables(equation)
        surface_flux[v, i, j] = fstarnode[v]
      end
    end
  end
end


# Calculate flux across interface with different states on both sides (surface version)
function Equations.riemann!(surface_flux::AbstractMatrix{Float64},
                            fstarnode::AbstractVector{Float64},
                            u_surfaces::AbstractArray{Float64, 4},
                            surface_id::Int,
                            equation::Mhd, n_nodes::Int,
                            orientations::Vector{Int})
  # Call pointwise Riemann solver
  for i = 1:n_nodes
    # Store flux in pre-allocated `fstarnode` to avoid allocations in loop
    riemann!(fstarnode,
             u_surfaces[1, 1, i, surface_id],
             u_surfaces[1, 2, i, surface_id],
             u_surfaces[1, 3, i, surface_id],
             u_surfaces[1, 4, i, surface_id],
             u_surfaces[1, 5, i, surface_id],
             u_surfaces[1, 6, i, surface_id],
             u_surfaces[1, 7, i, surface_id],
             u_surfaces[1, 8, i, surface_id],
             u_surfaces[1, 9, i, surface_id],
             u_surfaces[2, 1, i, surface_id],
             u_surfaces[2, 2, i, surface_id],
             u_surfaces[2, 3, i, surface_id],
             u_surfaces[2, 4, i, surface_id],
             u_surfaces[2, 5, i, surface_id],
             u_surfaces[2, 6, i, surface_id],
             u_surfaces[2, 7, i, surface_id],
             u_surfaces[2, 8, i, surface_id],
             u_surfaces[2, 9, i, surface_id],
             equation, orientations[surface_id])

    # Copy flux back to actual flux array
    for v in 1:nvariables(equation)
      surface_flux[v, i] = fstarnode[v]
    end
  end
end


# Calculate flux across interface with different states on both sides (pointwise version)
function Equations.riemann!(surface_flux,
                            rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll,
                            rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr,
                            equation::Mhd, orientation)

  # I'm not really sure where to hook into the call chain. This is just a first
  # implementation as proof of concept and should be discussed and improved.
  flux = equation.surface_flux(equation, orientation,
                               rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll,
                               rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr)

  for i in 1:9
    surface_flux[i] = flux[i]
  end

  return nothing
end


function lax_friedrichs_flux(equation::Mhd, orientation,
                             rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll,
                             rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr)
  # Calculate velocities and fast magnetoacoustic wave speeds
  # left
  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v3_ll = rho_v3_ll / rho_ll
  v_mag_ll = sqrt(v1_ll^2 + v2_ll^2 + v3_ll^2)
  cf_ll = calc_fast_wavespeed(equation, orientation, [rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll,
                              rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll])
  # right
  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  v3_rr = rho_v3_rr / rho_rr
  v_mag_rr = sqrt(v1_rr^2 + v2_rr^2 + v3_rr^2)
  cf_rr = calc_fast_wavespeed(equation, orientation, [rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr,
                              rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr])
  # Obtain left and right fluxes
  f_ll = zeros(MVector{9})
  f_rr = zeros(MVector{9})
  calcflux1D!(f_ll, equation, rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll,
              B1_ll, B2_ll, B3_ll, psi_ll, orientation)
  calcflux1D!(f_rr, equation, rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr,
              B1_rr, B2_rr, B3_rr, psi_rr, orientation)

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

  return (f1, f2, f3, f4, f5, f6, f7, f8, f9)
end


# strong form of nonconservative flux on a side, e.g., the Powell term
#     phi^L 1/2 (B^L+B^R) normal - phi^L B^L normal = phi^L 1/2 (B^R-B^L) normal
# OBS! 1) "weak" formulation of split DG already includes the contribution -1/2(phi^L B^L normal)
#         so this routine only adds 1/2(phi^L B^R nvec)
#         analogously for the Galilean nonconservative term
#      2) this is non-unique along a surface! normal direction is super important
function Equations.noncons_surface_flux!(noncons_flux::AbstractArray{Float64},
                                         u_left::AbstractArray{Float64},
                                         u_right::AbstractArray{Float64},
                                         surface_id::Int, equation::Mhd, n_nodes::Int,
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
function Equations.calc_max_dt(equation::Mhd, u::Array{Float64, 4},
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
      cf_x_direction = calc_fast_wavespeed(equation, 1, u[:,i, j, element_id])
      cf_y_direction = calc_fast_wavespeed(equation, 2, u[:,i, j, element_id])
      cf_max = max(cf_x_direction,cf_y_direction)
      equation.c_h = max(equation.c_h,cf_max) # GLM cleaning speed = c_f
      λ_max = max(λ_max, v_mag + cf_max)
    end
  end

  dt = cfl * 2 / (invjacobian * λ_max) / n_nodes

  return dt
end


# Convert conservative variables to primitive
function Equations.cons2prim(equation::Mhd, cons::Array{Float64, 4})
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
function Equations.cons2entropy(equation::Mhd, cons::Array{Float64, 4}, n_nodes::Int, n_elements::Int)
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
function prim2cons(equation::Mhd, prim::AbstractArray{Float64})
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
@inline function Equations.cons2indicator!(indicator::AbstractArray{Float64}, equation::Mhd,
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
@inline function Equations.cons2indicator(equation::Mhd, rho, rho_v1, rho_v2, rho_v3, rho_e,
                                          B1, B2, B3, psi, ::Val{:density})
  # Indicator variable is rho
  return rho
end



# Convert conservative variables to indicator variable for discontinuities (pointwise version)
@inline function Equations.cons2indicator(equation::Mhd, rho, rho_v1, rho_v2, rho_v3, rho_e,
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
@inline function Equations.cons2indicator(equation::Mhd, rho, rho_v1, rho_v2, rho_v3, rho_e,
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
@inline function calc_fast_wavespeed(equation::Mhd, direction::Int, cons::AbstractArray{Float64})
  rho    = cons[1]
  rho_v1 = cons[2]
  rho_v2 = cons[3]
  rho_v3 = cons[4]
  rho_e  = cons[5]
  B1     = cons[6]
  B2     = cons[7]
  B3     = cons[8]
  psi    = cons[9]
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


end # module
