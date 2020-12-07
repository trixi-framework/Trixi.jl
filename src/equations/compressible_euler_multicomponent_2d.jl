
@doc raw"""
    CompressibleEulerMulticomponentEquations2D

Multicomponent version of the compressible Euler equations for an ideal gas in two space dimensions.
"""
struct CompressibleEulerMulticomponentEquations2D{RealT<:Real} <: AbstractCompressibleEulerMulticomponentEquations{2, 5}
  gamma1::RealT 
  gamma2::RealT
  rs1   ::RealT
  rs2   ::RealT
  cvs1  ::RealT
  cvs2  ::RealT
  cps1  ::RealT
  cps2  ::RealT
end

#function CompressibleEulerMulticomponentEquations2D(N::int8, gamma::NTuple{2,<:Real}, rs::NTuple{2,<:Real}, cvs::NTuple{2,<:Real}, cps::NTuple{2,<:Real})
#  CompressibleEulerMulticomponentEquations2D(N, SVector(gamma), SVector(rs), SVector(cvs), SVector(cps))
#end

get_name(::CompressibleEulerMulticomponentEquations2D)      = "CompressibleEulerMulticomponentEquations2D"
varnames_cons(::CompressibleEulerMulticomponentEquations2D) = @SVector ["rho1", "rho2", "rho_v1", "rho_v2", "rho_e"]
varnames_prim(::CompressibleEulerMulticomponentEquations2D) = @SVector ["rho1", "rho2", "v1", "v2", "p"]


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
  ω       = 2 * pi * f
  ini     = c + A * sin(ω * (x[1] + x[2] - t))

  vel1    = 1.0
  vel2    = 1.0

  rho     = ini
  rho1    = 0.2 * rho
  rho2    = 0.8 * rho  
  rho_v1  = rho * vel1
  rho_v2  = rho * vel2
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
  # Same settings as in `initial_condition`
  c       = 2
  A       = 0.1
  L       = 2
  f       = 1/L
  ω       = 2 * pi * f

  rho1    = u[1]
  rho2    = u[2]
  rho     = rho1 + rho2

  γ       = (rho1/rho*equations.cvs1*equations.gamma1 + rho2/rho*equations.cvs2*equations.gamma2)/(rho1/rho*equations.cvs1 + rho2/rho*equations.cvs2)

  x1, x2  = x
  si, co  = sincos((x1 + x2 - t)*ω)
  tmp1    = co * A * ω
  tmp2    = si * A
  tmp3    = γ - 1
  tmp4    = (2*c - 1)*tmp3
  tmp5    = (2*tmp2*γ - 2*tmp2 + tmp4 + 1)*tmp1
  tmp6    = tmp2 + c

  du1     = 0.2 * tmp1
  du2     = 0.8 * tmp1
  du3     = tmp5
  du4     = tmp5
  du5     = 2*((tmp6 - 1.0)*tmp3 + tmp6*γ)*tmp1

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

function initial_condition_shock_bubble(x, t, equations::CompressibleEulerMulticomponentEquations2D)
  # bubble test case, see Agertz et al. https://arxiv.org/pdf/1904.00972
  # other reference: https://www.researchgate.net/profile/Pep_Mulet/publication/222675930_A_flux-split_algorithm_applied_to_conservative_models_for_multicomponent_compressible_flows/links/568da54508aeaa1481ae7af0.pdf
  # typical domain is rectangular, we change it to a square, as Trixi can only do squares

  # Region I 
  rho1_1  = 0.03
  rho2_1  = 1.225 * equations.rs1/equations.rs2 - 0.03
  v1_1    = 0.0
  v2_1    = 0.0
  p_1     = 101325

  # Region II 
  rho1_2  = 1.225-0.03
  rho2_2  = 0.03
  v1_2    = 0.0
  v2_2    = 0.0
  p_2     = 101325

  # Region III 
  rho1_3  = 1.6861 - 0.03
  rho2_3  = 0.03
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

# Not tested yet
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
* 
  # Calculate primitive variables.    !!! NOTE: EC flux does not transfer mass across an interface separating two different species
  rho1              = r > 0.5 ? 0.2*1.0 : 0.2*1.1691
  rho2              = r > 0.5 ? 0.8*1.0 : 0.8*1.1691
  rho               = rho1 + rho2
  v1                = r > 0.5 ? 0.0 : 0.1882 * cos_phi
  v2                = r > 0.5 ? 0.0 : 0.1882 * sin_phi
  p                 = r > 0.5 ? 1.0 : 1.245

  return prim2cons(SVector(rho1, rho2, v1, v2, p), equations)
end


# Not tested yet
"""
    initial_condition_khi(x, t, equations::CompressibleEulerMulticomponentEquations2D)

The classical Kelvin-Helmholtz instability based on
- https://rsaa.anu.edu.au/research/established-projects/fyris/2-d-kelvin-helmholtz-test
"""
function initial_condition_khi(x, t, equations::CompressibleEulerMulticomponentEquations2D)
  # https://rsaa.anu.edu.au/research/established-projects/fyris/2-d-kelvin-helmholtz-test
  # change discontinuity to tanh
  # typical resolution 128^2, 256^2
  # domain size is [-0.5,0.5]^2
  dens0 = 1.0 # outside density
  dens1 = 2.0 # inside density
  velx0 = -0.5 # outside velocity
  velx1 = 0.5 # inside velocity
  slope = 50 # used for tanh instead of discontinuous initial condition
  # pressure equilibrium
  p     = 2.5
  # density
  rho1  = 0.8 * (dens0 + (dens1-dens0) * 0.5*(1+(tanh(slope*(x[2]+0.25)) - (tanh(slope*(x[2]-0.25)) + 1))))
  rho2  = 0.2 * (dens1 + (dens0-dens1) * 0.5*(1+(tanh(slope*(x[2]+0.25)) - (tanh(slope*(x[2]-0.25)) + 1))))
  if iszero(t) # initial condition
    #  y velocity v2 is only white noise
    v2  = 0.01*(rand(Float64,1)[1]-0.5)
    #  x velocity is also augmented with noise
    v1  = velx0 + (velx1-velx0) * 0.5*(1+(tanh(slope*(x[2]+0.25)) - (tanh(slope*(x[2]-0.25)) + 1)))+0.01*(rand(Float64,1)[1]-0.5)
  else # background values to compute reference values for CI
    v2  = 0.0
    v1  = velx0 + (velx1-velx0) * 0.5*(1+(tanh(slope*(x[2]+0.25)) - (tanh(slope*(x[2]-0.25)) + 1)))
  end
  return prim2cons(SVector(rho1, rho2, v1, v2, p), equations)
end


# Calculate 1D flux for a single point
@inline function calcflux(u, orientation, equations::CompressibleEulerMulticomponentEquations2D)
  rho1, rho2, rho_v1, rho_v2, rho_e = u

  rho   = rho1 + rho2
  v1    = rho_v1/rho
  v2    = rho_v2/rho
  gamma = (rho1/rho*equations.cvs1*equations.gamma1 + rho2/rho*equations.cvs2*equations.gamma2)/(rho1/rho*equations.cvs1 + rho2/rho*equations.cvs2)
  p     = (gamma - 1) * (rho_e - 1/2 * rho * (v1^2 + v2^2))

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
  rho1_avg    = 0.5*(rho1_ll+rho1_rr)
  rho2_avg    = 0.5*(rho2_ll+rho2_rr)
  v1_avg      = 0.5 * (v1_ll + v1_rr)
  v2_avg      = 0.5 * (v2_ll + v2_rr)
  v1_square   = 0.5 * (v1_ll^2 + v1_rr^2)
  v2_square   = 0.5 * (v2_ll^2 + v2_rr^2)
  v_sum       = v1_avg + v2_avg

  # multicomponent specific values
  enth        = rho1_avg * equations.rs1 + rho2_avg * equations.rs2
  T_ll        = (rho_e_ll - 0.5 * rho_ll * (v1_ll^2 + v2_ll^2)) / (rho1_ll * equations.cvs1 + rho2_ll * equations.cvs2)
  T_rr        = (rho_e_rr - 0.5 * rho_rr * (v1_rr^2 + v2_rr^2)) / (rho1_rr * equations.cvs1 + rho2_rr * equations.cvs2)
  T           = 0.5 * (1.0/T_ll + 1.0/T_rr)
  T_log       = ln_mean(1.0/T_ll, 1.0/T_rr)

  gamma       = (rho1_ll/rho_ll*equations.cvs1*equations.gamma1 + rho2_ll/rho_ll*equations.cvs2*equations.gamma2)/(rho1_ll/rho_ll*equations.cvs1 + rho2_ll/rho_ll*equations.cvs2)
  p_ll        = (gamma - 1) * (rho_e_ll - 0.5 * (rho_v1_ll^2 + rho_v2_ll^2) / rho_ll)
  p_rr        = (gamma - 1) * (rho_e_rr - 0.5 * (rho_v1_rr^2 + rho_v2_rr^2) / rho_rr)

  # Calculate fluxes depending on orientation
  if orientation == 1
    f1 = rho1_mean * v1_avg
    f2 = rho2_mean * v1_avg
    f3 = (f1 + f2) * v1_avg + enth/T 
    f4 = (f1 + f2) * v2_avg
    f5 = (f1 * equations.cvs1 + f2 * equations.cvs2)/T_log - 0.5 * (v1_square + v2_square) * (f1 + f2) + v1_avg * f3 + v2_avg * f4
  else
    f1 = rho1_mean * v2_avg
    f2 = rho2_mean * v2_avg
    f3 = (f1 + f2) * v1_avg 
    f4 = (f1 + f2) * v2_avg + enth/T
    f5 = (f1 * equations.cvs1 + f2 * equations.cvs2)/T_log - 0.5 * (v1_square + v2_square) * (f1 + f2) + v1_avg * f3 + v2_avg * f4
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
  enth      = rho1_avg * equations.rs1 + rho2_avg * equations.rs2
  T_ll      = (rho_e_ll - 0.5 * rho_ll * (v1_ll^2 + v2_ll^2)) / (rho1_ll * equations.cvs1 + rho2_ll * equations.cvs2)
  T_rr      = (rho_e_rr - 0.5 * rho_rr * (v1_rr^2 + v2_rr^2)) / (rho1_rr * equations.cvs1 + rho2_rr * equations.cvs2)
  T         = 0.5 * (T_ll + T_rr)
  T_log     = ln_mean(1.0/T_ll, 1.0/T_rr)

  gamma = (rho1_avg/rho_avg*equations.cvs1*equations.gamma1 + rho2_avg/rho_avg*equations.cvs2*equations.gamma2)/(rho1_avg/rho_avg*equations.cvs1 + rho2_avg/rho_avg*equations.cvs2)

  p_ll = (gamma - 1) * (rho_e_ll - 0.5 * (rho_v1_ll^2 + rho_v2_ll^2) / rho_ll)
  p_rr = (gamma - 1) * (rho_e_rr - 0.5 * (rho_v1_rr^2 + rho_v2_rr^2) / rho_rr)

  # Calculate dissipation operator to be ES 
  R_x       = zeros(Float64, (5, 5))
  R_y       = zeros(Float64, (5, 5))
  #Lambda    = zeros(Float64, (5, 5))
  Lambda_x  = zeros(Float64, (5, 5))
  Lambda_y  = zeros(Float64, (5, 5))
  Tau       = zeros(Float64, (5, 5))
  W         = zeros(Float64, (5))
  Dv_x      = zeros(Float64, (5))
  Dv_y      = zeros(Float64, (5))

  # Help variables
  e1    = equations.cvs1 * T # check
  e2    = equations.cvs2 * T # check
  r     = (rho1_avg / rho_avg) * equations.rs1 + (rho2_avg / rho_avg) * equations.rs2 # check
  a     = sqrt(gamma * T * r) # check
  k_x   = 0.5 * (v_ll_sq + v_rr_sq) # Not Sure 
  k_y   = 0.5 * (v_ll_sq + v_rr_sq) # Not Sure
  h1    = e1 + equations.rs1 * T # check
  h2    = e2 + equations.rs2 * T # check
  h     = (rho1_avg / rho_avg) * h1 + (rho2_avg / rho_avg) * h2 # weighted sum correct?
  d1    = h1 - gamma * e1 # check
  d2    = h2 - gamma * e2 # check
  ht_x  = h + k_x # check
  ht_y  = h + k_y # check
  tauh  = sqrt(rho_avg/(gamma * r)) # check 

  # Matrices, 3e Spalte eventuell falsch bei R?
  # -------------------------------- #

  R_x[1,1]  = 1.0 # check
  R_x[1,2]  = 0.0 # check
  R_x[1,3]  = 0.0 # check
  R_x[1,4]  = rho1_avg / rho_avg # check
  R_x[1,5]  = R_x[1,4] # check

  R_x[2,1]  = 0.0 # check
  R_x[2,2]  = 1.0 # check
  R_x[2,3]  = 0.0 # check
  R_x[2,4]  = rho2_avg / rho_avg # check
  R_x[2,5]  = R_x[2,4] # check

  R_x[3,1]  = v1_avg # check
  R_x[3,2]  = v1_avg # check
  R_x[3,3]  = 0.0 
  R_x[3,4]  = v1_avg + a # check
  R_x[3,5]  = v1_avg - a # check

  R_x[4,1]  = v2_avg # check
  R_x[4,2]  = v2_avg # check
  R_x[4,3]  = a 
  R_x[4,4]  = v2_avg # check
  R_x[4,5]  = v2_avg # check

  R_x[5,1]  = k_x - (d1/(gamma - 1.0)) # check
  R_x[5,2]  = k_x - (d2/(gamma - 1.0)) # check
  R_x[5,3]  = a * v2_avg 
  R_x[5,4]  = ht_x + a * v1_avg # check
  R_x[5,5]  = ht_x - a * v1_avg # check

  # -------------------------------- #

  R_y[1,1]  = 1.0 # check
  R_y[1,2]  = 0.0 # check
  R_y[1,3]  = 0.0 # check
  R_y[1,4]  = rho1_avg / rho_avg # check
  R_y[1,5]  = R_y[1,4] # check

  R_y[2,1]  = 0.0 # check
  R_y[2,2]  = 1.0 # check
  R_y[2,3]  = 0.0 # check
  R_y[2,4]  = rho2_avg / rho_avg # check
  R_y[2,5]  = R_y[2,4] # check

  R_y[3,1]  = v1_avg # check
  R_y[3,2]  = v1_avg # check
  R_y[3,3]  = -1.0 * a 
  R_y[3,4]  = v1_avg # check
  R_y[3,5]  = v1_avg # check

  R_y[4,1]  = v2_avg # check
  R_y[4,2]  = v2_avg # check
  R_y[4,3]  = 0.0 
  R_y[4,4]  = v2_avg + a # check
  R_y[4,5]  = v2_avg - a # check

  R_y[5,1]  = k_y - (d1/(gamma - 1)) # check
  R_y[5,2]  = k_y - (d2/(gamma - 1)) # check
  R_y[5,3]  = -1.0 * a * v1_avg 
  R_y[5,4]  = ht_y + a * v2_avg # check
  R_y[5,5]  = ht_y - a * v2_avg # check

  # -------------------------------- #

  Lambda_x[1,1] = abs(v1_avg) # check
  Lambda_x[2,2] = abs(v1_avg) # check
  Lambda_x[3,3] = abs(v1_avg) # check
  Lambda_x[4,4] = abs(v1_avg + a) # check
  Lambda_x[5,5] = abs(v1_avg - a) # check

  # -------------------------------- #

  Lambda_y[1,1] = abs(v2_avg) # check
  Lambda_y[2,2] = abs(v2_avg) # check
  Lambda_y[3,3] = abs(v2_avg) # check
  Lambda_y[4,4] = abs(v2_avg + a) # check
  Lambda_y[5,5] = abs(v2_avg - a) # check

  # -------------------------------- #

  Tau[1,1]  = tauh * (-1.0 * sqrt((rho1_avg/rho_avg) * (rho2_avg/rho_avg)) * sqrt(gamma * (equations.rs2 / equations.rs1))) # check
  Tau[1,2]  = tauh * ((rho1_avg / rho_avg) * sqrt(gamma - 1.0)) # check
  Tau[1,3]  = 0.0 # check
  Tau[1,4]  = 0.0 # check
  Tau[1,5]  = 0.0 # check

  Tau[2,1]  = tauh * (sqrt((rho1_avg/rho_avg) * (rho2_avg/rho_avg)) * sqrt(gamma * (equations.rs1 / equations.rs2))) # check
  Tau[2,2]  = tauh * ((rho2_avg / rho_avg) * sqrt(gamma - 1.0)) # check
  Tau[2,3]  = 0.0 # check
  Tau[2,4]  = 0.0 # check
  Tau[2,5]  = 0.0 # check

  Tau[3,1]  = 0.0
  Tau[3,2]  = 0.0
  Tau[3,3]  = tauh / a# latest change
  Tau[3,4]  = 0.0
  Tau[3,5]  = 0.0

  Tau[4,1]  = 0.0 # check
  Tau[4,2]  = 0.0 # check
  Tau[4,3]  = 0.0 # check
  Tau[4,4]  = tauh / sqrt(2) # check
  Tau[4,5]  = 0.0 # check

  Tau[5,1]  = 0.0 # check
  Tau[5,2]  = 0.0 # check
  Tau[5,3]  = 0.0 # check
  Tau[5,4]  = 0.0 # check
  Tau[5,5]  = tauh / sqrt(2) # check

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
    f5 = ((f1 * equations.cvs1 + f2 * equations.cvs2)/T_log - 0.5 * (v1_square + v2_square) * (f1 + f2) + v1_avg * f3 + v2_avg * f4) - Dv_x[5]
  else
    f1 = (rho1_mean * v2_avg) - Dv_y[1]
    f2 = (rho2_mean * v2_avg) - Dv_y[2]
    f3 = ((f1 + f2) * v1_avg) - Dv_y[3] 
    f4 = ((f1 + f2) * v2_avg + enth * T) - Dv_y[4]
    f5 = ((f1 * equations.cvs1 + f2 * equations.cvs2)/T_log - 0.5 * (v1_square + v2_square) * (f1 + f2) + v1_avg * f3 + v2_avg * f4) - Dv_y[5]
  end

  return SVector(f1, f2, f3, f4, f5)
end


function flux_lax_friedrichs(u_ll, u_rr, orientation, equations::CompressibleEulerMulticomponentEquations2D)
  # Calculate primitive variables and speed of sound
  rho1_ll, rho2_ll, rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  rho1_rr, rho2_rr, rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr

  rho_ll = rho1_ll + rho2_ll
  rho_rr = rho1_rr + rho2_rr

  gamma =  (rho1_ll/rho_ll*equations.cvs1*equations.gamma1 + rho2_ll/rho_ll*equations.cvs2*equations.gamma2)/(rho1_ll/rho_ll*equations.cvs1 + rho2_ll/rho_ll*equations.cvs2)

  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v_mag_ll = sqrt(v1_ll^2 + v2_ll^2)
  p_ll = (gamma - 1) * (rho_e_ll - 1/2 * rho_ll * v_mag_ll^2)
  c_ll = sqrt(gamma * p_ll / rho_ll)
  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  v_mag_rr = sqrt(v1_rr^2 + v2_rr^2)
  p_rr = (gamma - 1) * (rho_e_rr - 1/2 * rho_rr * v_mag_rr^2)
  c_rr = sqrt(gamma * p_rr / rho_rr)

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
  rho1, rho2, rho_v1, rho_v2, rho_e = u

  rho   = rho1 + rho2
  v1    = rho_v1 / rho
  v2    = rho_v2 / rho

  gamma = (rho1/rho*equations.cvs1*equations.gamma1 + rho2/rho*equations.cvs2*equations.gamma2)/(rho1/rho*equations.cvs1 + rho2/rho*equations.cvs2)
  p     = (gamma - 1) * (rho_e - 1/2 * rho * (v1^2 + v2^2))
  c     = sqrt(gamma * p / rho)

  return abs(v1) + c, abs(v2) + c
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::CompressibleEulerMulticomponentEquations2D)
  rho1, rho2, rho_v1, rho_v2, rho_e = u

  rho   = rho1 + rho2
  v1    = rho_v1 / rho
  v2    = rho_v2 / rho
  gamma = (rho1/rho*equations.cvs1*equations.gamma1 + rho2/rho*equations.cvs2*equations.gamma2)/(rho1/rho*equations.cvs1 + rho2/rho*equations.cvs2)
  p     = (gamma - 1) * (rho_e - 0.5 * rho * (v1^2 + v2^2))

  return SVector(rho1, rho2, v1, v2, p)
end


# Convert conservative variables to entropy
@inline function cons2entropy(u, equations::CompressibleEulerMulticomponentEquations2D)
  rho1, rho2, rho_v1, rho_v2, rho_e = u

  rho       = rho1 + rho2
  v1        = rho_v1 / rho
  v2        = rho_v2 / rho
  v_square  = v1^2 + v2^2
  gamma     = (rho1/rho*equations.cvs1*equations.gamma1 + rho2/rho*equations.cvs2*equations.gamma2)/(rho1/rho*equations.cvs1 + rho2/rho*equations.cvs2)
  p         = (gamma - 1) * (rho_e - 0.5 * rho * v_square)
  s         = log(p) - gamma*log(rho)
  rho_p     = rho / p

  # Multicomponent stuff
  T         = (rho_e - 0.5 * rho * v_square) / (rho1 * equations.cvs1 + rho2 * equations.cvs2)
  s1        = equations.cvs1 * log(T) - equations.rs1 * log(rho1)
  s2        = equations.cvs2 * log(T) - equations.rs2 * log(rho2)

  # Entropy variables
  w1        = -s1 + equations.rs1 + equations.cvs1 - (v_square / (2*T)) # + e01/T (bec. e01 = 0 for compressible euler) (DONT confuse it with e1 which is e1 = cvs * T)
  w2        = -s2 + equations.rs2 + equations.cvs2 - (v_square / (2*T)) # + e02/T (bec. e02 = 0 for compressible euler)
  w3        = (v1)/T
  w4        = (v2)/T
  w5        = (-1.0)/T 

  return SVector(w1, w2, w3, w4, w5)
end


# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::CompressibleEulerMulticomponentEquations2D)
  rho1, rho2, v1, v2, p = prim
  rho     = rho1 + rho2

  gamma  = (rho1/rho*equations.cvs1*equations.gamma1 + rho2/rho*equations.cvs2*equations.gamma2)/(rho1/rho*equations.cvs1 + rho2/rho*equations.cvs2)
  rho_v1  = rho * v1
  rho_v2  = rho * v2
  rho_e   = p/(gamma-1) + 0.5 * (rho_v1 * v1 + rho_v2 * v2)

  return SVector(rho1, rho2, rho_v1, rho_v2, rho_e)
end


@inline function density(u, equations::CompressibleEulerMulticomponentEquations2D)
 rho = u[1] + u[2]
 return rho
end


@inline function pressure(u, equations::CompressibleEulerMulticomponentEquations2D)
 rho1, rho2, rho_v1, rho_v2, rho_e = u
 rho    = rho1 + rho2
 gamma  = (rho1/rho*equations.cvs1*equations.gamma1 + rho2/rho*equations.cvs2*equations.gamma2)/(rho1/rho*equations.cvs1 + rho2/rho*equations.cvs2)
 p      = (gamma - 1) * (rho_e - 0.5 * (rho_v1^2 + rho_v2^2) / rho)
 return p
end


@inline function density_pressure(u, equations::CompressibleEulerMulticomponentEquations2D)
 rho1, rho2, rho_v1, rho_v2, rho_e = u
 
 rho          = rho1 + rho2
 gamma        = (rho1/rho*equations.cvs1*equations.gamma1 + rho2/rho*equations.cvs2*equations.gamma2)/(rho1/rho*equations.cvs1 + rho2/rho*equations.cvs2)
 rho_times_p  = (gamma - 1) * (rho * rho_e - 0.5 * (rho_v1^2 + rho_v2^2))
 return rho_times_p
end


# Calculates the entropy flux in direction "orientation" and the entropy variables for a state cons
# NOTE: This method seems to work currently (b82534e) but is never used anywhere. Thus it is
# commented here until someone uses it or writes a test for it.
# @inline function cons2entropyvars_and_flux(gamma::Float64, cons, orientation::Int)
#   entropy = MVector{4, Float64}(undef)
#   v = (cons[2] / cons[1] , cons[3] / cons[1])
#   v_square= v[1]*v[1]+v[2]*v[2]
#   p = (gamma - 1) * (cons[4] - 1/2 * (cons[2] * v[1] + cons[3] * v[2]))
#   rho_p = cons[1] / p
#   # thermodynamic entropy
#   s = log(p) - gamma*log(cons[1])
#   # mathematical entropy
#   S = - s*cons[1]/(gamma-1)
#   # entropy variables
#   entropy[1] = (gamma - s)/(gamma-1) - 0.5*rho_p*v_square
#   entropy[2] = rho_p*v[1]
#   entropy[3] = rho_p*v[2]
#   entropy[4] = -rho_p
#   # entropy flux
#   entropy_flux = S*v[orientation]
#   return entropy, entropy_flux
# end


# Calculate thermodynamic entropy for a conservative state `cons`
@inline function entropy_thermodynamic(cons, equations::CompressibleEulerMulticomponentEquations2D)
  rho1, rho2, rho_v1, rho_v2, rho_e = cons
  rho = rho1 + rho2

  # Pressure
  gamma = (rho1/rho*equations.cvs1*equations.gamma1 + rho2/rho*equations.cvs2*equations.gamma2)/(rho1/rho*equations.cvs1 + rho2/rho*equations.cvs2)
  p = (gamma - 1) * (rho_e - 0.5 * (rho_v1^2 + rho_v2^2) / rho)

  # Thermodynamic entropy
  T   = (rho_e - 0.5 * rho * v_sq) / (rho1 * equations.cvs1 + rho2 * equations.cvs2)
  s1  = equations.cvs1 * log(T) - equations.rs1 * log(rho1)
  s2  = equations.cvs2 * log(T) - equations.rs2 * log(rho2) 

  s = rho1 * s1 + rho2 * s2

  return s
end



# Calculate mathematical entropy for a conservative state `cons`
@inline function entropy_math(cons, equations::CompressibleEulerMulticomponentEquations2D)
  # Mathematical entropy
  S = -entropy_thermodynamic(cons, equations)

  return S
end


# Default entropy is the mathematical entropy
@inline entropy(cons, equations::CompressibleEulerMulticomponentEquations2D) = entropy_math(cons, equations)


# Calculate total energy for a conservative state `cons`
@inline energy_total(cons, ::CompressibleEulerMulticomponentEquations2D) = cons[5]


# Calculate kinetic energy for a conservative state `cons`
@inline function energy_kinetic(u, equations::CompressibleEulerMulticomponentEquations2D)
  rho1, rho2, rho_v1, rho_v2, rho_e = u
  rho = rho1 + rho2
  return (rho_v1^2 + rho_v2^2) / (2 * rho)
end


# Calculate internal energy for a conservative state `cons`
@inline function energy_internal(cons, equations::CompressibleEulerMulticomponentEquations2D)
  return energy_total(cons, equations) - energy_kinetic(cons, equations)
end
