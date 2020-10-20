
@doc raw"""
    CompressibleEulerMulticomponentEquations2D

The compressible Euler Two!-Component equations for an ideal gas in two space dimensions.
"""
# Changed to mutable struct, this way it is possible to overwrite gamma in the initial conditions (still expensive, should be done in function CompressibleEulerMulticomponentEquations2D())
struct CompressibleEulerMulticomponentEquations2D <: AbstractCompressibleEulerMulticomponentEquations{2, 5}

  # Just for the moment:
  gamma1::Float64
  gamma2::Float64
  rs1::Float64
  rs2::Float64
  csv1::Float64
  csv2::Float64
  cps1::Float64
  cps2::Float64
  #gamma::Float64

  # Thats the next step:
  #gammas::SVector{2, Float64}
  #rs::SVector{2, Float64}
  #csv::SVector{2, Float64}
  #cpv::SVector{2, Float64}
  # gamma::Float64 # Maybe mutable still better?
end


function CompressibleEulerMulticomponentEquations2D()
  # Placing dummy variables, should be replaced by values from initial condition!!

  # Set ratio of specific heat of the gas per species 
  # (It holds: gamma = cp/cv) (Can be used for consistency check!)
  gamma1  = parameter("gamma1", 1.4)
  gamma2  = parameter("gamma2", 1.4)
  
  # Set specific gas constant with respect to the molar mass per species 
  # (It holds: rs = R/M, with R = molar gas constant, M = molar mass) 
  # (For a calorically and thermally perfect gas Mayer's relation holds: rs = cp - cv) (Can be used for consistency check!)
  rs1     = parameter("rs1", 0.4)
  rs2     = parameter("rs2", 0.4)

  # Set specific heat for a constant volume per species
  cvs1    = parameter("cvs1", 1.0)
  cvs2    = parameter("cvs2", 1.0)

  # Set specific heat for a constant pressure per species
  cps1    = parameter("cps1", 1.4)
  cps2    = parameter("cps2", 1.4)

  #gamma   = parameter("gamma", 1.4)

  CompressibleEulerMulticomponentEquations2D(gamma1, gamma2, rs1, rs2, cvs1, cvs2, cps1, cps2)#, gamma)
end


get_name(::CompressibleEulerMulticomponentEquations2D) = "CompressibleEulerMulticomponentEquations2D"
# Das muss irgendwie dynamisch angepasst werden, abhängig von Anzahl von Komponenten
varnames_cons(::CompressibleEulerMulticomponentEquations2D) = @SVector ["rho1", "rho2", "rho_v1", "rho_v2", "rho_e"]
varnames_prim(::CompressibleEulerMulticomponentEquations2D) = @SVector ["rho1", "rho2", "v1", "v2", "p"]


function initial_conditions_constant(x, t, equation::CompressibleEulerMulticomponentEquations2D)
  rho1 = 1.0
  rho2 = 1.0
  rho_v1 = 0.1
  rho_v2 = -0.2
  rho_e = 10.0
  return @SVector [rho1, rho2, rho_v1, rho_v2, rho_e]
end

function initial_conditions_convergence_test(x, t, equation::CompressibleEulerMulticomponentEquations2D)
  c = 2
  A = 0.1
  L = 2
  f = 1/L
  ω = 2 * pi * f
  ini = c + A * sin(ω * (x[1] + x[2] - t))

  vel1    = 1.0
  vel2    = 1.0

  rho     = ini
  rho1    = 0.5 * ini
  rho2    = 0.5 * ini
  rho_v1  = (rho1+rho2) * vel1
  rho_v2  = (rho1+rho2) * vel2

  rho_e = rho^2

  return @SVector [rho1, rho2, rho_v1, rho_v2, rho_e]
end


function initial_conditions_convergence_test_unsymmetrical(x, t, equation::CompressibleEulerMulticomponentEquations2D)
  c = 2
  A = 0.1
  L = 2
  f = 1/L
  ω = 2 * pi * f
  ini = c + A * sin(ω * (x[1] + x[2] - t))

  vel1    = 1.0
  vel2    = 1.0

  rho     = ini
  rho1    = 0.2 * ini
  rho2    = 0.8 * ini
  rho_v1  = (rho1+rho2) * vel1
  rho_v2  = (rho1+rho2) * vel2

  rho_e = rho^2

  return @SVector [rho1, rho2, rho_v1, rho_v2, rho_e]
end


function initial_conditions_weak_blast_wave(x, t, equation::CompressibleEulerMulticomponentEquations2D)
  # From Hennemann & Gassner JCP paper 2020 (Sec. 6.3)
  # Set up polar coordinates
  inicenter = SVector(0.0, 0.0)
  x_norm = x[1] - inicenter[1]
  y_norm = x[2] - inicenter[2]
  r = sqrt(x_norm^2 + y_norm^2)
  phi = atan(y_norm, x_norm)
  sin_phi, cos_phi = sincos(phi)

  # Calculate primitive variables
  # Alternative: rho1 = r > 0.5 ?  1.0 : 0.0 and rho2 = r > 0.5 ? 0.0 : 1.1691
  rho1 = r > 0.5 ? 1.0 : 1.1691   #Version 1
  rho2 = r > 0.5 ? 1.0 : 1.1691   #Version 1
  #rho1  = r > 0.5 ? 1.0 : 0.0     #Version 2
  #rho2  = r > 0.5 ? 0.0 : 1.1691  #Version 2
  #rho1  = r > 0.5 ? 1.0 : 1.1691  #Version 3
  #rho2  = 0.0                     #Version 3
  rho   = rho1 + rho2
  v1    = r > 0.5 ? 0.0 : 0.1882 * cos_phi
  v2    = r > 0.5 ? 0.0 : 0.1882 * sin_phi
  p     = r > 0.5 ? 1.0 : 1.245

  return prim2cons(SVector(rho1, rho2, v1, v2, p), equation)
end


function initial_conditions_kelvin_helmholtz(x, t, equation::CompressibleEulerMulticomponentEquations2D)
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
  #  y velocity v2 is only white noise
  v2  = 0.01*(rand(Float64,1)[1]-0.5)
  # density
  #rho1  = 0.5 * (dens0 + (dens1-dens0) * 0.5 *(1+(tanh(slope*(x[2]+0.25)) - (tanh(slope*(x[2]-0.25)) + 1))))
  #rho2  = 0.5 * (dens1 + (dens0-dens1) * 0.5 *(1+(tanh(slope*(x[2]+0.25)) - (tanh(slope*(x[2]-0.25)) + 1))))
  rho1  = 0.5 * (dens0 + (dens1-dens0) * 0.5 *(1+(tanh(slope*(x[2]+0.25)) - (tanh(slope*(x[2]-0.25)) + 1))))
  rho2  = 0.5 * (dens0 + (dens1-dens0) * 0.5 *(1+(tanh(slope*(x[2]+0.25)) - (tanh(slope*(x[2]-0.25)) + 1))))
  #  x velocity is also augmented with noise
  v1    = velx0 + (velx1-velx0) * 0.5*(1+(tanh(slope*(x[2]+0.25)) - (tanh(slope*(x[2]-0.25)) + 1)))+0.01*(rand(Float64,1)[1]-0.5)
  return prim2cons(SVector(rho1, rho2, v1, v2, p), equation)
end


# Apply source terms
function source_terms_convergence_test(ut, u, x, element_id, t, n_nodes, equation::CompressibleEulerMulticomponentEquations2D)
  # Same settings as in `initial_conditions`
  c = 2
  A = 0.1
  L = 2
  f = 1/L
  ω = 2 * pi * f
  
  rho1 = u[1]
  rho2 = u[2]
  rho = rho1+rho2

  γ = (rho1/rho*equation.csv1*equation.gamma1 + rho2/rho*equation.csv2*equation.gamma2)/(rho1/rho*equation.csv1 + rho2/rho*equation.csv2)

  for j in 1:n_nodes, i in 1:n_nodes
    x1 = x[1, i, j, element_id]
    x2 = x[2, i, j, element_id]
    si, co = sincos((x1 + x2 - t)*ω)
    tmp1 = co * A * ω
    tmp2 = si * A
    tmp3 = γ - 1
    tmp4 = (2*c - 1)*tmp3
    tmp5 = (2*tmp2*γ - 2*tmp2 + tmp4 + 1)*tmp1
    tmp6 = tmp2 + c

    ut[1, i, j, element_id] += 0.5 * tmp1 
    ut[2, i, j, element_id] += 0.5 * tmp1
    ut[3, i, j, element_id] += tmp5
    ut[4, i, j, element_id] += tmp5
    ut[5, i, j, element_id] += 2*((tmp6 - 1)*tmp3 + tmp6*γ)*tmp1

    # Original terms (without performanc enhancements)
    # ut[1, i, j, element_id] += cos((x1 + x2 - t)*ω)*A*ω
    # ut[2, i, j, element_id] += (2*sin((x1 + x2 - t)*ω)*A*γ - 2*sin((x1 + x2 - t)*ω)*A +
    #                             2*c*γ - 2*c - γ + 2)*cos((x1 + x2 - t)*ω)*A*ω
    # ut[3, i, j, element_id] += (2*sin((x1 + x2 - t)*ω)*A*γ - 2*sin((x1 + x2 - t)*ω)*A +
    #                             2*c*γ - 2*c - γ + 2)*cos((x1 + x2 - t)*ω)*A*ω
    # ut[4, i, j, element_id] += 2*((c - 1 + sin((x1 + x2 - t)*ω)*A)*(γ - 1) +
    #                               (sin((x1 + x2 - t)*ω)*A + c)*γ)*cos((x1 + x2 - t)*ω)*A*ω
  end

  return nothing
end


function source_terms_convergence_test_unsymmetrical(ut, u, x, element_id, t, n_nodes, equation::CompressibleEulerMulticomponentEquations2D)
  # Same settings as in `initial_conditions`
  c = 2
  A = 0.1
  L = 2
  f = 1/L
  ω = 2 * pi * f

  rho1 = u[1]
  rho2 = u[2]
  rho = rho1+rho2

  γ = (rho1/rho*equation.csv1*equation.gamma1 + rho2/rho*equation.csv2*equation.gamma2)/(rho1/rho*equation.csv1 + rho2/rho*equation.csv2)

  for j in 1:n_nodes, i in 1:n_nodes
    x1 = x[1, i, j, element_id]
    x2 = x[2, i, j, element_id]
    si, co = sincos((x1 + x2 - t)*ω)
    tmp1 = co * A * ω
    tmp2 = si * A
    tmp3 = γ - 1
    tmp4 = (2*c - 1)*tmp3
    tmp5 = (2*tmp2*γ - 2*tmp2 + tmp4 + 1)*tmp1
    tmp6 = tmp2 + c

    ut[1, i, j, element_id] += 0.2 * tmp1 
    ut[2, i, j, element_id] += 0.8 * tmp1
    ut[3, i, j, element_id] += tmp5
    ut[4, i, j, element_id] += tmp5
    ut[5, i, j, element_id] += 2*((tmp6 - 1)*tmp3 + tmp6*γ)*tmp1

    # Original terms (without performanc enhancements)
    # ut[1, i, j, element_id] += cos((x1 + x2 - t)*ω)*A*ω
    # ut[2, i, j, element_id] += (2*sin((x1 + x2 - t)*ω)*A*γ - 2*sin((x1 + x2 - t)*ω)*A +
    #                             2*c*γ - 2*c - γ + 2)*cos((x1 + x2 - t)*ω)*A*ω
    # ut[3, i, j, element_id] += (2*sin((x1 + x2 - t)*ω)*A*γ - 2*sin((x1 + x2 - t)*ω)*A +
    #                             2*c*γ - 2*c - γ + 2)*cos((x1 + x2 - t)*ω)*A*ω
    # ut[4, i, j, element_id] += 2*((c - 1 + sin((x1 + x2 - t)*ω)*A)*(γ - 1) +
    #                               (sin((x1 + x2 - t)*ω)*A + c)*γ)*cos((x1 + x2 - t)*ω)*A*ω
  end

  return nothing
end


# Calculate 1D flux for a single point
@inline function calcflux(u, orientation, equation::CompressibleEulerMulticomponentEquations2D)
  rho1, rho2, rho_v1, rho_v2, rho_e = u

  rho = rho1 + rho2
  
  # Has to be changed if its not possible to overwrite equation.gamma
  gamma = (rho1/rho*equation.csv1*equation.gamma1 + rho2/rho*equation.csv2*equation.gamma2)/(rho1/rho*equation.csv1 + rho2/rho*equation.csv2)
  p = (gamma - 1) * (rho_e - 0.5 * (rho_v1^2 + rho_v2^2) / rho)

  if orientation == 1
    f1 = rho1   * (rho_v1 / rho)
    f2 = rho2   * (rho_v1 / rho)
    f3 = rho_v1 * (rho_v1 / rho) + p
    f4 = rho_v2 * (rho_v1 / rho)
    f5 = (rho_v1 / rho) * (rho_e + p)
  else
    f1 = rho1   * (rho_v2 / rho)
    f2 = rho2   * (rho_v2 / rho)
    f3 = rho_v1 * (rho_v2 / rho)
    f4 = rho_v2 * (rho_v2 / rho) + p
    f5 = (rho_v2 / rho) * (rho_e + p)
  end
  return SVector(f1, f2, f3, f4, f5)
end


# Derived from paper
@inline function flux_chandrashekar(u_ll, u_rr, orientation, equation::CompressibleEulerMulticomponentEquations2D)
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
  rho1_avg  = 0.5*(rho1_ll+rho1_rr)
  rho2_avg  = 0.5*(rho2_ll+rho2_rr)
  v1_avg    = 0.5 * (v1_ll + v1_rr)
  v2_avg    = 0.5 * (v2_ll + v2_rr)
  v1_square = 0.5 * (v1_ll^2 + v1_rr^2)
  v2_square = 0.5 * (v2_ll^2 + v2_rr^2)
  v_sum     = v1_avg + v2_avg

  # multicomponent specific values
  enth      = rho1_avg * equation.rs1 + rho2_avg * equation.rs2
  T_ll      = (rho_e_ll - 0.5 * rho_ll * (v1_ll^2 + v2_ll^2)) / (rho1_ll * equation.csv1 + rho2_ll * equation.csv2)
  T_rr      = (rho_e_rr - 0.5 * rho_rr * (v1_rr^2 + v2_rr^2)) / (rho1_rr * equation.csv1 + rho2_rr * equation.csv2)
  T         = 0.5 * (1.0/T_ll + 1.0/T_rr)
  T_log     = ln_mean(1.0/T_ll, 1.0/T_rr)
  
  gamma = (rho1_ll/rho_ll*equation.csv1*equation.gamma1 + rho2_ll/rho_ll*equation.csv2*equation.gamma2)/(rho1_ll/rho_ll*equation.csv1 + rho2_ll/rho_ll*equation.csv2)

  p_ll =  (gamma - 1) * (rho_e_ll - 1/2 * rho_ll * (v1_ll^2 + v2_ll^2))
  p_rr =  (gamma - 1) * (rho_e_rr - 1/2 * rho_rr * (v1_rr^2 + v2_rr^2))
  beta_ll = 0.5*rho_ll/p_ll
  beta_rr = 0.5*rho_rr/p_rr
  specific_kin_ll = 0.5*(v1_ll^2 + v2_ll^2)
  specific_kin_rr = 0.5*(v1_rr^2 + v2_rr^2)

  # Calculate fluxes depending on orientation
  if orientation == 1
    f1 = rho1_mean * v1_avg
    f2 = rho2_mean * v1_avg
    f3 = (f1 + f2) * v1_avg + enth/T 
    f4 = (f1 + f2) * v2_avg
    f5 = (f1 * equation.csv1 + f2 * equation.csv2)/T_log - 0.5 * (v1_square + v2_square) * (f1 + f2) + v1_avg * f3 + v2_avg * f4
  else
    f1 = rho1_mean * v2_avg
    f2 = rho2_mean * v2_avg
    f3 = (f1 + f2) * v1_avg 
    f4 = (f1 + f2) * v2_avg + enth/T
    f5 = (f1 * equation.csv1 + f2 * equation.csv2)/T_log - 0.5 * (v1_square + v2_square) * (f1 + f2) + v1_avg * f3 + v2_avg * f4
  end
  
  return SVector(f1, f2, f3, f4, f5)
end



function flux_lax_friedrichs(u_ll, u_rr, orientation, equation::CompressibleEulerMulticomponentEquations2D)
  # Calculate primitive variables and speed of sound
  rho1_ll, rho2_ll, rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  rho1_rr, rho2_rr, rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr

  rho_ll = rho1_ll + rho2_ll
  rho_rr = rho1_rr + rho2_rr

  gamma = (rho1_ll/rho_ll*equation.csv1*equation.gamma1 + rho2_ll/rho_ll*equation.csv2*equation.gamma2)/(rho1_ll/rho_ll*equation.csv1 + rho2_ll/rho_ll*equation.csv2)

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
  f_ll = calcflux(u_ll, orientation, equation)
  f_rr = calcflux(u_rr, orientation, equation)

  λ_max = max(v_mag_ll, v_mag_rr) + max(c_ll, c_rr)
  f1 = 1/2 * (f_ll[1] + f_rr[1]) - 1/2 * λ_max * (rho1_rr   - rho1_ll)
  f2 = 1/2 * (f_ll[2] + f_rr[2]) - 1/2 * λ_max * (rho2_rr   - rho2_ll)
  f3 = 1/2 * (f_ll[3] + f_rr[3]) - 1/2 * λ_max * (rho_v1_rr - rho_v1_ll)
  f4 = 1/2 * (f_ll[4] + f_rr[4]) - 1/2 * λ_max * (rho_v2_rr - rho_v2_ll)
  f5 = 1/2 * (f_ll[5] + f_rr[5]) - 1/2 * λ_max * (rho_e_rr  - rho_e_ll)

  return SVector(f1, f2, f3, f4, f5)
end


# Determine maximum stable time step based on polynomial degree and CFL number
function calc_max_dt(u, element_id, invjacobian, cfl,
                     equation::CompressibleEulerMulticomponentEquations2D, dg)
  λ_max = 0.0
  for j in 1:nnodes(dg), i in 1:nnodes(dg)
    rho1, rho2, rho_v1, rho_v2, rho_e = get_node_vars(u, dg, i, j, element_id)

    rho = rho1 + rho2

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v_mag = sqrt(v1^2 + v2^2)
    # Again, has to be changed in case that equation.gamma doesnt work
    gamma = (rho1/rho*equation.csv1*equation.gamma1 + rho2/rho*equation.csv2*equation.gamma2)/(rho1/rho*equation.csv1 + rho2/rho*equation.csv2)
    p = (gamma - 1) * (rho_e - 0.5 * (rho_v1^2 + rho_v2^2) / rho)

    c = sqrt(gamma * p / rho)
    λ_max = max(λ_max, v_mag + c)
  end

  dt = cfl * 2 / (nnodes(dg) * invjacobian * λ_max)

  return dt
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equation::CompressibleEulerMulticomponentEquations2D)
  rho1, rho2, rho_v1, rho_v2, rho_e = u
  rho = rho1 + rho2
  
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  
  gamma = (rho1/rho*equation.csv1*equation.gamma1 + rho2/rho*equation.csv2*equation.gamma2)/(rho1/rho*equation.csv1 + rho2/rho*equation.csv2)
  p = (gamma - 1) * (rho_e - 0.5 * (rho_v1^2 + rho_v2^2) / rho)

  return SVector(rho1, rho2, v1, v2, p)
end


# Convert conservative variables to primitive
@inline function gamma(u, equation::CompressibleEulerMulticomponentEquations2D)
  rho1, rho2, rho_v1, rho_v2, rho_e = u
  rho = rho1 + rho2
 
  gamma = (rho1/rho*equation.csv1*equation.gamma1 + rho2/rho*equation.csv2*equation.gamma2)/(rho1/rho*equation.csv1 + rho2/rho*equation.csv2)

  return gamma
end


# Convert conservative variables to entropy
@inline function cons2entropy(u, equation::CompressibleEulerMulticomponentEquations2D)
  rho1, rho2, rho_v1, rho_v2, rho_e = u
  rho = rho1 + rho2

  v1    = rho_v1 / rho
  v2    = rho_v2 / rho
  v_sq  = v1^2 + v2^2
  
  gamma = (rho1/rho*equation.csv1*equation.gamma1 + rho2/rho*equation.csv2*equation.gamma2)/(rho1/rho*equation.csv1 + rho2/rho*equation.csv2)
  p = (gamma - 1) * (rho_e - 0.5 * (rho_v1^2 + rho_v2^2) / rho)

  # Multicomponent stuff
  T     = (rho_e - 0.5 * rho * v_sq) / (rho1 * equation.csv1 + rho2 * equation.csv2)
  s1    = equation.csv1 * log(T) - equation.rs1 * log(rho1)
  s2    = equation.csv2 * log(T) - equation.rs2 * log(rho2)

  # Entropy variables
  w1    = -s1 + equation.rs1 + equation.csv1 - (v_sq / (2*T)) # + e01/T (bec. e01 = 0 for compressible euler)
  w2    = -s2 + equation.rs2 + equation.csv2 - (v_sq / (2*T)) # + e02/T (bec. e02 = 0 for compressible euler)
  w3    = v1/T
  w4    = v2/T
  w5    = (-1.0)/T 

  #s = log(p) - equation.gamma*log(rho)
  #rho_p = rho / p

  # Just dummy input!
  #w1 = (equation.gamma - s) / (equation.gamma-1) - 0.5 * rho_p * v_square
  #w2 = w1
  #w3 = rho_p * v1
  #w4 = rho_p * v2
  #w5 = -rho_p

  return SVector(w1, w2, w3, w4, w5)
end


# Calculate thermodynamic entropy for a conservative state `cons`
@inline function entropy_thermodynamic(cons, equation::CompressibleEulerMulticomponentEquations2D)
  rho1, rho2, rho_v1, rho_v2, rho_e = cons
  rho = rho1 + rho2

  # Pressure
  gamma = (rho1/rho*equation.csv1*equation.gamma1 + rho2/rho*equation.csv2*equation.gamma2)/(rho1/rho*equation.csv1 + rho2/rho*equation.csv2)
  p = (gamma - 1) * (rho_e - 0.5 * (rho_v1^2 + rho_v2^2) / rho)

  # Thermodynamic entropy
  T   = (rho_e - 0.5 * rho * v_sq) / (rho1 * equation.csv1 + rho2 * equation.csv2)
  s1  = equation.csv1 * log(T) - equation.rs1 * log(rho1)
  s2  = equation.csv2 * log(T) - equation.rs2 * log(rho2) 

  s = rho1 * s1 + rho2 * s2

  return s
end


# Calculate mathematical entropy for a conservative state `cons`
@inline function entropy_math(cons, equation::CompressibleEulerMulticomponentEquations2D)
  # Mathematical entropy
  S = -entropy_thermodynamic(cons, equation)

  return S
end


# Default entropy is the mathematical entropy
@inline entropy(cons, equation::CompressibleEulerMulticomponentEquations2D) = entropy_math(cons, equation)


# Convert primitive to conservative variables
@inline function prim2cons(prim, equation::CompressibleEulerMulticomponentEquations2D)
  rho1, rho2, v1, v2, p = prim
  rho = rho1 + rho2

  gamma = (rho1/rho*equation.csv1*equation.gamma1 + rho2/rho*equation.csv2*equation.gamma2)/(rho1/rho*equation.csv1 + rho2/rho*equation.csv2)
  rho_v1 = rho * v1
  rho_v2 = rho * v2
  rho_e  = p/(gamma-1) + 0.5 * (rho_v1 * v1 + rho_v2 * v2)

  return SVector(rho1, rho2, rho_v1, rho_v2, rho_e)
end


@inline function density(u, equation::CompressibleEulerMulticomponentEquations2D)
 rho = u[1] + u[2]
 #rho = 5.0
 return rho
end


@inline function pressure(u, equation::CompressibleEulerMulticomponentEquations2D)
 rho1, rho2, rho_v1, rho_v2, rho_e = u
 rho = rho1 + rho2

 gamma = (rho1/rho*equation.csv1*equation.gamma1 + rho2/rho*equation.csv2*equation.gamma2)/(rho1/rho*equation.csv1 + rho2/rho*equation.csv2)
 p = (gamma - 1) * (rho_e - 0.5 * (rho_v1^2 + rho_v2^2) / rho)

 return p
end


@inline function density_pressure(u, equation::CompressibleEulerMulticomponentEquations2D)
 rho1, rho2, rho_v1, rho_v2, rho_e = u
 rho = rho1 + rho2

 gamma = (rho1/rho*equation.csv1*equation.gamma1 + rho2/rho*equation.csv2*equation.gamma2)/(rho1/rho*equation.csv1 + rho2/rho*equation.csv2)
 rho_times_p = (gamma - 1) * (rho * rho_e - 0.5 * (rho_v1^2 + rho_v2^2))

 return rho_times_p
end

