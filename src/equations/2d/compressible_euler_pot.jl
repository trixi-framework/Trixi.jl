@doc raw"""
    CompressibleMoistPotTempEulerEquations2D

The compressible Euler equations for a  gas with dry air and water vapor and a condensed phase 
in two space dimensions.
"""
struct CompressibleMoistPotTempEulerEquations2D <: AbstractCompressibleEulerEquations{2, 6}
  c_pd::Float64
  c_vd::Float64
  R_d::Float64
  c_pv::Float64
  c_vv::Float64
  R_v::Float64
  c_pl::Float64
  kappa::Float64
  gamma::Float64
  _grav::Float64
  p0::Float64
  L00::Float64
  cS::Float64
  delta_z::Float64
  z::Array{Float64,1}
  Val::Array{Float64,2}
end

function CompressibleMoistPotTempEulerEquations2D()
  c_pd = parameter("c_pd",1004)
  c_vd = parameter("c_vd",717)
  c_pv = parameter("c_pv",1885)
  c_vv = parameter("c_vv",1424)
  c_pl = parameter("c_pl",4186)
  R_d = parameter("R_d",c_pd-c_vd)
  R_v = parameter("R_v",c_pv-c_vv)
  kappa = parameter("kappa",R_d/c_pd)
  gamma = parameter("gamma", c_pd/c_vd)
  _grav = parameter("_grav",9.81)
  p0 = parameter("p0",1.e5)
  L00 = parameter("L00",2.5000e6 + (c_pl - c_pv) * 273.15)
  cS = parameter("cS",360.e0)

  n=1000
  z=zeros(n+1)
  Val=zeros(n+1,4)
  r_t0 = 2.e-2
  theta_e0 = 320
  delta_z = 10

  function ResMoisture(z, y, yPrime)

    p = y[1]
    rho = y[2]
    T = y[3]
    r_t = y[4]
    r_v = y[5]
    rho_qv = y[6]
    theta_e = y[7]
    pPrime = yPrime[1]
    F=zeros(7,1)

    rho_d = rho / (1 + r_t)
    p_d = R_d * rho_d * T
    T_C = T - 273.15
    p_vs = 611.2 * exp(17.62 * T_C / (243.12 + T_C))
    L = L00 - (c_pl - c_pv) * T
    F[1] = pPrime + _grav * rho
    F[2] = p - (R_d * rho_d + R_v * rho_qv) * T
    F[3] = theta_e - T * (p_d / p0)^(-R_d/
         (c_pd + c_pl * r_t)) * exp(L * r_v / ((c_pd + c_pl * r_t) * T))
    F[4] = r_t - r_t0
    F[5] = rho_qv-rho_d * r_v
    F[6] = theta_e - theta_e0
    a = p_vs / (R_v * T) - rho_qv
    b = rho - rho_qv - rho_d
    F[7]=a+b-sqrt(a*a+b*b)
    return F
  end
  function SetImplEuler(z,delta_z,y0)
    function ImplEuler(y)
      return ResMoisture(z,y,(y-y0)/delta_z)
    end
  end

  y=zeros(7)
  p=1.e5
  rho = 1.4
  r_t = r_t0
  r_v = r_t0
  rho_qv = rho * r_v
  theta_e = theta_e0
  T = theta_e

  yPrime = zeros(7)
  y0 = zeros(7)
  y0[1] = p
  y0[2] = rho
  y0[3] = T
  y0[4] = r_t
  y0[5] = r_v
  y0[6] = rho_qv
  y0[7] = theta_e


  z0 = 0.0
  delta_z = 0.01
  y=deepcopy(y0);
  F = SetImplEuler(z0,delta_z,y0)
  res = nlsolve(F,y0)
  p = res.zero[1]
  rho = res.zero[2]
  T = res.zero[3]
  r_t = res.zero[4]
  r_v = res.zero[5]
  rho_qv = res.zero[6]
  theta_e = res.zero[7]
  rho_d = rho / (1 + r_t)
  rho_qc = rho - rho_d - rho_qv
  kappa_M=(R_d * rho_d + R_v * rho_qv) / (c_pd * rho_d + c_pv * rho_qv + c_pl * rho_qc)
  rho_theta = rho * (p0 / p)^kappa_M * T * (1 + (R_v / R_d) *r_v) / (1 + r_t) 
  Val[1,1] = rho
  Val[1,2] = rho_theta
  Val[1,3] = rho_qv
  Val[1,4] = rho_qc
  delta_z = 10.0
  z[1] = 0
  Val[1,1] = res.zero[2]
  for i = 1 : n
    y0 = deepcopy(res.zero)
    F = SetImplEuler(z,delta_z,y0)
    res = nlsolve(F,y0)
    z[i+1] = z[i] + delta_z
    p = res.zero[1]
    rho = res.zero[2]
    T = res.zero[3]
    r_t = res.zero[4]
    r_v = res.zero[5]
    rho_qv = res.zero[6]
    theta_e = res.zero[7]
    rho_d = rho / (1 + r_t)
    rho_qc = rho - rho_d - rho_qv
    kappa_M=(R_d * rho_d + R_v * rho_qv) / (c_pd * rho_d + c_pv * rho_qv + c_pl * rho_qc)
    rho_theta = rho * (p0 / p)^kappa_M * T * (1 + (R_v / R_d) *r_v) / (1 + r_t) 
    Val[i+1,1] = rho
    Val[i+1,2] = rho_theta
    Val[i+1,3] = rho_qv
    Val[i+1,4] = rho_qc
  end
  CompressibleMoistPotTempEulerEquations2D(c_pd,c_vd,R_d,c_pv,c_vv,R_v,c_pl,kappa,gamma,_grav,p0,L00,cS,delta_z,z,Val)
end


get_name(::CompressibleMoistPotTempEulerEquations2D) = "CompressibleMoistPotTempEulerEquations2D"
varnames_pot(::CompressibleMoistPotTempEulerEquations2D) = @SVector ["rho", "v1", "v2", "theta", "qv", "qc"]
varnames_cons(::CompressibleMoistPotTempEulerEquations2D) = @SVector ["rho", "rho_v1", "rho_v2", "rho_theta", "rho_qv", "rho_qc"]
varnames_prim(::CompressibleMoistPotTempEulerEquations2D) = @SVector ["rho", "v1", "v2", "p", "qv", "qc"]


"""
Dry warm bubble test from paper:
Wicker, L. J., and W. C. Skamarock, 1998: A time-splitting scheme
for the elastic equations incorporating second-order Runge–Kutta
time differencing. Mon. Wea. Rev., 126, 1992–1999.
"""

function initial_conditions_warm_bubble(x, t, equation::CompressibleMoistPotTempEulerEquations2D)

  xc = 0
  zc = 2000
  r = sqrt((x[1] - xc)^2 + (x[2] - zc)^2)
  rc = 2000
  theta_ref = 300
  qv_ref = 0
  qc_ref = 0
  Δtheta = 0
  Δqv = 0
  Δqc = 0

  if r <= rc
     Δtheta = 2 * cospi(0.5*r/rc)^2
  end

  #Perturbed state:
  theta = theta_ref + Δtheta # potential temperature
  π_exner = 1 - equation._grav / (equation.c_pd * theta) * x[2] # exner pressure
  rho = equation.p0 / (equation.R_d * theta) * (π_exner)^(equation.c_vd / equation.R_d) # density

  v1 = 20
  v2 = 0
  qv = 0
  qc = 0
  rho_v1 = rho * v1
  rho_v2 = rho * v2
  rho_theta = rho * theta
  rho_qv = rho * qv
  rho_qc = rho * qc
  return @SVector [rho, rho_v1, rho_v2, rho_theta, rho_qv, rho_qc]
end

# Apply source terms
function source_terms_warm_bubble(ut, u, x, element_id, t, n_nodes, equation::CompressibleMoistPotTempEulerEquations2D)
  for j in 1:n_nodes, i in 1:n_nodes
    x1 = x[1, i, j, element_id]
    x2 = x[2, i, j, element_id]
    ut[3, i, j, element_id] +=  -equation._grav * u[1, i, j, element_id]
  end
  return nothing
end

function source_terms_moist_bubble(ut, u, x, element_id, t, n_nodes, equation::CompressibleMoistPotTempEulerEquations2D)

  RelCloud = 1
  for j in 1:n_nodes, i in 1:n_nodes
    x1 = x[1, i, j, element_id]
    x2 = x[2, i, j, element_id]
    rho = u[1, i, j, element_id]
    rho_theta = u[4, i, j, element_id]
    rho_qv = u[5, i, j, element_id]
    rho_qc = u[6, i, j, element_id]
    ut[3, i, j, element_id] +=  -equation._grav * rho

    rho_d = rho - rho_qv - rho_qc
    c_pml = equation.c_pd * rho_d + equation.c_pv * rho_qv + equation.c_pl * rho_qc
    c_vml = equation.c_vd * rho_d + equation.c_vv * rho_qv + equation.c_pl * rho_qc
    R_m   = equation.R_d * rho_d + equation.R_v * rho_qv
    kappa_M = R_m / c_pml
    p = (equation.R_d * rho_theta / equation.p0^kappa_M)^(1 / (1 - kappa_M))
    T = p / R_m
    T_C = T - 273.15
    p_vs = 611.2 * exp(17.62 * T_C / (243.12 + T_C))
    a = p_vs / (equation.R_v * T) - rho_qv
    b = rho_qc
    rho_q_cond = RelCloud * (a + b - sqrt(a * a + b * b))
    L = equation.L00 - (equation.c_pl - equation.c_pv) * T
    ut[4, i, j, element_id] += rho_theta * ((-L / (c_pml * T) -
                        log(p / equation.p0) * kappa_M * (equation.R_v / R_m - equation.c_pv / c_pml) +
                        equation.R_v / R_m) * rho_q_cond +
                        (log(p / equation.p0) * kappa_M * (equation.c_pl / c_pml)) * (-rho_q_cond))

    ut[5, i, j, element_id] +=  rho_q_cond
    ut[6, i, j, element_id] += -rho_q_cond
  end
  return nothing
end

"""
Moist bubble test from paper:
Bryan, G. H., and J. M. Fritsch, 2002: 
A Benchmark Simulation for Moist Nonhydrostatic Numerical Models. 
Mon. Wea. Rev., 130, 2917–2928, 
https://doi.org/10.1175/1520-0493(2002)130<2917:ABSFMN>2.0.CO;2.
"""

function initial_conditions_moist_bubble(x, t, equation::CompressibleMoistPotTempEulerEquations2D)


  z = x[2] 
  iz = 1000
  for i in 2:size(equation.z,1)
    if z <= equation.z[i]    
      iz = i - 1  
      break
    end
  end  
  z_l = equation.z[iz]
  rho_l = equation.Val[iz,1] 
  rho_theta_l = equation.Val[iz,2]
  rho_qv_l = equation.Val[iz,3]
  rho_qc_l = equation.Val[iz,4]
  z_r = equation.z[iz+1]
  rho_r = equation.Val[iz+1,1] 
  rho_theta_r = equation.Val[iz+1,2]
  rho_qv_r = equation.Val[iz+1,3]
  rho_qc_r = equation.Val[iz+1,4]

  rho = (rho_r * (z - z_l) + rho_l * (z_r - z)) / equation.delta_z
  rho_theta = rho * (rho_theta_r / rho_r * (z - z_l) + rho_theta_l / rho_l * (z_r - z)) / equation.delta_z
  rho_qv = rho * (rho_qv_r / rho_r * (z - z_l) + rho_qv_l / rho_l * (z_r - z)) / equation.delta_z
  rho_qc = rho * (rho_qc_r / rho_r * (z - z_l) + rho_qc_l / rho_l * (z_r - z)) / equation.delta_z

  rho, rho_theta, rho_qv, rho_qc = PerturbMoistProfile(x, rho, rho_theta, rho_qv, rho_qc, equation::CompressibleMoistPotTempEulerEquations2D)

  v1 = 20
  v2 = 0
  rho_v1 = rho * v1
  rho_v2 = rho * v2

  return @SVector [rho, rho_v1, rho_v2, rho_theta, rho_qv, rho_qc]
end


function PerturbMoistProfile(x, rho, rho_theta, rho_qv, rho_qc, equation::CompressibleMoistPotTempEulerEquations2D) 

  xc = 0
  zc = 2000
  rc = 2000
  Δtheta = 2

  r = sqrt((x[1] - xc)^2 + (x[2] - zc)^2)
  rho_d = rho - rho_qv - rho_qc
  kappa_M = (equation.R_d * rho_d + equation.R_v * rho_qv) / (equation.c_pd * rho_d + equation.c_pv * rho_qv + equation.c_pl * rho_qc)
  p_loc = equation.p0 *(equation.R_d * rho_theta / equation.p0)^(1/(1-kappa_M))
  T_loc = p_loc / (equation.R_d * rho_d + equation.R_v * rho_qv)

  if r < rc && Δtheta > 0 
    theta_dens = rho_theta / rho * (p_loc / equation.p0)^(kappa_M - equation.kappa)
    theta_dens_new = theta_dens * (1 + Δtheta * cospi(0.5*r/rc)^2 / 300)
    rt =(rho_qv + rho_qc) / rho_d 
    rv = rho_qv / rho_d
    theta_loc = theta_dens_new * (1 + rt)/(1 + (equation.R_v / equation.R_d) * rv)
    if rt > 0 
      while true 
        T_loc = theta_loc * (p_loc / equation.p0)^equation.kappa
        T_C = T_loc - 273.15
        # SaturVapor
        pvs = 611.2 * exp(17.62 * T_C / (243.12 + T_C))
        rho_d_new = (p_loc - pvs) / (equation.R_d * T_loc)
        rvs = pvs / (equation.R_v * rho_d_new * T_loc)
        theta_new = theta_dens_new * (1 + rt) / (1 + (equation.R_v / equation.R_d) * rvs)
        if abs(theta_new-theta_loc) <= theta_loc * 1.0e-12
          break
        else
          theta_loc=theta_new
        end
      end
    else
      rvs = 0
      T_loc = theta_loc * (p_loc / equation.p0)^equation.kappa
      rho_d_new = p_loc / (equation.R_d * T_loc)
      theta_new = theta_dens_new * (1 + rt) / (1 + (equation.R_v / equation.R_d) * rvs)
    end
    rho_qv = rvs * rho_d_new
    rho_qc = (rt - rvs) * rho_d_new
    rho = rho_d_new * (1 + rt)
    rho_d = rho - rho_qv - rho_qc
    kappa_M = (equation.R_d * rho_d + equation.R_v * rho_qv) / (equation.c_pd * rho_d + equation.c_pv * rho_qv + equation.c_pl * rho_qc)
    rho_theta = rho * theta_dens_new * (p_loc / equation.p0)^(equation.kappa - kappa_M)
  end
  return rho, rho_theta, rho_qv, rho_qc
end

function boundary_conditions_slip_wall(u_inner, orientation, direction, x, t,
                                       surface_flux_function,
                                       equation::CompressibleMoistPotTempEulerEquations2D)
  if orientation == 1 # interface in x-direction
    u_boundary = SVector(u_inner[1], -u_inner[2],  u_inner[3], u_inner[4], u_inner[5], u_inner[6])
  else # interface in y-direction
    u_boundary = SVector(u_inner[1],  u_inner[2], -u_inner[3], u_inner[4], u_inner[5], u_inner[6])
  end

  # Calculate boundary flux
  if direction in (2, 4) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equation)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equation)
  end

  return flux
end




# Calculate 1D flux for a single point
@inline function calcflux(u, orientation, equation::CompressibleMoistPotTempEulerEquations2D)
  rho, rho_v1, rho_v2, rho_theta, rho_qv, rho_qc = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  theta = rho_theta / rho
  qv = rho_qv / rho
  qc = rho_qc / rho
  rho_d = rho - rho_qv - rho_qc
  kappa_M = (equation.R_d * rho_d + equation.R_v * rho_qv) / (equation.c_pd * rho_d + equation.c_pv * rho_qv + equation.c_pl * rho_qc)
  p = (equation.R_d * rho_theta / equation.p0^kappa_M)^(1 / (1 - kappa_M))
  if orientation == 1
    f1 = rho_v1
    f2 = rho_v1 * v1 + p
    f3 = rho_v1 * v2
    f4 = rho_v1 * theta
    f5 = rho_v1 * qv
    f6 = rho_v1 * qc
  else
    f1 = rho_v2
    f2 = rho_v2 * v1
    f3 = rho_v2 * v2 + p
    f4 = rho_v2 * theta
    f5 = rho_v2 * qv
    f6 = rho_v2 * qc
  end
  return SVector(f1, f2, f3, f4, f5, f6)
end


"""
    function flux_shima_etal(u_ll, u_rr, orientation, equation::CompressibleMoistPotTempEulerEquations2D)

This flux is is a modification of the original kinetic energy preserving two-point flux by
Kuya, Totani and Kawai (2018)
  Kinetic energy and entropy preserving schemes for compressible flows
  by split convective forms
  [DOI: 10.1016/j.jcp.2018.08.058](https://doi.org/10.1016/j.jcp.2018.08.058)
The modification is in the energy flux to guarantee pressure equilibrium and was developed by
Nao Shima, Yuichi Kuya, Yoshiharu Tamaki, Soshi Kawai (JCP 2020)
  Preventing spurious pressure oscillations in split convective form discretizations for
  compressible flows
"""
@inline function flux_shima_etal(u_ll, u_rr, orientation, equation::CompressibleMoistPotTempEulerEquations2D)
  # Unpack left and right state
  rho_ll, rho_v1_ll, rho_v2_ll, rho_theta_ll, rho_qv_ll, rho_qc_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_theta_rr, rho_qv_rr, rho_qc_rr = u_rr

  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  theta_ll = rho_theta_ll / rho_ll
  qv_ll = rho_qv_ll / rho_ll
  qc_ll = rho_qc_ll / rho_ll
  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  theta_rr = rho_theta_rr / rho_rr
  qv_rr = rho_qv_rr / rho_rr
  qc_rr = rho_qc_rr / rho_rr

  # Average each factor of products in flux
  rho_avg = 1/2 * (rho_ll + rho_rr)
  v1_avg  = 1/2 * ( v1_ll +  v1_rr)
  v2_avg  = 1/2 * ( v2_ll +  v2_rr)
  theta_avg  = 1/2 * ( theta_ll +  theta_rr)
  qv_avg  = 1/2 * ( qv_ll +  qv_rr)
  qc_avg  = 1/2 * ( qc_ll +  qc_rr)

  # Calculate fluxes depending on orientation
  if orientation == 1
    f1 = rho_avg * v1_avg
    f2 = rho_avg * v1_avg * v1_avg + p_avg
    f3 = rho_avg * v1_avg * v2_avg
    f4 = rho_avg * v1_avg * theta_avg
    f5 = rho_avg * v1_avg * qv_avg
    f6 = rho_avg * v1_avg * qc_avg
  else
    f1 = rho_avg * v2_avg
    f2 = rho_avg * v2_avg * v1_avg
    f3 = rho_avg * v2_avg * v2_avg + p_avg
    f4 = rho_avg * v2_avg * theta_avg
    f5 = rho_avg * v2_avg * qv_avg
    f6 = rho_avg * v2_avg * qc_avg
  end

  return SVector(f1, f2, f3, f4, f5, f6)
end

"""
function flux_lmars(u_ll, u_rr, orientation, equation::CompressibleMoistPotTempEulerEquations2D)

Chen, X., N. Andronova, B. Van Leer, J. E. Penner, J. P. Boyd, C. Jablonowski, and S. Lin, 2013: 
A Control-Volume Model of the Compressible Euler Equations with a Vertical Lagrangian Coordinate. 
Mon. Wea. Rev., 141, 2526–2544, https://doi.org/10.1175/MWR-D-12-00129.1.

"""

function flux_lmars(u_ll, u_rr, orientation, equation::CompressibleMoistPotTempEulerEquations2D)
  # Calculate primitive variables and speed of sound
  rho_ll, rho_v1_ll, rho_v2_ll, rho_theta_ll, rho_qv_ll, rho_qc_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_theta_rr, rho_qv_rr, rho_qc_rr = u_rr

  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  theta_ll = rho_theta_ll / rho_ll
  qv_ll = rho_qv_ll / rho_ll
  qc_ll = rho_qc_ll / rho_ll
  rho_d_ll = rho_ll - rho_qv_ll - rho_qc_ll
  kappa_M = (equation.R_d * rho_d_ll + equation.R_v * rho_qv_ll) / (equation.c_pd * rho_d_ll + equation.c_pv * rho_qv_ll + equation.c_pl * rho_qc_ll)
  p_ll = (equation.R_d * rho_theta_ll / equation.p0^kappa_M)^(1 / (1 - kappa_M))
  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  theta_rr = rho_theta_rr / rho_rr
  qv_rr = rho_qv_rr / rho_rr
  qc_rr = rho_qc_rr / rho_rr
  rho_d_rr = rho_rr - rho_qv_rr - rho_qc_rr
  kappa_M = (equation.R_d * rho_d_rr + equation.R_v * rho_qv_rr) / (equation.c_pd * rho_d_rr + equation.c_pv * rho_qv_rr + equation.c_pl * rho_qc_rr)
  p_rr = (equation.R_d * rho_theta_rr / equation.p0^kappa_M)^(1 / (1 - kappa_M))

  rhoM = 0.5 * (rho_ll + rho_rr)
  if orientation == 1 # x-direction
    pM = 0.5 * (p_ll + p_rr) - 0.5 * rhoM * equation.cS * (v1_rr - v1_ll) 
    vM = 0.5 * (v1_ll + v1_rr) - 1 / (2 * rhoM * equation.cS) * (p_rr - p_ll) 
    if vM >= 0
      f = u_ll  * vM + pM * SVector(0, 1, 0, 0, 0, 0)
    else
      f = u_rr  * vM + pM * SVector(0, 1, 0, 0, 0, 0)
    end  
  else # y-direction
    pM = 0.5 * (p_ll + p_rr) - 0.5 * rhoM * equation.cS * (v2_rr - v2_ll) 
    vM = 0.5 * (v2_ll + v2_rr) - 1 / (2 * rhoM * equation.cS) * (p_rr - p_ll) 
    if vM >= 0
      f = u_ll * vM + pM * SVector(0, 0, 1, 0, 0, 0)
    else
      f = u_rr * vM + pM * SVector(0, 0, 1, 0, 0, 0)
    end  
  end
  return f
end


# Determine maximum stable time step based on polynomial degree and CFL number
function calc_max_dt(u, element_id, invjacobian, cfl,
                     equation::CompressibleMoistPotTempEulerEquations2D, dg)
  λ_max = 0.0
  for j in 1:nnodes(dg), i in 1:nnodes(dg)
    rho, rho_v1, rho_v2, rho_theta, rho_qv, rho_qc = get_node_vars(u, dg, i, j, element_id)
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v_mag = sqrt(v1^2 + v2^2)
    rho_d = rho - rho_qv - rho_qc
    kappa_M = (equation.R_d * rho_d + equation.R_v * rho_qv) / (equation.c_pd * rho_d + equation.c_pv * rho_qv + equation.c_pl * rho_qc)
    p = (equation.R_d * rho_theta / equation.p0^kappa_M)^(1 / (1 - kappa_M))
    c = sqrt(equation.gamma * p / rho)
    λ_max = max(λ_max, v_mag + c)
  end

  dt = cfl * 2 / (nnodes(dg) * invjacobian * λ_max)

  return dt
end


# Convert conservative variables to primitive
function cons2prim(cons, equation::CompressibleMoistPotTempEulerEquations2D)
  prim = similar(cons)
  @. prim[1, :, :, :] = cons[1, :, :, :]
  @. prim[2, :, :, :] = cons[2, :, :, :] / cons[1, :, :, :]
  @. prim[3, :, :, :] = cons[3, :, :, :] / cons[1, :, :, :]
  @. prim[4, :, :, :] = cons[4, :, :, :] / cons[1, :, :, :]
  @. prim[5, :, :, :] = cons[5, :, :, :] / cons[1, :, :, :]
  @. prim[6, :, :, :] = cons[6, :, :, :] / cons[1, :, :, :]
  return prim
end

# Convert conservative variables to potential
function cons2pot(u, equation::CompressibleMoistPotTempEulerEquations2D)
  rho, rho_v1, rho_v2, rho_theta, rho_qv, rho_qc = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  qv = rho_qv / rho
  qc = rho_qc / rho
  rho_d = rho - rho_qv - rho_qc 
  kappa_M = (equation.R_d * rho_d + equation.R_v * rho_qv) / (equation.c_pd * rho_d + equation.c_pv * rho_qv + 
             equation.c_pl * rho_qc)
  p = (equation.R_d * rho_theta / (equation.p0.^kappa_M)).^(1/(1-kappa_M)) 
  T = p / (equation.R_d * rho_d + equation.R_v * rho_qv)
  p_d = equation.R_d * rho_d * T
  theta_e = 
  T * (p_d / equation.p0)^(-equation.R_d * rho_d /
         (equation.c_pd * rho_d + equation.c_pl * (rho_qv + rho_qc))) * 
         exp((equation.L00 - (equation.c_pl - equation.c_pv) * T) * 
         rho_qv / ((equation.c_pd * rho_d + equation.c_pl * 
         (rho_qv + rho_qc)) * T))
   return SVector(rho, v1, v2, theta_e, qv, qc)      
end

# Convert conservative variables to entropy
@inline function cons2entropy(u, equation::CompressibleMoistPotTempEulerEquations2D)
  rho, rho_v1, rho_v2, rho_theta, rho_qv, rho_qc = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  v_square = v1^2 + v2^2
  rho_d = rho - rho_qv - rho_qc
  kappa_M = (equation.R_d * rho_d + equation.R_v * rho_qv) / (equation.c_pd * rho_d + equation.c_pv * rho_qv +
             equation.c_pl * rho_qc)
  p = (equation.R_d * rho_theta / (equation.p0.^kappa_M)).^(1/(1-kappa_M))
  s = log(p) - equation.gamma*log(rho)
  rho_p = rho / p

  w1 = (equation.gamma - s) / (equation.gamma-1) - 0.5 * rho_p * v_square
  w2 = rho_p * v1
  w3 = rho_p * v2
  w4 = -rho_p
  w5 = rho_p * rho_qv / rho
  w6 = rho_p * rho_qc / rho

  return SVector(w1, w2, w3, w4, w5, w6)
end

# Convert primitive to conservative variables
function prim2cons(prim, equation::CompressibleMoistPotTempEulerEquations2D)
  cons = similar(prim)
  cons[1] = prim[1]
  cons[2] = prim[2] * prim[1]
  cons[3] = prim[3] * prim[1]
  cons[4] = prim[4] * prim[1]
  cons[5] = prim[5] * prim[1]
  cons[6] = prim[6] * prim[1]
  return cons
end


# Convert conservative variables to indicator variable for discontinuities (elementwise version)
@inline function cons2indicator!(indicator, cons, element_id, n_nodes, indicator_variable,
                                 equation::CompressibleMoistPotTempEulerEquations2D)
  for j in 1:n_nodes
    for i in 1:n_nodes
      indicator[1, i, j] = cons2indicator(cons[1, i, j, element_id], cons[2, i, j, element_id],
                                          cons[3, i, j, element_id], cons[4, i, j, element_id],
                                          indicator_variable, equation)
    end
  end
end


# Convert conservative variables to indicator variable for discontinuities (pointwise version)
@inline function cons2indicator(rho, rho_v1, rho_v2, rho_theta, ::Val{:density},
                                equation::CompressibleMoistPotTempEulerEquations2D)
  # Indicator variable is rho
  return rho
end


# Convert conservative variables to indicator variable for discontinuities (pointwise version)
@inline function cons2indicator(rho, rho_v1, rho_v2, rho_theta, ::Val{:density_pressure},
                                equation::CompressibleMoistPotTempEulerEquations2D)
  # Calculate pressure
  p = (equation.R_d * rho_theta / equation.p0^equation.kappa)^(1 / (1 - equation.kappa))

  # Indicator variable is rho * p
  return rho * p
end


# Convert conservative variables to indicator variable for discontinuities (pointwise version)
@inline function cons2indicator(rho, rho_v1, rho_v2, rho_theta, ::Val{:pressure},
                                equation::CompressibleMoistPotTempEulerEquations2D)
  # Indicator variable is p
  return (equation.R_d * rho_theta / equation.p0^equation.kappa)^(1 / (1 - equation.kappa))
end


# Calculate thermodynamic entropy for a conservative state `cons`
@inline function entropy_thermodynamic(cons, equation::CompressibleMoistPotTempEulerEquations2D)
  # Pressure
  p = (equation.R_d * cons[4] / equation.p0^equation.kappa)^(1 / (1 - equation.kappa))

  # Thermodynamic entropy
  s = log(p) - equation.gamma*log(cons[1])

  return s
end

# Calculate potential temperature for a conservative state `cons`
@inline function pottemp_thermodynamic(cons, equation::CompressibleMoistPotTempEulerEquations2D)

  return cons[4] / cons[1]
end


# Calculate mathematical entropy for a conservative state `cons`
@inline function entropy_math(cons, equation::CompressibleMoistPotTempEulerEquations2D)
  # Mathematical entropy
  S = -entropy_thermodynamic(cons, equation) * cons[1] / (equation.gamma - 1)

  return S
end


# Default entropy is the mathematical entropy
@inline entropy(cons, equation::CompressibleMoistPotTempEulerEquations2D) = entropy_math(cons, equation)


# Calculate total energy for a conservative state `cons`
@inline energy_total(cons, ::CompressibleMoistPotTempEulerEquations2D) = cons[4]


# Calculate kinetic energy for a conservative state `cons`
@inline function energy_kinetic(cons, equation::CompressibleMoistPotTempEulerEquations2D)
  return 0.5 * (cons[2]^2 + cons[3]^2)/cons[1]
end


# Calculate internal energy for a conservative state `cons`
@inline function energy_internal(cons, equation::CompressibleMoistPotTempEulerEquations2D)
  return energy_total(cons, equation) - energy_kinetic(cons, equation)
end
