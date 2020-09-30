@doc raw"""
    CompressibleEulerPotEquations2D

The compressible Euler equations for an ideal gas in two space dimensions.
"""
struct CompressibleEulerPotEquations2D <: AbstractCompressibleEulerEquations{2, 6}
  c_pd::Float64
  c_vd::Float64
  R_d::Float64
  c_pv::Float64
  c_vv::Float64
  R_v::Float64
  c_pl::Float64
  κ::Float64
  gamma::Float64
  _grav::Float64
  p0::Float64
  L00::Float64
  cS::Float64
  Δz::Float64
  z::Array{Float64,1}
  Val::Array{Float64,2}
end

function CompressibleEulerPotEquations2D()
  c_pd = parameter("c_pd",1004)
  c_vd = parameter("c_vd",717)
  c_pv = parameter("c_pv",1885)
  c_vv = parameter("c_vv",1424)
  c_pl = parameter("c_pl",4186)
  R_d = parameter("R_d",c_pd-c_vd)
  R_v = parameter("R_v",c_pv-c_vv)
  κ = parameter("κ",R_d/c_pd)
  gamma = parameter("gamma", c_pd/c_vd)
  _grav = parameter("_grav",9.81)
  p0 = parameter("p0",1.e5)
  L00 = parameter("L00",2.5000e6 + (c_pl - c_pv) * 273.15)
  println("L00 ",L00)
  cS = parameter("cS",360.e0)

  n=1000
  z=zeros(n+1)
  Val=zeros(n+1,4)
  r_t0 = 2.e-2
  θ_e0 = 320
  Δz = 10

  function ResMoisture(z, y, yPrime)

    p = y[1]
    ρ = y[2]
    T = y[3]
    r_t = y[4]
    r_v = y[5]
    ρ_qv = y[6]
    θ_e = y[7]
    pPrime = yPrime[1]
    F=zeros(7,1)

    ρ_d = ρ / (1 + r_t)
    p_d = R_d * ρ_d * T
    T_C = T - 273.15
    p_vs = 611.2 * exp(17.62 * T_C / (243.12 + T_C))
    L = L00 - (c_pl - c_pv) * T
    F[1] = pPrime + _grav * ρ
    F[2] = p - (R_d * ρ_d + R_v * ρ_qv) * T
    F[3] = θ_e - T * (p_d / p0)^(-R_d/
         (c_pd + c_pl * r_t)) * exp(L * r_v / ((c_pd + c_pl * r_t) * T))
    F[4] = r_t - r_t0
    F[5] = ρ_qv-ρ_d * r_v
    F[6] = θ_e - θ_e0
    a = p_vs / (R_v * T) - ρ_qv
    b = ρ - ρ_qv - ρ_d
    F[7]=a+b-sqrt(a*a+b*b)
    return F
  end
  function SetImplEuler(z,Δz,y0)
    function ImplEuler(y)
      return ResMoisture(z,y,(y-y0)/Δz)
    end
  end

  y=zeros(7)
  p=1.e5
  ρ = 1.4
  r_t = r_t0
  r_v = r_t0
  ρ_qv = ρ * r_v
  θ_e = θ_e0
  T = θ_e

  yPrime = zeros(7)
  y0 = zeros(7)
  y0[1] = p
  y0[2] = ρ
  y0[3] = T
  y0[4] = r_t
  y0[5] = r_v
  y0[6] = ρ_qv
  y0[7] = θ_e


  z0 = 0.0
  Δz = 0.01
  y=deepcopy(y0);
  F = SetImplEuler(z0,Δz,y0)
  res = nlsolve(F,y0)
  p = res.zero[1]
  ρ = res.zero[2]
  T = res.zero[3]
  r_t = res.zero[4]
  r_v = res.zero[5]
  ρ_qv = res.zero[6]
  θ_e = res.zero[7]
  ρ_d = ρ / (1 + r_t)
  ρ_qc = ρ - ρ_d - ρ_qv
  κ_M=(R_d * ρ_d + R_v * ρ_qv) / (c_pd * ρ_d + c_pv * ρ_qv + c_pl * ρ_qc)
  ρ_θ = ρ * (p0 / p)^κ_M * T * (1 + (R_v / R_d) *r_v) / (1 + r_t) 
  Val[1,1] = ρ
  Val[1,2] = ρ_θ
  Val[1,3] = ρ_qv
  Val[1,4] = ρ_qc
  Δz = 10.0
  z[1] = 0
  Val[1,1] = res.zero[2]
  for i = 1 : n
    y0 = deepcopy(res.zero)
    F = SetImplEuler(z,Δz,y0)
    res = nlsolve(F,y0)
    z[i+1] = z[i] + Δz
    p = res.zero[1]
    ρ = res.zero[2]
    T = res.zero[3]
    r_t = res.zero[4]
    r_v = res.zero[5]
    ρ_qv = res.zero[6]
    θ_e = res.zero[7]
    ρ_d = ρ / (1 + r_t)
    ρ_qc = ρ - ρ_d - ρ_qv
    κ_M=(R_d * ρ_d + R_v * ρ_qv) / (c_pd * ρ_d + c_pv * ρ_qv + c_pl * ρ_qc)
    ρ_θ = ρ * (p0 / p)^κ_M * T * (1 + (R_v / R_d) *r_v) / (1 + r_t) 
    Val[i+1,1] = ρ
    Val[i+1,2] = ρ_θ
    Val[i+1,3] = ρ_qv
    Val[i+1,4] = ρ_qc
  end
  CompressibleEulerPotEquations2D(c_pd,c_vd,R_d,c_pv,c_vv,R_v,c_pl,κ,gamma,_grav,p0,L00,cS,Δz,z,Val)
end


get_name(::CompressibleEulerPotEquations2D) = "CompressibleEulerPotEquations2D"
varnames_cons(::CompressibleEulerPotEquations2D) = @SVector ["rho", "rho_v1", "rho_v2", "rho_pot", "rho_qv", "rho_qc"]
varnames_prim(::CompressibleEulerPotEquations2D) = @SVector ["rho", "v1", "v2", "pot", "qv", "qc"]


"""
Dry warm bubble test from paper:
Wicker, L. J., and W. C. Skamarock, 1998: A time-splitting scheme
for the elastic equations incorporating second-order Runge–Kutta
time differencing. Mon. Wea. Rev., 126, 1992–1999.
"""

function initial_conditions_warm_bubble(x, t, equation::CompressibleEulerPotEquations2D)

  xc = 0
  zc = 2000
  r = sqrt((x[1] - xc)^2 + (x[2] - zc)^2)
  rc = 2000
  θ_ref = 300
  qv_ref = 0
  qc_ref = 0
  Δθ = 0
  Δqv = 0
  Δqc = 0

  if r <= rc
     Δθ = 2 * cospi(0.5*r/rc)^2
  end
  xc = 1000
  zc = 2000
  r = sqrt((x[1] - xc)^2 + (x[2] - zc)^2)
  rc = 1000
  if r <= rc
     Δqv = 0 * cospi(0.5*r/rc)^2
     Δqc = 0 * cospi(0.5*r/rc)^2
  end

  #Perturbed state:
  θ = θ_ref + Δθ # potential temperature
  qv = qv_ref + Δqv
  qc = qc_ref + Δqc
  π_exner = 1 - equation._grav / (equation.c_pd * θ) * x[2] # exner pressure
  ρ = equation.p0 / (equation.R_d * θ) * (π_exner)^(equation.c_vd / equation.R_d) # density

  v1 = 20
  v2 = 0
  ρ_v1 = ρ * v1
  ρ_v2 = ρ * v2
  ρ_θ = ρ * θ
  ρ_qv = ρ * qv
  ρ_qc = ρ * qc
  return @SVector [ρ, ρ_v1, ρ_v2, ρ_θ, ρ_qv, ρ_qc]
end

# Apply source terms
function source_terms_warm_bubble(ut, u, x, element_id, t, n_nodes, equation::CompressibleEulerPotEquations2D)
  for j in 1:n_nodes, i in 1:n_nodes
    x1 = x[1, i, j, element_id]
    x2 = x[2, i, j, element_id]
    ut[3, i, j, element_id] +=  -equation._grav * u[1, i, j, element_id]
  end
  return nothing
end

function source_terms_moist_bubble(ut, u, x, element_id, t, n_nodes, equation::CompressibleEulerPotEquations2D)

  RelCloud = 1
  for j in 1:n_nodes, i in 1:n_nodes
    x1 = x[1, i, j, element_id]
    x2 = x[2, i, j, element_id]
    ρ = u[1, i, j, element_id]
    ρ_θ = u[4, i, j, element_id]
    ρ_qv = u[5, i, j, element_id]
    ρ_qc = u[6, i, j, element_id]
    ut[3, i, j, element_id] +=  -equation._grav * ρ

    ρ_d = ρ - ρ_qv - ρ_qc
    c_pml = equation.c_pd * ρ_d + equation.c_pv * ρ_qv + equation.c_pl * ρ_qc
    c_vml = equation.c_vd * ρ_d + equation.c_vv * ρ_qv + equation.c_pl * ρ_qc
    R_m   = equation.R_d * ρ_d + equation.R_v * ρ_qv
    κ_M = R_m / c_pml
    p = (equation.R_d * ρ_θ / equation.p0^κ_M)^(1 / (1 - κ_M))
    T = p / R_m
    T_C = T - 273.15
    p_vs = 611.2 * exp(17.62 * T_C / (243.12 + T_C))
    a = p_vs / (equation.R_v * T) - ρ_qv
    b = ρ_qc
    ρ_q_cond = RelCloud * (a + b - sqrt(a * a + b * b))
    L = equation.L00 - (equation.c_pl - equation.c_pv) * T
    ut[4, i, j, element_id] += ρ_θ * ((-L / (c_pml * T) -
                        log(p / equation.p0) * κ_M * (equation.R_v / R_m - equation.c_pv / c_pml) +
                        equation.R_v / R_m) * ρ_q_cond +
                        (log(p / equation.p0) * κ_M * (equation.c_pl / c_pml)) * (-ρ_q_cond))

    ut[5, i, j, element_id] +=  ρ_q_cond
    ut[6, i, j, element_id] += -ρ_q_cond
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

function initial_conditions_moist_bubble(x, t, equation::CompressibleEulerPotEquations2D)


  z = x[2] 
  iz = 1000
  for i in 2:size(equation.z,1)
    if z <= equation.z[i]    
      iz = i - 1  
      break
    end
  end  
  z_l = equation.z[iz]
  ρ_l = equation.Val[iz,1] 
  ρ_θ_l = equation.Val[iz,2]
  ρ_qv_l = equation.Val[iz,3]
  ρ_qc_l = equation.Val[iz,4]
  z_r = equation.z[iz+1]
  ρ_r = equation.Val[iz+1,1] 
  ρ_θ_r = equation.Val[iz+1,2]
  ρ_qv_r = equation.Val[iz+1,3]
  ρ_qc_r = equation.Val[iz+1,4]

  ρ = (ρ_r * (z - z_l) + ρ_l * (z_r - z)) / equation.Δz
  ρ_θ = ρ * (ρ_θ_r / ρ_r * (z - z_l) + ρ_θ_l / ρ_l * (z_r - z)) / equation.Δz
  ρ_qv = ρ * (ρ_qv_r / ρ_r * (z - z_l) + ρ_qv_l / ρ_l * (z_r - z)) / equation.Δz
  ρ_qc = ρ * (ρ_qc_r / ρ_r * (z - z_l) + ρ_qc_l / ρ_l * (z_r - z)) / equation.Δz

  ρ, ρ_θ, ρ_qv, ρ_qc = PerturbMoistProfile(x, ρ, ρ_θ, ρ_qv, ρ_qc, equation::CompressibleEulerPotEquations2D)

  v1 = 20
  v2 = 0
  ρ_v1 = ρ * v1
  ρ_v2 = ρ * v2

  return @SVector [ρ, ρ_v1, ρ_v2, ρ_θ, ρ_qv, ρ_qc]
end


function PerturbMoistProfile(x, ρ, ρ_θ, ρ_qv, ρ_qc, equation::CompressibleEulerPotEquations2D) 

  xc = 0
  zc = 2000
  rc = 2000
  Δθ = 2

  r = sqrt((x[1] - xc)^2 + (x[2] - zc)^2)
  ρ_d = ρ - ρ_qv - ρ_qc
  κ_M = (equation.R_d * ρ_d + equation.R_v * ρ_qv) / (equation.c_pd * ρ_d + equation.c_pv * ρ_qv + equation.c_pl * ρ_qc)
  p_loc = equation.p0 *(equation.R_d * ρ_θ / equation.p0)^(1/(1-κ_M))
  T_loc = p_loc / (equation.R_d * ρ_d + equation.R_v * ρ_qv)

  if r < rc && Δθ > 0 
    θ_dens = ρ_θ / ρ * (p_loc / equation.p0)^(κ_M - equation.κ)
    θ_dens_new = θ_dens * (1 + Δθ * cospi(0.5*r/rc)^2 / 300)
    rt =(ρ_qv + ρ_qc) / ρ_d 
    rv = ρ_qv / ρ_d
    θ_loc = θ_dens_new * (1 + rt)/(1 + (equation.R_v / equation.R_d) * rv)
    if rt > 0 
      while true 
        T_loc = θ_loc * (p_loc / equation.p0)^equation.κ
        T_C = T_loc - 273.15
        # SaturVapor
        pvs = 611.2 * exp(17.62 * T_C / (243.12 + T_C))
        ρ_d_new = (p_loc - pvs) / (equation.R_d * T_loc)
        rvs = pvs / (equation.R_v * ρ_d_new * T_loc)
        θ_new = θ_dens_new * (1 + rt) / (1 + (equation.R_v / equation.R_d) * rvs)
        if abs(θ_new-θ_loc) <= θ_loc * 1.0e-12
          break
        else
          θ_loc=θ_new
        end
      end
    else
      rvs = 0
      T_loc = θ_loc * (p_loc / equation.p0)^equation.κ
      ρ_d_new = p_loc / (equation.R_d * T_loc)
      θ_new = θ_dens_new * (1 + rt) / (1 + (equation.R_v / equation.R_d) * rvs)
    end
    ρ_qv = rvs * ρ_d_new
    ρ_qc = (rt - rvs) * ρ_d_new
    ρ = ρ_d_new * (1 + rt)
    ρ_d = ρ - ρ_qv - ρ_qc
    κ_M = (equation.R_d * ρ_d + equation.R_v * ρ_qv) / (equation.c_pd * ρ_d + equation.c_pv * ρ_qv + equation.c_pl * ρ_qc)
    ρ_θ = ρ * θ_dens_new * (p_loc / equation.p0)^(equation.κ - κ_M)
  end
  return ρ, ρ_θ, ρ_qv, ρ_qc
end




# Calculate 1D flux for a single point
@inline function calcflux(u, orientation, equation::CompressibleEulerPotEquations2D)
  ρ, ρ_v1, ρ_v2, ρ_θ, ρ_qv, ρ_qc = u

  v1 = ρ_v1 / ρ
  v2 = ρ_v2 / ρ
  θ = ρ_θ / ρ
  qv = ρ_qv / ρ
  qc = ρ_qc / ρ
  ρ_d = ρ - ρ_qv - ρ_qc
  κ_M = (equation.R_d * ρ_d + equation.R_v * ρ_qv) / (equation.c_pd * ρ_d + equation.c_pv * ρ_qv + equation.c_pl * ρ_qc)
  p = (equation.R_d * ρ_θ / equation.p0^κ_M)^(1 / (1 - κ_M))
  if orientation == 1
    f1 = ρ_v1
    f2 = ρ_v1 * v1 + p
    f3 = ρ_v1 * v2
    f4 = ρ_v1 * θ
    f5 = ρ_v1 * qv
    f6 = ρ_v1 * qc
  else
    f1 = ρ_v2
    f2 = ρ_v2 * v1
    f3 = ρ_v2 * v2 + p
    f4 = ρ_v2 * θ
    f5 = ρ_v2 * qv
    f6 = ρ_v2 * qc
  end
  return SVector(f1, f2, f3, f4, f5, f6)
end


"""
    function flux_shima_etal(u_ll, u_rr, orientation, equation::CompressibleEulerPotEquations2D)

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
@inline function flux_shima_etal(u_ll, u_rr, orientation, equation::CompressibleEulerPotEquations2D)
  # Unpack left and right state
  rho_ll, rho_v1_ll, rho_v2_ll, rho_θ_ll, rho_qv_ll, rho_qc_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_θ_rr, rho_qv_rr, rho_qc_rr = u_rr

  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  θ_ll = rho_θ_ll / rho_ll
  qv_ll = rho_qv_ll / rho_ll
  qc_ll = rho_qc_ll / rho_ll
  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  θ_rr = rho_θ_rr / rho_rr
  qv_rr = rho_qv_rr / rho_rr
  qc_rr = rho_qc_rr / rho_rr

  # Average each factor of products in flux
  rho_avg = 1/2 * (rho_ll + rho_rr)
  v1_avg  = 1/2 * ( v1_ll +  v1_rr)
  v2_avg  = 1/2 * ( v2_ll +  v2_rr)
  θ_avg  = 1/2 * ( θ_ll +  θ_rr)
  qv_avg  = 1/2 * ( qv_ll +  qv_rr)
  qc_avg  = 1/2 * ( qc_ll +  qc_rr)

  # Calculate fluxes depending on orientation
  if orientation == 1
    f1 = rho_avg * v1_avg
    f2 = rho_avg * v1_avg * v1_avg + p_avg
    f3 = rho_avg * v1_avg * v2_avg
    f4 = rho_avg * v1_avg * θ_avg
    f5 = rho_avg * v1_avg * qv_avg
    f6 = rho_avg * v1_avg * qc_avg
  else
    f1 = rho_avg * v2_avg
    f2 = rho_avg * v2_avg * v1_avg
    f3 = rho_avg * v2_avg * v2_avg + p_avg
    f4 = rho_avg * v2_avg * θ_avg
    f5 = rho_avg * v2_avg * qv_avg
    f6 = rho_avg * v2_avg * qc_avg
  end

  return SVector(f1, f2, f3, f4, f5, f6)
end

"""
function flux_lmars(u_ll, u_rr, orientation, equation::CompressibleEulerPotEquations2D)

Chen, X., N. Andronova, B. Van Leer, J. E. Penner, J. P. Boyd, C. Jablonowski, and S. Lin, 2013: 
A Control-Volume Model of the Compressible Euler Equations with a Vertical Lagrangian Coordinate. 
Mon. Wea. Rev., 141, 2526–2544, https://doi.org/10.1175/MWR-D-12-00129.1.

"""

function flux_lmars(u_ll, u_rr, orientation, equation::CompressibleEulerPotEquations2D)
  # Calculate primitive variables and speed of sound
  ρ_ll, ρ_v1_ll, ρ_v2_ll, ρ_θ_ll, ρ_qv_ll, ρ_qc_ll = u_ll
  ρ_rr, ρ_v1_rr, ρ_v2_rr, ρ_θ_rr, ρ_qv_rr, ρ_qc_rr = u_rr

  v1_ll = ρ_v1_ll / ρ_ll
  v2_ll = ρ_v2_ll / ρ_ll
  θ_ll = ρ_θ_ll / ρ_ll
  qv_ll = ρ_qv_ll / ρ_ll
  qc_ll = ρ_qc_ll / ρ_ll
  ρ_d_ll = ρ_ll - ρ_qv_ll - ρ_qc_ll
  κ_M = (equation.R_d * ρ_d_ll + equation.R_v * ρ_qv_ll) / (equation.c_pd * ρ_d_ll + equation.c_pv * ρ_qv_ll + equation.c_pl * ρ_qc_ll)
  p_ll = (equation.R_d * ρ_θ_ll / equation.p0^κ_M)^(1 / (1 - κ_M))
  v1_rr = ρ_v1_rr / ρ_rr
  v2_rr = ρ_v2_rr / ρ_rr
  θ_rr = ρ_θ_rr / ρ_rr
  qv_rr = ρ_qv_rr / ρ_rr
  qc_rr = ρ_qc_rr / ρ_rr
  ρ_d_rr = ρ_rr - ρ_qv_rr - ρ_qc_rr
  κ_M = (equation.R_d * ρ_d_rr + equation.R_v * ρ_qv_rr) / (equation.c_pd * ρ_d_rr + equation.c_pv * ρ_qv_rr + equation.c_pl * ρ_qc_rr)
  p_rr = (equation.R_d * ρ_θ_rr / equation.p0^κ_M)^(1 / (1 - κ_M))

  ρM = 0.5 * (ρ_ll + ρ_rr)
  if orientation == 1 # x-direction
    pM = 0.5 * (p_ll + p_rr) - 0.5 * ρM * equation.cS * (v1_rr - v1_ll) 
    vM = 0.5 * (v1_ll + v1_rr) - 1 / (2 * ρM * equation.cS) * (p_rr - p_ll) 
    if vM >= 0
      f = u_ll  * vM + pM * SVector(0, 1, 0, 0, 0, 0)
    else
      f = u_rr  * vM + pM * SVector(0, 1, 0, 0, 0, 0)
    end  
  else # y-direction
    pM = 0.5 * (p_ll + p_rr) - 0.5 * ρM * equation.cS * (v2_rr - v2_ll) 
    vM = 0.5 * (v2_ll + v2_rr) - 1 / (2 * ρM * equation.cS) * (p_rr - p_ll) 
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
                     equation::CompressibleEulerPotEquations2D, dg)
  λ_max = 0.0
  for j in 1:nnodes(dg), i in 1:nnodes(dg)
    ρ, ρ_v1, ρ_v2, ρ_θ, ρ_qv, ρ_qc = get_node_vars(u, dg, i, j, element_id)
    v1 = ρ_v1 / ρ
    v2 = ρ_v2 / ρ
    v_mag = sqrt(v1^2 + v2^2)
    ρ_d = ρ - ρ_qv - ρ_qc
    κ_M = (equation.R_d * ρ_d + equation.R_v * ρ_qv) / (equation.c_pd * ρ_d + equation.c_pv * ρ_qv + equation.c_pl * ρ_qc)
    p = (equation.R_d * ρ_θ / equation.p0^κ_M)^(1 / (1 - κ_M))
    c = sqrt(equation.gamma * p / ρ)
    λ_max = max(λ_max, v_mag + c)
  end

  dt = cfl * 2 / (nnodes(dg) * invjacobian * λ_max)

  return dt
end


# Convert conservative variables to primitive
function cons2prim(cons, equation::CompressibleEulerPotEquations2D)
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
function cons2pot(cons, equation::CompressibleEulerPotEquations2D)
  n_nodes = size(cons, 2)
  n_elements = size(cons, 4)
  pot = similar(cons)
  ρ_d = zeros(n_nodes,n_nodes,n_elements)
  κ_M  = zeros(n_nodes,n_nodes,n_elements)
  p = zeros(n_nodes,n_nodes,n_elements)
  p_d = zeros(n_nodes,n_nodes,n_elements)
  T = zeros(n_nodes,n_nodes,n_elements)
  @. ρ_d = cons[1, :, :, :] - cons[5, :, :, :] - cons[6, :, :, :]
  @. κ_M=(equation.R_d * ρ_d + equation.R_v * cons[5, :, :, :]) / (equation.c_pd * ρ_d + equation.c_pv * cons[5, :, :, :] + equation.c_pl * cons[6, :, :, :])
  @. p = (equation.R_d * cons[4, :, :, :] / (equation.p0.^κ_M)).^(1/(1-κ_M)) 
  @. T = p / (equation.R_d * ρ_d + equation.R_v * cons[5, :, :, :])
  @. p_d = equation.R_d * ρ_d * T
  @. pot[4, :, :, :] =       
  T * (p_d / equation.p0)^(-equation.R_d * ρ_d /
         (equation.c_pd * ρ_d + equation.c_pl * (cons[5, :, :, :] +cons[6, :, :, :]))) * 
         exp((equation.L00 - (equation.c_pl - equation.c_pv) * T) * 
         cons[5, :, :, :] / ((equation.c_pd * ρ_d + equation.c_pl * 
         (cons[5, :, :, :] +cons[6, :, :, :])) * T))
  @. pot[1, :, :, :] = cons[1, :, :, :]
  @. pot[2, :, :, :] = cons[2, :, :, :] / cons[1, :, :, :]
  @. pot[3, :, :, :] = cons[3, :, :, :] / cons[1, :, :, :]
# @. pot[4, :, :, :] = cons[4, :, :, :] / cons[1, :, :, :]
  @. pot[5, :, :, :] = cons[5, :, :, :] / cons[1, :, :, :]
  @. T = cons[1, :, :, :] - cons[5, :, :, :] - cons[6, :, :, :]
  @. pot[6, :, :, :] = (cons[6, :, :, :] + cons[5, :, :, :]) / T
  return pot
end

# Convert conservative variables to entropy
function cons2entropy(cons, n_nodes, n_elements, equation::CompressibleEulerPotEquations2D)
  entropy = similar(cons)
  v = zeros(2,n_nodes,n_nodes,n_elements)
  v_square = zeros(n_nodes,n_nodes,n_elements)
  p = zeros(n_nodes,n_nodes,n_elements)
  s = zeros(n_nodes,n_nodes,n_elements)
  rho_p = zeros(n_nodes,n_nodes,n_elements)

  @. v[1, :, :, :] = cons[2, :, :, :] / cons[1, :, :, :]
  @. v[2, :, :, :] = cons[3, :, :, :] / cons[1, :, :, :]
  @. v_square[ :, :, :] = v[1, :, :, :]*v[1, :, :, :]+v[2, :, :, :]*v[2, :, :, :]
  @. p[ :, :, :] = (equation.R_d * cons[4, :, :, :] / equation.p0^equation.κ)^(1 / (1 - equation.κ))
  @. s[ :, :, :] = log(p[:, :, :]) - equation.gamma*log(cons[1, :, :, :])
  @. rho_p[ :, :, :] = cons[1, :, :, :] / p[ :, :, :]

  @. entropy[1, :, :, :] = (equation.gamma - s[:,:,:])/(equation.gamma-1) -
                           0.5*rho_p[:,:,:]*v_square[:,:,:]
  @. entropy[2, :, :, :] = rho_p[:,:,:]*v[1,:,:,:]
  @. entropy[3, :, :, :] = rho_p[:,:,:]*v[2,:,:,:]
  @. entropy[4, :, :, :] = -rho_p[:,:,:]

  return entropy
end


# Convert primitive to conservative variables
function prim2cons(prim, equation::CompressibleEulerPotEquations2D)
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
                                 equation::CompressibleEulerPotEquations2D)
  for j in 1:n_nodes
    for i in 1:n_nodes
      indicator[1, i, j] = cons2indicator(cons[1, i, j, element_id], cons[2, i, j, element_id],
                                          cons[3, i, j, element_id], cons[4, i, j, element_id],
                                          indicator_variable, equation)
    end
  end
end


# Convert conservative variables to indicator variable for discontinuities (pointwise version)
@inline function cons2indicator(rho, rho_v1, rho_v2, rho_θ, ::Val{:density},
                                equation::CompressibleEulerPotEquations2D)
  # Indicator variable is rho
  return rho
end


# Convert conservative variables to indicator variable for discontinuities (pointwise version)
@inline function cons2indicator(rho, rho_v1, rho_v2, rho_θ, ::Val{:density_pressure},
                                equation::CompressibleEulerPotEquations2D)
  # Calculate pressure
  p = (equation.R_d * rho_θ / equation.p0^equation.κ)^(1 / (1 - equation.κ))

  # Indicator variable is rho * p
  return rho * p
end


# Convert conservative variables to indicator variable for discontinuities (pointwise version)
@inline function cons2indicator(rho, rho_v1, rho_v2, rho_θ, ::Val{:pressure},
                                equation::CompressibleEulerPotEquations2D)
  # Indicator variable is p
  return (equation.R_d * rho_θ / equation.p0^equation.κ)^(1 / (1 - equation.κ))
end


# Calculate thermodynamic entropy for a conservative state `cons`
@inline function entropy_thermodynamic(cons, equation::CompressibleEulerPotEquations2D)
  # Pressure
  p = (equation.R_d * cons[4] / equation.p0^equation.κ)^(1 / (1 - equation.κ))

  # Thermodynamic entropy
  s = log(p) - equation.gamma*log(cons[1])

  return s
end

# Calculate potential temperature for a conservative state `cons`
@inline function pottemp_thermodynamic(cons, equation::CompressibleEulerPotEquations2D)

  return cons[4] / cons[1]
end


# Calculate mathematical entropy for a conservative state `cons`
@inline function entropy_math(cons, equation::CompressibleEulerPotEquations2D)
  # Mathematical entropy
  S = -entropy_thermodynamic(cons, equation) * cons[1] / (equation.gamma - 1)

  return S
end


# Default entropy is the mathematical entropy
@inline entropy(cons, equation::CompressibleEulerPotEquations2D) = entropy_math(cons, equation)


# Calculate total energy for a conservative state `cons`
@inline energy_total(cons, ::CompressibleEulerPotEquations2D) = cons[4]


# Calculate kinetic energy for a conservative state `cons`
@inline function energy_kinetic(cons, equation::CompressibleEulerPotEquations2D)
  return 0.5 * (cons[2]^2 + cons[3]^2)/cons[1]
end


# Calculate internal energy for a conservative state `cons`
@inline function energy_internal(cons, equation::CompressibleEulerPotEquations2D)
  return energy_total(cons, equation) - energy_kinetic(cons, equation)
end
