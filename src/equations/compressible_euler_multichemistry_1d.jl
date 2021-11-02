using KROME
# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


@doc raw"""
    CompressibleEulerMultichemistryEquations1D(; gammas, gas_constants)
!!! warning "Experimental code"
    This system of equations is experimental and can change any time.
Multichemistry version of the compressible Euler equations
```math
\partial t
\begin{pmatrix}
\rho v_1 \\ E \\ \rho_1 \\ \rho_2 \\ \vdots \\ \rho_{n}
\end{pmatrix}
+
\partial x
\begin{pmatrix}
\rho v_1 \\ \rho v_1^2 + p \\ (E+p) v_1 \\ \rho_1 v_1 \\ \rho_2 v_1 \\ \vdots \\ \rho_{n} v_1
\end{pmatrix}
=
\begin{pmatrix}
0 \\ 0 \\ 0 \\ 0 \\ \vdots \\ 0
\end{pmatrix}
```
for calorically perfect gas in one space dimension.
In case of more than one component, the specific heat ratios `gammas` and the gas constants
`gas_constants` should be passed as tuples, e.g., `gammas=(1.4, 1.667)`.
The remaining variables like the specific heats at constant volume 'cv' or the specific heats at
constant pressure 'cp' are then calculated considering a calorically perfect gas.
"""
struct CompressibleEulerMultichemistryEquations1D{NVARS, NCOMP, RealT<:Real} <: AbstractCompressibleEulerMultichemistryEquations{1, NVARS, NCOMP}
  gammas                 ::SVector{NCOMP, RealT}
  gas_constants          ::SVector{NCOMP, RealT}
  cv                     ::SVector{NCOMP, RealT}
  cp                     ::SVector{NCOMP, RealT}
  heat_of_formations     ::SVector{NCOMP, RealT}

  function CompressibleEulerMultichemistryEquations1D{NVARS, NCOMP, RealT}(gammas                 ::SVector{NCOMP, RealT},
                                                                           gas_constants          ::SVector{NCOMP, RealT},
                                                                           heat_of_formations     ::SVector{NCOMP, RealT}) where {NVARS, NCOMP, RealT<:Real}

    NCOMP >= 1 || throw(DimensionMismatch("`gammas` and `gas_constants` have to be filled with at least one value"))

    cv = gas_constants ./ (gammas .- 1)
    cp = gas_constants + gas_constants ./ (gammas .- 1)

    new(gammas, gas_constants, cv, cp, heat_of_formations)
  end
end


function CompressibleEulerMultichemistryEquations1D(; gammas, gas_constants, heat_of_formations)

  _gammas                 = promote(gammas...)
  _gas_constants          = promote(gas_constants...)
  _heat_of_formations     = promote(heat_of_formations...)
  RealT                   = promote_type(eltype(_gammas), eltype(_gas_constants), eltype(_heat_of_formations))

  NVARS = length(_gammas) + 2
  NCOMP = length(_gammas)

  __gammas                = SVector(map(RealT, _gammas))
  __gas_constants         = SVector(map(RealT, _gas_constants))
  __heat_of_formations    = SVector(map(RealT, _heat_of_formations))

  return CompressibleEulerMultichemistryEquations1D{NVARS, NCOMP, RealT}(__gammas, __gas_constants, __heat_of_formations)
end


@inline Base.real(::CompressibleEulerMultichemistryEquations1D{NVARS, NCOMP, RealT}) where {NVARS, NCOMP, RealT} = RealT


function varnames(::typeof(cons2cons), equations::CompressibleEulerMultichemistryEquations1D)

  cons  = ("rho_v1", "rho_e")
  rhos  = ntuple(n -> "rho" * string(n), Val(ncomponents(equations)))
  return (cons..., rhos...)
end


function varnames(::typeof(cons2prim), equations::CompressibleEulerMultichemistryEquations1D)

  prim  = ("v1", "p")
  rhos  = ntuple(n -> "rho" * string(n), Val(ncomponents(equations)))
  return (prim..., rhos...)
end


# Set initial conditions at physical location `x` for time `t`

"""
    initial_condition_convergence_test(x, t, equations::CompressibleEulerMultichemistryEquations1D)
A smooth initial condition used for convergence tests in combination with
[`source_terms_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
function initial_condition_convergence_test(x, t, equations::CompressibleEulerMultichemistryEquations1D)
  c       = 2
  A       = 0.1
  L       = 2
  f       = 1/L
  omega   = 2 * pi * f
  ini     = c + A * sin(omega * (x[1] - t))

  v1      = 1.0

  rho     = ini

  # Here we compute an arbitrary number of different rhos. (one rho is double the next rho while the sum of all rhos is 1)
  prim_rho  = SVector{ncomponents(equations), real(equations)}(2^(i-1) * (1-2)/(1-2^ncomponents(equations)) * rho for i in eachcomponent(equations))

  prim1 = rho * v1
  prim2 = rho^2

  prim_other = SVector{2, real(equations)}(prim1, prim2)

  return vcat(prim_other, prim_rho)
end

"""
    source_terms_convergence_test(u, x, t, equations::CompressibleEulerMultichemistryEquations1D)
Source terms used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
@inline function source_terms_convergence_test(u, x, t, equations::CompressibleEulerMultichemistryEquations1D)
  # Same settings as in `initial_condition`
  c       = 2
  A       = 0.1
  L       = 2
  f       = 1/L
  omega   = 2 * pi * f

  gamma  = totalgamma(u, equations)

  x1,     = x
  si, co  = sincos((t - x1)*omega)
  tmp = (-((4 * si * A - 4c) + 1) * (gamma - 1) * co * A * omega) / 2

  # Here we compute an arbitrary number of different rhos. (one rho is double the next rho while the sum of all rhos is 1
  du_rho  = SVector{ncomponents(equations), real(equations)}(0.0 for i in eachcomponent(equations))

  du1 = tmp
  du2 = tmp

  du_other  = SVector{2, real(equations)}(du1, du2)

  return vcat(du_other, du_rho)
end


"""
    initial_condition_weak_blast_wave(x, t, equations::CompressibleEulerMultichemistryEquations1D)
A for Multichemistry adapted weak blast wave adapted to Multichemistry and taken from
- Sebastian Hennemann, Gregor J. Gassner (2020)
  A provably entropy stable subcell shock capturing approach for high order split form DG
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
function initial_condition_weak_blast_wave(x, t, equations::CompressibleEulerMultichemistryEquations1D)
  # From Hennemann & Gassner JCP paper 2020 (Sec. 6.3)
  inicenter         = SVector(0.0)
  x_norm            = x[1] - inicenter[1]
  r                 = abs(x_norm)
  cos_phi           = x_norm > 0 ? one(x_norm) : -one(x_norm)

  prim_rho          = SVector{ncomponents(equations), real(equations)}(r > 0.5 ? 2^(i-1) * (1-2)/(1-2^ncomponents(equations))*1.0 : 2^(i-1) * (1-2)/(1-2^ncomponents(equations))*1.1691 for i in eachcomponent(equations))

  v1                = r > 0.5 ? 0.0 : 0.1882 * cos_phi
  p                 = r > 0.5 ? 1.0 : 1.245

  prim_other         = SVector{2, real(equations)}(v1, p)

  return prim2cons(vcat(prim_other, prim_rho), equations)
end


function initial_condition_knallgas_detonation(x, t, equations::CompressibleEulerMultichemistryEquations1D)

  if x[1] <= 2.5
    v     = 8.0
    p     = 20.0  # T = 10
    H2    = 0.0
    O2    = 0.0
    H20   = 2.0
  else 
    v     = 0.0
    p     = 1.0   # T = 1
    H2    = 1/9
    O2    = 8/9
    H20   = 0.0
  end

  rho1  = H2
  rho2  = O2
  rho3  = H20 
  rho = rho1 + rho2 + rho3

  prim_rho    = SVector{3, real(equations)}(rho1, rho2, rho3)

  prim_other  = SVector{2, real(equations)}(v, p)

  return prim2cons(vcat(prim_other, prim_rho), equations)
end


function boundary_condition_knallgas_detonation(u_inner, orientation, direction, x, t,
                                                surface_flux_function, equations::CompressibleEulerMultichemistryEquations1D)

  u_boundary = initial_condition_knallgas_detonation(x, t, equations) # DIRICHLET
  #u_boundary = SVector{nvariables(equations), real(equations)}(u_inner[1], -u_inner[2], u_inner[3], u_inner[4], u_inner[5], u_inner[6]) # Reflect

  # Calculate boundary flux
  if direction == 2 # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
  end

  return flux
end


@inline function chemistry_knallgas_detonation(u, dt, equations::CompressibleEulerMultichemistryEquations1D)
  # Same settings as in `initial_condition`
  rho_v1, rho_e, H2, O2, H2O  = u
  @unpack heat_of_formations = equations     



  nmols = krome_nmols()[] # read Fortran module variable
  x = zeros(nmols) # default abundances (number density)

  idx_H2    = krome_idx_H2()[] + 1
  x[idx_H2] = H2/2.0
  idx_O2    = krome_idx_O2()[] + 1
  x[idx_O2] = O2/32.0
  idx_H2O   = krome_idx_H2O()[] + 1
  x[idx_H2O]= H2O/18.0
  
  rho   = H2 + O2 + H2O
  v1    = rho_v1/(rho)
  gamma = totalgamma(u, equations)

  p_chem = zero(rho)
  for i in eachcomponent(equations)                             
    p_chem += heat_of_formations[i] * u[i+2]                      
  end

  p     = (gamma - 1) * (rho_e - 0.5 * rho * v1^2 - p_chem)      

  Tgas  = fill(10.0 * (p/rho))

  dt = fill(dt) # time-step
  krome(x, Tgas, dt)

  x[idx_H2] = x[idx_H2]*2.0
  x[idx_O2] = x[idx_O2]*32.0
  x[idx_H2O]= x[idx_H2O]*18.0

  du_rho  = SVector{ncomponents(equations), real(equations)}(x[i]-u[i+2] for i in eachcomponent(equations))

  du_other  = SVector{2, real(equations)}(0.0, 0.0)

  return vcat(du_other, du_rho)
end


function initial_condition_knallgas_5_detonation(x, t, equations::CompressibleEulerMultichemistryEquations1D)

  if x[1] <= 0.5
    v     = 10.0
    p     = 40.0    # T = 10
    rho1  = 0.0
    rho2  = 0.0
    rho3  = 0.17 * 2.0
    rho4  = 0.63 * 2.0
    rho5  = 0.2  * 2.0
  else 
    v     = 0.0
    p     = 1.0
    rho1  = 0.08 * 1.0
    rho2  = 0.72 * 1.0
    rho3  = 0.0
    rho4  = 0.0
    rho5  = 0.2  * 1.0
  end

  prim_rho    = SVector{5, real(equations)}(rho1, rho2, rho3, rho4, rho5)

  prim_other  = SVector{2, real(equations)}(v, p)

  return prim2cons(vcat(prim_other, prim_rho), equations)
end


function boundary_condition_knallgas_5_detonation(u_inner, orientation, direction, x, t,
  surface_flux_function,
  equations::CompressibleEulerMultichemistryEquations1D)

  u_boundary = initial_condition_knallgas_5_detonation(x, t, equations) # DIRICHLET
  #u_boundary = SVector{nvariables(equations), real(equations)}(u_inner[1], -u_inner[2], u_inner[3], u_inner[4], u_inner[5], u_inner[6]) # Reflect

  # Calculate boundary flux
  if direction == 2 # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
  end

  return flux
end


@inline function chemistry_knallgas_5_detonation(u, dt, equations::CompressibleEulerMultichemistryEquations1D)
  # Same settings as in `initial_condition`
  rho_v1, rho_e, H2, O2, OH, H2O, N2  = u
  @unpack heat_of_formations = equations

  #krome_init() # init krome (mandatory)

  nmols = krome_nmols()[] # read Fortran module variable
  x = zeros(nmols) # default abundances (number density)

  rho = density(u, equations)

  idx_H2    = krome_idx_H2()[] + 1
  x[idx_H2] = H2/2.0
  idx_O2    = krome_idx_O2()[] + 1
  x[idx_O2] = O2/32.0
  idx_OH    = krome_idx_OH()[] + 1
  x[idx_OH] = OH/17.0
  idx_H2O   = krome_idx_H2O()[] + 1
  x[idx_H2O]= H2O/18.0
  
  v1    = rho_v1/rho
  gamma = totalgamma(u, equations)

  p_chem = zero(rho)
  for i in eachcomponent(equations)                                 # simplification!
    p_chem += heat_of_formations[i] * u[i+2]                      # simplification!
  end

  p     = (gamma - 1) * (rho_e - 0.5 * rho * v1^2 - p_chem)       # simplification!  
  
  Tgas  = fill(10.0 * (p/rho))
  dt = fill(dt) # time-step
  krome(x, Tgas, dt)

  x[idx_H2] = x[idx_H2]*2.0
  x[idx_O2] = x[idx_O2]*32.0
  x[idx_OH] = x[idx_OH]*17.0
  x[idx_H2O]= x[idx_H2O]*18.0
  
  chem = SVector{ncomponents(equations), real(equations)}(x[1], x[2], x[3], x[4], u[7])
  
  # Here we compute an arbitrary number of different rhos. (one rho is double the next rho while the sum of all rhos is 1
  du_rho  = SVector{ncomponents(equations), real(equations)}(chem[i]-u[i+2] for i in eachcomponent(equations))

  du_other  = SVector{2, real(equations)}(0.0, 0.0)

  return vcat(du_other, du_rho)

end



"""
    initial_condition_two_interacting_blast_waves(x, t, equations::CompressibleEulerMultichemistryEquations1D)
A Multichemistry two interacting blast wave test taken from
- T. Plewa & E. Müller (1999)
  The consistent multi-fluid advection method
  [arXiv: 9807241](https://arxiv.org/pdf/astro-ph/9807241.pdf)
"""
function initial_condition_two_interacting_blast_waves(x, t, equations::CompressibleEulerMultichemistryEquations1D)

  rho1        = 0.5 * x[1]^2
  rho2        = 0.5 * (sin(20 * x[1]))^2
  rho3        = 1 - rho1 - rho2

  prim_rho    = SVector{3, real(equations)}(rho1, rho2, rho3)

  v1          = 0.0

  if x[1] <= 0.1
    p = 1000
  elseif x[1] < 0.9
    p = 0.01
  else
    p = 100
  end

  prim_other  = SVector{2, real(equations)}(v1, p)

  return prim2cons(vcat(prim_other, prim_rho), equations)
end


function boundary_condition_two_interacting_blast_waves(u_inner, orientation, direction, x, t,
                                                        surface_flux_function,
                                                        equations::CompressibleEulerMultichemistryEquations1D)

u_inner_reflect = SVector{nvariables(equations), real(equations)}(-u_inner[1], u_inner[2], u_inner[3], u_inner[4], u_inner[5])
# Calculate boundary flux
if direction == 2 # u_inner is "left" of boundary, u_boundary is "right" of boundary
  flux = surface_flux_function(u_inner, u_inner_reflect, orientation, equations)
else # u_boundary is "left" of boundary, u_inner is "right" of boundary
  flux = surface_flux_function(u_inner_reflect, u_inner, orientation, equations)
end

  return flux
end


# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer, equations::CompressibleEulerMultichemistryEquations1D)
  rho_v1, rho_e  = u
  @unpack heat_of_formations = equations

  rho = density(u, equations)

  v1    = rho_v1/rho
  gamma = totalgamma(u, equations)
  
  p_chem = zero(rho)
  for i in eachcomponent(equations)                             
    p_chem += heat_of_formations[i] * u[i+2]                      
  end

  p     = (gamma - 1) * (rho_e - 0.5 * rho * v1^2 - p_chem)     

  f_rho = densities(u, v1, equations)
  f1  = rho_v1 * v1 + p
  f2  = (rho_e + p) * v1

  f_other  = SVector{2, real(equations)}(f1, f2)

  return vcat(f_other, f_rho)
end


"""
    flux_chandrashekar(u_ll, u_rr, orientation, equations::CompressibleEulerMultichemistryEquations1D)
Entropy conserving two-point flux by
- Ayoub Gouasmi, Karthik Duraisamy (2020)
  "Formulation of Entropy-Stable schemes for the Multichemistry compressible Euler equations""
  arXiv:1904.00972v3 [math.NA] 4 Feb 2020
"""
@inline function flux_chandrashekar(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerMultichemistryEquations1D)
  # Unpack left and right state
  @unpack gammas, gas_constants, cv, heat_of_formations = equations
  rho_v1_ll, rho_e_ll = u_ll
  rho_v1_rr, rho_e_rr = u_rr
  rhok_mean   = SVector{ncomponents(equations), real(equations)}(ln_mean(u_ll[i+2], u_rr[i+2]) for i in eachcomponent(equations))
  rhok_avg    = SVector{ncomponents(equations), real(equations)}(0.5 * (u_ll[i+2] + u_rr[i+2]) for i in eachcomponent(equations))

  # Iterating over all partial densities
  rho_ll      = density(u_ll, equations)
  rho_rr      = density(u_rr, equations)

  gamma_ll = totalgamma(u_ll, equations)
  gamma_rr = totalgamma(u_rr, equations)

  # extract velocities
  v1_ll       = rho_v1_ll/rho_ll
  v1_rr       = rho_v1_rr/rho_rr
  v1_avg      = 0.5 * (v1_ll + v1_rr)
  v1_square   = 0.5 * (v1_ll^2 + v1_rr^2)
  v_sum       = v1_avg

  enth      = zero(v_sum)
  help1_ll  = zero(v1_ll)
  help1_rr  = zero(v1_rr)

  for i in eachcomponent(equations)
    enth      += rhok_avg[i] * gas_constants[i]
    help1_ll  += u_ll[i+2] * cv[i]
    help1_rr  += u_rr[i+2] * cv[i]
  end

  T_ll        = (rho_e_ll - 0.5 * rho_ll * (v1_ll^2)) / help1_ll
  T_rr        = (rho_e_rr - 0.5 * rho_rr * (v1_rr^2)) / help1_rr
  T           = 0.5 * (1.0/T_ll + 1.0/T_rr)
  T_log       = ln_mean(1.0/T_ll, 1.0/T_rr)

  # Calculate fluxes depending on orientation
  help1       = zero(T_ll)
  help2       = zero(T_rr)

  f_rho       = SVector{ncomponents(equations), real(equations)}(rhok_mean[i]*v1_avg for i in eachcomponent(equations))
  for i in eachcomponent(equations)
    help1     += f_rho[i] * cv[i]
    help2     += f_rho[i]
  end
  f1 = (help2) * v1_avg + enth/T
  f2 = (help1)/T_log - 0.5 * (v1_square) * (help2) + v1_avg * f1

  f_other  = SVector{2, real(equations)}(f1, f2)

  return vcat(f_other, f_rho)
end


function flux_hllc(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerMultichemistryEquations1D)
  # Calculate primitive variables and speed of sound
  rho_v1_ll, rho_e_ll = u_ll
  rho_v1_rr, rho_e_rr = u_rr

  rho_ll = density(u_ll, equations)
  rho_rr = density(u_rr, equations)

  v1_ll = rho_v1_ll / rho_ll
  e_ll  = rho_e_ll / rho_ll
  p_ll = (equations.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * v1_ll^2)
  c_ll = sqrt(equations.gamma*p_ll/rho_ll)

  v1_rr = rho_v1_rr / rho_rr
  e_rr  = rho_e_rr / rho_rr
  p_rr = (equations.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * v1_rr^2 )
  c_rr = sqrt(equations.gamma*p_rr/rho_rr)

  # Obtain left and right fluxes
  f_ll = flux(u_ll, orientation, equations)
  f_rr = flux(u_rr, orientation, equations)

  # Compute Roe averages
  sqrt_rho_ll = sqrt(rho_ll)
  sqrt_rho_rr = sqrt(rho_rr)
  sum_sqrt_rho = sqrt_rho_ll + sqrt_rho_rr
  vel_L = v1_ll
  vel_R = v1_rr
  vel_roe = (sqrt_rho_ll * vel_L + sqrt_rho_rr * vel_R) / sum_sqrt_rho
  ekin_roe = 0.5 * vel_roe^2
  H_ll = (rho_e_ll + p_ll) / rho_ll
  H_rr = (rho_e_rr + p_rr) / rho_rr
  H_roe = (sqrt_rho_ll * H_ll + sqrt_rho_rr * H_rr) / sum_sqrt_rho
  c_roe = sqrt((equations.gamma - 1) * (H_roe - ekin_roe))

  Ssl = min(vel_L - c_ll, vel_roe - c_roe)
  Ssr = max(vel_R + c_rr, vel_roe + c_roe)
  sMu_L = Ssl - vel_L
  sMu_R = Ssr - vel_R
  if Ssl >= 0.0
    f1 = f_ll[1]
    f2 = f_ll[2]
    f3 = f_ll[3]
  elseif Ssr <= 0.0
    f1 = f_rr[1]
    f2 = f_rr[2]
    f3 = f_rr[3]
  else
    SStar = (p_rr - p_ll + rho_ll*vel_L*sMu_L - rho_rr*vel_R*sMu_R) / (rho_ll*sMu_L - rho_rr*sMu_R)
    if Ssl <= 0.0 <= SStar
      densStar = rho_ll*sMu_L / (Ssl-SStar)
      enerStar = e_ll + (SStar - vel_L) * (SStar + p_ll / (rho_ll * sMu_L))
      UStar1 = densStar
      UStar2 = densStar*SStar
      UStar3 = densStar*enerStar

      f1 = f_ll[1]+Ssl*(UStar1 - rho_ll)
      f2 = f_ll[2]+Ssl*(UStar2 - rho_v1_ll)
      f3 = f_ll[3]+Ssl*(UStar3 - rho_e_ll)
    else
      densStar = rho_rr*sMu_R / (Ssr-SStar)
      enerStar = e_rr + (SStar - vel_R) * (SStar + p_rr / (rho_rr * sMu_R))
      UStar1 = densStar
      UStar2 = densStar*SStar
      UStar3 = densStar*enerStar

      #end
      f1 = f_rr[1]+Ssr*(UStar1 - rho_rr)
      f2 = f_rr[2]+Ssr*(UStar2 - rho_v1_rr)
      f3 = f_rr[3]+Ssr*(UStar3 - rho_e_rr)
    end
  end
  return SVector(f1, f2, f3)
end


# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerMultichemistryEquations1D)
  rho_v1_ll, rho_e_ll = u_ll
  rho_v1_rr, rho_e_rr = u_rr
  @unpack heat_of_formations = equations

  # Calculate primitive variables and speed of sound
  rho_ll   = density(u_ll, equations)
  rho_rr   = density(u_rr, equations)
  gamma_ll = totalgamma(u_ll, equations)
  gamma_rr = totalgamma(u_rr, equations)

  v1_ll = rho_v1_ll / rho_ll
  v1_rr = rho_v1_rr / rho_rr
  v_mag_ll = sqrt(v1_ll^2)
  v_mag_rr = sqrt(v1_rr^2)

  p_chem_ll = zero(rho_ll)
  p_chem_rr = zero(rho_rr)
  for i in eachcomponent(equations)                                             # simplification!
    p_chem_ll += heat_of_formations[i] * u_ll[i+2]                            # simplification!
    p_chem_rr += heat_of_formations[i] * u_rr[i+2]                            # simplification!
  end                                                                         # simplification!
 
  p_ll      = (gamma_ll - 1) * (rho_e_ll - 0.5 * rho_ll * v1_ll^2 - p_chem_ll) # simplification!
  p_rr      = (gamma_rr - 1) * (rho_e_rr - 0.5 * rho_rr * v1_rr^2 - p_chem_rr) # simplification!

  c_ll = sqrt(gamma_ll * p_ll / rho_ll)
  c_rr = sqrt(gamma_rr * p_rr / rho_rr)

  λ_max = max(v_mag_ll, v_mag_rr) + max(c_ll, c_rr)
end


@inline function max_abs_speeds(u, equations::CompressibleEulerMultichemistryEquations1D)
  rho_v1, rho_e = u
  @unpack heat_of_formations = equations

  rho   = density(u, equations)
  v1    = rho_v1 / rho

  gamma = totalgamma(u, equations)
  
  p_chem = zero(rho)
  for i in eachcomponent(equations)                                 # simplification!
    p_chem += heat_of_formations[i] * u[i+2]                      # simplification!
  end                                                             # simplification!

  p     = (gamma - 1) * (rho_e - 0.5 * rho * v1^2 - p_chem)       # simplification!

  c     = sqrt(gamma * p / rho)

  return (abs(v1) + c, )
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::CompressibleEulerMultichemistryEquations1D)
  rho_v1, rho_e = u
  @unpack heat_of_formations = equations

  prim_rho = SVector{ncomponents(equations), real(equations)}(u[i+2] for i in eachcomponent(equations))

  rho   = density(u, equations)
  v1    = rho_v1 / rho
  gamma = totalgamma(u, equations)

  p_chem = zero(rho)
  for i in eachcomponent(equations)                                 # simplification!
    p_chem += heat_of_formations[i] * u[i+2]                      # simplification!
  end                                                             # simplification!

  p     = (gamma - 1) * (rho_e - 0.5 * rho * v1^2 - p_chem)       # simplification!
  
  prim_other =  SVector{2, real(equations)}(v1, p)

  return vcat(prim_other, prim_rho)
end


# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::CompressibleEulerMultichemistryEquations1D)
  @unpack cv, gammas, heat_of_formations = equations
  v1, p = prim

  RealT = eltype(prim)

  cons_rho = SVector{ncomponents(equations), RealT}(prim[i+2] for i in eachcomponent(equations))
  rho     = density(prim, equations)
  gamma   = totalgamma(prim, equations)

  rho_v1  = rho * v1

  p_chem = zero(rho)
  for i in eachcomponent(equations)                                 # simplification!
    p_chem += heat_of_formations[i] * prim[i+2]                   # simplification!
  end      

  rho_e     = p/(gamma - 1) + 0.5 * rho * v1^2 + p_chem           # simplification!

  cons_other = SVector{2, RealT}(rho_v1, rho_e)

  return vcat(cons_other, cons_rho)
end


# Convert conservative variables to entropy
@inline function cons2entropy(u, equations::CompressibleEulerMultichemistryEquations1D)
  @unpack cv, gammas, gas_constants, heat_of_formations = equations
  rho_v1, rho_e = u

  rho       = density(u, equations)

  v1        = rho_v1 / rho
  v_square  = v1^2
  gamma     = totalgamma(u, equations)

  p_chem = zero(rho)
  for i in eachcomponent(equations)                                 # simplification!
    p_chem += heat_of_formations[i] * u[i+2]                      # simplification!
  end                                                             # simplification!

  p     = (gamma - 1) * (rho_e - 0.5 * rho * v1^2 - p_chem)       # simplification!
  
  s         = log(p) - gamma * log(rho)
  rho_p     = rho / p

  # Multichemistry stuff
  help1 = zero(v1)

  for i in eachcomponent(equations)
    help1 += u[i+2] * cv[i]
  end

  T         = (rho_e - 0.5 * rho * v_square) / (help1)

  entrop_rho  = SVector{ncomponents(equations), real(equations)}(i for i in eachcomponent(equations)) #-1.0 * (cv[i] * log(T) - gas_constants[i] * log(u[i+2])) + gas_constants[i] + cv[i] - (v_square / (2*T)) for i in eachcomponent(equations))

  w1        = v1/T
  w2        = -1.0/T

  entrop_other = SVector{2, real(equations)}(w1, w2)

  return vcat(entrop_other, entrop_rho)
end


"""
    totalgamma(u, equations::CompressibleEulerMultichemistryEquations1D)
Function that calculates the total gamma out of all partial gammas using the
partial density fractions as well as the partial specific heats at constant volume.
"""
@inline function totalgamma(u, equations::CompressibleEulerMultichemistryEquations1D)
  @unpack cv, gammas = equations
  

  help1 = zero(u[1])
  help2 = zero(u[1])

  for i in eachcomponent(equations)
    help1 += u[i+2] * cv[i] * gammas[i]
    help2 += u[i+2] * cv[i]
  end

  return help1/help2
end


@inline function pressure(u, equations::CompressibleEulerMultichemistryEquations1D)
  rho_v1, rho_e = u
  @unpack heat_of_formations = equations

  rho          = density(u, equations)
  gamma        = totalgamma(u, equations)

  p_chem = zero(rho)
  for i in eachcomponent(equations)                                         # simplification!
    p_chem += heat_of_formations[i] * u[i+2]                              # simplification!
  end                                                                     # simplification!

  p  = (gamma - 1) * (rho_e - 0.5 * (rho_v1^2/rho) - p_chem)                  # simplification!

  return p
end


@inline function density(u, equations::CompressibleEulerMultichemistryEquations1D)
  rho = zero(u[1])

  for i in eachcomponent(equations)
    rho += u[i+2]
  end

  return rho
 end


 @inline function density_pressure(u, equations::CompressibleEulerMultichemistryEquations1D)
  rho_v1, rho_e = u
  @unpack heat_of_formations = equations

  rho          = density(u, equations)
  gamma        = totalgamma(u, equations)

  v1  = rho_v1 / rho

  p_chem = zero(rho)
  for i in eachcomponent(equations)                                         # simplification!
    p_chem += heat_of_formations[i] * u[i+2]                              # simplification!
  end                                                                     # simplification!

  p  = (gamma - 1) * (rho_e - 0.5 * (v1^2) - p_chem)                  # simplification!

  return p*rho
end


 @inline function densities(u, v, equations::CompressibleEulerMultichemistryEquations1D)

  return SVector{ncomponents(equations), real(equations)}(u[i+2]*v for i in eachcomponent(equations))
 end


end # @muladd