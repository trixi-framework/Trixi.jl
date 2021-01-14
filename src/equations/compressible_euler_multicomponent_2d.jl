
@doc raw"""
    CompressibleEulerMulticomponentEquations2D

!!! warning "Experimental code"
    This system of equations is experimental and can change any time.

Multicomponent version of the compressible Euler equations for an ideal gas in two space dimensions.
"""

struct CompressibleEulerMulticomponentEquations2D{NCOMP, NVARS} <: AbstractCompressibleEulerMulticomponentEquations{2, NVARS}
  gamma             ::SVector{NCOMP, Real} 
  gas_constant      ::SVector{NCOMP, Real}
  cv                ::SVector{NCOMP, Real}
  cp                ::SVector{NCOMP, Real}
  CompressibleEulerMulticomponentEquations2D{NCOMP, NVARS}(gamma, gas_constant) where {NCOMP, NVARS} = new(gamma, gas_constant, gas_constant./(gamma.-1), gas_constant + gas_constant./(gamma .-1))
end


function CompressibleEulerMulticomponentEquations2D(; gamma::AbstractVector{<:Real}, gas_constant::AbstractVector{<:Real})
                                                      
  # Set specific heat for a constant volume per species
  length(gamma) == length(gas_constant) || throw(DimensionMismatch())
  NVARS = length(gamma) + 3
  NCOMP = length(gamma)

  return CompressibleEulerMulticomponentEquations2D{NCOMP, NVARS}(gamma, gas_constant)
end


get_name(::CompressibleEulerMulticomponentEquations2D) = "CompressibleEulerMulticomponentEquations2D"

#cons  = SVector{3, String}
#prim  = SVector{3, String}

#for i in 1:3
# 
#  
# 
#end

# Option1: somehow append this vector, Option2: make a giant vector and let it be as it is (not efficient)

 varnames(::typeof(cons2cons), ::CompressibleEulerMulticomponentEquations2D) = @SVector ["rho_v1", "rho_v2", "rho_e", "rho1", "rho2", "rho3"]
 varnames(::typeof(cons2prim), ::CompressibleEulerMulticomponentEquations2D) = @SVector ["v1", "v2", "p", "rho1", "rho2", "rho3"]

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
  omega   = 2 * pi * f
  ini     = c + A * sin(omega * (x[1] + x[2] - t))

  v1      = 1.0
  v2      = 1.0

  rho     = ini
  rho1    = 0.2 * rho
  rho2    = 0.8 * rho  
  rho_v1  = rho * v1
  rho_v2  = rho * v2
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
  @unpack cv, gamma = equations
  # Same settings as in `initial_condition`
  c       = 2
  A       = 0.1
  L       = 2
  f       = 1/L
  omega   = 2 * pi * f

  rho1    = u[1]
  rho2    = u[2]

  gammas  = getgamma(u, equations)

  x1, x2  = x
  si, co  = sincos((x1 + x2 - t)*omega)
  tmp1    = co * A * omega
  tmp2    = si * A
  tmp3    = gammas - 1
  tmp4    = (2*c - 1)*tmp3
  tmp5    = (2*tmp2*gammas - 2*tmp2 + tmp4 + 1)*tmp1
  tmp6    = tmp2 + c

  du1     = 0.2 * tmp1
  du2     = 0.8 * tmp1
  du3     = tmp5
  du4     = tmp5
  du5     = 2*((tmp6 - 1.0)*tmp3 + tmp6*gammas)*tmp1

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


"""
    initial_condition_shock_bubble(x, t, equations::CompressibleEulerMulticomponentEquations2D)

A shock-bubble testcase for multicomponent Euler equations
- Ayoub Gouasmi, Karthik Duraisamy, Scott Murman
  Formulation of Entropy-Stable schemes for the multicomponent compressible Euler equations
  [arXiv: 1904.00972](https://arxiv.org/abs/1904.00972)
"""
function initial_condition_shock_bubble(x, t, equations::CompressibleEulerMulticomponentEquations2D)
  # bubble test case, see Agertz et al. https://arxiv.org/pdf/1904.00972
  # other reference: https://www.researchgate.net/profile/Pep_Mulet/publication/222675930_A_flux-split_algorithm_applied_to_conservative_models_for_multicomponent_compressible_flows/links/568da54508aeaa1481ae7af0.pdf
  # typical domain is rectangular, we change it to a square, as Trixi can only do squares
  @unpack gas_constant = equations

  # Positivity Preserving Parameter, can be set to zero if scheme is positivity preserving
  delta   = 0.03

  # Region I 
  rho1_1  = delta
  rho2_1  = 1.225 * gas_constant[1]/gas_constant[2] - delta
  v1_1    = 0.0
  v2_1    = 0.0
  p_1     = 101325

  # Region II 
  rho1_2  = 1.225-delta
  rho2_2  = delta
  v1_2    = 0.0
  v2_2    = 0.0
  p_2     = 101325

  # Region III 
  rho1_3  = 1.6861 - delta
  rho2_3  = delta
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

  return prim2cons(SVector(v1, v2, p, rho1, rho2), equations)
end


"""
    initial_condition_weak_blast_wave(x, t, equations::CompressibleEulerMulticomponentEquations2D)

A for multicomponent adapted weak blast wave taken from
- Sebastian Hennemann, Gregor J. Gassner (2020)
  A provably entropy stable subcell shock capturing approach for high order split form DG
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
function initial_condition_weak_blast_wave(x, t, equations::CompressibleEulerMulticomponentEquations2D)
  # From Hennemann & Gassner JCP paper 2020 (Sec. 6.3)
  @unpack gamma, cv, gas_constant, cp = equations
  prim              = zeros(length(gamma)+3)
  # Set up polar coordinates
  inicenter         = SVector(0.0, 0.0)
  x_norm            = x[1] - inicenter[1]
  y_norm            = x[2] - inicenter[2]
  r                 = sqrt(x_norm^2 + y_norm^2)
  phi               = atan(y_norm, x_norm)
  sin_phi, cos_phi  = sincos(phi)

  # Calculate primitive variables.    !!! NOTE: EC flux does not transfer mass across an interface separating two different species
  #rho1              = r > 0.5 ? 0.2*1.0 : 0.2*1.1691
  #rho2              = r > 0.5 ? 0.8*1.0 : 0.8*1.1691
  #rho               = rho1 + rho2

  #Three Components
  rho1              = r > 0.5 ? 0.2*1.0 : 0.2*1.1691
  rho2              = r > 0.5 ? 0.4*1.0 : 0.4*1.1691
  rho3              = r > 0.5 ? 0.3*1.0 : 0.3*1.1691
  v1                = r > 0.5 ? 0.0 : 0.1882 * cos_phi
  v2                = r > 0.5 ? 0.0 : 0.1882 * sin_phi
  p                 = r > 0.5 ? 1.0 : 1.245

  prim              = [v1, v2, p, rho1, rho2, rho3]

  result            = SVector{length(gamma)+3}(prim)

  return prim2cons(result, equations)
end


# Calculate 1D flux for a single point
@inline function calcflux(u, orientation, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack gamma, cv = equations
  rho_v1, rho_v2, rho_e  = u
  f         = zeros(length(gamma)+3)
  rho       = 0

  for i = 1:length(gamma)
    rho += u[i+3]
  end

  v1    = rho_v1/rho
  v2    = rho_v2/rho
  gammas= getgamma(u, equations)
  
  #p     = getpressure(u, gammas, equations)
  p     = (gammas - 1) * (rho_e - 0.5 * rho * (v1^2 + v2^2))



  if orientation == 1

    for i = 1:length(gamma)
      f[i+3] = u[i+3] * v1
    end

    f1  = rho_v1 * v1 + p
    f2  = rho_v2 * v1
    f3  = (rho_e + p) * v1
  else

    for i = 1:length(gamma)
      f[i+3] = u[i+3] * v2
    end

    f1  = rho_v1 * v2
    f2  = rho_v2 * v2 + p
    f3  = (rho_e + p) * v2
  end

  f[1:3]  = [f1, f2, f3]

  result  = SVector{length(gamma)+3}(f)

  return result
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
  @unpack gamma, gas_constant, cv = equations
  rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr
  rho_ll = 0
  rho_rr = 0
  rhok_mean   = zeros(length(gamma))
  rhok_avg    = zeros(length(gamma))
  f           = zeros(length(gamma)+3)

  # helpful primitive variables
  for i = 1:length(gamma)
    rho_ll += u_ll[i+3]
    rho_rr += u_rr[i+3]
    rhok_mean[i]  = ln_mean(u_ll[i+3], u_rr[i+3])
    rhok_avg[i]   = 0.5 * (u_ll[i+3] + u_rr[i+3])
  end 
  #rho_ll      = rho1_ll + rho2_ll
  #rho_rr      = rho1_rr + rho2_rr
  v1_ll       = rho_v1_ll/rho_ll
  v2_ll       = rho_v2_ll/rho_ll
  v1_rr       = rho_v1_rr/rho_rr
  v2_rr       = rho_v2_rr/rho_rr

  # helpful mean values
  #rho1_mean   = ln_mean(rho1_ll,rho1_rr)
  #rho2_mean   = ln_mean(rho2_ll,rho2_rr) 
  #rho1_avg    = 0.5 * (rho1_ll + rho1_rr)
  #rho2_avg    = 0.5 * (rho2_ll + rho2_rr)
  v1_avg      = 0.5 * (v1_ll + v1_rr)
  v2_avg      = 0.5 * (v2_ll + v2_rr)
  v1_square   = 0.5 * (v1_ll^2 + v1_rr^2)
  v2_square   = 0.5 * (v2_ll^2 + v2_rr^2)
  v_sum       = v1_avg + v2_avg

  # multicomponent specific values
  enth      = 0
  help1_ll  = 0
  help1_rr  = 0
  for i = 1:length(gamma)
    enth      += rhok_avg[i] * gas_constant[i] 
    help1_ll  += u_ll[i+3] * cv[i] 
    help1_rr  += u_rr[i+3] * cv[i]
  end
  #enth        = rho1_avg * gas_constant[1] + rho2_avg * gas_constant[2]
  T_ll        = (rho_e_ll - 0.5 * rho_ll * (v1_ll^2 + v2_ll^2)) / help1_ll #(rho1_ll * cv[1] + rho2_ll * cv[2])
  T_rr        = (rho_e_rr - 0.5 * rho_rr * (v1_rr^2 + v2_rr^2)) / help1_rr #(rho1_rr * cv[1] + rho2_rr * cv[2])
  T           = 0.5 * (1.0/T_ll + 1.0/T_rr)
  T_log       = ln_mean(1.0/T_ll, 1.0/T_rr)

  # Calculate fluxes depending on orientation
  help1       = 0
  help2       = 0
  if orientation == 1
    #f4 = rho1_mean * v1_avg
    #f5 = rho2_mean * v1_avg
    for i = 1:length(gamma)
      f[i+3]    = rhok_mean[i] * v1_avg
      help1     += f[i+3] * cv[i] 
      help2     += f[i+3]
    end
    f1 = (help2) * v1_avg + enth/T 
    f2 = (help2) * v2_avg
    f3 = (help1)/T_log - 0.5 * (v1_square + v2_square) * (help2) + v1_avg * f1 + v2_avg * f2
  else
    #f4 = rho1_mean * v2_avg
    #f5 = rho2_mean * v2_avg
    for i = 1:length(gamma)
      f[i+3]    = rhok_mean[i] * v2_avg
      help1     += f[i+3] * cv[i] 
      help2     += f[i+3]
    end
    f1 = (help2) * v1_avg 
    f2 = (help2) * v2_avg + enth/T
    f3 = (help1)/T_log - 0.5 * (v1_square + v2_square) * (help2) + v1_avg * f1 + v2_avg * f2
  end

  f[1:3]  = [f1, f2, f3]

  result  = SVector{length(gamma)+3}(f)

  return result
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
  @unpack gas_constant, cv, gamma = equations
  f = SVector{length(gamma)+3, Float64}
  rho_v1_ll, rho_v2_ll, rho_e_ll, rho1_ll, rho2_ll = u_ll
  rho_v1_rr, rho_v2_rr, rho_e_rr, rho1_rr, rho2_rr = u_rr
  rho_ll = 0
  rho_rr = 0

  # helpful primitive variables
  for i = 1:length(gamma)
    rho_ll += u_ll[i+3]
    rho_rr += u_rr[i+3]
  end 
  #rho_ll = rho1_ll + rho2_ll
  #rho_rr = rho1_rr + rho2_rr
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
  enth      = rho1_avg * gas_constant[1] + rho2_avg * gas_constant[22]
  T_ll      = (rho_e_ll - 0.5 * rho_ll * (v1_ll^2 + v2_ll^2)) / (rho1_ll * cv[1] + rho2_ll * cv[2])
  T_rr      = (rho_e_rr - 0.5 * rho_rr * (v1_rr^2 + v2_rr^2)) / (rho1_rr * cv[1] + rho2_rr * cv[2])
  T         = 0.5 * (T_ll + T_rr)
  T_log     = ln_mean(1.0/T_ll, 1.0/T_rr)

  # Not adapted yet
  gammas    = (rho1_avg*cv[1]*gamma[1] + rho2_avg*cv[2]*gamma[2]) / (rho1_avg*cv[1] + rho2_avg*cv[2])

  # Calculate dissipation operator to be ES 
  #R_x       = zeros(Float64, (5, 5))
  #R_y       = zeros(Float64, (5, 5))
  #Lambda_x  = zeros(Float64, (5, 5))
  #Lambda_y  = zeros(Float64, (5, 5))
  #Tau       = zeros(Float64, (5, 5))
  #W         = zeros(Float64, (5))
  #Dv_x      = zeros(Float64, (5))
  #Dv_y      = zeros(Float64, (5))


  # Help variables
  e1    = cv[1] * T
  e2    = cv[2] * T
  r     = (rho1_avg / rho_avg) * gas_constant[1] + (rho2_avg / rho_avg) * gas_constant[2]
  a     = sqrt(gammas * T * r) 
  k     = 0.5 * (v1_square + v2_square) 
  h1    = e1 + gas_constant[1] * T 
  h2    = e2 + gas_constant[2] * T 
  h     = (rho1_avg / rho_avg) * h1 + (rho2_avg / rho_avg) * h2 
  d1    = h1 - gammas * e1 
  d2    = h2 - gammas * e2 
  ht    = h + k
  tauh  = sqrt(rho_avg/(gammas * r))

  # -------------------------------- #

  R_x11  = 1.0 
  R_x12  = 0.0 
  R_x13  = 0.0 
  R_x14  = rho1_avg / rho_avg 
  R_x15  = R_x14

  R_x21  = 0.0 
  R_x22  = 1.0 
  R_x23  = 0.0 
  R_x24  = rho2_avg / rho_avg 
  R_x25  = R_x24

  R_x31  = v1_avg 
  R_x32  = v1_avg 
  R_x33  = 0.0 
  R_x34  = v1_avg + a 
  R_x35  = v1_avg - a 

  R_x41  = v2_avg 
  R_x42  = v2_avg 
  R_x43  = a 
  R_x44  = v2_avg 
  R_x45  = v2_avg 

  R_x51  = k - (d1/(gammas - 1.0)) 
  R_x52  = k - (d2/(gammas - 1.0)) 
  R_x53  = a * v2_avg 
  R_x54  = ht + a * v1_avg 
  R_x55  = ht - a * v1_avg 

  # -------------------------------- #

  R_y11  = 1.0 
  R_y12  = 0.0 
  R_y13  = 0.0 
  R_y14  = rho1_avg / rho_avg 
  R_y15  = R_y14 

  R_y21  = 0.0 
  R_y22  = 1.0 
  R_y23  = 0.0 
  R_y24  = rho2_avg / rho_avg
  R_y25  = R_y24 

  R_y31  = v1_avg 
  R_y32  = v1_avg 
  R_y33  = -1.0 * a 
  R_y34  = v1_avg 
  R_y35  = v1_avg 

  R_y41  = v2_avg 
  R_y42  = v2_avg 
  R_y43  = 0.0 
  R_y44  = v2_avg + a 
  R_y45  = v2_avg - a 

  R_y51  = k - (d1/(gammas - 1)) 
  R_y52  = k - (d2/(gammas - 1)) 
  R_y53  = -1.0 * a * v1_avg 
  R_y54  = ht + a * v2_avg 
  R_y55  = ht - a * v2_avg 

  # -------------------------------- #

  Lambda_x11 = abs(v1_avg) 
  Lambda_x22 = abs(v1_avg) 
  Lambda_x33 = abs(v1_avg) 
  Lambda_x44 = abs(v1_avg + a) 
  Lambda_x55 = abs(v1_avg - a) 

  # -------------------------------- #

  Lambda_y11 = abs(v2_avg)
  Lambda_y22 = abs(v2_avg) 
  Lambda_y33 = abs(v2_avg) 
  Lambda_y44 = abs(v2_avg + a) 
  Lambda_y55 = abs(v2_avg - a) 

  # -------------------------------- #

  Tau11  = tauh * (-1.0 * sqrt((rho1_avg/rho_avg) * (rho2_avg/rho_avg)) * sqrt(gammas * (gas_constant[2] / gas_constant[1])))
  Tau12  = tauh * ((rho1_avg / rho_avg) * sqrt(gammas - 1.0)) 
  Tau13  = 0.0 
  Tau14  = 0.0 
  Tau15  = 0.0 

  Tau21  = tauh * (sqrt((rho1_avg/rho_avg) * (rho2_avg/rho_avg)) * sqrt(gammas * (gas_constant[1] / gas_constant[2])))
  Tau22  = tauh * ((rho2_avg / rho_avg) * sqrt(gammas - 1.0)) 
  Tau23  = 0.0 
  Tau24  = 0.0 
  Tau25  = 0.0 

  Tau31  = 0.0
  Tau32  = 0.0
  Tau33  = tauh / a
  Tau34  = 0.0
  Tau35  = 0.0

  Tau41  = 0.0 
  Tau42  = 0.0 
  Tau43  = 0.0 
  Tau44  = tauh / sqrt(2) 
  Tau45  = 0.0 

  Tau51  = 0.0 
  Tau52  = 0.0 
  Tau53  = 0.0 
  Tau54  = 0.0 
  Tau55  = tauh / sqrt(2) 

  # -------------------------------- #

  R_x       = SMatrix{5,5}([R_x11 R_x12 R_x13 R_x14 R_x15 ; R_x21 R_x22 R_x23 R_x24 R_x25 ; R_x31 R_x32 R_x33 R_x34 R_x35 ; R_x41 R_x42 R_x43 R_x44 R_x45 ; R_x51 R_x52 R_x53 R_x54 R_x55])
  R_y       = SMatrix{5,5}([R_y11 R_y12 R_y13 R_y14 R_y15 ; R_y21 R_y22 R_y23 R_y24 R_y25 ; R_y31 R_y32 R_y33 R_y34 R_y35 ; R_y41 R_y42 R_y43 R_y44 R_y45 ; R_y51 R_y52 R_y53 R_y54 R_y55])
  Lambda_x  = SMatrix{5,5}([Lambda_x11 0.0 0.0 0.0 0.0    ; 0.0 Lambda_x22 0.0 0.0 0.0    ; 0.0 0.0 Lambda_x33 0.0 0.0    ; 0.0 0.0 0.0 Lambda_x44 0.0    ; 0.0 0.0 0.0 0.0 Lambda_x55])
  Lambda_y  = SMatrix{5,5}([Lambda_y11 0.0 0.0 0.0 0.0    ; 0.0 Lambda_y22 0.0 0.0 0.0    ; 0.0 0.0 Lambda_y33 0.0 0.0    ; 0.0 0.0 0.0 Lambda_y44 0.0    ; 0.0 0.0 0.0 0.0 Lambda_y55])
  Tau       = SMatrix{5,5}([Tau11 Tau12 Tau13 Tau14 Tau15 ; Tau21 Tau22 Tau23 Tau24 Tau25 ; Tau31 Tau32 Tau33 Tau34 Tau35 ; Tau41 Tau42 Tau43 Tau44 Tau45 ; Tau51 Tau52 Tau53 Tau54 Tau55])
  W         = SVector{5, Float64}
  Dv_x      = SVector{5, Float64}
  Dv_y      = SVector{5, Float64}

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
    f4 = (rho1_mean * v1_avg) - Dv_x[1]
    f5 = (rho2_mean * v1_avg) - Dv_x[2]
    f1 = ((f1 + f2) * v1_avg + enth * T) - Dv_x[3]
    f2 = ((f1 + f2) * v2_avg) - Dv_x[4]
    f3 = ((f1 * cv[1] + f2 * cv[2])/T_log - 0.5 * (v1_square + v2_square) * (f1 + f2) + v1_avg * f3 + v2_avg * f4) - Dv_x[5]
  else
    f4 = (rho1_mean * v2_avg) - Dv_y[1]
    f5 = (rho2_mean * v2_avg) - Dv_y[2]
    f1 = ((f1 + f2) * v1_avg) - Dv_y[3] 
    f2 = ((f1 + f2) * v2_avg + enth * T) - Dv_y[4]
    f3 = ((f1 * cv[1] + f2 * cv[2])/T_log - 0.5 * (v1_square + v2_square) * (f1 + f2) + v1_avg * f3 + v2_avg * f4) - Dv_y[5]
  end

  return SVector(f1, f2, f3, f4, f5)
end


function flux_lax_friedrichs(u_ll, u_rr, orientation, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack cv, gamma = equations
  f   = zeros(length(gamma)+3)
  # Calculate primitive variables and speed of sound
  rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr
  rho_ll = 0
  rho_rr = 0

  for i = 1:length(gamma)
    rho_ll += u_ll[i+3]
    rho_rr += u_rr[i+3]
  end 

  gamma_ll = getgamma(u_ll, equations)
  gamma_rr = getgamma(u_rr, equations)

  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v_mag_ll = sqrt(v1_ll^2 + v2_ll^2)

  p_ll = (gamma_ll - 1) * (rho_e_ll - 1/2 * rho_ll * v_mag_ll^2)
  #p_ll  = getpressure(u_ll, gamma_ll, equations)

  c_ll = sqrt(gamma_ll * p_ll / rho_ll)
  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  v_mag_rr = sqrt(v1_rr^2 + v2_rr^2)
  #p_rr  = getpressure(u_rr, gamma_rr, equations)
  p_rr = (gamma_rr - 1) * (rho_e_rr - 1/2 * rho_rr * v_mag_rr^2)
  c_rr = sqrt(gamma_rr * p_rr / rho_rr)

  # Obtain left and right fluxes
  f_ll = calcflux(u_ll, orientation, equations)
  f_rr = calcflux(u_rr, orientation, equations)

  λ_max = max(v_mag_ll, v_mag_rr) + max(c_ll, c_rr)
  #f1 = 1/2 * (f_ll[1] + f_rr[1]) - 1/2 * λ_max * (rho1_rr   - rho1_ll)
  #f2 = 1/2 * (f_ll[2] + f_rr[2]) - 1/2 * λ_max * (rho2_rr   - rho2_ll)
  f1 = 1/2 * (f_ll[3] + f_rr[3]) - 1/2 * λ_max * (rho_v1_rr - rho_v1_ll)
  f2 = 1/2 * (f_ll[4] + f_rr[4]) - 1/2 * λ_max * (rho_v2_rr - rho_v2_ll)
  f3 = 1/2 * (f_ll[5] + f_rr[5]) - 1/2 * λ_max * (rho_e_rr  - rho_e_ll)

  f[1:3]  = [f1, f2, f3]
  
  for i = 1:length(gamma)
    f[i+3]  = 1/2 * (f_ll[i+3] + f_rr[i+3]) - 1/2 * λ_max * (u_rr[i+3] - u_ll[i+3])
  end

  result  = SVector{length(gamma)+3}(f)

  return result
end


@inline function max_abs_speeds(u, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack cv, gamma = equations
  rho_v1, rho_v2, rho_e = u
  rho   = 0

  for i = 1:length(gamma)
    rho += u[i+3]
  end  

  v1    = rho_v1 / rho
  v2    = rho_v2 / rho

  gammas= getgamma(u, equations)
  #p     = getpressure(u, gammas, equations)
  p     = (gammas - 1) * (rho_e - 1/2 * rho * (v1^2 + v2^2))
  c     = sqrt(gammas * p / rho)

  return abs(v1) + c, abs(v2) + c
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack cv, gamma = equations
  rho_v1, rho_v2, rho_e = u
  prim  = zeros(length(gamma)+3)
  rho   = 0

  for i = 1:length(gamma)
    rho += u[i+3]
    prim[i+3] = u[i+3]
  end 

  v1    = rho_v1 / rho
  v2    = rho_v2 / rho
  gammas= getgamma(u, equations)
  #p     = getpressure(u, gammas, equations)
  p     = (gammas - 1) * (rho_e - 0.5 * rho * (v1^2 + v2^2))
  prim[1:3] = [v1, v2, p]

  result    = SVector{length(gamma)+3}(prim)

  return result
end


# Convert conservative variables to entropy
@inline function cons2entropy(u, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack cv, gamma, gas_constant = equations
  rho_v1, rho_v2, rho_e = u
  rho       = 0

  entrop  = zeros(length(gamma)+3)

  for i = 1:length(gamma)
    rho += u[i+3]
  end 

  v1        = rho_v1 / rho
  v2        = rho_v2 / rho
  v_square  = v1^2 + v2^2
  gammas    = getgamma(u, equations)
  #p         = getpressure(u, gammas, equations)
  p         = (gammas - 1) * (rho_e - 0.5 * rho * v_square)
  s         = log(p) - gammas*log(rho)
  rho_p     = rho / p

  # Multicomponent stuff
  help1 = 0
  for i = 1:length(gamma)
    help1 += u[i+3] * cv[i]
  end

  T         = (rho_e - 0.5 * rho * v_square) / (help1)

  for i = 1:length(gamma)
    entrop[i+3] = cv[i] * log(T) - gas_constant[i] * log(u[i+3]) + gas_constant[i] + cv[i] - (v_square / (2*T))
  end
  #s1        = cv[1] * log(T) - gas_constant[1] * log(rho1)
  #s2        = cv[2] * log(T) - gas_constant[2] * log(rho2)

  # Entropy variables
  #w4        = -s1 + gas_constant[1] + cv[1] - (v_square / (2*T)) # + e01/T (bec. e01 = 0 for compressible euler) (DONT confuse it with e1 which is e1 = cv * T)
  #w5        = -s2 + gas_constant[2] + cv[2] - (v_square / (2*T)) # + e02/T (bec. e02 = 0 for compressible euler)
  w1        = (v1)/T
  w2        = (v2)/T
  w3        = (-1.0)/T 

  entrop[1:3] = [w1, w2, w3]

  result      = SVector{length(gamma)+3}(entrop)

  return result
end


# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack cv, gamma = equations
  v1, v2, p = prim
  rho = 0
  cons = zeros(length(gamma)+3)
  
  for i = 1:length(gamma)
    rho       += prim[i+3]
    cons[i+3]   = prim[i+3]
  end

  gammas  = getgamma(prim, equations)

  rho_v1  = rho * v1
  rho_v2  = rho * v2
  rho_e   = p/(gammas-1) + 0.5 * (rho_v1 * v1 + rho_v2 * v2)

  cons[1:3] = [rho_v1, rho_v2, rho_e] 

  result    = SVector{length(gamma)+3}(cons)

  return result
end

@inline function getgamma(u, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack cv, gamma = equations

  help1 = 0.0
  help2 = 0.0
  for i in 1:length(gamma)
    help1 += u[i+3] * cv[i] * gamma[i]
    help2 += u[i+3] * cv[i]
  end

  gammas = help1/help2
  #println("gamma:",gammas)

  return gammas
end

@inline function getpressure(u, gamma, equations::CompressibleEulerMulticomponentEquations2D)
  rho_e, rho_v1, rho_v2 = u
  rho = 0

  for i = 1:length(gamma)
    rho += u[i+3]
  end

  v1  = rho_v1 / rho
  v2  = rho_v2 / rho

  p     = (gamma - 1) * (rho_e - 0.5 * rho * (v1^2 + v2^2))


  #println("pressure:",p)
  return p
end


@inline function density_pressure(u, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack cv, gamma = equations
  rho_v1, rho_v2, rho_e = u
  rho = 0
  
  for i = 1:length(gamma)
    rho += u[i+3]
  end

  gammas       = getgamma(u, equations)
  rho_times_p  = (gammas - 1) * (rho * rho_e - 0.5 * (rho_v1^2 + rho_v2^2))

  return rho_times_p
end


@inline function density(u, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack gamma = equations
  rho = 0

  for i = 1:length(gamma)
    rho += u[i+3]
  end

  return rho
 end