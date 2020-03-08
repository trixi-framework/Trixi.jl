module EulerEquations

using ...Trixi
using ..Equations # Use everything to allow method extension via "function Equations.<method>"
using ...Auxiliary: parameter
using StaticArrays: SVector, MVector, MMatrix, MArray

# Export all symbols that should be available from Equations
export Euler
export initial_conditions
export sources
export calcflux!
export riemann!
export calc_max_dt
export cons2prim
export cons2entropy
export cons2indicator


# Main data structure for system of equations "Euler"
struct Euler <: AbstractEquation{4}
  name::String
  initial_conditions::String
  sources::String
  varnames_cons::SVector{4, String}
  varnames_prim::SVector{4, String}
  gamma::Float64
  surface_flux_type::Symbol
  volume_flux_type::Symbol

  function Euler()
    name = "euler"
    initial_conditions = parameter("initial_conditions")
    sources = parameter("sources", "none")
    varnames_cons = ["rho", "rho_v1", "rho_v2", "rho_e"]
    varnames_prim = ["rho", "v1", "v2", "p"]
    gamma = 1.4
    surface_flux_type = Symbol(parameter("surface_flux_type", "hllc",
                                         valid=["hllc", "laxfriedrichs","central","kennedygruber","chandrashekar_ec"]))
    volume_flux_type = Symbol(parameter("volume_flux_type", "central",
                                        valid=["central","kennedygruber","chandrashekar_ec"]))
    new(name, initial_conditions, sources, varnames_cons, varnames_prim, gamma,
        surface_flux_type, volume_flux_type)
  end
end


# Set initial conditions at physical location `x` for time `t`
function Equations.initial_conditions(equation::Euler, x::AbstractArray{Float64}, t::Real)
  name = equation.initial_conditions
  if name == "density_pulse"
    rho = 1 + exp(-(x[1]^2 + x[2]^2))/2
    v1 = 1
    v2 = 1
    rho_v1 = rho * v1
    rho_v2 = rho * v2
    p = 1
    rho_e = p/(equation.gamma - 1) + 1/2 * rho * (v1^2 + v2^2)
    return [rho, rho_v1, rho_v2, rho_e] 
  elseif name == "pressure_pulse"
    rho = 1
    v1 = 1
    v2 = 1
    rho_v1 = rho * v1
    rho_v2 = rho * v2
    p = 1 + exp(-(x[1]^2 + x[2]^2))/2
    rho_e = p/(equation.gamma - 1) + 1/2 * rho * (v1^2 + v2^2)
    return [rho, rho_v1, rho_v2, rho_e] 
  elseif name == "density_pressure_pulse"
    rho = 1 + exp(-(x[1]^2 + x[2]^2))/2
    v1 = 1
    v2 = 1
    rho_v1 = rho * v1
    rho_v2 = rho * v2
    p = 1 + exp(-x^2)/2
    rho_e = p/(equation.gamma - 1) + 1/2 * rho * (v1^2 + v2^2)
    return [rho, rho_v1, rho_v2, rho_e] 
  elseif name == "constant"
    rho = 1.0
    rho_v1 = 0.1
    rho_v2 = -0.2
    rho_e = 10.0
    return [rho, rho_v1, rho_v2, rho_e] 
  elseif name == "convergence_test"
    c = 1.0
    A = 0.5
    a1 = 1.0
    a2 = 1.0
    L = 2 
    f = 1/L
    omega = 2 * pi * f
    p = 1.0
    rho = c + A * sin(omega * (x[1] + x[2] - (a1 + a2) * t))
    rho_v1 = rho * a1
    rho_v2 = rho * a2
    rho_e = p/(equation.gamma - 1) + 1/2 * rho * (a1^2 + a2^2)
    return [rho, rho_v1, rho_v2, rho_e]
  elseif name == "sod"
    if x < 0.0
      return [1.0, 0.0, 0.0, 2.5]
    else
      return [0.125, 0.0, 0.0, 0.25]
    end
  elseif name == "isentropic_vortex"
    # needs appropriate mesh size, e.g. [-10,-10]x[10,10]
    # make sure that the inicenter does not exit the domain, e.g. T=10.0
    # initial center of the vortex
    inicenter = [0,0]
    # size and strength of the vortex
    iniamplitude = 0.2
    # base flow
    prim=[1.0,1.0,1.0,10.0]
    vel=prim[2:3]
    rt=prim[4]/prim[1]                      # ideal gas equation
    cent=(inicenter+vel*t)                  # advection of center
    cent=x-cent                             # distance to centerpoint
    #cent=cross(iniaxis,cent)               # distance to axis, tangent vector, length r
    # cross product with iniaxis = [0,0,1]
    helper =  cent[1]
    cent[1] = -cent[2]               
    cent[2] = helper
    r2=cent[1]^2+cent[2]^2 
    du = iniamplitude/(2*π)*exp(0.5*(1-r2)) # vel. perturbation
    dtemp = -(equation.gamma-1)/(2*equation.gamma*rt)*du^2            # isentrop
    prim[1]=prim[1]*(1+dtemp)^(1\(equation.gamma-1))     
    prim[2:3]=prim[2:3]+du*cent #v
    prim[4]=prim[4]*(1+dtemp)^(equation.gamma/(equation.gamma-1))     
    rho,rho_v1,rho_v2,rho_e = prim2cons(equation,prim) 
    return [rho,rho_v1,rho_v2,rho_e]
  elseif name == "weak_blast_wave"
    # From Hennemann & Gassner JCP paper 2020 (Sec. 6.3)
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

    return prim2cons(equation, [rho, v1, v2, p])
  elseif name == "blast_wave"
    # From Hennemann & Gassner JCP paper 2020 (Sec. 6.3)
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
    p = r > 0.5 ? 1.0E-3 : 1.245

    return prim2cons(equation, [rho, v1, v2, p])
  else
    error("Unknown initial condition '$name'")
  end
end


# Apply source terms
function Equations.sources(equation::Euler, ut, u, x, element_id, t, n_nodes)
  name = equation.sources
  error("Unknown source term '$name'")
end


# Calculate 2D flux (element version)
@inline function Equations.calcflux!(f1::AbstractArray{Float64},
                                     f2::AbstractArray{Float64},
                                     equation::Euler,
                                     u::AbstractArray{Float64}, element_id::Int,
                                     n_nodes::Int)
  for j = 1:n_nodes
    for i = 1:n_nodes
      rho    = u[1, i, j, element_id]
      rho_v1 = u[2, i, j, element_id]
      rho_v2 = u[3, i, j, element_id]
      rho_e  = u[4, i, j, element_id]
      @views calcflux!(f1[:, i, j], f2[:, i, j], equation, rho, rho_v1, rho_v2, rho_e)
    end
  end
end


# Calculate 2D flux (pointwise version)
@inline function Equations.calcflux!(f1::AbstractArray{Float64},
                                     f2::AbstractArray{Float64},
                                     equation::Euler,
                                     rho::Float64, rho_v1::Float64,
                                     rho_v2::Float64, rho_e::Float64)
  v1 = rho_v1/rho
  v2 = rho_v2/rho
  p = (equation.gamma - 1) * (rho_e - 1/2 * rho * (v1^2 + v2^2))

  f1[1]  = rho_v1
  f1[2]  = rho_v1 * v1 + p
  f1[3]  = rho_v1 * v2
  f1[4]  = (rho_e + p) * v1

  f2[1]  = rho_v2
  f2[2]  = rho_v2 * v1
  f2[3]  = rho_v2 * v2 + p
  f2[4]  = (rho_e + p) * v2
end


# Calculate 2D two-point flux (decide which volume flux type to use)
@inline function Equations.calcflux_twopoint!(f1::AbstractArray{Float64},
                                              f2::AbstractArray{Float64},
                                              equation::Euler,
                                              u::AbstractArray{Float64},
                                              element_id::Int, n_nodes::Int)
  calcflux_twopoint!(f1, f2, Val(equation.volume_flux_type), equation, u, element_id, n_nodes)
end

# Calculate 2D two-point flux (element version)
@inline function Equations.calcflux_twopoint!(f1::AbstractArray{Float64},
                                              f2::AbstractArray{Float64},
                                              twopoint_flux_type::Val,
                                              equation::Euler,
                                              u::AbstractArray{Float64},
                                              element_id::Int, n_nodes::Int)
  # Calculate regular volume fluxes
  f1_diag = MArray{Tuple{nvariables(equation), n_nodes, n_nodes}, Float64}(undef)
  f2_diag = MArray{Tuple{nvariables(equation), n_nodes, n_nodes}, Float64}(undef)
  calcflux!(f1_diag, f2_diag, equation, u, element_id, n_nodes)


  for j = 1:n_nodes
    for i = 1:n_nodes
      # Set diagonal entries (= regular volume fluxes due to consistency)
      @views f1[:, i, i, j] .= f1_diag[:, i, j]
      @views f2[:, j, i, j] .= f2_diag[:, i, j]

      # Flux in x-direction
      for l = i + 1:n_nodes
        @views symmetric_twopoint_flux!(f1[:, l, i, j], twopoint_flux_type,
                                        equation, 1, # 1-> x-direction
                                        u[:, i, j, element_id],
                                        u[:, l, j, element_id])
        @views f1[:, i, l, j] .= f1[:, l, i, j]
      end

      # Flux in y-direction
      for l = j + 1:n_nodes
        @views symmetric_twopoint_flux!(f2[:, l, i, j], twopoint_flux_type,
                                        equation, 2, # 2 -> y-direction
                                        u[:, i, j, element_id],
                                        u[:, i, l, element_id])
        @views f2[:, j, i, l] .= f2[:, l, i, j]
      end
    end
  end
end


# Central two-point flux (identical to weak form volume integral, except for floating point errors)
@inline function symmetric_twopoint_flux!(f::AbstractArray{Float64}, ::Val{:central},
                                          equation::Euler, orientation::Int,
                                          u_ll::AbstractArray{Float64},
                                          u_rr::AbstractArray{Float64})
  # Calculate regular 1D fluxes
  f_ll = MVector{4, Float64}(undef)
  f_rr = MVector{4, Float64}(undef)
  calcflux1D!(f_ll, equation, u_ll[1], u_ll[2], u_ll[3], u_ll[4], orientation)
  calcflux1D!(f_rr, equation, u_rr[1], u_rr[2], u_rr[3], u_rr[4], orientation)

  # Average regular fluxes
  @. f[:] = 1/2 * (f_ll + f_rr)
end


# Kinetic energy preserving two-point flux by Kennedy and Gruber
@inline function symmetric_twopoint_flux!(f::AbstractArray{Float64}, ::Val{:kennedygruber},
                                          equation::Euler, orientation::Int,
                                          u_ll::AbstractArray{Float64},
                                          u_rr::AbstractArray{Float64})
  # Unpack left and right state
  rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr
  v1_ll = rho_v1_ll/rho_ll
  v2_ll = rho_v2_ll/rho_ll
  v1_rr = rho_v1_rr/rho_rr
  v2_rr = rho_v2_rr/rho_rr

  # Average each factor of products in flux
  rho_avg = 1/2 * (rho_ll + rho_rr)
  v1_avg = 1/2 * (v1_ll + v1_rr)
  v2_avg = 1/2 * (v2_ll + v2_rr)
  p_avg = 1/2 * ((equation.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * (v1_ll^2 + v2_ll^2)) +
                 (equation.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * (v1_rr^2 + v2_rr^2)))
  e_avg = 1/2 * (rho_e_ll/rho_ll + rho_e_rr/rho_rr)

  # Calculate fluxes depending on orientation
  if orientation == 1
    f[1]  = rho_avg * v1_avg
    f[2]  = rho_avg * v1_avg * v1_avg + p_avg
    f[3]  = rho_avg * v1_avg * v2_avg
    f[4]  = (rho_avg * e_avg + p_avg) * v1_avg
  else
    f[1]  = rho_avg * v2_avg
    f[2]  = rho_avg * v2_avg * v1_avg
    f[3]  = rho_avg * v2_avg * v2_avg + p_avg
    f[4]  = (rho_avg * e_avg + p_avg) * v2_avg
  end
end

# Entropy conserving two-point flux by Chandrashekar
@inline function symmetric_twopoint_flux!(f::AbstractArray{Float64}, ::Val{:chandrashekar_ec},
                                          equation::Euler, orientation::Int,
                                          u_ll::AbstractArray{Float64},
                                          u_rr::AbstractArray{Float64})
  # Unpack left and right state
  rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr
  v1_ll = rho_v1_ll/rho_ll
  v2_ll = rho_v2_ll/rho_ll
  v1_rr = rho_v1_rr/rho_rr
  v2_rr = rho_v2_rr/rho_rr
  p_ll =  (equation.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * (v1_ll^2 + v2_ll^2))
  p_rr =  (equation.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * (v1_rr^2 + v2_rr^2))
  beta_ll = 0.5*rho_ll/p_ll
  beta_rr = 0.5*rho_rr/p_rr
  specific_kin_ll = 0.5*(v1_ll^2 + v2_ll^2)
  specific_kin_rr = 0.5*(v1_rr^2 + v2_rr^2)
     
  # Compute the necessary mean values
  rho_avg  = 0.5*(rho_ll+rho_rr)
  rho_mean = ln_mean(rho_ll,rho_rr)
  beta_mean = ln_mean(beta_ll,beta_rr)
  beta_avg = 0.5*(beta_ll+beta_rr)
  v1_avg = 0.5*(v1_ll+v1_rr)
  v2_avg = 0.5*(v2_ll+v2_rr)
  p_mean = 0.5*rho_avg/beta_avg
  velocity_square_avg = specific_kin_ll + specific_kin_rr

  # Calculate fluxes depending on orientation
  if orientation == 1
    f[1]  = rho_mean * v1_avg
    f[2]  = f[1] * v1_avg + p_mean
    f[3]  = f[1] * v2_avg
    f[4]  = f[1] *0.5*(1/(equation.gamma-1)/beta_mean - velocity_square_avg)+f[2]*v1_avg + f[3]*v2_avg 
  else
    f[1]  = rho_mean * v2_avg
    f[2]  = f[1] * v1_avg
    f[3]  = f[1] * v2_avg + p_mean
    f[4]  = f[1] *0.5*(1/(equation.gamma-1)/beta_mean - velocity_square_avg)+f[2]*v1_avg + f[3]*v2_avg 
  end
end

# Computes the logarithmic mean: (aR-aL)/(LOG(aR)-LOG(aL)) = (aR-aL)/LOG(aR/aL)
# Problem: if aL~= aR, then 0/0, but should tend to --> 0.5*(aR+aL)
#
# introduce xi=aR/aL and f=(aR-aL)/(aR+aL) = (xi-1)/(xi+1)
# => xi=(1+f)/(1-f)
# => Log(xi) = log(1+f)-log(1-f), and for small f (f^2<1.0E-02) :
#
#    Log(xi) ~=     (f - 1/2 f^2 + 1/3 f^3 - 1/4 f^4 + 1/5 f^5 - 1/6 f^6 + 1/7 f^7)
#                  +(f + 1/2 f^2 + 1/3 f^3 + 1/4 f^4 + 1/5 f^5 + 1/6 f^6 + 1/7 f^7)
#             = 2*f*(1           + 1/3 f^2           + 1/5 f^4           + 1/7 f^6)
#  (aR-aL)/Log(xi) = (aR+aL)*f/(2*f*(1 + 1/3 f^2 + 1/5 f^4 + 1/7 f^6)) = (aR+aL)/(2 + 2/3 f^2 + 2/5 f^4 + 2/7 f^6)
#  (aR-aL)/Log(xi) = 0.5*(aR+aL)*(105/ (105+35 f^2+ 21 f^4 + 15 f^6)
function ln_mean(value1::Float64,value2::Float64)
  epsilon_f2 = 1.0e-4
  ratio = value2/value1
  # f2 = f^2
  f2=(ratio*(ratio-2.)+1.)/(ratio*(ratio+2.)+1.) 
  if (f2<epsilon_f2)
    return (value1+value2)*52.5/(105.0 + f2*(35.0 + f2*(21.0 +f2*15.0)))
  else
    return (value2-value1)/log(ratio)
  end
end


# Calculate 1D flux in for a single point
@inline function calcflux1D!(f::AbstractArray{Float64}, equation::Euler, rho::Float64,
                             rho_v1::Float64, rho_v2::Float64,
                             rho_e::Float64, orientation::Int)
  v1 = rho_v1/rho
  v2 = rho_v2/rho
  p = (equation.gamma - 1) * (rho_e - 1/2 * rho * (v1^2 + v2^2))
  if orientation == 1
    f[1]  = rho_v1
    f[2]  = rho_v1 * v1 + p
    f[3]  = rho_v1 * v2
    f[4]  = (rho_e + p) * v1
  else
    f[1]  = rho_v2
    f[2]  = rho_v2 * v1
    f[3]  = rho_v2 * v2 + p
    f[4]  = (rho_e + p) * v2
  end
end


# Calculate flux across interface with different states on both sides (surface version)
function Equations.riemann!(surface_flux::Matrix{Float64},
                            u_surfaces::Array{Float64, 4}, surface_id::Int,
                            equation::Euler, n_nodes::Int,
                            orientations::Vector{Int})
  for i = 1:n_nodes
    @views riemann!(surface_flux[:, i], u_surfaces[:, :, i, surface_id],
                    equation, orientations[surface_id])
  end
end


# Calculate flux across interface with different states on both sides (pointwise version)
function Equations.riemann!(surface_flux::AbstractArray{Float64, 1},
                            u_surfaces::AbstractArray{Float64, 2},
                            equation::Euler, orientation::Int)

  # Store for convenience
  rho_ll    = u_surfaces[1, 1]
  rho_v1_ll = u_surfaces[1, 2]
  rho_v2_ll = u_surfaces[1, 3]
  rho_e_ll  = u_surfaces[1, 4]
  rho_rr    = u_surfaces[2, 1]
  rho_v1_rr = u_surfaces[2, 2]
  rho_v2_rr = u_surfaces[2, 3]
  rho_e_rr  = u_surfaces[2, 4]

  # Calculate primitive variables and speed of sound
  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v_mag_ll = sqrt(v1_ll^2 + v2_ll^2)
  p_ll = (equation.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * v_mag_ll^2)
  c_ll = sqrt(equation.gamma * p_ll / rho_ll)
  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  v_mag_rr = sqrt(v1_rr^2 + v2_rr^2)
  p_rr = (equation.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * v_mag_rr^2)
  c_rr = sqrt(equation.gamma * p_rr / rho_rr)

  # Obtain left and right fluxes
  f_ll = zeros(MVector{4})
  f_rr = zeros(MVector{4})
  calcflux1D!(f_ll, equation, rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll, orientation)
  calcflux1D!(f_rr, equation, rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr, orientation)
 
  if equation.surface_flux_type == :laxfriedrichs
    λ_max = max(v_mag_ll, v_mag_rr) + max(c_ll, c_rr)
    surface_flux[1] = 1/2 * (f_ll[1] + f_rr[1]) - 1/2 * λ_max * (rho_rr    - rho_ll)
    surface_flux[2] = 1/2 * (f_ll[2] + f_rr[2]) - 1/2 * λ_max * (rho_v1_rr - rho_v1_ll)
    surface_flux[3] = 1/2 * (f_ll[3] + f_rr[3]) - 1/2 * λ_max * (rho_v2_rr - rho_v2_ll)
    surface_flux[4] = 1/2 * (f_ll[4] + f_rr[4]) - 1/2 * λ_max * (rho_e_rr  - rho_e_ll)
  elseif equation.surface_flux_type in [:central,:kennedygruber,:chandrashekar_ec]
    @views symmetric_twopoint_flux!(surface_flux[:], Val(equation.surface_flux_type),
                             equation, orientation,
                             u_surfaces[1,:], u_surfaces[2,:])
     
  elseif equation.surface_flux_type == :hllc
    error("not yet implemented or tested")
    v_tilde = (sqrt(rho_ll) * v_ll + sqrt(rho_rr) * v_rr) / (sqrt(rho_ll) + sqrt(rho_rr))
    h_ll = (rho_e_ll + p_ll) / rho_ll
    h_rr = (rho_e_rr + p_rr) / rho_rr
    h_tilde = (sqrt(rho_ll) * h_ll + sqrt(rho_rr) * h_rr) / (sqrt(rho_ll) + sqrt(rho_rr))
    c_tilde = sqrt((equation.gamma - 1) * (h_tilde - 1/2 * v_tilde^2))
    s_ll = v_tilde - c_tilde
    s_rr = v_tilde + c_tilde

    if s_ll > 0
      surface_flux[1, surface_id] = f_ll[1]
      surface_flux[2, surface_id] = f_ll[2]
      surface_flux[3, surface_id] = f_ll[3]
    elseif s_rr < 0
      surface_flux[1, surface_id] = f_rr[1]
      surface_flux[2, surface_id] = f_rr[2]
      surface_flux[3, surface_id] = f_rr[3]
    else
      s_star = ((p_rr - p_ll + rho_ll * v_ll * (s_ll - v_ll) - rho_rr * v_rr * (s_rr - v_rr))
                / (rho_ll * (s_ll - v_ll) - rho_rr * (s_rr - v_rr)))
      if s_ll <= 0 && 0 <= s_star
        surface_flux[1, surface_id] = (f_ll[1] + s_ll *
            (rho_ll * (s_ll - v_ll)/(s_ll - s_star) - rho_ll))
        surface_flux[2, surface_id] = (f_ll[2] + s_ll *
            (rho_ll * (s_ll - v_ll)/(s_ll - s_star) * s_star - rho_v_ll))
        surface_flux[3, surface_id] = (f_ll[3] + s_ll *
            (rho_ll * (s_ll - v_ll)/(s_ll - s_star) *
            (rho_e_ll/rho_ll + (s_star - v_ll) * (s_star + rho_ll/(rho_ll * (s_ll - v_ll))))
            - rho_e_ll))
      else
        surface_flux[1, surface_id] = (f_rr[1] + s_rr *
            (rho_rr * (s_rr - v_rr)/(s_rr - s_star) - rho_rr))
        surface_flux[2, surface_id] = (f_rr[2] + s_rr *
            (rho_rr * (s_rr - v_rr)/(s_rr - s_star) * s_star - rho_v_rr))
        surface_flux[3, surface_id] = (f_rr[3] + s_rr *
            (rho_rr * (s_rr - v_rr)/(s_rr - s_star) *
            (rho_e_rr/rho_rr + (s_star - v_rr) * (s_star + rho_rr/(rho_rr * (s_rr - v_rr))))
            - rho_e_rr))
      end
    end
  else
    error("unknown Riemann solver '$(string(equation.surface_flux_type))'")
  end
end

# Original riemann! implementation, non-optimized but easier to understand
# function Equations.riemann!(surface_flux::Array{Float64, 2},
#                             u_surfaces::Array{Float64, 3}, surface_id::Int,
#                             equation::Euler, n_nodes::Int)
#   u_ll     = u_surfaces[1, :, surface_id]
#   u_rr     = u_surfaces[2, :, surface_id]
# 
#   rho_ll   = u_ll[1]
#   rho_v_ll = u_ll[2]
#   rho_e_ll = u_ll[3]
#   rho_rr   = u_rr[1]
#   rho_v_rr = u_rr[2]
#   rho_e_rr = u_rr[3]
# 
#   v_ll = rho_v_ll / rho_ll
#   p_ll = (equation.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * v_ll^2)
#   c_ll = sqrt(equation.gamma * p_ll / rho_ll)
#   v_rr = rho_v_rr / rho_rr
#   p_rr = (equation.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * v_rr^2)
#   c_rr = sqrt(equation.gamma * p_rr / rho_rr)
# 
#   f_ll = zeros(MVector{3})
#   f_rr = zeros(MVector{3})
#   calcflux!(f_ll, equation, rho_ll, rho_v_ll, rho_e_ll)
#   calcflux!(f_rr, equation, rho_rr, rho_v_rr, rho_e_rr)
# 
#   if equation.surface_flux_type == :laxfriedrichs
#     λ_max = max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr)
# 
#     @. surface_flux[:, surface_id] = 1/2 * (f_ll + f_rr) - 1/2 * λ_max * (u_rr - u_ll)
#   elseif equation.surface_flux_type == :hllc
#     v_tilde = (sqrt(rho_ll) * v_ll + sqrt(rho_rr) * v_rr) / (sqrt(rho_ll) + sqrt(rho_rr))
#     h_ll = (rho_e_ll + p_ll) / rho_ll
#     h_rr = (rho_e_rr + p_rr) / rho_rr
#     h_tilde = (sqrt(rho_ll) * h_ll + sqrt(rho_rr) * h_rr) / (sqrt(rho_ll) + sqrt(rho_rr))
#     c_tilde = sqrt((equation.gamma - 1) * (h_tilde - 1/2 * v_tilde^2))
#     s_ll = v_tilde - c_tilde
#     s_rr = v_tilde + c_tilde
# 
#     if s_ll > 0
#       @. surface_flux[:, surface_id] = f_ll
#     elseif s_rr < 0
#       @. surface_flux[:, surface_id] = f_rr
#     else
#       s_star = ((p_rr - p_ll + rho_ll * v_ll * (s_ll - v_ll) - rho_rr * v_rr * (s_rr - v_rr))
#                 / (rho_ll * (s_ll - v_ll) - rho_rr * (s_rr - v_rr)))
#       if s_ll <= 0 && 0 <= s_star
#         u_star_ll = rho_ll * (s_ll - v_ll)/(s_ll - s_star) .* (
#             [1, s_star,
#              rho_e_ll/rho_ll + (s_star - v_ll) * (s_star + rho_ll/(rho_ll * (s_ll - v_ll)))])
#         @. surface_flux[:, surface_id] = f_ll + s_ll * (u_star_ll - u_ll)
#       else
#         u_star_rr = rho_rr * (s_rr - v_rr)/(s_rr - s_star) .* (
#             [1, s_star,
#              rho_e_rr/rho_rr + (s_star - v_rr) * (s_star + rho_rr/(rho_rr * (s_rr - v_rr)))])
#         @. surface_flux[:, surface_id] = f_rr + s_rr * (u_star_rr - u_rr)
#       end
#     end
#   else
#     error("unknown Riemann solver '$(string(equation.surface_flux_type))'")
#   end
# end


# Determine maximum stable time step based on polynomial degree and CFL number
function Equations.calc_max_dt(equation::Euler, u::Array{Float64, 4},
                               element_id::Int, n_nodes::Int,
                               invjacobian::Float64, cfl::Float64)
  λ_max = 0.0
  for j = 1:n_nodes
    for i = 1:n_nodes
      rho    = u[1, i, j, element_id]
      rho_v1 = u[2, i, j, element_id]
      rho_v2 = u[3, i, j, element_id]
      rho_e  = u[4, i, j, element_id]
      v1 = rho_v1/rho
      v2 = rho_v2/rho
      v_mag = sqrt(v1^2 + v2^2)
      p = (equation.gamma - 1) * (rho_e - 1/2 * rho * v_mag^2)
      c = sqrt(equation.gamma * p / rho)
      λ_max = max(λ_max, v_mag + c)
    end
  end

  dt = cfl * 2 / (invjacobian * λ_max) / n_nodes

  return dt
end


# Convert conservative variables to primitive
function Equations.cons2prim(equation::Euler, cons::Array{Float64, 4})
  prim = similar(cons)
  @. prim[1, :, :, :] = cons[1, :, :, :]
  @. prim[2, :, :, :] = cons[2, :, :, :] / cons[1, :, :, :]
  @. prim[3, :, :, :] = cons[3, :, :, :] / cons[1, :, :, :]
  @. prim[4, :, :, :] = ((equation.gamma - 1)
                         * (cons[4, :, :, :] - 1/2 * (cons[2, :, :, :] * prim[2, :, :, :] +
                                                      cons[3, :, :, :] * prim[3, :, :, :])))
  return prim
end

# Convert conservative variables to entropy
function Equations.cons2entropy(equation::Euler, cons::Array{Float64, 4}, n_nodes::Int, n_elements::Int)
  entropy = similar(cons)
  v = zeros(2,n_nodes,n_nodes,n_elements)
  v_square = zeros(n_nodes,n_nodes,n_elements)
  p = zeros(n_nodes,n_nodes,n_elements)
  s = zeros(n_nodes,n_nodes,n_elements)
  rho_p = zeros(n_nodes,n_nodes,n_elements)

  @. v[1, :, :, :] = cons[2, :, :, :] / cons[1, :, :, :] 
  @. v[2, :, :, :] = cons[3, :, :, :] / cons[1, :, :, :] 
  @. v_square[ :, :, :] = v[1, :, :, :]*v[1, :, :, :]+v[2, :, :, :]*v[2, :, :, :]
  @. p[ :, :, :] = ((equation.gamma - 1)
                         * (cons[4, :, :, :] - 
		     1/2 * (cons[2, :, :, :] * v[1, :, :, :] +
                            cons[3, :, :, :] * v[2, :, :, :])))
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
function prim2cons(equation::Euler, prim::AbstractArray{Float64})
  cons = similar(prim)
  cons[1] = prim[1]
  cons[2] = prim[2] * prim[1]
  cons[3] = prim[3] * prim[1]
  cons[4] = prim[4]/(equation.gamma-1)+1/2*(cons[2] * prim[2] + cons[3] * prim[3])
  return cons
end


# Convert conservative variables to indicator variable for discontinuities
function Equations.cons2indicator(equation::Euler, cons::AbstractArray{Float64})
  rho, rho_v1, rho_v2, rho_e = cons
  v1 = rho_v1/rho
  v2 = rho_v2/rho

  # Calculate pressure
  p = (equation.gamma - 1) * (rho_e - 1/2 * rho * (v1^2 + v2^2))

  # Indicator variable is rho * p
  return rho * p
end

       

end # module
