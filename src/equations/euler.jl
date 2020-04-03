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
export cons2indicator!


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
    gamma = parameter("gamma", 1.4)
    surface_flux_type = Symbol(parameter("surface_flux_type", "hllc",
                                         valid=["hllc", "laxfriedrichs","central", 
                                                "kennedygruber", "chandrashekar_ec","yuichi"]))
    volume_flux_type = Symbol(parameter("volume_flux_type", "central",
                                        valid=["central", "kennedygruber", "chandrashekar_ec","yuichi"]))
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
    # Modified From Hennemann & Gassner JCP paper 2020 (Sec. 6.3) -> "medium blast wave"
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
  elseif name == "sedov_blast_wave"
    # Set up polar coordinates
    inicenter = [0, 0]
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)

    # Setup based on http://flash.uchicago.edu/site/flashcode/user_support/flash_ug_devel/node184.html#SECTION010114000000000000000
    r0 = 0.21875 # = 3.5 * smallest dx (for domain length=4 and max-ref=6)
    # r0 = 0.5 # = more reasonable setup
    E = 1.0
    p0_inner = 3 * (equation.gamma - 1) * E / (3 * pi * r0^2)
    p0_outer = 1.0e-5 # = true Sedov setup
    # p0_outer = 1.0e-3 # = more reasonable setup

    # Calculate primitive variables
    rho = 1.0
    v1 = 0.0
    v2 = 0.0
    p = r > r0 ? p0_outer : p0_inner

    return prim2cons(equation, [rho, v1, v2, p])
  elseif name == "medium_sedov_blast_wave"
    # Set up polar coordinates
    inicenter = [0, 0]
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)

    # Setup based on http://flash.uchicago.edu/site/flashcode/user_support/flash_ug_devel/node184.html#SECTION010114000000000000000
    r0 = 0.21875 # = 3.5 * smallest dx (for domain length=4 and max-ref=6)
    # r0 = 0.5 # = more reasonable setup
    E = 1.0
    p0_inner = 3 * (equation.gamma - 1) * E / (3 * pi * r0^2)
    # p0_outer = 1.0e-5 # = true Sedov setup
    p0_outer = 1.0e-3 # = more reasonable setup

    # Calculate primitive variables
    rho = 1.0
    v1 = 0.0
    v2 = 0.0
    p = r > r0 ? p0_outer : p0_inner

    return prim2cons(equation, [rho, v1, v2, p])
  elseif name == "khi" 
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
    rho = dens0 + (dens1-dens0) * 0.5*(1+(tanh(slope*(x[2]+0.25)) - (tanh(slope*(x[2]-0.25)) + 1)))
    #  x velocity is also augmented with noise
    v1 = velx0 + (velx1-velx0) * 0.5*(1+(tanh(slope*(x[2]+0.25)) - (tanh(slope*(x[2]-0.25)) + 1)))+0.01*(rand(Float64,1)[1]-0.5)
    return prim2cons(equation, [rho, v1, v2, p])
  elseif name == "blob" 
    # blob test case, see Agertz et al. https://arxiv.org/pdf/astro-ph/0610051.pdf
    # other reference: https://arxiv.org/pdf/astro-ph/0610051.pdf
    # change discontinuity to tanh 
    # typical domain is rectangular, we change it to a square, as Trixi can only do squares
    # resolution 128^2, 256^2
    # domain size is [-20.0,20.0]^2
    # gamma = 5/3 for this test case
    R = 1.0 # radius of the blob
    # background density
    dens0 = 1.0
    Chi = 10.0 # density contrast
    # reference time of characteristic growth of KH instability equal to 1.0
    tau_kh = 1.0
    tau_cr = tau_kh/1.6 # crushing time
    # determine background velocity 
    velx0 = 2*R*sqrt(Chi)/tau_cr
    vely0 = 0.0
    Ma0 = 2.7 # background flow Mach number Ma=v/c
    c = velx0/Ma0 # sound speed
    # use perfect gas assumption to compute background pressure via the sound speed c^2 = gamma * pressure/density
    p0 = c*c*dens0/equation.gamma
    # initial center of the blob
    inicenter = [-15,0]
    x_rel = x-inicenter
    r = sqrt(x_rel[1]^2 + x_rel[2]^2)
    # steepness of the tanh transition zone
    slope = 5
    # density blob
    dens = dens0 + (Chi-1) * 0.5*(1+(tanh(slope*(r+R)) - (tanh(slope*(r-R)) + 1)))
    # velocity blob is zero
    velx = velx0 - velx0 * 0.5*(1+(tanh(slope*(r+R)) - (tanh(slope*(r-R)) + 1)))
    return prim2cons(equation, [dens, velx, vely0, p0])
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
                                              f1_diag::AbstractArray{Float64},
                                              f2_diag::AbstractArray{Float64},
                                              equation::Euler,
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
                                              equation::Euler,
                                              u::AbstractArray{Float64},
                                              element_id::Int, n_nodes::Int)
  # Calculate regular volume fluxes
  calcflux!(f1_diag, f2_diag, equation, u, element_id, n_nodes)


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
                                        u[1, l, j, element_id], u[2, l, j, element_id],
                                        u[3, l, j, element_id], u[4, l, j, element_id])
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
                                        u[1, i, l, element_id], u[2, i, l, element_id],
                                        u[3, i, l, element_id], u[4, i, l, element_id])
        for v in 1:nvariables(equation)
          f2[v, j, i, l] = f2[v, l, i, j]
        end
      end
    end
  end
end


# Central two-point flux (identical to weak form volume integral, except for floating point errors)
@inline function symmetric_twopoint_flux!(f::AbstractArray{Float64}, ::Val{:central},
                                          equation::Euler, orientation::Int,
                                          rho_ll::Float64,
                                          rho_v1_ll::Float64,
                                          rho_v2_ll::Float64,
                                          rho_e_ll::Float64,
                                          rho_rr::Float64,
                                          rho_v1_rr::Float64,
                                          rho_v2_rr::Float64,
                                          rho_e_rr::Float64)
  # Calculate regular 1D fluxes
  f_ll = MVector{4, Float64}(undef)
  f_rr = MVector{4, Float64}(undef)
  calcflux1D!(f_ll, equation, rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll, orientation)
  calcflux1D!(f_rr, equation, rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr, orientation)

  # Average regular fluxes
  @. f[:] = 1/2 * (f_ll + f_rr)
end

# Kinetic energy preserving two-point flux by Yuichi et al. with pressure oscillation fix
@inline function symmetric_twopoint_flux!(f::AbstractArray{Float64}, ::Val{:yuichi},
                                          equation::Euler, orientation::Int,
                                          rho_ll::Float64,
                                          rho_v1_ll::Float64,
                                          rho_v2_ll::Float64,
                                          rho_e_ll::Float64,
                                          rho_rr::Float64,
                                          rho_v1_rr::Float64,
                                          rho_v2_rr::Float64,
                                          rho_e_rr::Float64)
  # Unpack left and right state
  v1_ll = rho_v1_ll/rho_ll
  v2_ll = rho_v2_ll/rho_ll
  v1_rr = rho_v1_rr/rho_rr
  v2_rr = rho_v2_rr/rho_rr
  p_ll =  (equation.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * (v1_ll^2 + v2_ll^2))
  p_rr =  (equation.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * (v1_rr^2 + v2_rr^2))

  # Average each factor of products in flux
  rho_avg = 1/2 * (rho_ll + rho_rr)
  v1_avg = 1/2 * (v1_ll + v1_rr)
  v2_avg = 1/2 * (v2_ll + v2_rr)
  p_avg = 1/2 * (p_ll + p_rr)
  kin_avg = 1/2 * (v1_ll*v1_rr + v2_ll*v2_rr)

  # Calculate fluxes depending on orientation
  if orientation == 1
    pv1_avg = 1/2 * ( p_ll*v1_ll + p_rr*v1_rr)
    f[1]  = rho_avg * v1_avg
    f[2]  = rho_avg * v1_avg * v1_avg + p_avg
    f[3]  = rho_avg * v1_avg * v2_avg
    f[4]  = p_avg*v1_avg/(equation.gamma-1) + rho_avg*v1_avg*kin_avg + pv1_avg 
  else
    pv2_avg = 1/2 * ( p_ll*v2_ll + p_rr*v2_rr)
    f[1]  = rho_avg * v2_avg
    f[2]  = rho_avg * v2_avg * v1_avg
    f[3]  = rho_avg * v2_avg * v2_avg + p_avg
    f[4]  = p_avg*v2_avg/(equation.gamma-1) + rho_avg*v2_avg*kin_avg + pv2_avg 
  end
end


# Kinetic energy preserving two-point flux by Kennedy and Gruber
@inline function symmetric_twopoint_flux!(f::AbstractArray{Float64}, ::Val{:kennedygruber},
                                          equation::Euler, orientation::Int,
                                          rho_ll::Float64,
                                          rho_v1_ll::Float64,
                                          rho_v2_ll::Float64,
                                          rho_e_ll::Float64,
                                          rho_rr::Float64,
                                          rho_v1_rr::Float64,
                                          rho_v2_rr::Float64,
                                          rho_e_rr::Float64)
  # Unpack left and right state
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
                                          rho_ll::Float64,
                                          rho_v1_ll::Float64,
                                          rho_v2_ll::Float64,
                                          rho_e_ll::Float64,
                                          rho_rr::Float64,
                                          rho_v1_rr::Float64,
                                          rho_v2_rr::Float64,
                                          rho_e_rr::Float64)
  # Unpack left and right state
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


# Calculate flux across interface with different states on both sides (EC mortar version)
function Equations.riemann!(surface_flux::AbstractArray{Float64, 3},
                            fstarnode::AbstractVector{Float64},
                            u_surfaces_left::AbstractArray{Float64, 3},
                            u_surfaces_right::AbstractArray{Float64, 3},
                            surface_id::Int,
                            equation::Euler, n_nodes::Int,
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
               u_surfaces_right[1, j, surface_id],
               u_surfaces_right[2, j, surface_id],
               u_surfaces_right[3, j, surface_id],
               u_surfaces_right[4, j, surface_id],
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
                            equation::Euler, n_nodes::Int,
                            orientations::Vector{Int})
  # Call pointwise Riemann solver
  for i = 1:n_nodes
    # Store flux in pre-allocated `fstarnode` to avoid allocations in loop
    riemann!(fstarnode,
             u_surfaces[1, 1, i, surface_id],
             u_surfaces[1, 2, i, surface_id],
             u_surfaces[1, 3, i, surface_id],
             u_surfaces[1, 4, i, surface_id],
             u_surfaces[2, 1, i, surface_id],
             u_surfaces[2, 2, i, surface_id],
             u_surfaces[2, 3, i, surface_id],
             u_surfaces[2, 4, i, surface_id],
             equation, orientations[surface_id])

    # Copy flux back to actual flux array
    for v in 1:nvariables(equation)
      surface_flux[v, i] = fstarnode[v]
    end
  end
end


# Calculate flux across interface with different states on both sides (pointwise version)
function Equations.riemann!(surface_flux::AbstractArray{Float64, 1},
                            rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll,
                            rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr,
                            equation::Euler, orientation::Int)
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
  elseif equation.surface_flux_type in (:central,:kennedygruber,:chandrashekar_ec,:yuichi)
    symmetric_twopoint_flux!(surface_flux, Val(equation.surface_flux_type),
                             equation, orientation,
                             rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll,
                             rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr)
     
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


# Convert conservative variables to indicator variable for discontinuities (elementwise version)
@inline function Equations.cons2indicator!(indicator::AbstractArray{Float64}, equation::Euler,
                                           cons::AbstractArray{Float64},
                                           element_id::Int, n_nodes::Int, indicator_variable)
  for j in 1:n_nodes
    for i in 1:n_nodes
      indicator[1, i, j] = cons2indicator(equation,
                                          cons[1, i, j, element_id], cons[2, i, j, element_id],
					  cons[3, i, j, element_id], cons[4, i, j, element_id], indicator_variable)
    end
  end
end

# Convert conservative variables to indicator variable for discontinuities (pointwise version)
@inline function Equations.cons2indicator(equation::Euler, rho, rho_v1, rho_v2, rho_e,
                                          ::Val{:density})
  # Indicator variable is rho 
  return rho 
end

# Convert conservative variables to indicator variable for discontinuities (pointwise version)
@inline function Equations.cons2indicator(equation::Euler, rho, rho_v1, rho_v2, rho_e,
                                          ::Val{:density_pressure})
  v1 = rho_v1/rho
  v2 = rho_v2/rho

  # Calculate pressure
  p = (equation.gamma - 1) * (rho_e - 1/2 * rho * (v1^2 + v2^2))

  # Indicator variable is rho * p
  return rho * p
end
#
# Convert conservative variables to indicator variable for discontinuities (pointwise version)
@inline function Equations.cons2indicator(equation::Euler, rho, rho_v1, rho_v2, rho_e,
                                          ::Val{:pressure})
  v1 = rho_v1/rho
  v2 = rho_v2/rho

  # Indicator variable is p
  return (equation.gamma - 1) * (rho_e - 1/2 * rho * (v1^2 + v2^2))
end

# Calculates the entropy flux in direction "orientation" and the entropy variables for a state cons
@inline function cons2entropyvars_and_flux(gamma::Float64, cons, orientation::Int)  
  entropy = MVector{4, Float64}(undef)
  v = (cons[2] / cons[1] , cons[3] / cons[1]) 
  v_square= v[1]*v[1]+v[2]*v[2]
  p = (gamma - 1) * (cons[4] - 1/2 * (cons[2] * v[1] + cons[3] * v[2]))
  rho_p = cons[1] / p 
  # thermodynamic entropy
  s = log(p) - gamma*log(cons[1])
  # mathematical entropy
  S = - s*cons[1]/(gamma-1)
  # entropy variables
  entropy[1] = (gamma - s)/(gamma-1) - 0.5*rho_p*v_square 
  entropy[2] = rho_p*v[1]
  entropy[3] = rho_p*v[2]
  entropy[4] = -rho_p
  # entropy flux
  entropy_flux = S*v[orientation]
  return entropy, entropy_flux
end       


end # module
