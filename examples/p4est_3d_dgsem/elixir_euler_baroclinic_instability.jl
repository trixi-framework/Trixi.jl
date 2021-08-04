
using Random: seed!
using OrdinaryDiffEq
using Trixi
using LinearAlgebra

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations3D(gamma)

function cart_to_sphere(x)
  r = norm(x)
  lambda = atan(x[2], x[1])
  if x[2] < 0
    lambda += 8.0 * atan(1.0)
  end
  phi = asin(x[3] / r)

  return lambda, phi, r
end

function initial_condition_acoustic_wave(x, t, equations::CompressibleEulerEquations3D)
  p = 1e5

  Cpd = 1004.0e0 # Specific heat by constant pressure
  Cvd = 717.0e0 # Specific heat by constant volume
  Rd = Cpd - Cvd

  rho = p / (Rd * 300)
  v1 = v2 = v3 = 0.0

  lambda, phi, z = cart_to_sphere(x)
  z -= 63712
  R = 63712 / 3
  r = 63712 * acos(sin(0) * sin(phi) + cos(0) * cos(phi) * cos(lambda - 0))
  if r < R
    f = 0.5 * (1 + cos(pi * r / R))
  else
    f = 0.0
  end

  n_v = 1.0
  g = sin(n_v * pi * z / 10000)

  p += 100 * f * g

  return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end

function evaluate_pressure_temperature(lon, lat, z)
  p0 = 1.0e5 # Surface pressure

  RadEarth = 6371220.0 # Earth radius
  OmegaEarth = 0.0000729212 # Coriolis parameter
  Grav = 9.81
  Cpd = 1004.0 # Specific heat by constant pressure
  Cvd = 717.0 # Specific heat by constant volume
  Rd = Cpd - Cvd

  KJet = 3.0
  BJet = 2.0
  T0E = 310.0
  T0P = 240.0
  lapse = 0.005
  pertu0 = 0.50
  pertr = 1.0/6.0
  pertup = 1.0
  pertexpr = 0.1
  pertlon = pi/9.0
  pertlat = 2.0*pi/9.0
  pertz = 15000.0

  aref = RadEarth
  omegaref = OmegaEarth

  T0 = 0.5*(T0E + T0P)
  constA = 1.0/lapse
  constB = (T0 - T0P)/(T0*T0P)
  constC = 0.5*(KJet + 2.0)*(T0E - T0P)/(T0E*T0P)
  constH = Rd*T0/Grav

  scaledZ = z/(BJet*constH)

  # tau values
  tau1 = constA*lapse/T0*exp(lapse*z/T0) + constB*(1.0 - 2.0*scaledZ^2)*exp(-scaledZ^2)
  tau2 = constC*(1.0 - 2.0*scaledZ^2)*exp(-scaledZ^2)

  inttau1 = constA*(exp(lapse*z/T0) - 1.0) + constB*z*exp(-scaledZ^2)
  inttau2 = constC*z*exp(-scaledZ^2)

  # radius ratio
  rratio = (z + aref)/aref;

  # interior term on temperature expression
  inttermT = (rratio*cos(lat))^KJet - KJet/(KJet + 2.0)*(rratio*cos(lat))^(KJet + 2.0)

  # temperature
  t = 1.0/(rratio^2*(tau1 - tau2*inttermT))

  # hydrostatic pressure
  p = p0*exp(-Grav/Rd*(inttau1 - inttau2 * inttermT))

  return p, t
end

function evaluate_exponential(lon,lat,z)
  KJet = 3.0
  BJet = 2.0
  T0E = 310.0
  T0P = 240.0
  lapse = 0.005
  pertu0 = 0.50
  pertr = 1.0/6.0
  pertup = 1.0
  pertexpr = 0.1
  pertlon = pi/9.0
  pertlat = 2.0*pi/9.0
  pertz = 15000.0

  # Great circle distance
  greatcircler = 1.0/pertexpr * acos(sin(pertlat)*sin(lat) + cos(pertlat)*cos(lat)*cos(lon - pertlon))

  # Vertical tapering of stream function
  if z < pertz
    perttaper = 1.0 - 3.0*z^2/pertz^2 + 2.0*z^3/pertz^3
  else
    perttaper = 0.0
  end

  # Zonal velocity perturbation
  if greatcircler < 1.0
    result = pertup * perttaper * exp(- greatcircler^2)
  else
    result = 0.0
  end

  return result
end

function VelSphToCa(u_lam,u_phi,u_r,lam,phi)
  r11 =          -sin(lam)
  r21 = -sin(phi)*cos(lam)
  r31 =  cos(phi)*cos(lam)
  r12 =           cos(lam)
  r22 = -sin(phi)*sin(lam)
  r32 =  cos(phi)*sin(lam)
  r13 =           0.0
  r23 =  cos(phi)
  r33 =  sin(phi)

  u_x = r11*u_lam + r21*u_phi + r31*u_r
  u_y = r12*u_lam + r22*u_phi + r32*u_r
  u_z = r13*u_lam + r23*u_phi + r33*u_r

  return u_x, u_y, u_z
end

function initial_condition_baroclinic_instability(x, t, equations::CompressibleEulerEquations3D)
  RadEarth = 6371220.0 # Earth radius
  OmegaEarth = 0.0000729212 # Coriolis parameter
  Grav = 9.81
  Cpd = 1004.0 # Specific heat by constant pressure
  Cvd = 717.0 # Specific heat by constant volume
  Rd = Cpd - Cvd

  KJet = 3.0
  BJet = 2.0
  T0E = 310.0
  T0P = 240.0
  lapse = 0.005
  pertu0 = 0.50
  pertr = 1.0/6.0
  pertup = 1.0
  pertexpr = 0.1
  pertlon = pi/9.0
  pertlat = 2.0*pi/9.0
  pertz = 15000.0


  lon, lat, Radius = cart_to_sphere(x)
  zLoc = max(Radius - RadEarth, 0.0)
  pLoc, tLoc = evaluate_pressure_temperature(lon, lat, zLoc)
  T0 = 0.5*(T0E + T0P)
  constH = Rd*T0/Grav
  constC = 0.50*(KJet + 2.0)*(T0E -T0P)/(T0E*T0P)
  scaledZ = zLoc/(BJet*constH)
  inttau2 = constC*zLoc*exp(-scaledZ^2)
  rratio = Radius/RadEarth
  inttermU = (rratio*cos(lat))^(KJet-1.0)-(rratio*cos(lat))^(KJet+1.0)
  bigU = Grav/RadEarth*KJet*inttau2*inttermU*tLoc
  rcoslat = Radius*cos(lat)
  omegarcoslat = OmegaEarth*rcoslat
  uS = -omegarcoslat+sqrt(omegarcoslat^2+rcoslat*bigU)
  uS = uS + evaluate_exponential(lon,lat,zLoc)
  vS = 0.0
  wS = 0.0
  v1, v2, v3 = VelSphToCa(uS,vS,wS,lon,lat)
  rho = pLoc/(Rd*tLoc)
  p = pLoc

  return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end

DampF(alpha) = sin(0.5*pi*alpha)^2

@inline function source_terms_baroclinic_instability(u, x, t, equations::CompressibleEulerEquations3D)
  RadEarth = 6371220.0 # Earth radius
  OmegaEarth = 7.29212E-5 # Coriolis parameter
  Height = 30000.0 # Atmosphere height
  Grav = 9.81
  Cpd = 1004.0e0 # Specific heat by constant pressure
  Cvd = 717.0e0 # Specific heat by constant volume
  Rd = Cpd - Cvd
  StrideDamp = 6000.0
  Relax = 1.E-3

  Radius = norm(x)
  zPLoc = Radius-RadEarth
  du2 = - Grav*(RadEarth/Radius)^2 * x[1]/Radius*u[1]

  du3 = - Grav*(RadEarth/Radius)^2 * x[2]/Radius*u[1]
  du4 = - Grav*(RadEarth/Radius)^2 * x[3]/Radius*u[1]
  du5 = - Grav*(RadEarth/Radius)^2 * (u[2] * x[1] + u[3] * x[2] + u[4] * x[3])/Radius
  # coriolis_term = - 2.0*OmegaEarth*cross([0.0,0.0,1.0], u[2:4])
  # du2 += coriolis_term[1]
  # du3 += coriolis_term[2]
  # du4 += coriolis_term[3]
  du2 += 2.0 * OmegaEarth * u[3]
  du3 -= 2.0 * OmegaEarth * u[2]

  if zPLoc >= Height-StrideDamp
    Damp = Relax*DampF((1.0 - (Height - zPLoc)/StrideDamp))
    damp_vec = - Damp*(u[2] * x[1] + u[3] * x[2] + u[4] * x[3])/Radius*x/Radius
    du2 += damp_vec[1]
    du3 += damp_vec[2]
    du4 += damp_vec[3]
  end

  du1 = zero(eltype(u))

  return SVector(du1, du2, du3, du4, du5)
end

function flux_mars(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerEquations3D)
  # This only works in x-direction
  @assert orientation == 1

  cS = 360

  sRho_L = 1 / u_ll[1]
  sRho_R = 1 / u_rr[1]

  Vel_L_1 = u_ll[2] * sRho_L
  Vel_L_2 = u_ll[3] * sRho_L
  Vel_L_3 = u_ll[4] * sRho_L
  Vel_R_1 = u_rr[2] * sRho_R
  Vel_R_2 = u_rr[3] * sRho_R
  Vel_R_3 = u_rr[4] * sRho_R

  p_L = (equations.gamma - 1) * (u_ll[5] - 0.5 * (u_ll[2] * Vel_L_1 + u_ll[3] * Vel_L_2 + u_ll[4] * Vel_L_3))
  p_R = (equations.gamma - 1) * (u_rr[5] - 0.5 * (u_rr[2] * Vel_R_1 + u_rr[3] * Vel_R_2 + u_rr[4] * Vel_R_3))
  rhoM = 0.5 * (u_ll[1] + u_rr[1])
  pM = 0.5*(p_L + p_R) -0.5*cS*rhoM*(Vel_R_1 - Vel_L_1)
  vM = 0.5*(Vel_R_1 + Vel_L_1) -1.0/(2.0*rhoM*cS)*(p_R - p_L)
  if vM >= 0
    f1 = u_ll[1] * vM
    f2 = u_ll[2] * vM
    f3 = u_ll[3] * vM
    f4 = u_ll[4] * vM
    f5 = u_ll[5] * vM

    f2 += pM
    f5 += pM*vM
  else
    f1 = u_rr[1] * vM
    f2 = u_rr[2] * vM
    f3 = u_rr[3] * vM
    f4 = u_rr[4] * vM
    f5 = u_rr[5] * vM

    f2 += pM
    f5 += pM*vM
  end

  return SVector(f1, f2, f3, f4, f5)
end

function indicator_test(u::AbstractArray{<:Any,5},
                        equations, dg::DGSEM, cache;
                        kwargs...)
  alpha = zeros(Int, nelements(dg, cache))

  for element in eachelement(dg, cache)
    # Calculate coordinates at Gauss-Lobatto nodes
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      x = Trixi.get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, k, element)
      lambda, phi, r = cart_to_sphere(x)
      if 0.22 < lambda < 3.3 && 0.45 < phi < 1.3
        alpha[element] = 1
      end
    end
  end

  return alpha
end

function Trixi.get_element_variables!(element_variables, indicator::typeof(indicator_test), ::AMRCallback)
  return nothing
end

initial_condition = initial_condition_baroclinic_instability

boundary_condition_slip_wall = BoundaryConditionWall(boundary_state_slip_wall)
# boundary_condition_slip_wall = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = Dict(
  :inside  => boundary_condition_slip_wall,
  :outside => boundary_condition_slip_wall,
)

# surface_flux = flux_lax_friedrichs
surface_flux = FluxRotated(flux_mars)
volume_flux  = flux_kennedy_gruber
solver = DGSEM(polydeg=5, surface_flux=surface_flux, volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

mesh = Trixi.P4estMeshCubedSphere(8, 4, 6371220.0, 30000.0,
                                  polydeg=5, initial_refinement_level=0)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms=source_terms_baroclinic_instability,
                                    boundary_conditions=boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10 * 24 * 60 * 60.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=0.5)

amr_controller = ControllerThreeLevel(semi, indicator_test,
                                      base_level=0,
                                      max_level=1, max_threshold=0.6)
amr_callback = AMRCallback(semi, amr_controller,
                           interval=typemax(Int),
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        amr_callback,
                        stepsize_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

# use a Runge-Kutta method with automatic (error based) time step size control
# sol = solve(ode, RDPK3SpFSAL49(), abstol=1.0e-6, reltol=1.0e-6,
#             save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary
