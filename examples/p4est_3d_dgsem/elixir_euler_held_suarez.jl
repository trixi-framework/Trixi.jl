# Held-Suarez test case
# Following Souza et al.:
# The Flux-Differencing Discontinuous Galerkin Method Applied to an Idealized Fully
# Compressible Nonhydrostatic Dry Atmosphere

using OrdinaryDiffEq
using Trixi
using LinearAlgebra


###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations3D(gamma)

function initial_condition_isothermal(x, t, equations::CompressibleEulerEquations3D)
    # equation (60) in the paper
    temperature = 285                     # T_I
    gas_constant = 287                    # R_d
    surface_pressure = 1e5                # p_0
    radius_earth = 6.371229e6             # r_planet
    gravitational_acceleration = 9.80616  # g

    r = norm(x)
    # Make sure that r is not smaller than radius_earth
    z = max(r - radius_earth, 0.0)
    r = z + radius_earth

    # pressure
    # geopotential formulation?
    p = surface_pressure *
        exp(gravitational_acceleration *
            (radius_earth^2 / r - radius_earth) /
            (gas_constant * temperature))

    # density (via ideal gas law)
    rho = p / (gas_constant * temperature)

    return prim2cons(SVector(rho, 0, 0, 0, p), equations)
end

@inline function source_terms_gravity_coriolis(u, x, t,
                                               equations::CompressibleEulerEquations3D)
    radius_earth = 6.371229e6             # r_planet
    gravitational_acceleration = 9.80616  # g
    angular_velocity = 7.29212e-5         # Ω
    #                  7.27220521664e-05  (Giraldo)

    r = norm(x)
    # Make sure that r is not smaller than radius_earth
    z = max(r - radius_earth, 0.0)
    r = z + radius_earth

    du1 = zero(eltype(u))

    # Gravity term
    temp = -gravitational_acceleration * radius_earth^2 / r^3
    du2 = temp * u[1] * x[1]
    du3 = temp * u[1] * x[2]
    du4 = temp * u[1] * x[3]
    du5 = temp * (u[2] * x[1] + u[3] * x[2] + u[4] * x[3])

    # Coriolis term, -2Ω × ρv = -2 * angular_velocity * (0, 0, 1) × u[2:4]
    du2 -= -2 * angular_velocity * u[3]
    du3 -= 2 * angular_velocity * u[2]

    return SVector(du1, du2, du3, du4, du5)
end

function cartesian_to_sphere(x)
    r = norm(x)
    lambda = atan(x[2], x[1])
    if lambda < 0
        lambda += 2 * pi
    end
    phi = asin(x[3] / r)

    return lambda, phi, r
end

@inline function source_terms_hs_relaxation(u, x, t,
                                            equations::CompressibleEulerEquations3D)
    # equations (55)-(58) in the paper
    secondsperday = 60*60*24
    radius_earth = 6.371229e6   # r_planet
    k_f = 1/secondsperday       # Damping scale for momentum
    k_a = 1/(40*secondsperday)  # Polar relaxation scale
    k_s = 1/(4*secondsperday)   # Equatorial relaxation scale
    T_min = 200                 # Minimum equilibrium temperature
    T_equator = 315             # Equatorial equilibrium temperature
    surface_pressure = 1e5      # p_0
    deltaT = 60                 # Latitudinal temperature difference
    deltaTheta = 10             # Vertical temperature difference
    sigma_b = 0.7               # Dimensionless damping height
    gas_constant = 287          # R_d
    c_v = 717.5                 # Specific heat capacity of dry air at constant volume
    c_p = 1004.5                # Specific heat capacity of dry air at constant pressur
 
    _, _, _, _, pressure = cons2prim(u, equations)
    lon, lat, r = cartesian_to_sphere(x)
    temperature = pressure / (u[1] * gas_constant)

    sigma = pressure / surface_pressure   # "p_0 instead of instantaneous surface pressure"
    delta_sigma = max(0, (sigma-sigma_b)/(1-sigma_b))   # "height factor"
    k_v = k_f * delta_sigma
    k_T = k_a + (k_s - k_a) * delta_sigma * cos(lat)^4

    T_equi = max(T_min,
                 (T_equator - deltaT * sin(lat)^2 - deltaTheta * log(sigma) * cos(lat)^2) *
                  sigma^(gas_constant/c_p))
    # cos^2! Clima: abs2(cos(ᶜφ))

    # project onto r, normalize! @. Yₜ.c.uₕ -= k_v * Y.c.uₕ
    # Make sure that r is not smaller than radius_earth
    z = max(r - radius_earth, 0.0)
    r = z + radius_earth
    dotprod = (u[2] * x[1] + u[3] * x[2] + u[4] * x[3]) / (r*r)
    
    du2 = -k_v * (u[2] - dotprod * x[1])
    du3 = -k_v * (u[3] - dotprod * x[2])
    du4 = -k_v * (u[4] - dotprod * x[3])

    du5 = -k_T * u[1] * c_v * (temperature - T_equi)
 
    return SVector(zero(eltype(u)), du2, du3, du4, du5)
end

@inline function source_terms_held_suarez(u, x, t,
                                          equations::CompressibleEulerEquations3D)
    return source_terms_gravity_coriolis(u,x,t,equations) +
           source_terms_hs_relaxation(u,x,t,equations)
end

initial_condition = initial_condition_isothermal

boundary_conditions = Dict(:inside => boundary_condition_slip_wall,
                           :outside => boundary_condition_slip_wall)

# This is a good estimate for the speed of sound in this example.
# Other values between 300 and 400 should work as well.
surface_flux = FluxLMARS(340)
volume_flux = flux_kennedy_gruber
solver = DGSEM(polydeg = 3, surface_flux = surface_flux,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

# For optimal results, use (16, 8) here
# Giraldo: (10,8), polydeg 4
trees_per_cube_face = (10, 4)
mesh = Trixi.P4estMeshCubedSphere(trees_per_cube_face..., 6.371229e6, 30000.0,
                                  polydeg = 3, initial_refinement_level = 0)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_held_suarez,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1000 * 24 * 60 * 60.0) # time in seconds for 10 days
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 5000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 5000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution)
                        #stepsize_callback)

###############################################################################
# run the simulation

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
sol = solve(ode,
            RDPK3SpFSAL49(thread = OrdinaryDiffEq.True()); abstol = 1.0e-5, reltol = 1.0e-5,
            ode_default_options()...,
            callback = callbacks);

summary_callback() # print the timer summary
