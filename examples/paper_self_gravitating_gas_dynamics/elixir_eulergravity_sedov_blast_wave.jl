
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations_euler = CompressibleEulerEquations2D(gamma)

"""
    initial_condition_sedov_self_gravity(x, t, equations::CompressibleEulerEquations2D)

Adaptation of the Sedov blast wave with self-gravity taken from
- Michael Schlottke-Lakemper, Andrew R. Winters, Hendrik Ranocha, Gregor J. Gassner (2020)
  A purely hyperbolic discontinuous Galerkin approach for self-gravitating gas dynamics
  [arXiv: 2008.10593](https://arxiv.org/abs/2008.10593)
based on
- https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node187.html#SECTION010114100000000000000
Should be used together with [`boundary_condition_sedov_self_gravity`](@ref).
"""
function initial_condition_sedov_self_gravity(x, t, equations::CompressibleEulerEquations2D)
    # Set up polar coordinates
    r = sqrt(x[1]^2 + x[2]^2)

    # Setup based on https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node187.html#SECTION010114100000000000000
    r0 = 0.125 # = 4.0 * smallest dx (for domain length=8 and max-ref=8)
    E = 1.0
    p_inner = (equations.gamma - 1) * E / (pi * r0^2)
    p_ambient = 1e-5 # = true Sedov setup

    # Calculate primitive variables
    # use a logistic function to transfer density value smoothly
    L = 1.0    # maximum of function
    x0 = 1.0    # center point of function
    k = -150.0 # sharpness of transfer
    logistic_function_rho = L / (1.0 + exp(-k * (r - x0)))
    rho_ambient = 1e-5
    rho = max(logistic_function_rho, rho_ambient) # clip background density to not be so tiny

    # velocities are zero
    v1 = 0.0
    v2 = 0.0

    # use a logistic function to transfer pressure value smoothly
    logistic_function_p = p_inner / (1.0 + exp(-k * (r - r0)))
    p = max(logistic_function_p, p_ambient)

    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_sedov_self_gravity

"""
    boundary_condition_sedov_self_gravity(u_inner, orientation, direction, x, t,
                                          surface_flux_function,
                                          equations::CompressibleEulerEquations2D)

Adaptation of the Sedov blast wave with self-gravity taken from
- Michael Schlottke-Lakemper, Andrew R. Winters, Hendrik Ranocha, Gregor J. Gassner (2020)
  A purely hyperbolic discontinuous Galerkin approach for self-gravitating gas dynamics
  [arXiv: 2008.10593](https://arxiv.org/abs/2008.10593)
based on
- https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node187.html#SECTION010114100000000000000
Should be used together with [`initial_condition_sedov_self_gravity`](@ref).
"""
function boundary_condition_sedov_self_gravity(u_inner, orientation, direction, x, t,
                                               surface_flux_function,
                                               equations::CompressibleEulerEquations2D)
    # velocities are zero, density/pressure are ambient values according to
    # initial_condition_sedov_self_gravity
    rho = 1e-5
    v1 = 0.0
    v2 = 0.0
    p = 1e-5

    u_boundary = prim2cons(SVector(rho, v1, v2, p), equations)

    # Calculate boundary flux
    if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end

    return flux
end
boundary_conditions = boundary_condition_sedov_self_gravity

surface_flux = FluxHLL(min_max_speed_naive)
volume_flux = flux_chandrashekar
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations_euler, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
solver_euler = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-4, -4)
coordinates_max = (4, 4)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                n_cells_max = 100_000,
                periodicity = false)

semi_euler = SemidiscretizationHyperbolic(mesh, equations_euler, initial_condition,
                                          solver_euler,
                                          boundary_conditions = boundary_conditions)

###############################################################################
# semidiscretization of the hyperbolic diffusion equations
equations_gravity = HyperbolicDiffusionEquations2D()

"""
    initial_condition_sedov_self_gravity(x, t, equations::HyperbolicDiffusionEquations2D)

Adaptation of the Sedov blast wave with self-gravity taken from
- Michael Schlottke-Lakemper, Andrew R. Winters, Hendrik Ranocha, Gregor J. Gassner (2020)
  A purely hyperbolic discontinuous Galerkin approach for self-gravitating gas dynamics
  [arXiv: 2008.10593](https://arxiv.org/abs/2008.10593)
based on
- https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node187.html#SECTION010114100000000000000
Should be used together with [`boundary_condition_sedov_self_gravity`](@ref).
"""
function initial_condition_sedov_self_gravity(x, t,
                                              equations::HyperbolicDiffusionEquations2D)
    # for now just use constant initial condition for sedov blast wave (can likely be improved)
    phi = 0.0
    q1 = 0.0
    q2 = 0.0
    return SVector(phi, q1, q2)
end

"""
    boundary_condition_sedov_self_gravity(u_inner, orientation, direction, x, t,
                                          surface_flux_function,
                                          equations::HyperbolicDiffusionEquations2D)

Adaptation of the Sedov blast wave with self-gravity taken from
- Michael Schlottke-Lakemper, Andrew R. Winters, Hendrik Ranocha, Gregor J. Gassner (2020)
  A purely hyperbolic discontinuous Galerkin approach for self-gravitating gas dynamics
  [arXiv: 2008.10593](https://arxiv.org/abs/2008.10593)
based on
- https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node187.html#SECTION010114100000000000000
Should be used together with [`initial_condition_sedov_self_gravity`](@ref).
"""
function boundary_condition_sedov_self_gravity(u_inner, orientation, direction, x, t,
                                               surface_flux_function,
                                               equations::HyperbolicDiffusionEquations2D)
    u_boundary = initial_condition_sedov_self_gravity(x, t, equations)

    # Calculate boundary flux
    if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end

    return flux
end

solver_gravity = DGSEM(polydeg, flux_lax_friedrichs)

semi_gravity = SemidiscretizationHyperbolic(mesh, equations_gravity, initial_condition,
                                            solver_gravity,
                                            boundary_conditions = boundary_conditions,
                                            source_terms = source_terms_harmonic)

###############################################################################
# combining both semidiscretizations for Euler + self-gravity
parameters = ParametersEulerGravity(background_density = 0.0, # aka rho0
                                    gravitational_constant = 6.674e-8, # aka G
                                    cfl = 2.4,
                                    resid_tol = 1.0e-4,
                                    n_iterations_max = 100,
                                    timestep_gravity = timestep_gravity_erk52_3Sstar!)

semi = SemidiscretizationEulerGravity(semi_euler, semi_gravity, parameters)

###############################################################################
# ODE solvers, callbacks etc.
tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max = 1.0,
                                          alpha_min = 0.0,
                                          alpha_smooth = false,
                                          variable = density_pressure)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 2,
                                      max_level = 8, max_threshold = 0.0003)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 1,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

stepsize_callback = StepsizeCallback(cfl = 1.0)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

analysis_interval = 100
alive_callback = AliveCallback(analysis_interval = analysis_interval)

analysis_callback = AnalysisCallback(semi_euler, interval = analysis_interval,
                                     save_analysis = true,
                                     extra_analysis_integrals = (energy_total,
                                                                 energy_kinetic,
                                                                 energy_internal))

callbacks = CallbackSet(summary_callback, amr_callback, stepsize_callback,
                        save_solution,
                        analysis_callback, alive_callback)

###############################################################################
# run the simulation
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
println("Number of gravity subcycles: ", semi.gravity_counter.ncalls_since_readout)
