
using OrdinaryDiffEq
using Trixi

"""
    initial_condition_jeans_instability(x, t,
                                        equations::Union{CompressibleEulerEquations2D,
                                                         HyperbolicDiffusionEquations2D})

The classical Jeans instability taken from
- Michael Schlottke-Lakemper, Andrew R. Winters, Hendrik Ranocha, Gregor J. Gassner (2020)
  A purely hyperbolic discontinuous Galerkin approach for self-gravitating gas dynamics
  [arXiv: 2008.10593](https://arxiv.org/abs/2008.10593)
- Dominik Derigs, Andrew R. Winters, Gregor J. Gassner, Stefanie Walch (2016)
  A Novel High-Order, Entropy Stable, 3D AMR MHD Solver with Guaranteed Positive Pressure
  [arXiv: 1605.03572](https://arxiv.org/abs/1605.03572)
- Flash manual https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node189.html#SECTION010131000000000000000
in CGS (centimeter, gram, second) units.
"""
function initial_condition_jeans_instability(x, t,
                                             equations::CompressibleEulerEquations2D)
    # Jeans gravitational instability test case
    # see Derigs et al. https://arxiv.org/abs/1605.03572; Sec. 4.6
    # OBS! this uses cgs (centimeter, gram, second) units
    # periodic boundaries
    # domain size [0,L]^2 depends on the wave number chosen for the perturbation
    # OBS! Be very careful here L must be chosen such that problem is periodic
    # typical final time is T = 5
    # gamma = 5/3
    dens0 = 1.5e7 # g/cm^3
    pres0 = 1.5e7 # dyn/cm^2
    delta0 = 1e-3
    # set wave vector values for perturbation (units 1/cm)
    # see FLASH manual: https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node189.html#SECTION010131000000000000000
    kx = 2.0 * pi / 0.5 # 2π/λ_x, λ_x = 0.5
    ky = 0.0   # 2π/λ_y, λ_y = 1e10
    k_dot_x = kx * x[1] + ky * x[2]
    # perturb density and pressure away from reference states ρ_0 and p_0
    dens = dens0 * (1.0 + delta0 * cos(k_dot_x))                 # g/cm^3
    pres = pres0 * (1.0 + equations.gamma * delta0 * cos(k_dot_x)) # dyn/cm^2
    # flow starts as stationary
    velx = 0.0 # cm/s
    vely = 0.0 # cm/s
    return prim2cons((dens, velx, vely, pres), equations)
end

function initial_condition_jeans_instability(x, t,
                                             equations::HyperbolicDiffusionEquations2D)
    # gravity equation: -Δϕ = -4πGρ
    # Constants taken from the FLASH manual
    # https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node189.html#SECTION010131000000000000000
    rho0 = 1.5e7
    delta0 = 1e-3

    phi = rho0 * delta0 # constant background perturbation magnitude
    q1 = 0.0
    q2 = 0.0
    return (phi, q1, q2)
end

initial_condition = initial_condition_jeans_instability

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 5 / 3
equations_euler = CompressibleEulerEquations2D(gamma)

polydeg = 3
solver_euler = DGSEM(polydeg, FluxHLL(min_max_speed_naive))

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 10_000)

semi_euler = SemidiscretizationHyperbolic(mesh, equations_euler, initial_condition,
                                          solver_euler)

###############################################################################
# semidiscretization of the hyperbolic diffusion equations
equations_gravity = HyperbolicDiffusionEquations2D()

solver_gravity = DGSEM(polydeg, flux_lax_friedrichs)

semi_gravity = SemidiscretizationHyperbolic(mesh, equations_gravity, initial_condition,
                                            solver_gravity,
                                            source_terms = source_terms_harmonic)

###############################################################################
# combining both semidiscretizations for Euler + self-gravity
parameters = ParametersEulerGravity(background_density = 1.5e7, # aka rho0
                                    gravitational_constant = 6.674e-8, # aka G
                                    cfl = 0.8, # value as used in the paper
                                    resid_tol = 1.0e-4,
                                    n_iterations_max = 1000,
                                    timestep_gravity = timestep_gravity_carpenter_kennedy_erk54_2N!)

semi = SemidiscretizationEulerGravity(semi_euler, semi_gravity, parameters)

###############################################################################
# ODE solvers, callbacks etc.
tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()

stepsize_callback = StepsizeCallback(cfl = 0.5) # value as used in the paper

save_solution = SaveSolutionCallback(interval = 10,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

save_restart = SaveRestartCallback(interval = 100,
                                   save_final_restart = true)

analysis_interval = 100
alive_callback = AliveCallback(analysis_interval = analysis_interval)

Trixi.pretty_form_utf(::Val{:energy_potential}) = "∑e_potential"
Trixi.pretty_form_ascii(::Val{:energy_potential}) = "e_potential"

function Trixi.analyze(::Val{:energy_potential}, du, u_euler, t,
                       semi::SemidiscretizationEulerGravity)
    u_gravity = Trixi.wrap_array(semi.cache.u_ode, semi.semi_gravity)

    mesh, equations_euler, dg, cache = Trixi.mesh_equations_solver_cache(semi.semi_euler)
    _, equations_gravity, _, _ = Trixi.mesh_equations_solver_cache(semi.semi_gravity)

    e_potential = Trixi.integrate_via_indices(u_euler, mesh, equations_euler, dg, cache,
                                              equations_gravity,
                                              u_gravity) do u, i, j, element,
                                                            equations_euler, dg,
                                                            equations_gravity, u_gravity
        u_euler_local = Trixi.get_node_vars(u_euler, equations_euler, dg, i, j, element)
        u_gravity_local = Trixi.get_node_vars(u_gravity, equations_gravity, dg, i, j,
                                              element)
        # OBS! subtraction is specific to Jeans instability test where rho0 = 1.5e7
        return (u_euler_local[1] - 1.5e7) * u_gravity_local[1]
    end
    return e_potential
end

analysis_callback = AnalysisCallback(semi_euler, interval = analysis_interval,
                                     save_analysis = true,
                                     extra_analysis_integrals = (energy_total,
                                                                 energy_kinetic,
                                                                 energy_internal,
                                                                 Val(:energy_potential)))

callbacks = CallbackSet(summary_callback, stepsize_callback,
                        save_restart, save_solution,
                        analysis_callback, alive_callback)

###############################################################################
# run the simulation
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
println("Number of gravity subcycles: ", semi.gravity_counter.ncalls_since_readout)
