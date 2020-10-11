
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 5/3
equations_euler = CompressibleEulerEquations2D(gamma)

# TODO: Taal, define initial_conditions_jeans_instability here for Euler
initial_conditions = Trixi.initial_conditions_jeans_instability

polydeg = 3
solver_euler = DGSEM(polydeg, flux_hll)

coordinates_min = (0, 0)
coordinates_max = (1, 1)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=10_000)

semi_euler = SemidiscretizationHyperbolic(mesh, equations_euler, initial_conditions, solver_euler)


###############################################################################
# semidiscretization of the hyperbolic diffusion equations
resid_tol = 1.0e-4
equations_gravity = HyperbolicDiffusionEquations2D(resid_tol)

# TODO: Taal, define initial_conditions_jeans_instability here for gravity

solver_gravity = DGSEM(polydeg, flux_lax_friedrichs)

semi_gravity = SemidiscretizationHyperbolic(mesh, equations_gravity, initial_conditions, solver_gravity,
                                            source_terms=source_terms_harmonic)


###############################################################################
# combining both semidiscretizations for Euler + self-gravity
parameters = ParametersEulerGravity(background_density=1.5e7, # aka rho0
                                    gravitational_constant=6.674e-8, # aka G
                                    cfl=2.4,
                                    n_iterations_max=1000,
                                    timestep_gravity=timestep_gravity_erk52_3Sstar!)

semi = SemidiscretizationEulerGravity(semi_euler, semi_gravity, parameters)


###############################################################################
# ODE solvers, callbacks etc.
tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()

stepsize_callback = StepsizeCallback(cfl=1.0)

save_solution = SaveSolutionCallback(interval=10,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=:primitive)
# TODO: Taal, IO
# restart_interval = 10

analysis_interval = 100
alive_callback = AliveCallback(analysis_interval=analysis_interval)

Trixi.pretty_form_repl(::Val{:energy_potential}) = "∑e_potential"
Trixi.pretty_form_file(::Val{:energy_potential}) = "e_potential"

function Trixi.analyze(::Val{:energy_potential}, du, u_euler, t, semi::SemidiscretizationEulerGravity)

  u_gravity = Trixi.wrap_array(semi.cache.u_ode, semi.semi_gravity)

  mesh, equations_euler, dg, cache = Trixi.mesh_equations_solver_cache(semi.semi_euler)
  _, equations_gravity, _, _ = Trixi.mesh_equations_solver_cache(semi.semi_gravity)

  e_potential = Trixi.integrate(mesh, equations_euler, dg, cache, u_euler, equations_gravity, u_gravity) do u, i, j, element, equations_euler, dg, equations_gravity, u_gravity
    u_euler_local   = Trixi.get_node_vars(u_euler,   equations_euler,   dg, i, j, element)
    u_gravity_local = Trixi.get_node_vars(u_gravity, equations_gravity, dg, i, j, element)
    # OBS! subtraction is specific to Jeans instability test where rho0 = 1.5e7
    return (u_euler_local[1] - 1.5e7) * u_gravity_local[1]
  end
  return e_potential
end

analysis_callback = AnalysisCallback(semi_euler, interval=analysis_interval,
                                     save_analysis=true,
                                     extra_analysis_integrals=(entropy, energy_total, energy_kinetic, energy_internal, Val(:energy_potential)))

callbacks = CallbackSet(summary_callback, stepsize_callback, save_solution, analysis_callback, alive_callback)


###############################################################################
# run the simulation
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), dt=stepsize_callback(ode),
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
println("Number of gravity subcycles: ", semi.gravity_counter.ncalls_since_readout)
