using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler

equations = CompressibleEulerEquations2D(1.4)

@inline function uniform_flow_state(x, t, equations::CompressibleEulerEquations2D)
    # set the freestream flow parameters
    rho_freestream = 1.0
    u_freestream = 0.3
    p_freestream = inv(equations.gamma)

    theta = pi / 90.0 # analogous with a two degree angle of attack
    si, co = sincos(theta)
    v1 = u_freestream * co
    v2 = u_freestream * si

    prim = SVector(rho_freestream, v1, v2, p_freestream)
    return prim2cons(prim, equations)
end

initial_condition = uniform_flow_state

boundary_condition_uniform_flow = BoundaryConditionDirichlet(uniform_flow_state)
boundary_conditions = Dict(:Body => boundary_condition_uniform_flow,
                           :Button1 => boundary_condition_slip_wall,
                           :Button2 => boundary_condition_slip_wall,
                           :Eye1 => boundary_condition_slip_wall,
                           :Eye2 => boundary_condition_slip_wall,
                           :Smile => boundary_condition_slip_wall,
                           :Bowtie => boundary_condition_slip_wall)

volume_flux = flux_ranocha
# Up to version 0.13.0, `max_abs_speed_naive` was used as the default wave speed estimate of
# `const flux_lax_friedrichs = FluxLaxFriedrichs(), i.e., `FluxLaxFriedrichs(max_abs_speed = max_abs_speed_naive)`.
# In the `StepsizeCallback`, though, the less diffusive `max_abs_speeds` is employed which is consistent with `max_abs_speed`.
# Thus, we exchanged in PR#2458 the default wave speed used in the LLF flux to `max_abs_speed`.
# To ensure that every example still runs we specify explicitly `FluxLaxFriedrichs(max_abs_speed_naive)`.
# We remark, however, that the now default `max_abs_speed` is in general recommended due to compliance with the 
# `StepsizeCallback` (CFL-Condition) and less diffusion.
solver = DGSEM(polydeg = 5, surface_flux = FluxLaxFriedrichs(max_abs_speed_naive),
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

# Get the unstructured quad mesh from a file (downloads the file if not available locally)
mesh_file = Trixi.download("https://gist.githubusercontent.com/andrewwinters5000/0e9e990a04b5105d1d2e3096a6e41272/raw/0d924b1d7e7d3cc1070a6cc22fe1d501687aa6dd/abaqus_gingerbread_man.inp",
                           joinpath(@__DIR__, "abaqus_gingerbread_man.inp"))

mesh = P4estMesh{2}(mesh_file)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

amr_indicator = IndicatorLöhner(semi, variable = density)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 0,
                                      med_level = 1, med_threshold = 0.05,
                                      max_level = 3, max_threshold = 0.1)

amr_callback = AMRCallback(semi, amr_controller,
                           interval = 5,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        amr_callback)

###############################################################################
# run the simulation
sol = solve(ode, RDPK3SpFSAL49(); abstol = 1.0e-7, reltol = 1.0e-7,
            ode_default_options()..., callback = callbacks);
