
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Navier-Stokes equations

prandtl_number() = 0.72
mu = 6.25e-4 # equivalent to Re = 1600

equations = CompressibleEulerEquations3D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion3D(equations, mu = mu,
                                                          Prandtl = prandtl_number())

function initial_condition_3d_blast_wave(x, t, equations::CompressibleEulerEquations3D)
    rho_c = 1.0
    p_c = 1.0
    u_c = 0.0

    rho_o = 0.125
    p_o = 0.1
    u_o = 0.0

    rc = 0.5
    r = sqrt(x[1]^2 + x[2]^2 + x[3]^2)
    if r < rc
        rho = rho_c
        v1 = u_c
        v2 = u_c
        v3 = u_c
        p = p_c
    else
        rho = rho_o
        v1 = u_o
        v2 = u_o
        v3 = u_o
        p = p_o
    end

    return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end
initial_condition = initial_condition_3d_blast_wave

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 1.0,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

coordinates_min = (-1.0, -1.0, -1.0) .* pi
coordinates_max = (1.0, 1.0, 1.0) .* pi

trees_per_dimension = (4, 4, 4)

mesh = P4estMesh(trees_per_dimension, polydeg = 3,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 periodicity = (true, true, true), initial_refinement_level = 1)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.8)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
save_solution = SaveSolutionCallback(interval = analysis_interval,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

amr_indicator = IndicatorLöhner(semi, variable = Trixi.density)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 0,
                                      med_level = 1, med_threshold = 0.05,
                                      max_level = 3, max_threshold = 0.1)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 10,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        amr_callback,
                        save_solution)

###############################################################################
# run the simulation

time_int_tol = 1e-8
sol = solve(ode, RDPK3SpFSAL49(); abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)
summary_callback() # print the timer summary
