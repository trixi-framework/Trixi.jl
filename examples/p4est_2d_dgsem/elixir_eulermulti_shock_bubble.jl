using OrdinaryDiffEqSSPRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler multicomponent equations

# 1) Dry Air  2) Helium + 28% Air
equations = CompressibleEulerMulticomponentEquations2D(gammas = (1.4, 1.648),
                                                       #gas_constants = (0.287 * 10^3, 1.578 * 10^3)
                                                       gas_constants = (0.287, 1.578)
                                                       )

"""
    initial_condition_shock_bubble(x, t, equations::CompressibleEulerMulticomponentEquations2D{5, 2})

A shock-bubble testcase for multicomponent Euler equations
- Ayoub Gouasmi, Karthik Duraisamy, Scott Murman
  Formulation of Entropy-Stable schemes for the multicomponent compressible Euler equations
  [arXiv: 1904.00972](https://arxiv.org/abs/1904.00972)
"""
function initial_condition_shock_bubble(x, t,
                                        equations::CompressibleEulerMulticomponentEquations2D{5,
                                                                                              2})
    RealT = eltype(x)
    @unpack gas_constants = equations

    # Positivity Preserving Parameter, can be set to zero if scheme is positivity preserving
    delta = convert(RealT, 0.03)

    # Region I
    rho1_1 = delta
    rho2_1 = RealT(1.225) * gas_constants[1] / gas_constants[2] - delta
    v1_1 = zero(RealT)
    v2_1 = zero(RealT)
    p_1 = 101325

    # Region II
    rho1_2 = RealT(1.225) - delta
    rho2_2 = delta
    v1_2 = zero(RealT)
    v2_2 = zero(RealT)
    p_2 = 101325

    # Region III
    rho1_3 = RealT(1.6861) - delta
    rho2_3 = delta
    v1_3 = -RealT(113.5243)
    v2_3 = zero(RealT)
    p_3 = 159060

    # Set up Region I & II:
    inicenter = SVector(0.225, 0.0445) # center of bubble
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)

    if (x[1] > 0.275)
        # Set up Region III
        rho1 = rho1_3
        rho2 = rho2_3
        v1 = v1_3
        v2 = v2_3
        p = p_3
    elseif (r < 0.025f0)
        # Set up Region I
        rho1 = rho1_1
        rho2 = rho2_1
        v1 = v1_1
        v2 = v2_1
        p = p_1
    else
        # Set up Region II
        rho1 = rho1_2
        rho2 = rho2_2
        v1 = v1_2
        v2 = v2_2
        p = p_2
    end

    return prim2cons(SVector(v1, v2, p, rho1, rho2), equations)
end
initial_condition = initial_condition_shock_bubble

surface_flux = flux_lax_friedrichs

volume_flux = flux_ranocha
basis = LobattoLegendreBasis(5)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral_stabilized = VolumeIntegralShockCapturingRRG(basis, indicator_sc;
                                                             volume_flux_dg = volume_flux,
                                                             volume_flux_fv = surface_flux,
                                                             slope_limiter = vanleer)

volume_integral = VolumeIntegralAdaptive(volume_integral_default = VolumeIntegralWeakForm(),
                                         volume_integral_stabilized = volume_integral_stabilized,
                                         indicator = nothing) # Indicator taken from `volume_integral_stabilized`

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (0.0, 0.0)
coordinates_max = (0.445, 0.089)
trees_per_dimension = (8, 2)
mesh = P4estMesh(trees_per_dimension, polydeg = 1,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 initial_refinement_level = 3, periodicity = false)
    
restart_file = "out/restart_t4.h5"
mesh = load_mesh(restart_file)

bc_LR = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = Dict(:x_neg => bc_LR, :x_pos => bc_LR,
                           :y_neg => boundary_condition_slip_wall,
                           :y_pos => boundary_condition_slip_wall,)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

# Plot times in paper above
t1 = 23.32e-6
tspan = (0.0, t1)
#ode = semidiscretize(semi, tspan)

t2 = 42.98e-6
t3 = 52.81e-6
t4 = 67.55e-6
t5 = 77.38e-6
t6 = 101.95e-6
t7 = 259.21e-6

tspan = (load_time(restart_file), t5)
ode = semidiscretize(semi, tspan, restart_file)

summary_callback = SummaryCallback()

analysis_interval = 2000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (Trixi.density,))

alive_callback = AliveCallback(alive_interval = 100)

amr_indicator = IndicatorLöhner(semi, variable = Trixi.density_pressure)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 0,
                                      med_level = 3, med_threshold = 0.0005,
                                      max_level = 5, max_threshold = 0.001)

# For first (non-restarted) run
#=
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 50,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)
=#

# For restarted runs

amr_callback = AMRCallback(semi, amr_controller,
                           interval = 50,
                           adapt_initial_condition = false)


save_solution = SaveSolutionCallback(interval = 2000)

save_restart = SaveRestartCallback(interval = 100_000,
                                   save_final_restart = true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_restart,
                        amr_callback,
                        #save_solution,
                        )

###############################################################################
# run the simulation

sol = solve(ode, SSPRK43(thread = Trixi.True());
            dt = 2e-10, ode_default_options()...,
            maxiters = Inf, callback = callbacks);
