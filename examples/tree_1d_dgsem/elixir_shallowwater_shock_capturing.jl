
using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the shallow water equations

equations = ShallowWaterEquations1D(gravity_constant = 9.812, H0 = 1.75)

# Initial condition with a truly discontinuous velocity and bottom topography.
# Works as intended for TreeMesh1D with `initial_refinement_level=3`. If the mesh
# refinement level is changed the initial condition below may need changed as well to
# ensure that the discontinuities lie on an element interface.
function initial_condition_stone_throw_discontinuous_bottom(x, t,
                                                            equations::ShallowWaterEquations1D)

    # Calculate primitive variables

    # flat lake
    H = equations.H0

    # Discontinuous velocity
    v = 0.0
    if x[1] >= -0.75 && x[1] <= 0.0
        v = -1.0
    elseif x[1] >= 0.0 && x[1] <= 0.75
        v = 1.0
    end

    b = (1.5 / exp(0.5 * ((x[1] - 1.0)^2)) +
         0.75 / exp(0.5 * ((x[1] + 1.0)^2)))

    # Force a discontinuous bottom topography
    if x[1] >= -1.5 && x[1] <= 0.0
        b = 0.5
    end

    return prim2cons(SVector(H, v, b), equations)
end

initial_condition = initial_condition_stone_throw_discontinuous_bottom

boundary_condition = boundary_condition_slip_wall

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
surface_flux = (FluxHydrostaticReconstruction(flux_lax_friedrichs,
                                              hydrostatic_reconstruction_audusse_etal),
                flux_nonconservative_audusse_etal)
basis = LobattoLegendreBasis(4)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = waterheight_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

###############################################################################
# Create the TreeMesh for the domain [-3, 3]

coordinates_min = -3.0
coordinates_max = 3.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 10_000,
                periodicity = false)

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_condition)

###############################################################################
# ODE solver

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

###############################################################################
# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = false,
                                     extra_analysis_integrals = (energy_kinetic,
                                                                 energy_internal,
                                                                 lake_at_rest_error))

# Enable in-situ visualization with a new plot generated every 50 time steps
# and we explicitly pass that the plot data will be one-dimensional
# visualization = VisualizationCallback(interval=50, plot_data_creator=PlotData1D)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution)#,
# visualization)

###############################################################################
# run the simulation

# use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(); abstol = 1.0e-7, reltol = 1.0e-7,
            ode_default_options()..., callback = callbacks);
summary_callback() # print the timer summary
