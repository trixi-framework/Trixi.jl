
using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the shallow water equations

equations = ShallowWaterEquations1D(gravity_constant=9.812, H0=1.75)

function initial_condition_stone_throw(x, t, equations::ShallowWaterEquations1D)
    # Set up polar coordinates
    inicenter = 0.15
    x_norm = x[1] - inicenter[1]
    r = abs(x_norm)

    # Calculate primitive variables
    H = equations.H0
    # v = 0.0 # for well-balanced test
    v = r < 0.6 ? 1.75 : 0.0 # for stone throw

    b = (  1.5 / exp( 0.5 * ((x[1] - 1.0)^2 ) )
       + 0.75 / exp( 0.5 * ((x[1] + 1.0)^2 ) ) )

    return prim2cons(SVector(H, v, b), equations)
end

initial_condition = initial_condition_stone_throw

boundary_condition = boundary_condition_slip_wall

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
surface_flux = (FluxHydrostaticReconstruction(flux_lax_friedrichs, hydrostatic_reconstruction_audusse_etal),
                flux_nonconservative_audusse_etal)
basis = LobattoLegendreBasis(4)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=waterheight_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

###############################################################################
# Create the TreeMesh for the domain [-3, 3]

coordinates_min = -3.0
coordinates_max = 3.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=3,
                n_cells_max=10_000,
                periodicity=false)

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_condition)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

# Hack in a discontinuous bottom for a more interesting test
function initial_condition_stone_throw_discontinuous_bottom(x, t, element_id, equations::ShallowWaterEquations1D)

    inicenter = 0.15
    x_norm = x[1] - inicenter[1]
    r = abs(x_norm)

    # Calculate primitive variables
    H = equations.H0 # flat lake
    # Discontinuous velocity set via element id number
    v = 0.0
    if element_id == 4
        v = -1.0
    elseif element_id == 5
        v = 1.0
    end

    b = (  1.5 / exp( 0.5 * ((x[1] - 1.0)^2 ) )
       + 0.75 / exp( 0.5 * ((x[1] + 1.0)^2 ) ) )

    # Setup a discontinuous bottom topography using the element id number
    if element_id == 3 || element_id == 4
      b = 0.5
    end

    return prim2cons(SVector(H, v, b), equations)
end

# point to the data we want to augment
u = Trixi.wrap_array(ode.u0, semi)
# reset the initial condition
for element in eachelement(semi.solver, semi.cache)
  for i in eachnode(semi.solver)
    x_node = Trixi.get_node_coords(semi.cache.elements.node_coordinates, equations, semi.solver, i, element)
    u_node = initial_condition_stone_throw_discontinuous_bottom(x_node, first(tspan), element, equations)
    Trixi.set_node_vars!(u, u_node, equations, semi.solver, i, element)
  end
end

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=false,
                                    extra_analysis_integrals=(energy_kinetic,
                                                              energy_internal,
                                                              lake_at_rest_error))

# Enable in-situ visualization with a new plot generated every 50 time steps
# and we explicitly pass that the plot data will be one-dimensional
# visualization = VisualizationCallback(interval=50, plot_data_creator=PlotData1D)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution)#,
                        # visualization)

###############################################################################
# run the simulation

# use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(); abstol=1.0e-7, reltol=1.0e-7,
            ode_default_options()..., callback=callbacks);
summary_callback() # print the timer summary
