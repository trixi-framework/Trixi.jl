using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the shallow water equations with a discontinuous
# bottom topography function

equations = ShallowWaterEquations2D(gravity_constant = 9.81, H0 = 3.25)

# An initial condition with constant total water height and zero velocities to test well-balancedness.
# Note, this routine is used to compute errors in the analysis callback but the initialization is
# overwritten by `initial_condition_discontinuous_well_balancedness` below.
function initial_condition_well_balancedness(x, t, equations::ShallowWaterEquations2D)
    # Set the background values
    H = equations.H0
    v1 = 0.0
    v2 = 0.0
    # bottom topography taken from Pond.control in [HOHQMesh](https://github.com/trixi-framework/HOHQMesh)
    x1, x2 = x
    b = (1.5 / exp(0.5 * ((x1 - 1.0)^2 + (x2 - 1.0)^2)) +
         0.75 / exp(0.5 * ((x1 + 1.0)^2 + (x2 + 1.0)^2)))
    return prim2cons(SVector(H, v1, v2, b), equations)
end

initial_condition = initial_condition_well_balancedness

boundary_condition = boundary_condition_slip_wall

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
surface_flux = (flux_fjordholm_etal, flux_nonconservative_fjordholm_etal)
solver = DGSEM(polydeg = 4, surface_flux = surface_flux,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Get the TreeMesh and setup a periodic mesh

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                n_cells_max = 10_000,
                periodicity = false)

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_condition)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 100.0)
ode = semidiscretize(semi, tspan)

###############################################################################
# Workaround to set a discontinuous bottom topography and initial condition for debugging and testing.

# alternative version of the initial conditinon used to setup a truly discontinuous
# bottom topography function for this academic testcase of well-balancedness.
# The errors from the analysis callback are not important but the error for this lake at rest test case
# `∑|H0-(h+b)|` should be around machine roundoff
# In contrast to the usual signature of initial conditions, this one get passed the
# `element_id` explicitly. In particular, this initial conditions works as intended
# only for the TreeMesh2D with initial_refinement_level=2.
function initial_condition_discontinuous_well_balancedness(x, t, element_id,
                                                           equations::ShallowWaterEquations2D)
    # Set the background values
    H = equations.H0
    v1 = 0.0
    v2 = 0.0
    b = 0.0

    # Setup a discontinuous bottom topography using the element id number
    if element_id == 7
        b = 2.0 + 0.5 * sin(2.0 * pi * x[1]) + 0.5 * cos(2.0 * pi * x[2])
    end

    return prim2cons(SVector(H, v1, v2, b), equations)
end

# point to the data we want to augment
u = Trixi.wrap_array(ode.u0, semi)
# reset the initial condition
for element in eachelement(semi.solver, semi.cache)
    for j in eachnode(semi.solver), i in eachnode(semi.solver)
        x_node = Trixi.get_node_coords(semi.cache.elements.node_coordinates, equations,
                                       semi.solver, i, j, element)
        u_node = initial_condition_discontinuous_well_balancedness(x_node, first(tspan),
                                                                   element, equations)
        Trixi.set_node_vars!(u, u_node, equations, semi.solver, i, j, element)
    end
end

###############################################################################
# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (lake_at_rest_error,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 1000,
                                     save_initial_solution = true,
                                     save_final_solution = true)

stepsize_callback = StepsizeCallback(cfl = 1.0)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
