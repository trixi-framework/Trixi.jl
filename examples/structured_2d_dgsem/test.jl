# #Ignore this
# using Pkg
# Pkg.activate("..")

using Trixi
using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Plots

equations = ShallowWaterEquations2D(gravity_constant = 9.81)
# Define the initial condition

function initial_condition_test(x, t, equations::ShallowWaterEquations2D)
        x1, x2 = x
        v1 = 0.0
        v2 = 0.0
        b = 0.0
        H = 1.0
        return  prim2cons(SVector(H, v1, v2, b), equations)
end

initial_condition = initial_condition_test
    
# Define the dirichlet boundary condition
boundary_conditions = BoundaryConditionDirichlet(initial_condition_test)

# Define the problem solver

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
solver = DGSEM(polydeg = 3,
               surface_flux = (flux_lax_friedrichs, flux_nonconservative_fjordholm_etal),
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

xl = 0.0;   yl = 0.0;
xr = 3.0;   yr = 3.0;
coordinates_min = (xl, yl)
coordinates_max = (xr, yr)
cells_per_dimension = (30, 30)


# Define the mesh


# If we use tree mesh, it is good.
#mesh = TreeMesh(coordinates_min, coordinates_max, initial_refinement_level = 5, n_cells_max = 10_000, periodicity = false)         

# If we use a structured mesh is wrong.
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max, periodicity = false)


# Create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions = boundary_conditions)

###############################################################################
# ODE solver

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

###############################################################################
# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (lake_at_rest_error,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
