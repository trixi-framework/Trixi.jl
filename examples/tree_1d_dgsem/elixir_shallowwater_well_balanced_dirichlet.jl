
using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the shalow water equations

equations = ShallowWaterEquations1D(gravity_constant=1.0, H0=3.0)

# An initial condition with constant total water height and zero velocities to test well-balancedness.
function initial_condition_well_balancedness(x, t, equations::ShallowWaterEquations1D)
    # Set the background values
    H = equations.H0
    v = 0.0
    
    b = (  1.5 / exp( 0.5 * ((x[1] - 1.0)^2))
            + 0.75 / exp( 0.5 * ((x[1] + 1.0)^2) ) )
    return prim2cons(SVector(H, v, b), equations)
end
  
initial_condition = initial_condition_well_balancedness

boundary_condition = BoundaryConditionDirichlet(initial_condition)

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
solver = DGSEM(polydeg=4, surface_flux=(flux_hll, flux_nonconservative_fjordholm_etal),
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Get the TreeMesh and setup a periodic mesh

coordinates_min = 0.0
coordinates_max = sqrt(2.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=3,
                n_cells_max=10_000,
                periodicity=false)

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_condition)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 100.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     save_analysis=true,
                                     extra_analysis_integrals=(lake_at_rest_error,))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1000,
                                     save_initial_solution=true,
                                     save_final_solution=true)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution)

###############################################################################
# run the simulation

# use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(), abstol=1.0e-11, reltol=1.0e-11,
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
