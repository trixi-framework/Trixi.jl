
using OrdinaryDiffEq
using Trixi

###############################################################################

equations = TrafficFlowLWREquations1D()

# Use first order finite volume to prevent oscillations at the shock
solver = DGSEM(polydeg = 0, surface_flux = flux_lax_friedrichs)

coordinates_min = -1.0 # minimum coordinate
coordinates_max = 1.0 # maximum coordinate

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 9,
                n_cells_max = 30_000,
                periodicity = false)

# Example taken from http://www.clawpack.org/riemann_book/html/Traffic_flow.html#Example:-Traffic-jam                
# Discontinuous initial condition (Riemann Problem) leading to a shock that moves to the left.
# The shock corresponds to the traffic congestion.
function initial_condition_traffic_jam(x, t, equation::TrafficFlowLWREquations1D)
    scalar = x[1] < 0.0 ? 0.5 : 1.0

    return SVector(scalar)
end

###############################################################################
# Specify non-periodic boundary conditions

function outflow(x, t, equations::TrafficFlowLWREquations1D)
    return initial_condition_traffic_jam(coordinates_min, t, equations)
end
boundary_condition_outflow = BoundaryConditionDirichlet(outflow)

function boundary_condition_inflow(u_inner, orientation, normal_direction, x, t,
                                   surface_flux_function,
                                   equations::TrafficFlowLWREquations1D)
    # Calculate the boundary flux entirely from the internal solution state
    flux = Trixi.flux(u_inner, orientation, equations)

    return flux
end

boundary_conditions = (x_neg = boundary_condition_outflow,
                       x_pos = boundary_condition_inflow)

initial_condition = initial_condition_traffic_jam

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 1.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

# Note: Be careful when increasing the polynomial degree and switching from first order finite volume 
# to some actual DG method - in that case, you should also exchange the ODE solver.
sol = solve(ode, Euler(),
            dt = 42, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

summary_callback() # print the timer summary
