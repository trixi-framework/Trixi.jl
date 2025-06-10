using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################

equations = TrafficFlowLWREquations1D()

solver = DGSEM(polydeg = 3, surface_flux = FluxHLL(min_max_speed_davis))

coordinates_min = -1.0 # minimum coordinate
coordinates_max = 1.0 # maximum coordinate
cells_per_dimension = (64,)

mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max,
                      periodicity = false)

# Specify the initial condition as a discontinuous initial condition (see docstring of 
# `DiscontinuousInitialCondition` for more information) which comes with a specialized 
# initialization routine suited for Riemann problems.
# In short, if a discontinuity is right at an interface, the boundary nodes (which are at the same location)
# on that interface will be initialized with the left and right state of the discontinuity, i.e., 
#                         { u_1, if element = left element and x_{element}^{(n)} = x_jump
# u(x_jump, t, element) = {
#                         { u_2, if element = right element and x_{element}^{(1)} = x_jump
# This is realized by shifting the outer DG nodes inwards, i.e., on reference element
# the outer nodes at `[-1, 1]` are shifted inwards to `[-1 + ε, 1 - ε]` with machine precision `ε`.
struct InitialConditionGreenlight <: DiscontinuousInitialCondition end

# Example inspired from http://www.clawpack.org/riemann_book/html/Traffic_flow.html#Example:-green-light
# Green light that at x = 0 which switches at t = 0 from red to green.
# To the left there are cars bumper to bumper, to the right there are no cars.
function (initial_condition_greenlight::InitialConditionGreenlight)(x, t,
                                                                    equation::TrafficFlowLWREquations1D)
    RealT = eltype(x)
    scalar = x[1] < 0 ? one(RealT) : zero(RealT)

    return SVector(scalar)
end
# Note calling the constructor of the struct: `InitialConditionGreenlight()` instead of
# `initial_condition_greenlight` !
const initial_condition = InitialConditionGreenlight()

###############################################################################
# Specify non-periodic boundary conditions

# Assume that there are always cars waiting at the left
function inflow(x, t, equations::TrafficFlowLWREquations1D)
    # -1.0 = coordinates_min
    return initial_condition(-1.0, t, equations)
end
boundary_condition_inflow = BoundaryConditionDirichlet(inflow)

# Cars may leave the modeled domain
function boundary_condition_outflow(u_inner, orientation, normal_direction, x, t,
                                    surface_flux_function,
                                    equations::TrafficFlowLWREquations1D)
    # Calculate the boundary flux entirely from the internal solution state
    flux = Trixi.flux(u_inner, orientation, equations)

    return flux
end

boundary_conditions = (x_neg = boundary_condition_inflow,
                       x_pos = boundary_condition_outflow)

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

stepsize_callback = StepsizeCallback(cfl = 1.2)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 42, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
