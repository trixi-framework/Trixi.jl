using OrdinaryDiffEqSSPRK
using Trixi

###############################################################################
## Semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

# Variant of the 4-quadrant Riemann problem considered in 
# - Carsten W. Schulz-Rinne:
#   Classification of the Riemann Problem for Two-Dimensional Gas Dynamics
#   https://doi.org/10.1137/0524006
# and 
# - Carsten W. Schulz-Rinne, James P. Collins, and Harland M. Glaz
#   Numerical Solution of the Riemann Problem for Two-Dimensional Gas Dynamics
#   https://doi.org/10.1137/0914082

# Specify the initial condition as a discontinuous initial condition (see docstring of 
# `DiscontinuousFunction` for more information) which comes with a specialized 
# initialization routine suited for Riemann problems.
# In short, if a discontinuity is right at an interface, the boundary nodes (which are at the same location)
# on that interface will be initialized with the left and right state of the discontinuity, i.e., 
#                         { u_1, if element = left element and x_{element}^{(n)} = x_jump
# u(x_jump, t, element) = {
#                         { u_2, if element = right element and x_{element}^{(1)} = x_jump
# This is realized by shifting the outer DG nodes inwards, i.e., on reference element
# the outer nodes at `[-1, 1]` are shifted inwards to `[-1 + ε, 1 - ε]` with machine precision `ε`.
struct InitialConditionRP <: DiscontinuousFunction end

function (initial_condition_rp::InitialConditionRP)(x_, t,
                                                    equations::CompressibleEulerEquations2D)
    x, y = x_[1], x_[2]

    if x >= 0.5 && y >= 0.5
        rho, v1, v2, p = (0.5313, 0.0, 0.0, 0.4)
    elseif x < 0.5 && y >= 0.5
        rho, v1, v2, p = (1.0, 0.7276, 0.0, 1.0)
    elseif x < 0.5 && y < 0.5
        rho, v1, v2, p = (0.8, 0.0, 0.0, 1.0)
    elseif x >= 0.5 && y < 0.5
        rho, v1, v2, p = (1.0, 0.0, 0.7276, 1.0)
    end

    prim = SVector(rho, v1, v2, p)
    return prim2cons(prim, equations)
end
# Note calling the constructor of the struct: `InitialConditionRP()` instead of `initial_condition_rp` !
initial_condition = InitialConditionRP()

# Extend domain by specifying free outflow
function boundary_condition_outflow(u_inner,
                                    orientation::Integer, normal_direction,
                                    x, t,
                                    surface_flux_function,
                                    equations::CompressibleEulerEquations2D)
    flux = Trixi.flux(u_inner, orientation, equations)

    return flux
end

boundary_conditions = (x_neg = boundary_condition_outflow,
                       x_pos = boundary_condition_outflow,
                       y_neg = boundary_condition_outflow,
                       y_pos = boundary_condition_outflow)

coordinates_min = (0.0, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 100_000,
                periodicity = false)

surface_flux = flux_hllc
volume_flux = flux_ranocha

polydeg = 3
basis = LobattoLegendreBasis(polydeg)
shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max = 0.5,
                                            alpha_min = 0.001,
                                            alpha_smooth = true,
                                            variable = Trixi.density)

volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_conditions)

###############################################################################
## ODE solvers, callbacks etc.

tspan = (0.0, 0.25)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 1.8)

amr_indicator = IndicatorLöhner(semi, variable = Trixi.density)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 2,
                                      med_level = 5, med_threshold = 0.02,
                                      max_level = 8, max_threshold = 0.04)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 10,
                           adapt_initial_condition = true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        amr_callback,
                        stepsize_callback)

###############################################################################
## Run the simulation

sol = solve(ode, SSPRK54();
            dt = 1.0, save_everystep = false, callback = callbacks);
