
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

"""
    initial_condition_double_mach_reflection(x, t, equations::CompressibleEulerEquations2D)

Compressible Euler setup for a double Mach reflection problem.
Involves strong shock interactions as well as steady / unsteady flow structures.
Also exercises special boundary conditions along the bottom of the domain that is a mixture of
Dirichlet and slip wall.
See Section IV c on the paper below for details.

- Paul Woodward and Phillip Colella (1984)
  The Numerical Simulation of Two-Dimensional Fluid Flows with Strong Shocks.
  [DOI: 10.1016/0021-9991(84)90142-6](https://doi.org/10.1016/0021-9991(84)90142-6)
"""
# @inline function initial_condition_double_mach_reflection(x, t, equations::CompressibleEulerEquations2D)

#   if x[1] < 1 / 6 + (x[2] + 20 * t) / sqrt(3)
#     phi = pi / 6
#     sin_phi, cos_phi = sincos(phi)

#     rho =  8
#     v1  =  8.25 * cos_phi
#     v2  = -8.25 * sin_phi
#     p   =  116.5
#   else
#     rho = 1.4
#     v1  = 0
#     v2  = 0
#     p   = 1
#   end

#   prim = SVector(rho, v1, v2, p)
#   return prim2cons(prim, equations)
# end
initial_condition = Trixi.initial_condition_double_mach_reflection


# boundary_condition_inflow = BoundaryConditionDirichlet(initial_condition_double_mach_reflection)
boundary_condition_inflow_outflow = BoundaryConditionCharacteristic(initial_condition)


# Supersonic outflow boundary condition. Solution is taken entirely from the internal state.
# See `examples/p4est_2d_dgsem/elixir_euler_forward_step_amr.jl` for complete documentation.
# @inline function boundary_condition_outflow(u_inner, normal_direction::AbstractVector, direction, x, t,
#                                             surface_flux_function, equations::CompressibleEulerEquations2D)
#   # NOTE: Only for the supersonic outflow is this strategy valid
#   # Calculate the boundary flux entirely from the internal solution state
#   return flux(u_inner, normal_direction, equations)
# end

# Special mixed boundary condition type for the :Bottom of the domain.
# It is Dirichlet when x < 1/6 and a slip wall when x >= 1/6
# @inline function boundary_condition_mixed_dirichlet_wall(u_inner, normal_direction::AbstractVector, direction,
#                                                          x, t, surface_flux_function,
#                                                          equations::CompressibleEulerEquations2D)
#   if x[1] < 1 / 6
#     # # From the BoundaryConditionDirichlet
#     # # get the external value of the solution
#     # u_boundary = initial_condition_double_mach_reflection(x, t, equations)
#     # # Calculate boundary flux
#     # flux = surface_flux_function(u_inner, u_boundary, normal_direction, equations)

#     # From the BoundaryConditionCharacteristic
#     # get the external state of the solution
#     u_boundary = Trixi.characteristic_boundary_value_function(initial_condition,
#                                                               u_inner, normal_direction, direction, x, t, equations)
#     # Calculate boundary flux
#     flux = surface_flux_function(u_boundary, u_inner, normal_direction, equations)
#   else # x[1] >= 1 / 6
#     # Use the free slip wall BC otherwise
#     flux = boundary_condition_slip_wall(u_inner, normal_direction, direction, x, t, surface_flux_function, equations)
#   end

#   return flux
# end

boundary_conditions = (y_neg=Trixi.boundary_condition_mixed_dirichlet_wall,
                       y_pos=boundary_condition_inflow_outflow,
                       x_pos=boundary_condition_inflow_outflow,
                       x_neg=boundary_condition_inflow_outflow)

surface_flux = flux_lax_friedrichs
volume_flux  = flux_ranocha
polydeg = 4
basis = LobattoLegendreBasis(polydeg)

limiter_mcl = SubcellLimiterMCL(equations, basis;
                                DensityLimiter=true,
                                DensityAlphaForAll=false,
                                SequentialLimiter=true,
                                ConservativeLimiter=false,
                                DensityPositivityLimiter=false,
                                PressurePositivityLimiterKuzmin=false,
                                SemiDiscEntropyLimiter=false,
                                Plotting=true)
volume_integral = VolumeIntegralSubcellLimiting(limiter_mcl;
                                                volume_flux_dg=volume_flux,
                                                volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

initial_refinement_level = 6
cells_per_dimension = (4 * 2^initial_refinement_level, 2^initial_refinement_level)
coordinates_min = (0.0, 0.0)
coordinates_max = (4.0, 1.0)
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max, periodicity=false)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.2)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 500
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_integrals=(entropy,))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1000,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=0.9)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback,
                        save_solution)

###############################################################################
# run the simulation

stage_callbacks = (BoundsCheckCallback(save_errors=false),)

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks=stage_callbacks);
                  dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  callback=callbacks);
summary_callback() # print the timer summary
