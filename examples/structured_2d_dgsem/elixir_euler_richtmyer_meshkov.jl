using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

@doc raw"""
    smoothed_discontinuity(x, a, b; slope = 15)

A smoothed discontinuity function that transitions from `a` to `b` around `x = 0`.
The `slope` parameter controls how sharp the transition is; larger values result in a steeper transition.
This is the function ``d_{a,b}`` from the paper
- Chan, Ranocha, Rueda-Ramírez, Gassner, Warburton (2022):
  On the entropy projection and the robustness of high order entropy stable discontinuous Galerkin schemes for under-resolved flows.
  [DOI: 10.3389/fphy.2022.898028](https://doi.org/10.3389/fphy.2022.898028)
"""
@inline function smoothed_discontinuity(x, a, b; slope = 15)
    return a + 0.5f0 * (1 + tanh(slope * x)) * (b - a)
end

"""
    initial_condition_richtmyer_meshkov_instability(x, t,
                                                    equations::CompressibleEulerEquations2D)

Initial condition for the Richtmyer-Meshkov instability test case in two dimensions.
Taken from section 3.1.3 of the paper
- Chan, Ranocha, Rueda-Ramírez, Gassner, Warburton (2022):
  On the entropy projection and the robustness of high order entropy stable discontinuous Galerkin schemes for under-resolved flows.
  [DOI: 10.3389/fphy.2022.898028](https://doi.org/10.3389/fphy.2022.898028)
"""
@inline function initial_condition_richtmyer_meshkov_instability(x, t,
                                                                 equations::CompressibleEulerEquations2D)
    slope = 2

    L_x = 40.0
    argument_1 = x[2] - (18 + 2 * cospi(6 * x[1] / L_x))
    rho_summand_1 = smoothed_discontinuity(argument_1, 1, 0.25, slope = slope)

    argument_2 = abs(x[2] - 4) - 2
    rho_summand_2 = smoothed_discontinuity(argument_2, 3.22, 0, slope = slope)

    rho = rho_summand_1 + rho_summand_2

    p = smoothed_discontinuity(argument_2, 4.9, 1.0, slope = slope)

    u = 0
    v = 0
    return prim2cons(SVector(rho, u, v, p), equations)
end

polydeg = 3
surface_flux = flux_hllc
volume_flux = flux_ranocha

basis = LobattoLegendreBasis(polydeg)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.01,
                                         alpha_smooth = false,
                                         variable = Trixi.density_pressure)

volume_integral = VolumeIntegralShockCapturingRRG(basis, indicator_sc;
                                                  volume_flux_dg = volume_flux,
                                                  volume_flux_fv = surface_flux,
                                                  slope_limiter = monotonized_central)

dg = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
           volume_integral = volume_integral)

num_elements_x = 38
cells_per_dimension = (num_elements_x, 3 * num_elements_x)
coordinates_min = (0.0, 0.0)
coordinates_max = (40.0 / 3, 40.0)
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max,
                      periodicity = false)

initial_condition = initial_condition_richtmyer_meshkov_instability

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg;
                                    boundary_conditions = boundary_condition_slip_wall)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 30.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval, uEltype = real(dg))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = analysis_interval,
                                     solution_variables = cons2prim)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        save_solution,
                        alive_callback)

###############################################################################
# run the simulation

sol = solve(ode, RDPK3SpFSAL49(); abstol = 1.0e-5, reltol = 1.0e-5,
            adaptive = true, dt = 1e-2,
            ode_default_options()..., callback = callbacks)
