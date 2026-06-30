using OrdinaryDiffEqLowStorageRK
using OrdinaryDiffEqSSPRK
using Trixi

###############################################################################
# semidiscretization of the ideal compressible Navier-Stokes equations

prandtl_number() = 0.73

function initial_condition_kelvin_helmholtz_instability(x, t,
                                                        equations::CompressibleEulerEquations2D)
    # change discontinuity to tanh
    # typical resolution 128^2, 256^2
    # domain size is [-1,+1]^2
    slope = 15
    B = tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5)
    rho = 0.5 + 0.75 * B
    v1 = 0.5 * (B - 1)
    v2 = 0.1 * sin(2 * pi * x[1])
    p = 1.0
    return prim2cons(SVector(rho, v1, v2, p), equations)
end

initial_condition = initial_condition_kelvin_helmholtz_instability
tspan = (0.0, 0.5)
periodicity = (true, true)
mu() = 1e-6

equations = CompressibleEulerEquations2D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesEntropy())

dg = DGSEM(polydeg = 3, surface_flux = FluxLaxFriedrichs(max_abs_speed),
           #    volume_integral = VolumeIntegralWeakForm())
           volume_integral = VolumeIntegralFluxDifferencing(flux_shima_etal))

# surface_flux = FluxLaxFriedrichs(max_abs_speed)
# basis = LobattoLegendreBasis(3)
# indicator_sc = IndicatorHennemannGassner(equations, basis,
#                                          alpha_max = 0.25,
#                                          alpha_min = 0.001,
#                                          alpha_smooth = true,
#                                          variable = density_pressure)
# volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
#                                                  volume_flux_dg = flux_shima_etal,
#                                                 #  volume_flux_dg = flux_central,
#                                                  volume_flux_fv = surface_flux)           
# dg = DGSEM(basis, surface_flux, volume_integral)

# Create a uniformly refined mesh with periodic boundaries
coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
initial_refinement_level = 5
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                periodicity = periodicity, n_cells_max = 400_000)

# BC types
boundary_condition_noslip_wall = BoundaryConditionNavierStokesWall(NoSlip((x, t, equations_parabolic) -> (0.0,
                                                                                                          0.0)),
                                                                   Adiabatic((x, t, equations_parabolic) -> 0.0))

# define inviscid boundary conditions
boundary_conditions_hyperbolic = (; x_neg = boundary_condition_slip_wall,
                                  x_pos = boundary_condition_slip_wall,
                                  y_neg = boundary_condition_slip_wall,
                                  y_pos = boundary_condition_slip_wall)

# define viscous boundary conditions
boundary_conditions_parabolic = (; x_neg = boundary_condition_noslip_wall,
                                 x_pos = boundary_condition_noslip_wall,
                                 y_neg = boundary_condition_noslip_wall,
                                 y_pos = boundary_condition_noslip_wall)

# solver_parabolic = ParabolicFormulationBassiRebay1()
solver_parabolic = ParabolicFormulationLocalDG()

if all(mesh.tree.periodicity .== true)
    semi = SemidiscretizationArtificialViscosity(mesh, (equations, equations_parabolic),
                                                 initial_condition, dg;
                                                 combine_rhs = Trixi.True(),
                                                 solver_parabolic = solver_parabolic,
                                                 boundary_conditions = (boundary_condition_periodic,
                                                                        boundary_condition_periodic))
    # semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
    #                                              initial_condition, dg;
    #                                              solver_parabolic = solver_parabolic)

else
    semi = SemidiscretizationArtificialViscosity(mesh, (equations, equations_parabolic),
                                                 initial_condition, dg;
                                                 combine_rhs = Trixi.True(),
                                                 solver_parabolic = solver_parabolic,
                                                 boundary_conditions = (boundary_conditions_hyperbolic,
                                                                        boundary_conditions_parabolic))

    #     semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
    #                                                 initial_condition, dg;
    #                                                 solver_parabolic = solver_parabolic,
    #                                                 boundary_conditions = (boundary_conditions_hyperbolic, 
    #                                                                         boundary_conditions_parabolic))
end

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 100)
callbacks = CallbackSet(summary_callback, alive_callback)

###############################################################################
# run the simulation

solver = SSPRK43()

sol = solve(ode, solver; abstol = 1e-6, reltol = 1e-4, # dt = 1e-8,
            ode_default_options()..., callback = callbacks)
