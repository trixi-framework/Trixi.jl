using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

# 1) Dry Air  2) SF_6
equations = CompressibleEulerMulticomponentEquations2D(gammas = (1.4, 1.648),
                                                       gas_constants = (0.287, 1.578))

"""
    initial_condition_shock(coordinates, t, equations::CompressibleEulerEquations2D)

Shock traveling from left to right where it interacts with a Perturbed interface.
"""
@inline function initial_condition_shock(x, t,
                                        equations::CompressibleEulerMulticomponentEquations2D)
    rho_0 = 1.25 # kg/m^3
    p_0 = 101325 # Pa
    T_0 = 293 # K
    u_0 = 352 # m/s
    d = 20
    w = 40

    if x[1] < 25
        # Shock region.
        v1 = 0.35
        v2 = 0.0
        rho1 = 1.72
        rho2 = 0.03
        p = 1.57
    elseif (x[1] <= 30) || (x[1] <= 70 && abs(125 - x[2]) > w/2)
        # Intermediate region.
        v1 = 0.0
        v2 = 0.0
        rho1 = 1.25
        rho2 = 0.03
        p = 1.0
    else (x[1] <= 70 + d)
        # SF_6 region.
        v1 = 0.0
        v2 = 0.0
        rho1 = 0.03
        rho2 = 6.03 #SF_6
        p = 1.0
    end

    return prim2cons(SVector(v1, v2, p, rho1, rho2), equations)
end

# Define the simulation.
initial_condition = initial_condition_shock

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha
basis = LobattoLegendreBasis(3)

limiter_idp = SubcellLimiterIDP(equations, basis;
                                positivity_variables_cons = ["rho" * string(i)
                                                             for i in eachcomponent(equations)])

volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (0.0, 0.0)
coordinates_max = (250.0, 250.0)
mesh = P4estMesh((32, 32), polydeg=3, coordinates_min=(0.0, 0.0),
                 coordinates_max=(250.0, 250.0), periodicity=(false, true),
                 initial_refinement_level=0)

# Completely free outflow
function boundary_condition_outflow(u_inner, normal_direction::AbstractVector,
                                    x, t,
                                    surface_flux_function,
                                    equations)
    # Calculate the boundary flux entirely from the internal solution state
    flux = Trixi.flux(u_inner, normal_direction, equations)

    return flux
end

boundary_conditions = Dict(:x_neg => BoundaryConditionDirichlet(initial_condition),
                           :x_pos => boundary_condition_outflow)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 10
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(dt = 2.0,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.2)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

stage_callbacks = (SubcellLimiterIDPCorrection(),
                   BoundsCheckCallback(save_errors = false, interval = 100))
# `interval` is used when calling this elixir in the tests with `save_errors=true`.

@time begin
    sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                    dt = 0.1, # solve needs some value here but it will be overwritten by the stepsize_callback
                    ode_default_options()..., callback = callbacks);
end
