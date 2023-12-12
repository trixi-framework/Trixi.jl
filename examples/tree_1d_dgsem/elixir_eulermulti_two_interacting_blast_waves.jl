
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerMulticomponentEquations1D(gammas = (1.4, 1.4, 1.4),
                                                       gas_constants = (0.4, 0.4, 0.4))

"""
    initial_condition_two_interacting_blast_waves(x, t, equations::CompressibleEulerMulticomponentEquations1D)

A multicomponent two interacting blast wave test taken from
- T. Plewa & E. Müller (1999)
  The consistent multi-fluid advection method
  [arXiv: 9807241](https://arxiv.org/pdf/astro-ph/9807241.pdf)
"""
function initial_condition_two_interacting_blast_waves(x, t,
                                                       equations::CompressibleEulerMulticomponentEquations1D)
    rho1 = 0.5 * x[1]^2
    rho2 = 0.5 * (sin(20 * x[1]))^2
    rho3 = 1 - rho1 - rho2

    prim_rho = SVector{3, real(equations)}(rho1, rho2, rho3)

    v1 = 0.0

    if x[1] <= 0.1
        p = 1000
    elseif x[1] < 0.9
        p = 0.01
    else
        p = 100
    end

    prim_other = SVector{2, real(equations)}(v1, p)

    return prim2cons(vcat(prim_other, prim_rho), equations)
end
initial_condition = initial_condition_two_interacting_blast_waves

function boundary_condition_two_interacting_blast_waves(u_inner, orientation, direction,
                                                        x, t, surface_flux_function,
                                                        equations::CompressibleEulerMulticomponentEquations1D)
    u_inner_reflect = SVector(-u_inner[1], u_inner[2], u_inner[3], u_inner[4], u_inner[5])
    # Calculate boundary flux
    if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_inner_reflect, orientation, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_inner_reflect, u_inner, orientation, equations)
    end

    return flux
end
boundary_conditions = boundary_condition_two_interacting_blast_waves

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha
basis = LobattoLegendreBasis(3)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.8,
                                         alpha_min = 0.0,
                                         alpha_smooth = true,
                                         variable = pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (0.0,)
coordinates_max = (1.0,)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 9,
                n_cells_max = 10_000,
                periodicity = false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.038)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000

analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.1)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
