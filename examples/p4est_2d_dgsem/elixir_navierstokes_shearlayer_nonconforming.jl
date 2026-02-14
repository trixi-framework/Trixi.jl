using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Navier-Stokes equations

prandtl_number() = 0.72
mu = 1.0 / 3.0 * 10^(-4) # equivalent to Re = 30,000

equations = CompressibleEulerEquations2D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu,
                                                          Prandtl = prandtl_number())

"""
A compressible version of the double shear layer initial condition. Adapted from
Brown and Minion (1995).

- David L. Brown and Michael L. Minion (1995)
  Performance of Under-resolved Two-Dimensional Incompressible Flow Simulations.
  [DOI: 10.1006/jcph.1995.1205](https://doi.org/10.1006/jcph.1995.1205)
"""
function initial_condition_shear_layer(x, t, equations::CompressibleEulerEquations2D)
    # Shear layer parameters
    RealT = eltype(x)
    k = 80
    delta = convert(RealT, 0.05)
    u0 = 1

    Ms = convert(RealT, 0.1) # maximum Mach number

    rho = 1
    v1 = x[2] <= 0.5f0 ? u0 * tanh(k * (x[2] - 0.25f0)) : u0 * tanh(k * (0.75f0 - x[2]))
    v2 = u0 * delta * sinpi(2 * (x[1] + 0.25f0))
    p = (u0 / Ms)^2 * rho / equations.gamma # scaling to get Ms

    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_shear_layer

volume_flux = flux_ranocha
solver = DGSEM(polydeg = 3, surface_flux = flux_hllc,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
# This setup is identical to the one for the `TreeMesh`, allowing for error comparison.
trees_per_dimension = (1, 1)
mesh = P4estMesh(trees_per_dimension, polydeg = 1,
                 initial_refinement_level = 4,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 periodicity = true)

# Refine bottom left cell
function refine_fn(p4est, which_tree, quadrant)
    quadrant_obj = unsafe_load(quadrant)
    if quadrant_obj.x == 0 && quadrant_obj.y == 0 && quadrant_obj.level < 5
        # return true (refine)
        return Cint(1)
    else
        # return false (don't refine)
        return Cint(0)
    end
end
refine_fn_c = @cfunction(refine_fn, Cint,
                         (Ptr{Trixi.p4est_t}, Ptr{Trixi.p4est_topidx_t},
                          Ptr{Trixi.p4est_quadrant_t}))
Trixi.refine_p4est!(mesh.p4est, true, refine_fn_c, C_NULL)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             solver_parabolic = ViscousFormulationBassiRebay1(),
                                             boundary_conditions = (boundary_condition_periodic,
                                                                    boundary_condition_periodic))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 2000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 1.3)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
