using OrdinaryDiffEqLowStorageRK
using Trixi
using Trixi: ForwardDiff

###############################################################################
# semidiscretization of the compressible Euler equations

eos = VanDerWaals(; a = 10, b = 1e-2, gamma = 1.4, R = 287)
equations = NonIdealCompressibleEulerEquations2D(eos)

# the default amplitude and frequency k are chosen to be consistent with 
# initial_condition_density_wave for CompressibleEulerEquations1D
function Trixi.initial_condition_density_wave(x, t,
                                              equations::NonIdealCompressibleEulerEquations2D;
                                              amplitude = 0.98, k = 2)
    RealT = eltype(x)

    eos = equations.equation_of_state

    v1 = convert(RealT, 0.1)
    v2 = convert(RealT, 0.2)
    rho = 1 + convert(RealT, amplitude) * sinpi(k * (x[1] + x[2] - t * (v1 + v2)))
    p = 20

    V = inv(rho)

    # invert for temperature given V, p
    T = temperature_given_Vp(V, p, eos; initial_T = one(RealT),
                             tol = 100 * eps(RealT), maxiter = 100)

    return thermo2cons(SVector(V, v1, v2, T), equations)
end
initial_condition = initial_condition_density_wave

polydeg = 3
basis = LobattoLegendreBasis(polydeg)
volume_flux = flux_terashima_etal
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
surface_flux = flux_lax_friedrichs
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 30_000,
                periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 2000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.75)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = stepsize_callback(ode), # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
