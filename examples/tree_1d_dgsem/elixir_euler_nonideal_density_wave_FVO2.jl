using OrdinaryDiffEqLowStorageRK
using Trixi
using Trixi: ForwardDiff

###############################################################################
# semidiscretization of the compressible Euler equations

eos = VanDerWaals(; a = 10, b = 1e-2, gamma = 1.4, R = 287)
equations = NonIdealCompressibleEulerEquations1D(eos)

# Reduce amplitude of the density wave to avoid crashes due to 
# overshoots at `reconstruction_O2_full`
function Trixi.initial_condition_density_wave(x, t,
                                              equations::NonIdealCompressibleEulerEquations1D;
                                              amplitude = 0.6, k = 2)
    RealT = eltype(x)
    eos = equations.equation_of_state

    v1 = convert(RealT, 0.1)
    rho = 1 + convert(RealT, amplitude) * sinpi(k * (x[1] - v1 * t))
    p = 20

    V = inv(rho)

    # invert for temperature given p, V
    T = 1
    tol = 100 * eps(RealT)
    dp = pressure(V, T, eos) - p
    iter = 1
    while abs(dp) / abs(p) > tol && iter < 100
        dp = pressure(V, T, eos) - p
        dpdT_V = ForwardDiff.derivative(T -> pressure(V, T, eos), T)
        T = max(tol, T - dp / dpdT_V)
        iter += 1
    end
    if iter == 100
        @warn "Solver for temperature(V, p) did not converge"
    end

    return thermo2cons(SVector(V, v1, T), equations)
end
initial_condition = initial_condition_density_wave

polydeg = 6 # governs in this case only the number of subcells
basis = LobattoLegendreBasis(polydeg)
surface_flux = flux_hll

volume_integral = VolumeIntegralPureLGLFiniteVolumeO2(basis,
                                                      volume_flux_fv = surface_flux,
                                                      reconstruction_mode = reconstruction_O2_full,
                                                      slope_limiter = minmod,
                                                      cons2recon = cons2thermo,
                                                      recon2cons = thermo2cons)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

coordinates_min = -1.0
coordinates_max = 1.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 30_000, periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 2000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:conservation_error,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = stepsize_callback(ode), # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
