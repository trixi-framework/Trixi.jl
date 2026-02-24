using OrdinaryDiffEqLowStorageRK
using OrdinaryDiffEqSSPRK
using Trixi
using Trixi: ForwardDiff

###############################################################################
# semidiscretization of the compressible Euler equations

eos = PengRobinson()
equations = NonIdealCompressibleEulerEquations2D(eos)

# A transcritical mixing layer adapted from "Kinetic-energy- and pressure-equilibrium-preserving 
# schemes for real-gas turbulence in the transcritical regime" by Bernades, Jofre, Capuano (2023). 
# <https://doi.org/10.1016/j.jcp.2023.112477>
function initial_condition_transcritical_mixing(x, t,
                                                equations::NonIdealCompressibleEulerEquations2D)
    eos = equations.equation_of_state

    RealT = eltype(x)

    x, y = x

    pc = 3.4e6 # critical pressure for N2
    p = 2 * pc

    # from Bernades et al
    epsilon, delta, A = 1, convert(RealT, 1 / 20), convert(RealT, 3 / 8)
    u0 = 25 # m/s
    T = eos.Tc * (3 * A - A * tanh(y / delta)) # Tc is 126.2 for N2

    tol = Trixi.eos_newton_tol(eos)

    # invert for V given p, T. Initialize V so that the denominator 
    # (V - b) in Peng-Robinson is positive. 
    V = eos.b + tol
    dp = pressure(V, T, eos) - p
    iter = 1
    while abs(dp) > tol * abs(p) && iter < 100
        dp = pressure(V, T, eos) - p
        dpdV_T = ForwardDiff.derivative(V -> pressure(V, T, eos), V)
        V = max(eos.b + tol, V - dp / dpdV_T)
        iter += 1
    end
    if iter == 100
        @warn "Solver for temperature(V, p) did not converge"
    end

    k = 6
    dv = epsilon * sinpi(k * x) * (tanh(100 * (y + 0.1)) - tanh(100 * (y - 0.1))) / 2
    v1 = u0 * (1 + convert(RealT, 0.2) * tanh(y / delta)) + dv
    v2 = dv

    return thermo2cons(SVector(V, v1, v2, T), equations)
end

initial_condition = initial_condition_transcritical_mixing

volume_flux = flux_terashima_etal
surface_flux = FluxPlusDissipation(volume_flux, DissipationLocalLaxFriedrichs())

basis = LobattoLegendreBasis(3)
volume_integral_default = VolumeIntegralFluxDifferencing(volume_flux)
volume_integral_entropy_stable = VolumeIntegralPureLGLFiniteVolume(surface_flux)
volume_integral = VolumeIntegralAdaptive(IndicatorEntropyCorrection(equations, basis),
                                         volume_integral_default,
                                         volume_integral_entropy_stable)
solver = DGSEM(basis, surface_flux, volume_integral)

cells_per_dimension = (32, 16)
coordinates_min = (-0.5, -0.25)
coordinates_max = (0.5, 0.25)
mesh = StructuredMesh(cells_per_dimension,
                      coordinates_min, coordinates_max,
                      periodicity = (true, false))

boundary_conditions = (x_neg = boundary_condition_periodic,
                       x_pos = boundary_condition_periodic,
                       y_neg = boundary_condition_slip_wall,
                       y_pos = boundary_condition_slip_wall)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0, 0.033 * 2)

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 2000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.25)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

solver = CarpenterKennedy2N54(williamson_condition = false)
sol = solve(ode, solver;
            dt = stepsize_callback(ode), # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
