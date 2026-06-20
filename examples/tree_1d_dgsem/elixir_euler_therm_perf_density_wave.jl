using OrdinaryDiffEqLowStorageRK
using Trixi
using Trixi: ForwardDiff

###############################################################################
# semidiscretization of the compressible Euler equations

#=
Data taken from https://ntrs.nasa.gov/api/citations/20020085330/downloads/20020085330.pdf page 276/284

Air Mole%:N2 78.084,O2 20.9476,Ar .9365,CO2 .0319.Gordon,1982.Reac
2 g 9/95 N 1.5617O .41959AR.00937C .00032 .00000 0 28.9651159 -125.530
200.000 1000.0007 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 8649.264
1.009950160D+04-1.968275610D+02 5.009155110D+00-5.761013730D-03 1.066859930D-05
-7.940297970D-09 2.185231910D-12 -1.767967310D+02-3.921504225D+00
1000.000 6000.0007 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 8649.264
2.415214430D+05-1.257874600D+03 5.144558670D+00-2.138541790D-04 7.065227840D-08
-1.071483490D-11 6.577800150D-16 6.462263190D+03-8.147411905D+00
=#

M = 0.0289651159 # [kg/mol]
R_universal = 8.31446261815324 # [J/(mol K)]
R_specific = R_universal / M # [J/(kg K)]

temp_bounds = SVector(200.0, 1000.0, 6000.0) # [K]

a_cold = [1.009950160e+04; -1.968275610e+02; 5.009155110e+00; -5.761013730e-03;
          1.066859930e-05; -7.940297970e-09; 2.185231910e-12; -1.767967310e+02;
          -3.921504225e+00]
a_hot = [2.415214430e+05; -1.257874600e+03; 5.144558670e+00; -2.138541790e-04;
         7.065227840e-08; -1.071483490e-11; 6.577800150e-16; 6.462263190e+03;
         -8.147411905e+00]
a_ = hcat(a_cold, a_hot)
a = Trixi.SMatrix{9, 2}(a_)

function eos()
    ThermallyPerfectGas9PolyFit(R_specific = R_specific,
                                temperature_bounds = temp_bounds,
                                a = a)
end
equations = NonIdealCompressibleEulerEquations1D(eos())

# The default amplitude and frequency k are consistent with initial_condition_density_wave 
# for CompressibleEulerEquations1D. Note that this initial condition may not define admissible 
# solution states for all non-ideal equations of state!
function Trixi.initial_condition_density_wave(x, t,
                                              equations::NonIdealCompressibleEulerEquations1D;
                                              amplitude = 0.98, k = 2)
    RealT = eltype(x)
    eos = equations.equation_of_state

    v1 = convert(RealT, 0.1) # [m/s]
    rho = 1.225 + convert(RealT, amplitude) * sinpi(k * (x[1] - v1 * t)) # [kg/m^3]
    p = 101325 # [Pa]

    V = inv(rho)

    # invert for temperature given p, V
    T = 300 # [K]
    tol = 100 * eps(RealT)
    dp = pressure(V, T, eos()) - p
    iter = 1
    while abs(dp) / abs(p) > tol && iter < 100
        dp = pressure(V, T, eos()) - p
        dpdT_V = ForwardDiff.derivative(T -> pressure(V, T, eos()), T)
        T = max(tol, T - dp / dpdT_V)
        iter += 1
    end
    if iter == 100
        @warn "Solver for temperature(V, p) did not converge"
    end

    return thermo2cons(SVector(V, v1, T), equations)
end
initial_condition = initial_condition_density_wave

solver = DGSEM(polydeg = 3, surface_flux = flux_hll)

coordinates_min = -1.0 # [m]
coordinates_max = 1.0  # [m]
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 30_000, periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.1) # [s]
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 2000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 1.2)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = stepsize_callback(ode), # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
