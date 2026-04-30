using OrdinaryDiffEqLowStorageRK
using OrdinaryDiffEqSSPRK
using Trixi
using Trixi: ForwardDiff

###############################################################################
# semidiscretization of the compressible Euler equations

eos = StiffenedGas()
equations = NonIdealCompressibleEulerEquations1D(eos)

# A basic density wave with initial condition scaled to match physical units of liquid water
function initial_condition_density_wave(x, t,
                                        equations::NonIdealCompressibleEulerEquations1D)
    RealT = eltype(x)
    eos = equations.equation_of_state
    (; pInf, cv0, gamma) = eos
    v1 = 100.0 # m/s
    rho = 1000 * (1 + 0.5 * sin(2 * pi * (x[1] - v1 * t))) # kg/m^3
    p = 1e6 # Pa
    T = (p + pInf) * inv(rho) / (cv0 * (gamma - 1))
    return thermo2cons(SVector(inv(rho), v1, T), equations)
end

initial_condition = initial_condition_density_wave

surface_flux = FluxPlusDissipation(volume_flux,
                                   DissipationLocalLaxFriedrichs(max_abs_speed))

basis = LobattoLegendreBasis(3)
volume_integral = VolumeIntegralWeakForm()

solver = DGSEM(basis, surface_flux, volume_integral)

cells_per_dimension = (100,);
coordinates_min = (-1.0,);
coordinates_max = (1.0,);
mesh = StructuredMesh(cells_per_dimension,
                      coordinates_min, coordinates_max,
                      periodicity = true)

boundary_conditions = (x_neg = boundary_condition_periodic,
                       x_pos = boundary_condition_periodic);

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0, 2.0)

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 50000
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
            dt = stepsize_callback(ode), saveat = LinRange(tspan..., 100),# solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
