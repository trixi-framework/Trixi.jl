# An elixir that has an alternative convergence test that uses
# the `TimeSeriesCallback` on several gauge points. Many of the
# gauge points are selected as "stress tests" for the element
# identification, e.g., a gauge point that lies on an
# element corner of a curvilinear mesh

using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

# Modify the manufactured solution test to use `L = sqrt(2)`
# in the initial condition and source terms
function initial_condition_convergence_shifted(x, t,
                                               equations::CompressibleEulerEquations2D)
    c = 2
    A = 0.1
    L = sqrt(2)
    f = 1 / L
    ω = 2 * pi * f
    ini = c + A * sin(ω * (x[1] + x[2] - t))

    rho = ini
    rho_v1 = ini
    rho_v2 = ini
    rho_e = ini^2

    return SVector(rho, rho_v1, rho_v2, rho_e)
end

@inline function source_terms_convergence_shifted(u, x, t,
                                                  equations::CompressibleEulerEquations2D)
    # Same settings as in `initial_condition`
    c = 2
    A = 0.1
    L = sqrt(2)
    f = 1 / L
    ω = 2 * pi * f
    γ = equations.gamma

    x1, x2 = x
    si, co = sincos(ω * (x1 + x2 - t))
    rho = c + A * si
    rho_x = ω * A * co
    # Note that d/dt rho = -d/dx rho = -d/dy rho.

    tmp = (2 * rho - 1) * (γ - 1)

    du1 = rho_x
    du2 = rho_x * (1 + tmp)
    du3 = du2
    du4 = 2 * rho_x * (rho + tmp)

    return SVector(du1, du2, du3, du4)
end

initial_condition = initial_condition_convergence_shifted

source_term = source_terms_convergence_shifted

###############################################################################
# Get the DG approximation space

solver = DGSEM(polydeg = 6, surface_flux = flux_lax_friedrichs)

###############################################################################
# Get the curved quad mesh from a file (downloads the file if not available locally)

mesh_file = Trixi.download("https://gist.githubusercontent.com/andrewwinters5000/b434e724e3972a9c4ee48d58c80cdcdb/raw/55c916cd8c0294a2d4a836e960dac7247b7c8ccf/mesh_multiple_flips.mesh",
                           joinpath(@__DIR__, "mesh_multiple_flips.mesh"))

mesh = UnstructuredMesh2D(mesh_file, periodicity = true)

###############################################################################
# create the semi discretization object

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_term)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

time_series = TimeSeriesCallback(semi,
                                 [(0.75, 0.7), (1.23, 0.302), (0.8, 1.0),
                                     (0.353553390593274, 0.353553390593274),
                                     (0.505, 1.125), (1.37, 0.89), (0.349, 0.7153),
                                     (0.883883476483184, 0.406586401289607),
                                     (sqrt(2), sqrt(2))];
                                 interval = 10)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        time_series,
                        alive_callback)

###############################################################################
# run the simulation

sol = solve(ode, RDPK3SpFSAL49(); abstol = 1.0e-6, reltol = 1.0e-6,
            ode_default_options()..., callback = callbacks);

summary_callback() # print the timer summary
