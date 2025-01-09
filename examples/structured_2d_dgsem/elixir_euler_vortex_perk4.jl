
using OrdinaryDiffEq # Required for `CallbackSet`
using Trixi

# Ratio of specific heats
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

solver = DGSEM(polydeg = 3, surface_flux = flux_hllc)

"""
    initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)

The classical isentropic vortex test case as presented in Section 5.1 of

- Brian Vermeire (2019).
  Paired Explicit Runge-Kutta Schemes for Stiff Systems of Equations
  [DOI:10.1016/j.jcp.2019.05.014](https://doi.org/10.1016/j.jcp.2019.05.014)
  https://spectrum.library.concordia.ca/id/eprint/985444/1/Paired-explicit-Runge-Kutta-schemes-for-stiff-sy_2019_Journal-of-Computation.pdf
"""
function initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)
    # Evaluate error after full domain traversion
    if t == t_end
        t = 0
    end

    # Initial center of the vortex
    inicenter = SVector(0.0, 0.0)
    # Strength of the vortex
    S = 13.5
    # Radius of vortex
    R = 1.5
    # Free-stream Mach 
    M = 0.4
    # Base flow
    v1 = 1.0
    v2 = 1.0
    vel = SVector(v1, v2)

    center = inicenter + vel * t # Advection of center
    center = x - center          # Distance to centerpoint
    center = SVector(center[2], -center[1])
    r2 = center[1]^2 + center[2]^2

    f = (1 - r2) / (2 * R^2)

    rho = (1 - (S * M / pi)^2 * (gamma - 1) * exp(2 * f) / 8)^(1 / (gamma - 1))

    du = S / (2 * π * R) * exp(f) # Vel. perturbation
    vel = vel + du * center
    v1, v2 = vel

    p = rho^gamma / (gamma * M^2)

    prim = SVector(rho, v1, v2, p)
    return prim2cons(prim, equations)
end
initial_condition = initial_condition_isentropic_vortex

edge_length = 20.0

N_passes = 1
t_end = edge_length * N_passes
tspan = (0.0, t_end)

coordinates_min = (-edge_length / 2, -edge_length / 2)
coordinates_max = (edge_length / 2, edge_length / 2)

cells_per_dimension = (32, 32)
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

ode = semidiscretize(semi, tspan)

###############################################################################
# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_integrals = (entropy,))

# Note quite large CFL number
stepsize_callback = StepsizeCallback(cfl = 9.1)

callbacks = CallbackSet(summary_callback,
                        stepsize_callback,
                        analysis_callback)

###############################################################################
# set up time integration algorithm

num_stages = 19
coefficient_file = "a_" * string(num_stages) * ".txt"

# Download the optimized PERK4 coefficients
path_coeff_file = mktempdir()
Trixi.download("https://gist.githubusercontent.com/DanielDoehring/84f266ff61f0a69a0127cec64056275e/raw/1a66adbe1b425d33daf502311ecbdd4b191b89cc/a_19.txt",
               joinpath(path_coeff_file, coefficient_file))

ode_algorithm = Trixi.PairedExplicitRK4(num_stages, path_coeff_file)

###############################################################################
# run the simulation

sol = Trixi.solve(ode, ode_algorithm,
                  dt = 42.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  save_everystep = false, callback = callbacks);

summary_callback()
