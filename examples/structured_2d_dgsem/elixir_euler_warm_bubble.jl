using OrdinaryDiffEq
using Trixi

# Physical constants
g::Float64 = 9.81          # gravity of earth
p_0::Float64 = 100_000.0   # reference pressure
c_p::Float64 = 1004.0      # heat capacity for constant pressure (dry air)
c_v::Float64 = 717.0       # heat capacity for constant volume (dry air)
R = c_p - c_v              # gas constant (dry air)
gamma = c_p / c_v          # heat capacity ratio (dry air)

# Warm bubble test from
# Wicker, L. J., and Skamarock, W. C., 1998: A time-splitting scheme
# for the elastic equations incorporating second-order Runge–Kutta
# time differencing. Mon. Wea. Rev., 126, 1992–1999.
function initial_condition_warm_bubble(x, t, equations::CompressibleEulerEquations2D)
    # center of perturbation
    xc = 10000.0
    zc = 2000.0
    # radius of perturbation
    rc = 2000.0
    # distance of current x to center of perturbation
    r = sqrt((x[1] - xc)^2 + (x[2] - zc)^2)

    # perturbation in potential temperature
    θ_ref = 300.0
    Δθ = 0.0
    if r <= rc
        Δθ = 2 * cospi(0.5 * r / rc)^2
    end
    θ = θ_ref + Δθ # potential temperature

    # Exner pressure, solves hydrostatic equation for x[2]
    exner = 1 - g / (c_p * θ) * x[2]

    # pressure
    p = p_0 * exner^(c_p / R)

    # temperature
    T = θ * exner

    # density
    rho = p / (R * T)

    v1 = 20.0
    v2 = 0.0
    E = c_v * T + 0.5 * (v1^2 + v2^2)
    return SVector(rho, rho * v1, rho * v2, rho * E)
end

@inline function source_terms_gravity(u, x, t,
                                      equations::CompressibleEulerEquations2D)
    rho, _, rho_v2, _ = u
    return SVector(zero(eltype(u)), zero(eltype(u)), -g * rho, -g * rho_v2)
end

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(gamma)

boundary_conditions = (x_neg = boundary_condition_periodic,
                       x_pos = boundary_condition_periodic,
                       y_neg = boundary_condition_slip_wall,
                       y_pos = boundary_condition_slip_wall)

polydeg = 4
basis = LobattoLegendreBasis(polydeg)

surface_flux = FluxLMARS(340.0)
volume_flux = flux_ranocha
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (0.0, 0.0)
coordinates_max = (20_000.0, 10_000.0)

cells_per_dimension = (64, 32)
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_warm_bubble, solver,
                                    source_terms = source_terms_gravity,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1000.0)

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000

analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:entropy_conservation_error,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = analysis_interval,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     output_directory = "out.struct_lmars_ra",
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.2)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            maxiters = 1.0e7,
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

summary_callback()
