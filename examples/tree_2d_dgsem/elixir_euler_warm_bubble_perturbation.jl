using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi
using Plots

# Warm bubble test case from
# - Wicker, L. J., and Skamarock, W. C. (1998)
#   A time-splitting scheme for the elastic equations incorporating
#   second-order Runge–Kutta time differencing
#   [DOI: 10.1175/1520-0493(1998)126%3C1992:ATSSFT%3E2.0.CO;2](https://doi.org/10.1175/1520-0493(1998)126%3C1992:ATSSFT%3E2.0.CO;2)
# See also
# - Bryan and Fritsch (2002)
#   A Benchmark Simulation for Moist Nonhydrostatic Numerical Models
#   [DOI: 10.1175/1520-0493(2002)130<2917:ABSFMN>2.0.CO;2](https://doi.org/10.1175/1520-0493(2002)130<2917:ABSFMN>2.0.CO;2)
# - Carpenter, Droegemeier, Woodward, Hane (1990)
#   Application of the Piecewise Parabolic Method (PPM) to
#   Meteorological Modeling
#   [DOI: 10.1175/1520-0493(1990)118<0586:AOTPPM>2.0.CO;2](https://doi.org/10.1175/1520-0493(1990)118<0586:AOTPPM>2.0.CO;2)
struct WarmBubbleSetup
    # Physical constants
    g::Float64       # gravity of earth
    c_p::Float64     # heat capacity for constant pressure (dry air)
    c_v::Float64     # heat capacity for constant volume (dry air)
    gamma::Float64   # heat capacity ratio (dry air)

    function WarmBubbleSetup(; g = 9.81, c_p = 1004.0, c_v = 717.0, gamma = c_p / c_v)
        new(g, c_p, c_v, gamma)
    end
end

# Initial condition
function (setup::WarmBubbleSetup)(x, t, equations::CompressibleEulerEquationsPerturbation2D)
    RealT = eltype(x)
    @unpack g, c_p, c_v = setup

    # center of perturbation
    center_x = 10000
    center_z = 2000
    # radius of perturbation
    radius = 2000
    # distance of current x to center of perturbation
    r = sqrt((x[1] - center_x)^2 + (x[2] - center_z)^2)

    # perturbation in potential temperature
    potential_temperature_ref = 300
    potential_temperature_perturbation = 0
    if r <= radius
        potential_temperature_perturbation = 2 * cospi(0.5f0 * r / radius)^2
    end
    potential_temperature = potential_temperature_ref + potential_temperature_perturbation

    # Exner pressure, solves hydrostatic equation for x[2]
    exner = 1 - g / (c_p * potential_temperature) * x[2]
    exner_ref = 1 - g / (c_p * potential_temperature_ref) * x[2]

    # pressure
    p_0 = 100_000  # reference pressure
    R = c_p - c_v  # gas constant (dry air)
    p = p_0 * exner^(c_p / R)
    p_ref = p_0 * exner_ref^(c_p / R)

    # temperature
    T = potential_temperature * exner
    T_ref = potential_temperature_ref * exner_ref

    # density
    rho = p / (R * T)
    rho_ref = p_ref / (R * T_ref)

    v1 = 20.0
    v2 = 0

    E = c_v * T + 0.5f0 * (v1^2 + v2^2)
    E_ref = c_v * T_ref

    return SVector(rho - rho_ref, rho * v1, rho * v2, rho * E - rho_ref * E_ref)
    #return SVector(rho, rho * v1, rho * v2, rho * E)
end

# Steady state
function (setup::WarmBubbleSetup)(x)
    @unpack g, c_p, c_v = setup

    potential_temperature_ref = 300

    # Exner pressure, solves hydrostatic equation for x[2]
    exner_ref = 1 - g / (c_p * potential_temperature_ref) * x[2]

    # pressure
    p_0 = 100_000  # reference pressure
    R = c_p - c_v  # gas constant (dry air)
    p_ref = p_0 * exner_ref^(c_p / R)

    # temperature
    T_ref = potential_temperature_ref * exner_ref

    rho_ref = p_ref / (R * T_ref)
    E_ref = c_v * T_ref

    return SVector(rho_ref, rho_ref * E_ref)
    #return SVector(0, 0)
end

# Source terms
@inline function (setup::WarmBubbleSetup)(u, aux, x, t,
                                          ::CompressibleEulerEquationsPerturbation2D)
    @unpack g = setup
    rho_pert, _, rho_v2, _ = u

    # TODO: use perturbation
    rho_total = rho_pert + aux[1]

    return SVector(zero(eltype(u)), zero(eltype(u)), -g * rho_pert, -g * rho_v2)
end

###############################################################################
# semidiscretization of the compressible Euler equations
warm_bubble_setup = WarmBubbleSetup()

equations = CompressibleEulerEquationsPerturbation2D(warm_bubble_setup.gamma)

boundary_conditions = (x_neg = boundary_condition_periodic,
                       x_pos = boundary_condition_periodic,
                       y_neg = boundary_condition_slip_wall_simple,
                       y_pos = boundary_condition_slip_wall_simple)

polydeg = 3
basis = LobattoLegendreBasis(polydeg)

# This is a good estimate for the speed of sound in this example.
# Other values between 300 and 400 should work as well.
#surface_flux = FluxLMARS(340.0)
surface_flux = flux_lax_friedrichs

#volume_flux = flux_kennedy_gruber
#volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

solver = DGSEM(basis, surface_flux) #, volume_integral)

coordinates_min = (0.0, -5000.0)
coordinates_max = (20_000.0, 15_000.0)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4, #. 6,
                n_cells_max = 10_000,
                periodicity = (true, false))

semi = SemidiscretizationHyperbolic(mesh, equations, warm_bubble_setup, solver,
                                    source_terms = warm_bubble_setup,
                                    boundary_conditions = boundary_conditions,
                                    auxiliary_field = warm_bubble_setup)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 150.0)  # 1000 seconds final time

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000

analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = analysis_interval,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     output_directory = "out_warm_bubble_testing",
                                     solution_variables = cons2cons)

stepsize_callback = StepsizeCallback(cfl = 1.0)

visualization = VisualizationCallback(interval = 100, show_mesh = false,
                                      suspend = true,
                                      #plot_creator = Trixi.show_plot_makie,
                                      solution_variables = cons2prim_pert,
                                      variable_names = ["rho_pert"],
                                      #aspect_ratio = 4
                                      )

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        visualization,
                        save_solution)
                        #stepsize_callback)

###############################################################################
# run the simulation
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            maxiters = 1.0e7,
            dt = 1e-2, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
