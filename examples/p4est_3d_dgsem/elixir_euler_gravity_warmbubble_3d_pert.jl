
using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

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
function (setup::WarmBubbleSetup)(x, t,
                                  ::CompressibleEulerEquationsPerturbationGravity3D)
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
    potential_temperature_perturbation = zero(RealT)
    if r <= radius
        potential_temperature_perturbation = 2 * cospi(0.5f0 * r / radius)^2
    end
    potential_temperature = potential_temperature_ref + potential_temperature_perturbation

    # Exner pressure, solves hydrostatic equation for x[2]
    exner = 1 - g / (c_p * potential_temperature) * x[2]
    exner_ref = 1 - g / (c_p * potential_temperature_ref) * x[2]

    # pressure
    p_0 = 100_000  # reference pressure
    R = c_p - c_v    # gas constant (dry air)
    p = p_0 * exner^(c_p / R)
    p_ref = p_0 * exner_ref^(c_p / R)

    # temperature
    T = potential_temperature * exner
    T_ref = potential_temperature_ref * exner_ref

    # density
    rho = p / (R * T)
    rho_ref = p_ref / (R * T_ref)

    # density
    v1 = 20.0
    v2 = 0.0
    v3 = 0.0
    v1_ref = 20.0
    v2_ref =  0.0
    v3_ref =  0.0

    # geopotential, steady as well
    phi = g * x[2]

    # energy
    E = c_v * T + 0.5f0 * (v1^2 + v2^2 + v3^2) + phi
    E_ref = c_v * T_ref + 0.5f0 * (v1_ref^2 + v2_ref^2 + v3_ref^2) + phi

    return SVector(rho - rho_ref,
                   rho * v1 - rho_ref * v1_ref,
                   rho * v2 - rho_ref * v2_ref,
                   rho * v3 - rho_ref * v3_ref,
                   rho * E - rho_ref * E_ref)
end

# Steady state
function (setup::WarmBubbleSetup)(x, ::CompressibleEulerEquationsPerturbationGravity3D)
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

    # geopotential, steady as well
    phi = g * x[2]

    rho_ref = p_ref / (R * T_ref)
    v1_ref = 20.0
    v2_ref =  0.0
    v3_ref =  0.0

    E_ref = c_v * T_ref + 0.5f0 * (v1_ref^2 + v2_ref^2 + v3_ref^2) + phi

    return SVector(rho_ref,
                   rho_ref * v1_ref,
                   rho_ref * v2_ref,
                   rho_ref * v3_ref,
                   rho_ref * E_ref,
                   phi)
end

###############################################################################
# semidiscretization of the compressible Euler equations
warm_bubble_setup = WarmBubbleSetup()

equations = CompressibleEulerEquationsPerturbationGravity3D(warm_bubble_setup.gamma)

initial_condition = warm_bubble_setup

volume_flux = (flux_kennedy_gruber, flux_nonconservative_waruszewski)
surface_flux = (FluxLMARS(340.0), flux_nonconservative_waruszewski)
#surface_flux = (flux_central, flux_nonconservative_waruszewski)

polydeg = 3
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.002,
                                         alpha_min = 0.0001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
#volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
#                                                 volume_flux_dg = volume_flux,
#                                                 volume_flux_fv = surface_flux)
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

trees_per_dimension = (40, 20, 2)
coordinates_min = (0.0, 0.0, 0.0)
coordinates_max = (20_000.0, 10_000.0, 2_000.0)
mesh = P4estMesh(trees_per_dimension;
                 polydeg = 1,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 initial_refinement_level = 0,
                 periodicity = (true, false, true))

boundary_conditions = Dict(:y_neg => boundary_condition_slip_wall,
                           :y_pos => boundary_condition_slip_wall)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions,
                                    aux_field = warm_bubble_setup)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1000.0)  # 1000 seconds final time
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_polydeg = polydeg)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(dt = 10.0, #interval = 1, #dt = 10.0,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim_total,
                                     output_directory="out_bubble_3d_pert")

stepsize_callback = StepsizeCallback(cfl = 1.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
