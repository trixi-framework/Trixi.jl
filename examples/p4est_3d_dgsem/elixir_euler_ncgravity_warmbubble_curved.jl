
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
function (setup::WarmBubbleSetup)(x, t, equations::CompressibleEulerEquationsNCGravity3D)
    RealT = eltype(x)
    @unpack g, c_p, c_v = setup
    lambda, phi, r = cart_to_sphere(x)

    z = r - 1e5

    # center of perturbation
    center_x = -1e5 - 2000
    center_y = 0
    center_z = 0
    # radius of perturbation
    radius = 2000
    # distance of current x to center of perturbation
    dist = sqrt((x[1] - center_x)^2 + (x[2] - center_y)^2 + (x[3] - center_z)^2)

    # perturbation in potential temperature
    potential_temperature_ref = 300
    potential_temperature_perturbation = zero(RealT)
    if dist <= radius
        potential_temperature_perturbation = 2 * cospi(0.5f0 * dist / radius)^2
    end
    potential_temperature = potential_temperature_ref + potential_temperature_perturbation

    # Exner pressure, solves hydrostatic equation for x[2]
    exner = 1 - g / (c_p * potential_temperature) * z

    # pressure
    p_0 = 100_000  # reference pressure
    R = c_p - c_v    # gas constant (dry air)
    p = p_0 * exner^(c_p / R)

    # temperature
    T = potential_temperature * exner
    # T = potential_temperature - g / (c_p) * x[2]

    # density
    rho = p / (R * T)

    v1 = 0
    v2 = 0
    E = c_v * T + 0.5f0 * (v1^2 + v2^2)
    return SVector(rho, rho * v1, rho * v2, 0.0f0, rho * E)
end

# Steady state
function (setup::WarmBubbleSetup)(x, ::CompressibleEulerEquationsNCGravity3D)
    @unpack g, c_p, c_v = setup
lambda, phi, r = cart_to_sphere(x)
z = r - 1e5
    # Geopotential
    phi = g * z

    return SVector(phi)
end

function cart_to_sphere(x)
    r = norm(x)
    lambda = atan(x[2], x[1])
    if lambda < 0
        lambda += 2 * pi
    end
    phi = asin(x[3] / r)

    return lambda, phi, r
end

@inline function Tpotpert(u, aux, equations::CompressibleEulerEquationsNCGravity3D)
    rho, _, _, _, p = cons2prim(u, aux, equations)
    exner = (p / 100_000)^(287 / 1004)
    T = p / (rho * 287)
    return abs(T / exner - 300)
end

###############################################################################
# semidiscretization of the compressible Euler equations
warm_bubble_setup = WarmBubbleSetup()

equations = CompressibleEulerEquationsNCGravity3D(warm_bubble_setup.gamma)

initial_condition = warm_bubble_setup

volume_flux = (flux_kennedy_gruber, flux_nonconservative_waruszewski_arithmean)
surface_flux = (FluxLMARS(340.0), flux_nonconservative_waruszewski_arithmean)
#surface_flux = (flux_central, flux_nonconservative_waruszewski)

polydeg = 3
basis = LobattoLegendreBasis(polydeg)
#indicator_sc = IndicatorHennemannGassner(equations, basis,
#                                         alpha_max = 0.002,
#                                         alpha_min = 0.0001,
#                                         alpha_smooth = true,
#                                         variable = density_pressure)
#volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
#                                                 volume_flux_dg = volume_flux,
#                                                 volume_flux_fv = surface_flux)
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

# One face of the cubed sphere
#mapping(xi, eta, zeta) = Trixi.cubed_sphere_mapping(xi, eta, zeta, 1e5, 10000.0, 1)
function mapping(xi, eta, zeta)
    
    inner_radius = 1e5
    thickness = 10_000

    alpha = xi * pi / 16
    beta = eta * pi / 16

    # Equiangular projection
    x = tan(alpha)
    y = tan(beta)

    # Radius on cube surface
    r = sqrt(1 + x^2 + y^2)

    # Radius of the sphere
    R = inner_radius + thickness * (0.5f0 * (zeta + 1))

    # Projection onto the sphere
    return R / r * SVector(-1, -x, y)
end

trees_per_dimension = (20, 20, 10)
mesh = P4estMesh(trees_per_dimension, polydeg=3,
                 mapping=mapping,
                 initial_refinement_level=0,
                 periodicity=false)

boundary_conditions = Dict(:x_neg => boundary_condition_slip_wall,
                           :x_pos => boundary_condition_slip_wall,
                           :y_neg => boundary_condition_slip_wall,
                           :y_pos => boundary_condition_slip_wall,
                           :z_neg => boundary_condition_slip_wall,
                           :z_pos => boundary_condition_slip_wall)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions,
                                    aux_field = warm_bubble_setup)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1000)  # 1000 seconds final time
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_polydeg = polydeg)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable = Tpotpert),
                                      base_level = 0,
                                      med_level = 1, med_threshold = 0.5,
                                      max_level = 2, max_threshold = 2.0)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 100,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

save_solution = SaveSolutionCallback(dt = 10.0, #interval = 1, #dt = 10.0,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory="out_bubble_3d_nc_curved_amr")

stepsize_callback = StepsizeCallback(cfl = 1.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        amr_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
