
using OrdinaryDiffEq
using Trixi
using NLsolve: nlsolve
using Infiltrator

# Warm moist bubble test case from
# Bryan and Fritsch (2002)
# A Benchmark Simulation for Moist Nonhydrostatic Numerical Models
# [DOI: 10.1175/1520-0493(2002)130<2917:ABSFMN>2.0.CO;2](https://doi.org/10.1175/1520-0493(2002)130<2917:ABSFMN>2.0.CO;2)
struct WarmBubbleSetup
    # Physical constants
    g::Float64         # gravity of earth
    c_pd::Float64      # heat capacity for constant pressure (dry air)
    c_vd::Float64      # heat capacity for constant volume (dry air)
    R_d::Float64       # gas constant (dry air)
    c_pv::Float64      # heat capacity for constant pressure (water vapor)
    c_vv::Float64      # heat capacity for constant volume (water vapor)
    R_v::Float64       # gas constant (water vapor)
    c_pl::Float64      # heat capacity for constant pressure (liquid water)
    p_0::Float64       # reference pressure 1000 hPa
    L_00::Float64      # latent heat of evaporation at 0 K
    # TODO value ???
    # L00 = parameter("L00",2.5000e6 + (c_pl - c_pv) * 273.15)

    function WarmBubbleSetup(; g = 9.81,
                             c_pd = 1004.0, c_vd = 717.0, R_d = c_pd - c_vd,
                             c_pv = 1885, c_vv = 1424.0, R_v = c_pv - c_vv,
                             c_pl = 4186.0, p_0 = 100_000.0, L_00 = 3147620.0)
        new(g, c_pd, c_vd, R_d, c_pv, c_vv, R_v, c_pl, p_0, L_00)
    end
end

function get_c_p(setup::WarmBubbleSetup)
    return (setup.c_pd, setup.c_pv, setup.c_pl)
end

function get_c_v(setup::WarmBubbleSetup)
    return (setup.c_vd, setup.c_vv, setup.c_pl)  # c_pl = c_vl for liquid phase
end

###############################################################################
# source terms
@inline function source_terms_geopotential(u, setup::WarmBubbleSetup,
                                           equations::CompressibleMoistEulerEquations2D)
    @unpack g = setup
    _, rho_v2 = u
    rho = density(u, equations)

    return SVector(zero(eltype(u)), -g * rho, -g * rho_v2,
                   zero(eltype(u)), zero(eltype(u)), zero(eltype(u)))
end

@inline function source_terms_phase_change(u, setup::WarmBubbleSetup,
                                           equations::CompressibleMoistEulerEquations2D)

    # This source term models the phase chance between could water and vapor.
    @unpack R_v = setup
    rho_v1, rho_v2, rho_e, rho_qd, rho_qv, rho_ql = u
    rho = density(u, equations)
    T = temperature(u, equations)

    T_C = T - 273.15
    # saturation vapor pressure
    p_vs = 611.2 * exp(17.62 * T_C / (243.12 + T_C))

    # saturation density of vapor 
    rho_star_qv = p_vs / (R_v * T)

    # Fisher-Burgmeister-Function
    a = rho_star_qv - rho_qv
    b = rho - rho_qv - rho_qd

    # saturation control factor  
    # < 1: stronger saturation effect
    # > 1: weaker saturation effect
    C = 1

    Q_ph = (a + b - sqrt(a^2 + b^2)) * C

    return SVector(zero(eltype(u)), zero(eltype(u)), zero(eltype(u)),
                   zero(eltype(u)), Q_ph, -Q_ph)
end

@inline function (setup::WarmBubbleSetup)(u, x, t,
                                          equations::CompressibleMoistEulerEquations2D)
    return source_terms_geopotential(u, setup, equations) +
           source_terms_phase_change(u, setup, equations)
end

###############################################################################
# Initial condition
struct AtmossphereLayers{RealT <: Real}
    setup::WarmBubbleSetup
    # structure:  1--> i-layer (z = total_hight/precision *(i-1)),  2--> rho, rho_theta, rho_qv, rho_ql
    LayerData::Matrix{RealT}
    total_hight::RealT
    preciseness::Int
    layers::Int
    ground_state::NTuple{2, RealT}
    equivalentpotential_temperature::RealT
    mixing_ratios::NTuple{2, RealT}
end

function AtmossphereLayers(setup; total_hight = 10010.0, preciseness = 10,
                           ground_state = (1.4, 100000.0),
                           equivalentpotential_temperature = 320,
                           mixing_ratios = (0.02, 0.02), RealT = Float64)
    @unpack c_pd, R_d, c_pv, R_v, c_pl = setup
    rho0, p0 = ground_state
    r_t0, r_v0 = mixing_ratios
    theta_e0 = equivalentpotential_temperature

    rho_qv0 = rho0 * r_v0
    T0 = theta_e0
    y0 = [p0, rho0, T0, r_t0, r_v0, rho_qv0, theta_e0]

    n = convert(Int, total_hight / preciseness)
    dz = 0.01
    LayerData = zeros(RealT, n + 1, 4)

    F = generate_function_of_y(dz, y0, r_t0, theta_e0, setup)
    sol = nlsolve(F, y0)
    p, rho, T, r_t, r_v, rho_qv, theta_e = sol.zero

    rho_d = rho / (1 + r_t)
    rho_ql = rho - rho_d - rho_qv
    kappa_M = (R_d * rho_d + R_v * rho_qv) / (c_pd * rho_d + c_pv * rho_qv + c_pl * rho_ql)
    rho_theta = rho * (p0 / p)^kappa_M * T * (1 + (R_v / R_d) * r_v) / (1 + r_t)

    LayerData[1, :] = [rho, rho_theta, rho_qv, rho_ql]
    for i in (1:n)
        #println("Layer", i)
        y0 = deepcopy(sol.zero)
        dz = preciseness
        F = generate_function_of_y(dz, y0, r_t0, theta_e0, setup)
        sol = nlsolve(F, y0)
        p, rho, T, r_t, r_v, rho_qv, theta_e = sol.zero
        rho_d = rho / (1 + r_t)
        rho_ql = rho - rho_d - rho_qv
        kappa_M = (R_d * rho_d + R_v * rho_qv) /
                  (c_pd * rho_d + c_pv * rho_qv + c_pl * rho_ql)
        rho_theta = rho * (p0 / p)^kappa_M * T * (1 + (R_v / R_d) * r_v) / (1 + r_t)

        LayerData[i + 1, :] = [rho, rho_theta, rho_qv, rho_ql]
    end

    return AtmossphereLayers{RealT}(setup, LayerData, total_hight, dz, n, ground_state,
                                    theta_e0, mixing_ratios)
end

function moist_state(y, dz, y0, r_t0, theta_e0, setup::WarmBubbleSetup)
    @unpack g, c_pd, R_d, c_pv, R_v, c_pl, p_0, L_00 = setup
    (p, rho, T, r_t, r_v, rho_qv, theta_e) = y
    p0 = y0[1]

    F = zeros(7, 1)
    rho_d = rho / (1 + r_t)
    p_d = R_d * rho_d * T
    T_C = T - 273.15
    p_vs = 611.2 * exp(17.62 * T_C / (243.12 + T_C))
    L = L_00 - (c_pl - c_pv) * T

    F[1] = (p - p0) / dz + g * rho
    F[2] = p - (R_d * rho_d + R_v * rho_qv) * T
    # H = 1 is assumed
    F[3] = (theta_e -
            T * (p_d / p_0)^(-R_d / (c_pd + c_pl * r_t)) *
            exp(L * r_v / ((c_pd + c_pl * r_t) * T)))
    F[4] = r_t - r_t0
    F[5] = rho_qv - rho_d * r_v
    F[6] = theta_e - theta_e0
    a = p_vs / (R_v * T) - rho_qv
    b = rho - rho_qv - rho_d
    # H=1 => phi=0
    F[7] = a + b - sqrt(a * a + b * b)

    return F
end

function generate_function_of_y(dz, y0, r_t0, theta_e0, setup::WarmBubbleSetup)
    function function_of_y(y)
        return moist_state(y, dz, y0, r_t0, theta_e0, setup)
    end
end

# Moist bubble test case from paper:
# G.H. Bryan, J.M. Fritsch, A Benchmark Simulation for Moist Nonhydrostatic Numerical
# Models, MonthlyWeather Review Vol.130, 2917–2928, 2002, 
# https://journals.ametsoc.org/view/journals/mwre/130/12/1520-0493_2002_130_2917_absfmn_2.0.co_2.xml.
function initial_condition_moist_bubble(x, t, setup::WarmBubbleSetup,
                                        AtmosphereLayers::AtmossphereLayers)
    @unpack LayerData, preciseness, total_hight = AtmosphereLayers
    dz = preciseness
    z = x[2]
    if (z > total_hight && !(isapprox(z, total_hight)))
        error("The atmossphere does not match the simulation domain")
    end
    n = convert(Int, floor((z + eps()) / dz)) + 1
    z_l = (n - 1) * dz
    (rho_l, rho_theta_l, rho_qv_l, rho_ql_l) = LayerData[n, :]
    z_r = n * dz
    if (z_l == total_hight)
        z_r = z_l + dz
        n = n - 1
    end
    (rho_r, rho_theta_r, rho_qv_r, rho_ql_r) = LayerData[n + 1, :]
    rho = (rho_r * (z - z_l) + rho_l * (z_r - z)) / dz
    rho_theta = rho * (rho_theta_r / rho_r * (z - z_l) + rho_theta_l / rho_l * (z_r - z)) /
                dz
    rho_qv = rho * (rho_qv_r / rho_r * (z - z_l) + rho_qv_l / rho_l * (z_r - z)) / dz
    rho_ql = rho * (rho_ql_r / rho_r * (z - z_l) + rho_ql_l / rho_l * (z_r - z)) / dz

    rho, rho_e, rho_qv, rho_ql = PerturbMoistProfile(x, rho, rho_theta, rho_qv, rho_ql,
                                                     setup)

    v1 = 0.0
    v2 = 0.0
    rho_v1 = rho * v1
    rho_v2 = rho * v2
    rho_E = rho_e + 1 / 2 * rho * (v1^2 + v2^2)

    rho_qd = rho - rho_qv - rho_ql

    return SVector(rho_v1, rho_v2, rho_E, rho_qd, rho_qv, rho_ql)
end

function PerturbMoistProfile(x, rho, rho_theta, rho_qv, rho_ql, setup::WarmBubbleSetup)
    @unpack c_pd, c_vd, R_d, c_pv, c_vv, R_v, c_pl, p_0, L_00 = setup
    xc = 10000.0
    zc = 2000.0
    rc = 2000.0
    Δθ = 2.0

    r = sqrt((x[1] - xc)^2 + (x[2] - zc)^2)
    rho_d = rho - rho_qv - rho_ql
    kappa_M = (R_d * rho_d + R_v * rho_qv) / (c_pd * rho_d + c_pv * rho_qv + c_pl * rho_ql)
    p_loc = p_0 * (R_d * rho_theta / p_0)^(1 / (1 - kappa_M))
    T_loc = p_loc / (R_d * rho_d + R_v * rho_qv)
    rho_e = (c_vd * rho_d + c_vv * rho_qv + c_pl * rho_ql) * T_loc + L_00 * rho_qv

    p_v = rho_qv * R_v * T_loc
    p_d = p_loc - p_v
    T_C = T_loc - 273.15
    p_vs = 611.2 * exp(17.62 * T_C / (243.12 + T_C))
    H = p_v / p_vs
    r_v = rho_qv / rho_d
    r_l = rho_ql / rho_d
    r_t = r_v + r_l

    # Aequivalentpotential temperature
    a = T_loc * (p_0 / p_d)^(R_d / (c_pd + r_t * c_pl))
    b = H^(-r_v * R_v / c_pd)
    L_v = L_00 + (c_pv - c_pl) * T_loc
    c = exp(L_v * r_v / ((c_pd + r_t * c_pl) * T_loc))
    aeq_pot = (a * b * c)

    # Assume pressure stays constant
    if (r < rc && Δθ > 0)
        kappa = 1 - c_vd / c_pd
        # Calculate background density potential temperature
        θ_dens = rho_theta / rho * (p_loc / p_0)^(kappa_M - kappa)
        # Calculate perturbed density potential temperature
        θ_dens_new = θ_dens * (1 + Δθ * cospi(0.5 * r / rc)^2 / 300)
        rt = (rho_qv + rho_ql) / rho_d
        rv = rho_qv / rho_d
        # Calculate moist potential temperature
        θ_loc = θ_dens_new * (1 + rt) / (1 + (R_v / R_d) * rv)
        # Adjust varuables until the temperature is met
        if rt > 0
            while true
                T_loc = θ_loc * (p_loc / p_0)^kappa
                T_C = T_loc - 273.15
                # SaturVapor
                pvs = 611.2 * exp(17.62 * T_C / (243.12 + T_C))
                rho_d_new = (p_loc - pvs) / (R_d * T_loc)
                rvs = pvs / (R_v * rho_d_new * T_loc)
                θ_new = θ_dens_new * (1 + rt) / (1 + (R_v / R_d) * rvs)
                if abs(θ_new - θ_loc) <= θ_loc * 1.0e-12
                    break
                else
                    θ_loc = θ_new
                end
            end
        else
            rvs = 0
            T_loc = θ_loc * (p_loc / p_0)^kappa
            rho_d_new = p_loc / (R_d * T_loc)
            θ_new = θ_dens_new * (1 + rt) / (1 + (R_v / R_d) * rvs)
        end
        rho_qv = rvs * rho_d_new
        rho_ql = (rt - rvs) * rho_d_new
        rho = rho_d_new * (1 + rt)
        rho_d = rho - rho_qv - rho_ql
        kappa_M = (R_d * rho_d + R_v * rho_qv) /
                  (c_pd * rho_d + c_pv * rho_qv + c_pl * rho_ql)
        rho_theta = rho * θ_dens_new * (p_loc / p_0)^(kappa - kappa_M)
        rho_e = (c_vd * rho_d + c_vv * rho_qv + c_pl * rho_ql) * T_loc + L_00 * rho_qv
    end
    return SVector(rho, rho_e, rho_qv, rho_ql)
end

###############################################################################
# semidiscretization of the compressible moist Euler equations
warm_bubble_setup = WarmBubbleSetup()

AtmossphereData = AtmossphereLayers(warm_bubble_setup)

# Create the initial condition with the initial data set
function initial_condition_moist(x, t, equations)
    return initial_condition_moist_bubble(x, t, warm_bubble_setup, AtmossphereData)
end

equations = CompressibleMoistEulerEquations2D(; c_p = get_c_p(warm_bubble_setup),
                                              c_v = get_c_v(warm_bubble_setup),
                                              L_00 = warm_bubble_setup.L_00)

boundary_condition = (x_neg = boundary_condition_slip_wall,
                      x_pos = boundary_condition_slip_wall,
                      y_neg = boundary_condition_slip_wall,
                      y_pos = boundary_condition_slip_wall)

polydeg = 4
basis = LobattoLegendreBasis(polydeg)

surface_flux = FluxLMARS(360.0)
volume_flux = flux_chandrashekar
#volume_flux = flux_ranocha
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (0.0, 0.0)
coordinates_max = (20000.0, 10000.0)

cells_per_dimension = (64, 32)

# Create curved mesh with 64 x 32 elements
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

###############################################################################
# create the semi discretization object

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_moist, solver,
                                    boundary_conditions = boundary_condition,
                                    source_terms = warm_bubble_setup)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1000.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
#solution_variables = cons2aeqpot
solution_variables = cons2prim

analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
#analysis_callback = AnalysisCallback(semi, interval=analysis_interval, extra_analysis_errors=(:entropy_conservation_error, ), extra_analysis_integrals=(entropy, energy_total, Trixi.saturation_pressure))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 1000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = solution_variables)

stepsize_callback = StepsizeCallback(cfl = 0.2)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
