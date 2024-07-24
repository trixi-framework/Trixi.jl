
using OrdinaryDiffEq
using Trixi
using NLsolve: nlsolve
using Infiltrator

# Warm moist bubble test case from
# Bryan and Fritsch (2002)
# A Benchmark Simulation for Moist Nonhydrostatic Numerical Models
# [DOI: 10.1175/1520-0493(2002)130<2917:ABSFMN>2.0.CO;2](https://doi.org/10.1175/1520-0493(2002)130<2917:ABSFMN>2.0.CO;2)

@inline function getqvs(eps, p, t)
    # Equation for saturation vapor pressure
    # Bolton (1980, MWR, p. 1047)
    es = 611.2 * exp(17.67 * (t - 273.15) / (t - 29.65))
    return eps * es / (p - es)
end

# Compute a base state in hydrostatic equilibrium
# The hydrostatic ODE is integrated upwards from surface (z=0) to `height`
# A one dimensional distribution (in z) with `nz` points is stored
# Adapted from https://www2.mmm.ucar.edu/people/bryan/Code/mbm.F
function calculate_hydrostatic_base_state(g, c_pd, R_d, c_pv, R_v, c_pl, L_00,
                                          p_ref, theta_ref, r_t, nz, height)
    eps = R_d / R_v
    tolerance = 0.0001                        # tolerance for iterative scheme
    relaxation = 0.3                          # relaxation factor for iterative scheme

    theta_0 = Vector{Float64}(undef, nz)      # potential temperature
    theta_rho_0 = Vector{Float64}(undef, nz)  # density potential temperature
    p_0 = Vector{Float64}(undef, nz)          # pressure
    q_v0 = Vector{Float64}(undef, nz)         # vapor mixing ratio
    q_l0 = Vector{Float64}(undef, nz)         # liquid water mixing ratio

    # Start at the surface
    p_0[1] = p_ref                            # ==> exner = 1
    q_v0[1] = getqvs(eps, p_ref, theta_ref)   # T = theta_ref * exner = theta_ref
    q_l0[1] = r_t - q_v0[1]
    theta_0[1] = theta_ref
    theta_rho_0[1] = theta_ref * (1.0 + inv(eps) * q_v0[1]) / (1.0 + q_v0[1] + q_l0[1])

    T_last = theta_0[1]                       # T = theta_ref * exner = theta_ref
    exner_last = 1

    # Integrate upwards
    dz = height / (nz - 1)
    for k in 2:nz
        theta_new = theta_0[k - 1]
        theta_tmp = theta_0[k - 1]
        theta_rho_tmp = theta_rho_0[k - 1]

        exner_tmp = 1.0
        p_tmp = 1.0
        T_tmp = 1.0
        q_v_tmp = 1.0
        q_l_tmp = 1.0
        steps = 0
        converged = false
        while !converged
            steps = steps + 1

            # Trapezoidal rule? Only applied to theta_rho?
            exner_tmp = exner_last -
                        dz * g / (c_pd * 0.5 * (theta_rho_0[k - 1] + theta_rho_tmp))
            p_tmp = p_ref * (exner_tmp^(c_pd / R_d))
            T_tmp = theta_new * exner_tmp
            q_v_tmp = getqvs(eps, p_tmp, T_tmp)
            q_l_tmp = r_t - q_v_tmp
            theta_rho_tmp = theta_tmp * (1.0 + inv(eps) * q_v_tmp) /
                            (1.0 + q_v_tmp + q_l_tmp)

            # Trapezoidal rule?
            T_bar = 0.5 * (T_last + T_tmp)
            q_vbar = 0.5 * (q_v0[k - 1] + q_v_tmp)
            q_lbar = 0.5 * (q_l0[k - 1] + q_l_tmp)

            lhv = L_00 - (c_pl - c_pv) * T_bar           # latent heat
            R_m = R_d + R_v * q_vbar                     # gas constant for moist air
            c_pm = c_pd + c_pv * q_vbar + c_pl * q_lbar  # c_p for moist air

            # An exact thermodynamic equation (e.g., for CM1)
            theta_tmp = theta_0[k - 1] *
                        exp(-lhv * (q_v_tmp - q_v0[k - 1]) / (c_pm * T_bar) +
                            (R_m / c_pm - R_d / c_pd) * log(p_tmp / p_0[k - 1]))

            if abs(theta_tmp - theta_new) > tolerance
                theta_new = theta_new + relaxation * (theta_tmp - theta_new)
                if steps > 100
                    error("calculate_hydrostatic_base_state: not converging!")
                end
            else
                converged = true
                if q_l_tmp < 0.0
                    error("calculate_hydrostatic_base_state: liquid ratio negative!")
                end
            end
        end

        theta_0[k] = theta_tmp
        theta_rho_0[k] = theta_tmp * (1.0 + inv(eps) * q_v_tmp) / (1.0 + q_v_tmp + q_l_tmp)
        p_0[k] = p_tmp
        q_v0[k] = q_v_tmp
        q_l0[k] = q_l_tmp

        T_last = T_tmp
        exner_last = exner_tmp
    end

    return theta_0, theta_rho_0, p_0, q_v0, q_l0
end

struct WarmBubbleSetup
    g::Float64                    # gravity of earth
    c_pd::Float64                 # heat capacity for constant pressure (dry air)
    c_vd::Float64                 # heat capacity for constant volume (dry air)
    R_d::Float64                  # gas constant (dry air)
    c_pv::Float64                 # heat capacity for constant pressure (water vapor)
    c_vv::Float64                 # heat capacity for constant volume (water vapor)
    R_v::Float64                  # gas constant (water vapor)
    c_pl::Float64                 # heat capacity for constant pressure (liquid water)
    p_ref::Float64                # reference pressure
    theta_ref::Float64            # potential temperature at surface
    L_00::Float64                 # latent heat of evaporation at 0 K
    r_t::Float64                  # total water mixing ratio
    theta_0::Vector{Float64}      # potential temperature
    theta_rho_0::Vector{Float64}  # density potential temperature
    p_0::Vector{Float64}          # pressure
    q_v0::Vector{Float64}         # vapor mixing ratio
    q_l0::Vector{Float64}         # liquid water mixing ratio
    nz::Int64                     # resolution for vertical hydrostatic distribution
    height::Float64               # maximal height in vertical hydrostatic distribution

    function WarmBubbleSetup(; g = 9.81,
                             c_pd = 1004.0, c_vd = 717.0, R_d = c_pd - c_vd,
                             c_pv = 1885, c_vv = 1424.0, R_v = c_pv - c_vv,
                             c_pl = 4186.0,
                             p_ref = 100_000.0, theta_ref = 289.8486,
                             L_00 = 2.5e6 + (c_pl - c_pv) * 273.15,
                             r_t = 0.020,
                             nz, height)
        theta_0,
        theta_rho_0,
        p_0,
        q_v0,
        q_l0 = calculate_hydrostatic_base_state(g, c_pd, R_d, c_pv, R_v, c_pl, L_00, p_ref,
                                                theta_ref, r_t, nz, height)
        new(g, c_pd, c_vd, R_d, c_pv, c_vv, R_v, c_pl, p_ref, theta_ref, L_00, r_t, theta_0,
            theta_rho_0, p_0, q_v0, q_l0, nz, height)
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

# condensation, equation by Rutledge and Hobbs (1983)
@inline function source_terms_phase_change_rh(u, setup::WarmBubbleSetup,
                                              equations::CompressibleMoistEulerEquations2D)
    @unpack c_pd, R_d, c_pv, R_v, c_pl, L_00 = setup
    eps = R_d / R_v
    rho_v1, rho_v2, rho_e, rho_qd, rho_qv, rho_ql = u
    rho = density(u, equations)
    T = temperature(u, equations)
    p = pressure(u, equations)

    q_vs = getqvs(eps, p, T)
    q_v = rho_qv / rho
    lhv = L_00 - (c_pl - c_pv) * T
    rcond = (q_v - q_vs) / (1.0 + q_vs * lhv^2 / (c_pd * R_v * T^2))
    #qcond(mgs) = Min( Max( 0.0, tmp ), (qvap(mgs)-qvs(mgs)) )
    return SVector(zero(eltype(u)), zero(eltype(u)), zero(eltype(u)),
                   zero(eltype(u)), -rcond, rcond)
end

@inline function source_terms_phase_change_fb(u, setup::WarmBubbleSetup,
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
    C = 94.0

    Q_ph = (a + b - sqrt(a^2 + b^2)) * C

    return SVector(zero(eltype(u)), zero(eltype(u)), zero(eltype(u)),
                   zero(eltype(u)), Q_ph, -Q_ph)
end

@inline function (setup::WarmBubbleSetup)(u, x, t,
                                          equations::CompressibleMoistEulerEquations2D)
    return source_terms_geopotential(u, setup, equations) +
           source_terms_phase_change_fb(u, setup, equations)
end

###############################################################################
# Initial condition

# Use the precomputed vertical distribution in `WarmBubbleSetup` to interpolate and perturb
# the initial solution
@inline function (setup::WarmBubbleSetup)(x, t,
                                          equations::CompressibleMoistEulerEquations2D)
    @unpack c_pd, c_vd, R_d, c_vv, R_v, c_pl, L_00, r_t, p_ref, theta_0,
    theta_rho_0, p_0, q_v0, q_l0, nz, height = setup
    eps = R_d / R_v
    tolerance = 0.0001                        # tolerance for iterative scheme

    # get value at current z via interpolation
    dz = height / (nz - 1)
    k_l = convert(Int, floor(x[2] / dz)) + 1
    z_l = (k_l - 1) * dz
    z_u = k_l * dz
    if k_l == nz
        k_u = nz
    else
        k_u = k_l + 1
    end
    theta, theta_rho, p, q_v, q_l = map((a, b) -> (z_u - x[2]) * a / dz +
                                                  (x[2] - z_l) * b / dz,
                                        [
                                            theta_0[k_l],
                                            theta_rho_0[k_l],
                                            p_0[k_l],
                                            q_v0[k_l],
                                            q_l0[k_l],
                                        ],
                                        [
                                            theta_0[k_u],
                                            theta_rho_0[k_u],
                                            p_0[k_u],
                                            q_v0[k_u],
                                            q_l0[k_u],
                                        ])
    # center of perturbation
    center_x = 10000.0
    center_z = 2000.0
    # radius of perturbation
    radius = 2000.0
    # distance of current x to center of perturbation
    r = sqrt((x[1] - center_x)^2 + (x[2] - center_z)^2)

    exner = (p / p_ref)^(R_d / c_pd)
    T = theta * exner

    if r <= radius
        theta_perturbation = 2 * cospi(0.5 * r / radius)^2
        theta_tmp = theta
        converged = false
        steps = 0
        while !converged
            steps = steps + 1
            theta = ((theta_perturbation / 300.0) + (1.0 + r_t) / (1.0 + q_v)) *
                    theta_rho * (1.0 + q_v) / (1.0 + inv(eps) * q_v)
            T = theta * exner
            q_v = getqvs(eps, p, T)
            q_l = r_t - q_v
            if abs(theta - theta_tmp) < tolerance
                converged = true
                if q_l < 0.0
                    error("calculate_hydrostatic_base_state: liquid ratio negative!")
                end
            else
                theta_tmp = theta
                if steps > 100
                    error("initial condition perturbation: not converging!")
                end
            end
        end
    end

    rho_d = p / (R_d * T * (1.0 + q_v * inv(eps)))  # p = rho_d * R_d * T * (1 + q_v / eps)
    rho_v = rho_d * q_v
    rho_l = rho_d * q_l
    rho = rho_d + rho_l + rho_v
    v1 = 0.0
    v2 = 0.0
    rho_v1 = rho * v1
    rho_v2 = rho * v2
    rho_e = (c_vd * rho_d + c_vv * rho_v + c_pl * rho_l) * T +
            L_00 * rho_v +
            0.5 * rho * (v1^2 + v2^2)
    return SVector(rho_v1, rho_v2, rho_e, rho_d, rho_v, rho_l)
end

# Initial condition Knoth
struct AtmossphereLayers{RealT <: Real}
    setup::WarmBubbleSetup
    # structure:  1--> i-layer (z = total_height/precision *(i-1)),  2--> rho, rho_theta, rho_qv, rho_ql
    LayerData::Matrix{RealT}
    total_height::RealT
    preciseness::Int
    layers::Int
    ground_state::NTuple{2, RealT}
    equivalentpotential_temperature::RealT
    mixing_ratios::NTuple{2, RealT}
end

function AtmossphereLayers(setup; total_height = 10010.0, preciseness = 10,
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

    n = convert(Int, total_height / preciseness)
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

    return AtmossphereLayers{RealT}(setup, LayerData, total_height, dz, n, ground_state,
                                    theta_e0, mixing_ratios)
end

function moist_state(y, dz, y0, r_t0, theta_e0, setup::WarmBubbleSetup)
    @unpack g, c_pd, R_d, c_pv, R_v, c_pl, p_ref, theta_ref, L_00 = setup
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
            T * (p_d / p_ref)^(-R_d / (c_pd + c_pl * r_t)) *
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

function initial_condition_moist_bubble(x, t, setup::WarmBubbleSetup,
                                        AtmosphereLayers::AtmossphereLayers)
    @unpack LayerData, preciseness, total_height = AtmosphereLayers
    dz = preciseness
    z = x[2]
    if (z > total_height && !(isapprox(z, total_height)))
        error("The atmossphere does not match the simulation domain")
    end
    n = convert(Int, floor((z + eps()) / dz)) + 1
    z_l = (n - 1) * dz
    (rho_l, rho_theta_l, rho_qv_l, rho_ql_l) = LayerData[n, :]
    z_r = n * dz
    if (z_l == total_height)
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
    @unpack c_pd, c_vd, R_d, c_pv, c_vv, R_v, c_pl, p_ref, L_00 = setup
    xc = 10000.0
    zc = 2000.0
    rc = 2000.0
    Δθ = 2.0

    r = sqrt((x[1] - xc)^2 + (x[2] - zc)^2)
    rho_d = rho - rho_qv - rho_ql
    kappa_M = (R_d * rho_d + R_v * rho_qv) / (c_pd * rho_d + c_pv * rho_qv + c_pl * rho_ql)
    p_loc = p_ref * (R_d * rho_theta / p_ref)^(1 / (1 - kappa_M))
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
    a = T_loc * (p_ref / p_d)^(R_d / (c_pd + r_t * c_pl))
    b = H^(-r_v * R_v / c_pd)
    L_v = L_00 + (c_pv - c_pl) * T_loc
    c = exp(L_v * r_v / ((c_pd + r_t * c_pl) * T_loc))
    aeq_pot = (a * b * c)

    # Assume pressure stays constant
    if (r < rc && Δθ > 0)
        kappa = 1 - c_vd / c_pd
        # Calculate background density potential temperature
        θ_dens = rho_theta / rho * (p_loc / p_ref)^(kappa_M - kappa)
        # Calculate perturbed density potential temperature
        θ_dens_new = θ_dens * (1 + Δθ * cospi(0.5 * r / rc)^2 / 300)
        rt = (rho_qv + rho_ql) / rho_d
        rv = rho_qv / rho_d
        # Calculate moist potential temperature
        θ_loc = θ_dens_new * (1 + rt) / (1 + (R_v / R_d) * rv)
        # Adjust varuables until the temperature is met
        if rt > 0
            while true
                T_loc = θ_loc * (p_loc / p_ref)^kappa
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
            T_loc = θ_loc * (p_loc / p_ref)^kappa
            rho_d_new = p_loc / (R_d * T_loc)
            θ_new = θ_dens_new * (1 + rt) / (1 + (R_v / R_d) * rvs)
        end
        rho_qv = rvs * rho_d_new
        rho_ql = (rt - rvs) * rho_d_new
        rho = rho_d_new * (1 + rt)
        rho_d = rho - rho_qv - rho_ql
        kappa_M = (R_d * rho_d + R_v * rho_qv) /
                  (c_pd * rho_d + c_pv * rho_qv + c_pl * rho_ql)
        rho_theta = rho * θ_dens_new * (p_loc / p_ref)^(kappa - kappa_M)
        rho_e = (c_vd * rho_d + c_vv * rho_qv + c_pl * rho_ql) * T_loc + L_00 * rho_qv
    end
    return SVector(rho, rho_e, rho_qv, rho_ql)
end

###############################################################################
# semidiscretization of the compressible moist Euler equations
nelements_z = 100
polydeg = 1
height = 10_000.0

warm_bubble_setup = WarmBubbleSetup(; nz = nelements_z * (polydeg + 1), height = height)

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

basis = LobattoLegendreBasis(polydeg)

#surface_flux = FluxLMARS(340.0)
surface_flux = flux_lax_friedrichs
#volume_flux = flux_chandrashekar
#volume_flux = flux_ranocha
volume_flux = flux_kennedy_gruber
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (0.0, 0.0)
coordinates_max = (20_000.0, height)

cells_per_dimension = (2 * nelements_z, nelements_z)

# Create curved mesh with 64 x 32 elements
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

###############################################################################
# create the semi discretization object

semi = SemidiscretizationHyperbolic(mesh, equations, warm_bubble_setup, solver,
                                    boundary_conditions = boundary_condition,
                                    source_terms = warm_bubble_setup)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1200.0)
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
                                     output_directory = "out_bf_struct200_p1_cfl02_lf_kennedy_gruber",
                                     solution_variables = solution_variables)

stepsize_callback = StepsizeCallback(cfl = 0.2)

callbacks = CallbackSet(summary_callback,
                        #analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false,
            maxiters = 1e6,
            callback = callbacks);
summary_callback() # print the timer summary
