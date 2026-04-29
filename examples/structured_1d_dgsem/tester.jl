using Clapeyron
using StaticArrays, ForwardDiff
using LinearAlgebra
using Trixi: ln_mean, inv_ln_mean

struct QuasiEuler1D{T}
    eos::T
end

struct MyEuler1D{T}
    eos::T
end

#struct to match 
# https://www.sciencedirect.com/science/article/pii/S0045793015001887

struct MySG{T}
    pInf::T
    q::T # is the reference specific internal energy
    R::T #(takes in J/kg-K), so need to adjust for given specie
    gamma::T
    cv::T
end

Base.broadcastable(model::MySG) = Ref(model)

Base.broadcastable(equations::MyEuler1D) = Ref(equations)

Base.broadcastable(equations::QuasiEuler1D) = Ref(equations)

function psi(u, equations::QuasiEuler1D)
    _, _, _, a = u
    (; pInf, cv, gamma) = equations.eos
    rho, v, p, a = cons2prim(u, equations)
    T = (p + pInf) / ((gamma - 1) * cv * rho)
    rho = density_pT(p, T, equations)
    #return a * rho * v
    return a * p * v / T
end

function psi(u, equations::MyEuler1D)
    (; pInf, cv, gamma) = equations.eos
    rho, v, p = cons2prim(u, equations)
    T = (p + pInf) / ((gamma - 1) * cv * rho)
    return p * v / T
end
# constructor for MySG, default values are for water
# R = 0.08314 J/mol-K -> 
Rwater = 0.08314 / 0.01802 # J/(kg-K)
function MySG(pInf = 1e9, q = -1167 * 1e3, R = Rwater, gamma = 2.35,
              cv = 1816.0)
    return MySG(pInf, q, R, gamma, cv)
end

#this is needed for initial condition nozzle
function density_pT(p, T, equations::QuasiEuler1D)
    (; pInf, gamma, cv) = equations.eos
    return (p + pInf) / ((gamma - 1) * cv * T)
end

function density_pT(p, T, equations::MyEuler1D)
    (; pInf, gamma, cv) = equations.eos
    return (p + pInf) / ((gamma - 1) * cv * T)
end

# uP = uM[md.mapP]
# uP[1] = inlet_stagnation_nozzle(uM[1], 1e6, 453, equations)
# uP[end] = nozzle_outlet_subsonic(uM[end], 0.5e6, equations)
function pressure(u, equations::QuasiEuler1D)
    (; pInf, q, gamma) = equations.eos
    rho, v, p, _ = cons2prim(u, equations)
    return p
end

function pressure(u, equations::MyEuler1D)
    (; pInf, q, gamma) = equations.eos
    rho, v, p = cons2prim(u, equations)
    return p
end

function rho_e_rhoP(rho, p, equations::MyEuler1D)
    (; pInf, q, gamma) = equations.eos
    return (p + gamma * pInf) / (gamma - 1) + rho * q
end

function rho_e_rhoP(rho, p, equations::QuasiEuler1D)
    (; pInf, q, gamma) = equations.eos
    return (p + gamma * pInf) / (gamma - 1) + rho * q
end

function cons2prim(u, equations::QuasiEuler1D)
    (; gamma, pInf, q) = equations.eos
    a_rho, a_rho_v, a_rho_E, a = u
    rho = a_rho / a
    v = a_rho_v / a_rho
    E = a_rho_E / a_rho
    e = (E - 0.5 * v^2)
    p = (gamma - 1) * rho * (e - q) - gamma * pInf
    return SVector(rho, v, p, a)
end

function cons2prim(u, equations::MyEuler1D)
    (; gamma, pInf, q) = equations.eos
    rho, rho_v, rho_E = u
    v = rho_v / rho
    E = rho_E / rho
    e = (E - 0.5 * v^2)
    p = (gamma - 1) * rho * (e - q) - gamma * pInf
    return SVector(rho, v, p)
end

function energy_internal_specific(u, equations::MyEuler1D)
    (; gamma, pInf, q) = equations.eos
    rho, rho_v, rho_E = u
    v = rho_v / rho
    E = rho_E / rho
    e = (E - 0.5 * v^2)
    return e
end

function enthalpy(u, equations::MyEuler1D)
    (; gamma, cv, q) = model
    T = temperature(u, equations)
    return gamma * cv * T + q
end

function enthalpy(u, equations::QuasiEuler1D)
    (; gamma, cv, q) = model
    T = temperature(u, equations)
    return gamma * cv * T + q
end

function physical_entropy(u, equations::QuasiEuler1D)
    (; cv, gamma, pInf) = equations.eos
    rho, v, p, a = cons2prim(u, equations)
    return cv * log((p + pInf) / rho^(gamma))
end

function physical_entropy(u, equations::MyEuler1D)
    (; cv, gamma, pInf) = equations.eos
    rho, v, p = cons2prim(u, equations)
    return cv * log((p + pInf) / rho^(gamma))
end

function entropy(u, equations::QuasiEuler1D)
    (; gamma) = equations.eos
    a_rho, _, _, _ = u
    s = physical_entropy(u, equations)
    return -a_rho * s
end

function entropy(u, equations::MyEuler1D)
    (; gamma) = equations.eos
    rho, _, _ = u
    s = physical_entropy(u, equations)
    return -rho * s
end

# speed of sound: https://inldigitallibrary.inl.gov/sites/sti/sti/5753425.pdf
function speed_of_sound(u, equations::QuasiEuler1D)
    (; gamma, pInf) = equations.eos
    rho, _, p = cons2prim(u, equations)
    return sqrt(gamma * (p + pInf) / rho)
end

function speed_of_sound(u, equations::MyEuler1D)
    (; gamma, pInf) = equations.eos
    rho, _, p = cons2prim(u, equations)
    return sqrt(gamma * (p + pInf) / rho)
end

function temperature(u, equations::MyEuler1D)
    (; pInf, gamma, cv) = equations.eos
    rho, v, p = cons2prim(u, equations)
    return (p + pInf) / ((gamma - 1) * cv * rho)
end

function temperature(u, equations::QuasiEuler1D)
    (; pInf, gamma, cv) = equations.eos
    rho, v, p, a = cons2prim(u, equations)
    return (p + pInf) / ((gamma - 1) * cv * rho)
end

function mycons2entropy(u, equations::MyEuler1D)
    model = equations.eos
    (; gamma) = model
    T = temperature(u, equations)
    g = gibbs_free_energy(u, equations)
    v = u[2] / u[1]
    return SVector(g - 0.5 * v^2, v, -1) / T
end

function mycons2entropy(u, equations::QuasiEuler1D)
    model = equations.eos
    (; gamma) = model
    T = temperature(u, equations)
    g = gibbs_free_energy(u, equations)
    v = u[2] / u[1]
    return SVector(g - 0.5 * v^2, v, -1) / T
end

function cons2entropy(u, equations::QuasiEuler1D)
    return ForwardDiff.gradient(u -> entropy(u, equations), u)
end

function cons2entropy(u, equations::MyEuler1D)
    return ForwardDiff.gradient(u -> entropy(u, equations), u)
end

function gibbs_free_energy(u, equations::MyEuler1D)
    s = physical_entropy(u, equations)
    T = temperature(u, equations)
    h = enthalpy(u, equations)
    return h - T * s
end

function gibbs_free_energy(u, equations::QuasiEuler1D)
    s = physical_entropy(u, equations)
    T = temperature(u, equations)
    h = enthalpy(u, equations)
    return h - T * s
end

function gibbs_free_energy2(u, equations::QuasiEuler1D)
    V = u[4] / u[1]
    p = pressure(u, equations)
    s = physical_entropy(u, equations)
    T = temperature(u, equations)
    e = rho_e_rhoP(1 / V, p, equations) * V
    h = e + p * V
    return h - T * s
end

function flux(u, equations::QuasiEuler1D)
    rho_a, rho_v1_a, rho_E_a, a = u
    v1 = rho_v1_a / rho_a
    rho = rho_a / a
    rho_E = rho_E_a / a
    p = pressure(u, equations)
    return SVector(rho_v1_a, a * (rho * v1 * v1), a * v1 * (rho_E + p), 0)
end

function flux(u, equations::MyEuler1D)
    rho, rho_v1, rho_E = u
    v1 = rho_v1 / rho
    p = pressure(u, equations)
    return SVector(rho_v1, rho_v1 * v1 + p, v1 * (rho_E + p))
end

function flux_central(u_ll, u_rr, normal, equations::QuasiEuler1D)
    model = equations.eos
    return 0.5 * (flux(u_ll, equations) + flux(u_rr, equations))
end

function flux_central(u_ll, u_rr, normal, equations::MyEuler1D)
    model = equations.eos
    return 0.5 * (flux(u_ll, equations) + flux(u_rr, equations))
end

function flux_nonconservative_chan_etal(u_ll, u_rr, equations::QuasiEuler1D)
    _, _, _, a_ll = u_ll
    _, _, _, a_rr = u_rr
    p_rr = pressure(u_rr, equations)
    p_ll = pressure(u_ll, equations)
    p_avg = (p_ll + p_rr)
    return SVector(0.0, (a_ll + a_rr) / 2 * p_avg, 0.0, 0.0)
end

function flux_nonsym(uA_ll, uA_rr, equations::QuasiEuler1D)
    A_ll, A_rr = uA_ll[4], uA_rr[4]

    rho_E_ll = uA_ll[3] / A_ll
    rho_E_rr = uA_rr[3] / A_rr
    rho_ll, v1_ll, p_ll, _ = cons2prim(uA_ll, equations)
    rho_rr, v1_rr, p_rr, _ = cons2prim(uA_rr, equations)

    p_avg = 0.5 * (p_ll + p_rr)

    f1 = 0.5 * (A_ll * rho_ll * v1_ll + A_rr * rho_rr * v1_rr)
    f2 = 0.5 * (A_ll * rho_ll * v1_ll^2 + A_rr * rho_rr * v1_rr^2) + A_ll * p_avg
    f3 = 0.5 * (A_ll * (rho_E_ll + p_ll) * v1_ll + A_rr * (rho_E_rr + p_rr) * v1_rr)

    return SVector(f1, f2, f3, 0.0)
end

function flux_LXF(a_uM, a_uP, normal, equations::QuasiEuler1D)
    model = equations.eos
    _, vM, _ = cons2prim(a_uM, equations)
    _, vP, _ = cons2prim(a_uP, equations)
    lambda = max(abs(vM) + speed_of_sound(a_uM, equations),
                 abs(vP) + speed_of_sound(a_uP, equations))
    # equations = MyEuler1D(model)
    # uM = a_uM / a_uM[4]
    # uP = a_uP / a_uP[4]
    # f = flux_central(uM[1:3], uP[1:3], 1, equations) * normal - 0.5 * c * (a_uP[1:3] - a_uM[1:3])
    # return SVector(f[1], f[2], f[3], 0.0)
    #return (flux_central(a_uM, a_uP, 1, equations) + 0.5 *
    #     flux_nonconservative_chan_etal(a_uM, a_uP, equations))* normal - 0.5 * lambda * (a_uP- a_uM)
    return flux_ec(a_uM, a_uP, equations) * normal - 0.5 * lambda * (a_uP - a_uM)
end

function flux_LXF_SG(a_uM, a_uP, normal, equations::QuasiEuler1D)
    model = equations.eos
    _, vM, _ = cons2prim(a_uM, equations)
    _, vP, _ = cons2prim(a_uP, equations)
    lambda = max(abs(vM) + speed_of_sound(a_uM, equations),
                 abs(vP) + speed_of_sound(a_uP, equations))
    # equations = MyEuler1D(model)
    # uM = a_uM / a_uM[4]
    # uP = a_uP / a_uP[4]
    # f = flux_central(uM[1:3], uP[1:3], 1, equations) * normal - 0.5 * c * (a_uP[1:3] - a_uM[1:3])
    # return SVector(f[1], f[2], f[3], 0.0)
    #return (flux_central(a_uM, a_uP, 1, equations) + 0.5 *
    #     flux_nonconservative_chan_etal(a_uM, a_uP, equations))* normal - 0.5 * lambda * (a_uP- a_uM)
    a = flux_nonsym(a_uM, a_uP, equations) * normal - 0.5 * lambda * (a_uP - a_uM)
    return SVector(a[1], a[2], a[3], 0.0)
end

function flux_LXF(uM, uP, normal, equations::MyEuler1D)
    _, vM, _ = cons2prim(uM, equations)
    _, vP, _ = cons2prim(uP, equations)
    c = max(abs(vM) + speed_of_sound(uM, equations),
            abs(vP) + speed_of_sound(uP, equations))
    return flux_central(uM, uP, 1, equations) * normal - 0.5 * c * (uP - uM)
end

function flux_APEC_LXF(uM, uP, normal, equations::MyEuler1D)
    model = equations.eos
    c = max(speed_of_sound(uM, equations),
            speed_of_sound(uP, equations))
    return flux_terashima(uM, uP, 1, equations) * normal
    -0.5 * c * (uP - uM)
end

function drho_e_drho_p(u, equations::MyEuler1D)
    (; pInf, q, gamma) = equations.eos
    return q
end

function drho_e_drho_p(u, equations::QuasiEuler1D)
    (; pInf, gamma, q) = equations.eos
    return q
end

function flux_terashima(u_ll, u_rr, orientation::Int, equations::MyEuler1D)
    rho_ll, v1_ll, p_ll = cons2prim(u_ll, equations)
    rho_rr, v1_rr, p_rr = cons2prim(u_rr, equations)

    rho_avg = 0.5 * (rho_ll + rho_rr)
    v1_avg = 0.5 * (v1_ll + v1_rr)
    p_avg = 0.5 * (p_ll + p_rr)
    #rho_e_avg = 0.5 * (rho_ll * e_ll + rho_rr * e_rr)

    rho_e_avg = 0.5 *
                (rho_e_rhoP(rho_ll, p_ll, equations) + rho_e_rhoP(rho_rr, p_rr, equations))
    p_v1_avg = 0.5 * (p_ll * v1_rr + p_rr * v1_ll)

    # chain rule from Terashima
    drho_e_drho_p_ll = drho_e_drho_p(u_ll, equations)
    drho_e_drho_p_rr = drho_e_drho_p(u_rr, equations)
    rho_e_v1_avg = (rho_e_avg -
                    0.25 * (drho_e_drho_p_rr - drho_e_drho_p_ll) * (rho_rr - rho_ll)) *
                   v1_avg

    f_rho = rho_avg * v1_avg
    f_rho_v1 = rho_avg * v1_avg * v1_avg + p_avg
    f_rho_E = rho_e_v1_avg + rho_avg * 0.5 * (v1_ll * v1_rr) * v1_avg + p_v1_avg

    return SVector(f_rho, f_rho_v1, f_rho_E)
end

function flux_nonconservative(u, equations::QuasiEuler1D)
    p = pressure(u, equations)
    return SVector(0.0, p, 0.0, 0.0)
end

function flux_nonconservative2(u, equations::QuasiEuler1D)
    _, _, _, a = u
    return SVector(0.0, a, 0.0, 0.0)
end

function flux_ec(uA_ll, uA_rr, equations::QuasiEuler1D)
    model = equations.eos
    gamma = model.gamma
    A_ll, A_rr = uA_ll[4], uA_rr[4]

    rho_ll, v1_ll, p_ll, _ = cons2prim(uA_ll, equations)
    rho_rr, v1_rr, p_rr, _ = cons2prim(uA_rr, equations)

    rho_mean = Trixi.ln_mean(rho_ll, rho_rr)
    inv_rho_p_mean = p_ll * p_rr * Trixi.inv_ln_mean(rho_ll * p_rr, rho_rr * p_ll)
    v1_avg = 0.5 * (v1_ll + v1_rr)
    p_avg = 0.5 * (p_ll + p_rr)
    velocity_square_avg = 0.5 * (v1_ll * v1_rr)

    v1A_ll = v1_ll * A_ll
    v1A_rr = v1_rr * A_rr
    v1A_avg = 0.5 * (v1A_ll + v1A_rr)

    f1 = rho_mean * v1A_avg
    f2 = f1 * v1_avg + A_ll * p_avg
    f3 = f1 * (velocity_square_avg + inv_rho_p_mean / (gamma - 1)) +
         0.5 * (p_ll * v1A_rr + p_rr * v1A_ll)

    return SVector(f1, f2, f3, 0.0)
end

# A basic density wave with initial condition scaled to match physical units of liquid water
function initial_condition_density_wave(x, t,
                                        equations::MyEuler1D)
    RealT = eltype(x)
    eos = equations.equation_of_state
    v1 = 100.0 # m/s
    rho = 1000 * (1 + 0.5 * sin(2 * pi * (x[1] - v1 * t))) # kg/m^3
    p = 1e6 # Pa
    T = temperature(inv(rho), p, equations)
    return thermo2cons(SVector(inv(rho), v1, T), equations)
end

function rhs!(du_voa, u_voa, params, t)
    (; rd, md, invMQTr, equations, C12, Qr_skew) = params
    du = parent(du_voa)
    u = parent(u_voa)
    # calc volume residual

    fill!(du, zero(eltype(du)))
    for e in axes(u, 2)
        du[i, e] .-= invMQr * flux(u[i, e], 1, equations)
    end

    uM = rd.Vf * u
    uP = uM[md.mapP]
    du .+= rd.LIFT * @. flux(uM, uP, md.nx, equations)

    @. du = -du / md.J
    return du
end

N = 3
M = 100
rd = RefElemData(Line(), SBP(), N)
(VX,), EToV = uniform_mesh(Line(), M)

md = MeshData((VX,), EToV, rd, is_periodic = true)

(Qrh,), VhP, Ph = hybridized_SBP_operators(rd)
invMQTr = rd.M \ (rd.Dr' * rd.M)
Qr_skew = rd.M * rd.Dr - rd.Dr' * rd.M

params = (; rd, md, C12 = 1, initial_condition, equations,
          invMQTr = rd.M \ (rd.Dr' * rd.M),
          Qr_skew, Qrh_skew = Qrh - Qrh', VhP, Vh = VhP * rd.Vq, Ph,
          epsilon_save = Float64[], t_save = Float64[], unorm_save = Float64[])

u = Pq * initial_condition_density_wave.(md.x, 0.0, trixisg);
tspan = (0.0, 0.01)
ode = ODEProblem(rhs!, VectorOfArray(u), tspan, params)

sg = MyEuler1D(MySG())
trixisg = StiffenedGas()

rho = 1 / 0.0005
v1 = 1.0;
T = 650;
V = inv(rho)
e = energy_internal_specific(V, T, trixisg)
rhoE = rho * (e + 0.5 * v1 * v1)
u = SVector(rho, rho * v1, rhoE)
e = energy_internal_specific(u, equations_1)

T = temperature(u, MyEuler1D(sg))
T_trixi = temperature(V, e, trixisg)

speed_of_sound(V, T, trixisg)
speed_of_sound(u, MyEuler1D(sg))
