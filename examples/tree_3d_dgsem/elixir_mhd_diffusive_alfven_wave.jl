using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the visco-resistive compressible MHD equations

prandtl_number() = 0.72
mu_const = 1e-2
eta_const = 1e-2

equations = IdealGlmMhdEquations3D(5 / 3)
equations_parabolic = ViscoResistiveMhd3D(equations, mu = mu_const,
                                          Prandtl = prandtl_number(),
                                          eta = eta_const,
                                          gradient_variables = GradientVariablesPrimitive())

volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
solver = DGSEM(polydeg = 3,
               surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell),
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-1.0, -1.0, -1.0) # minimum coordinates (min(x), min(y), min(z))
coordinates_max = (1.0, 1.0, 1.0) # maximum coordinates (max(x), max(y), max(z))

# Create a uniformly refined mesh
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                n_cells_max = 100_000) # set maximum capacity of tree data structure

function initial_condition_constant_alfven(x, t, equations)
    # Alfvén wave in three space dimensions modified by a periodic density variation.
    # For the system without the density variations see: Altmann thesis http://dx.doi.org/10.18419/opus-3895.
    # Domain must be set to [-1, 1]^3, γ = 5/3.
    p = 1
    omega = 2 * pi # may be multiplied by frequency
    # r: length-variable = length of computational domain
    r = 2
    # e: epsilon = 0.02
    e = 0.02
    nx = 1 / sqrt(r^2 + 1)
    ny = r / sqrt(r^2 + 1)
    sqr = 1
    Va = omega / (ny * sqr)
    phi_alv = omega / ny * (nx * (x[1] - 0.5 * r) + ny * (x[2] - 0.5 * r)) - Va * t

    rho = 1 + e*cos(phi_alv + 1)
    v1 = -e * ny * cos(phi_alv) / rho
    v2 = e * nx * cos(phi_alv) / rho
    v3 = e * sin(phi_alv) / rho
    B1 = nx - rho * v1 * sqr
    B2 = ny - rho * v2 * sqr
    B3 = -rho * v3 * sqr
    psi = 0

    return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end

@inline function source_terms_mhd_convergence_test(u, x, t, equations)
    dTdx2 = pi^2*(0.0133333333333333*(0.0002*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2 + 0.0002*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2)*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^2 + 0.00533333333333333*(0.01*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) - 1)*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) + 0.00533333333333333*(0.04*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) + 1)*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) - 0.000266666666666667*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2 - (0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 0.0008*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)^2/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1))*((0.000133333333333333*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2 + 0.000133333333333333*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2)/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1) + 0.266666666666667*(0.01*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) - 1)^2 + 0.0666666666666667*(0.04*cos(pi*(sqrt(5)*t -    x[1] - 2*x[2] + 3.0)) + 1)^2 + 0.000133333333333333*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2 - 0.666666666666667)/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1) - 0.04*(-0.0133333333333333*(0.0002*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2 + 0.0002*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2)*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^2 + 0.00533333333333333*(0.01*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) - 1)*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) + 0.00533333333333333*(0.04*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) + 1)*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) - 0.000266666666666667*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) + (-3.61400724161835e-20*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) + 5.33333333333333e-6*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1) + 5.33333333333333e-6*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1))/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1))*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1) - (-3.61400724161835e-20*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2 + 3.61400724161835e-20*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2 + 5.33333333333333e-6*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1) - 3.70435742265881e-21*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1) + 5.33333333333333e-6*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1) + 3.2e-7*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)^2/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^2 + 3.2e-7*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)^2*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^2)/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1) + 0.0266666666666667*(-5.42101086242752e-20*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) + 8.0e-6*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1) + 8.0e-6*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1))*sin(-sqrt(5)*pi*t + pi *(x[1] + 2*x[2] - 3.0) + 1)/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^2 + 5.29395592033938e-23*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)^2/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^2)/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)

    dTdy2 = pi^2*(0.0533333333333333*(0.0002*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2 + 0.0002*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2)*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^2 + 0.0213333333333333*(0.01*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) - 1)*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) + 0.0213333333333333*(0.04*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) + 1)*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) - 0.00106666666666667*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2 - (0.08*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 0.0032*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)^2/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1))*((0.000133333333333333*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2 + 0.000133333333333333*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2)/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1) + 0.266666666666667*(0.01*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) - 1)^2 + 0.0666666666666667*(0.04*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) + 1)^2 + 0.000133333333333333*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2 - 0.666666666666667)/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1) - 0.08*(-0.0266666666666667*(0.0002*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2 + 0.0002*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2)*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^2 + 0.0106666666666667*(0.01*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) - 1)*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) + 0.0106666666666667*(0.04*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) + 1)*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) - 0.000533333333333333*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) + (-7.2280144832367e-20*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) + 1.06666666666667e-5*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1) + 1.06666666666667e-5*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1))/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1))*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1) - (-1.44560289664734e-19*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2 + 1.44560289664734e-19*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2 + 2.13333333333333e-5*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1) - 1.48174296906352e-20*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1) + 2.13333333333333e-5*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1) + 1.28e-6*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)^2/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^2 + 1.28e-6*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)^2*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^2)/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1) + 0.0533333333333333*(-1.0842021724855e-19*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) + 1.6e-5*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1) + 1.6e-5*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1))*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^2 + 2.11758236813575e-22*sin(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)^2/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^2)/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)

    dTdz2 = 0

    r_1 = 0.02*sqrt(5)*pi*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)

    r_2 = pi*(sqrt(5)*pi*mu_const*(-0.04*(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^2*cos(pi*(sqrt(5)*t - x[1]- 2*x[2] + 3.0)) + (0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) +1) + 1)*(0.0012*cos(-2*sqrt(5)*pi*t + 2*pi*x[1] + 4*pi*x[2] + 1) - 0.0004*cos(1)) + 3.2e-5*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)^2*cos(pi*(sqrt(5)*t - x[1]- 2*x[2] + 3.0))) + (3.61400724161835e-20*(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)*sin(pi*(-sqrt(5)*t + x[1] + 2*x[2]))*cos(pi*(sqrt(5)*t - x[1]- 2*x[2] + 3.0)) - 5.33333333333333e-6*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1))*(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1) - (8.13151629364128e-20*cos(pi*(sqrt(5)*t - x[1]- 2*x[2] + 3.0)) + 8.67361737988404e-19)*(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^3*sin(pi*(-sqrt(5)*t + x[1] + 2*x[2])) + 2.66666666666667e-6*(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1))/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^3

    r_3 = pi*(-sqrt(5)*pi*mu_const*(-0.02*(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^2*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) + (0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)*(0.0006*cos(-2*sqrt(5)*pi*t + 2*pi*x[1] + 4*pi*x[2] + 1) - 0.0002*cos(1)) + 1.6e-5*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)^2*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))) + (3.46944695195361e-18 - 5.42101086242752e-20*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)))*(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^3*sin(pi*(-sqrt(5)*t + x[1] + 2*x[2])) + (7.2280144832367e-20*(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)*sin(pi*(-sqrt(5)*t + x[1] + 2*x[2]))*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) - 1.06666666666667e-5*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1))*(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1) + 5.33333333333333e-6*(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1))/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^3

    r_4 = -pi^2*mu_const*((0.003*sin(-2*sqrt(5)*pi*t + 2*pi*x[1] + 4*pi*x[2] + 1) + 0.001*sin(1))*(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1) + 0.1*(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^2*sin(pi*(-sqrt(5)*t + x[1] + 2*x[2])) - 8.0e-5*sin(pi*(-sqrt(5)*t + x[1] + 2*x[2]))*sin(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1)^2)/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^3

    r_5 = 0.2*(1.91641204316285e-21*pi^2*mu_const*(cos(-4*sqrt(5)*pi*t + 4*pi*x[1] + 8*pi*x[2] + 2)/8 - cos(2)/8) - 1.58818677610181e-20*pi^2*mu_const*(cos(-3*sqrt(5)*pi*t + 3*pi*x[1] + 6*pi*x[2] + 1)- cos(sqrt(5)*pi*t - pi*x[1] - 2*pi*x[2] + 1)) - 1.2e-5*pi^2*mu_const*sin(pi*(-sqrt(5)*t + x[1] + 2*x[2]))^2*sin(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)^2 - 4.0e-6*pi^2*mu_const*sin(pi*(-sqrt(5)*t + x[1] + 2*x[2]))^2*cos(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)^2 + 0.0002*pi^2*mu_const*sin(pi*(-sqrt(5)*t + x[1] + 2*x[2]))^2*cos(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1) - 1.0842021724855e-18*pi^2*mu_const*sin(pi*(-sqrt(5)*t + x[1] + 2*x[2]))^2 - 1.2e-5*pi^2*mu_const*sin(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)^2*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2 - 4.0e-6*pi^2*mu_const*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2*cos(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)^2 + 0.0002*pi^2*mu_const*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2*cos(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1) - 2.77777777777778e-6*mu_const*cos(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)^4*dTdx2 - 2.77777777777778e-6*mu_const*cos(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)^4*dTdy2 - 2.77777777777778e-6*mu_const*cos(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)^4*dTdz2 + 0.000555555555555556*mu_const*cos(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)^3*dTdx2 + 0.000555555555555556*mu_const*cos(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)^3*dTdy2 + 0.000555555555555556*mu_const*cos(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)^3*dTdz2- 0.0416666666666667*mu_const*cos(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)^2*dTdx2 - 0.0416666666666667*mu_const*cos(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)^2*dTdy2 - 0.0416666666666667*mu_const*cos(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)^2*dTdz2 + 1.38888888888889*mu_const*cos(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)*dTdx2 + 1.38888888888889*mu_const*cos(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)*dTdy2 + 1.38888888888889*mu_const*cos(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)*dTdz2 - 17.3611111111111*mu_const*dTdx2 - 17.3611111111111*mu_const*dTdy2 - 17.3611111111111*mu_const*dTdz2 + 4.06575814682064e-21*sqrt(5)*pi*(-sin(-3*sqrt(5)*pi*t + 3*pi*x[1] + 6*pi*x[2] + 1) + sin(sqrt(5)*pi*t - pi*x[1] - 2*pi*x[2] + 1)) - 4.06575814682064e-23*sqrt(5)*pi*(-2*sin(pi*(-2*sqrt(5)*t + 2*x[1] + 4*x[2])) - sin(-4*sqrt(5)*pi*t + 4*pi*x[1] + 8*pi*x[2] + 2) + sin(2)) + 2.0e-7*sqrt(5)*pi*(-sin(-4*sqrt(5)*pi*t + 4*pi*x[1] + 8*pi*x[2] + 2) + 2*sin(-2*sqrt(5)*pi*t + 2*pi*x[1] + 4*pi*x[2] + 2) - sin(2)) + 2.0e-7*sqrt(5)*pi*(sin(-4*sqrt(5)*pi*t + 4*pi*x[1] + 8*pi*x[2] + 2) + 2*sin(-2*sqrt(5)*pi*t + 2*pi*x[1] + 4*pi*x[2] + 2) + sin(2)) + 1.35525271560688e-19*sqrt(5)*pi*sin(pi*(-2*sqrt(5)*t + 2*x[1] + 4*x[2])) - 1.6e-8*sqrt(5)*pi*sin(pi*(-sqrt(5)*t + x[1] + 2*x[2]))^2*sin(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)*cos(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)^2 - 4.0e-5*sqrt(5)*pi*sin(pi*(-sqrt(5)*t + x[1] + 2*x[2]))^2*sin(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1) + 2.16840434497101e-24*sqrt(5)*pi*sin(pi*(-sqrt(5)*t + x[1] + 2*x[2]))*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))*cos(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)^3 - 1.6e-8*sqrt(5)*pi*sin(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2*cos(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)^2 - 4.0e-5*sqrt(5)*pi*sin(-sqrt(5)*pi*t + pi*x[1] + 2*pi*x[2] + 1)*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0))^2)/(-0.04*sin(pi*x[2])*sin(-sqrt(5)*pi*t + pi*x[1] + pi*x[2] + 1) + 0.02*cos(-sqrt(5)*pi*t + pi*x[1] + 1) - 1.0)^4

    r_6 = pi*(-0.04*(sqrt(5)*pi*eta_const*cos(pi*(-sqrt(5)*t + x[1] + 2*x[2])) + sin(pi*(-sqrt(5)*t + x[1] + 2*x[2])))*(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^2 + 0.04*(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)*sin(pi*(-sqrt(5)*t + x[1] + 2*x[2])) + 0.0004*sin(-2*sqrt(5)*pi*t + 2*pi*x[1] + 4*pi*x[2] + 1) + 0.0004*sin(1))/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0)+ 1) + 1)^2

    r_7 = pi*(0.02*(sqrt(5)*pi*eta_const*cos(pi*(-sqrt(5)*t + x[1] + 2*x[2])) + sin(pi*(-sqrt(5)*t + x[1] + 2*x[2])))*(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^2 - 0.02*(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)*sin(pi*(-sqrt(5)*t + x[1] + 2*x[2])) - 0.0002*sin(-2*sqrt(5)*pi*t + 2*pi*x[1] + 4*pi*x[2] + 1) - 0.0002*sin(1))/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^2

    r_8 = pi*((0.1*pi*eta_const*sin(pi*(-sqrt(5)*t + x[1] + 2*x[2])) + 0.02*sqrt(5)*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)))*(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^2 - 0.02*sqrt(5)*(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)*cos(pi*(sqrt(5)*t - x[1] - 2*x[2] + 3.0)) + 0.0002*sqrt(5)*(cos(-2*sqrt(5)*pi*t + 2*pi*x[1] + 4*pi*x[2] + 1) - cos(1)))/(0.02*cos(-sqrt(5)*pi*t + pi*(x[1] + 2*x[2] - 3.0) + 1) + 1)^2

    r_9 = 0

    return SVector(r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8, r_9)
end

initial_condition = initial_condition_constant_alfven

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver,
                                             source_terms = source_terms_mhd_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 0.001)
ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 100)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 100,
                                     solution_variables = cons2prim)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
cfl = 0.25
stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        save_solution,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1e-5, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

# Print the timer summary.
summary_callback()
