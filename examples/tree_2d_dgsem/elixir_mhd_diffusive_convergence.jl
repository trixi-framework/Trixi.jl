using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the visco-resistive compressible MHD equations

prandtl_number() = 0.72
mu_const = 1e-2
eta_const = 1e-2
prandtl_const = prandtl_number()

equations = IdealGlmMhdEquations2D(5 / 3)
equations_parabolic = ViscoResistiveMhd2D(equations, mu = mu_const,
                                          Prandtl = prandtl_number(),
                                          eta = eta_const,
                                          gradient_variables = GradientVariablesPrimitive())

volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
solver = DGSEM(polydeg = 3,
               surface_flux = (flux_hindenlang_gassner, flux_nonconservative_powell),
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y), min(z))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y), max(z))

# Create a uniformly refined mesh
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 1,
                n_cells_max = 150_000) # set maximum capacity of tree data structure

function initial_condition_constant_alfven_3d(x, t, equations)
    # Alfvén wave in three space dimensions modified by a periodic density variation.
    # For the system without the density variations see: Altmann thesis http://dx.doi.org/10.18419/opus-3895.
    # Domain must be set to [-1, 1]^3, γ = 5/3.
    omega = 2.0 * pi # may be multiplied by frequency
    # r = length-variable = length of computational domain
    r = 2.0
    # e = epsilon
    e = 0.02
    nx = 1 / sqrt(r^2 + 1.0)
    ny = r / sqrt(r^2 + 1.0)
    sqr = 1.0
    Va = omega / (ny * sqr)
    phi_alv = omega / ny * (nx * (x[1] - 0.5 * r) + ny * (x[2] - 0.5 * r)) - Va * t

    # 3d Alfven wave
    rho = 1.0 + e * cos(phi_alv + 1.0)
    v1 = -e * ny * cos(phi_alv) / rho
    v2 = e * nx * cos(phi_alv) / rho
    v3 = e * sin(phi_alv) / rho
    p = 1.0# + e*cos(phi_alv + 1.0)
    B1 = nx - rho * v1 * sqr
    B2 = ny - rho * v2 * sqr
    B3 = -rho * v3 * sqr
    psi = 0.0

    return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end

@inline function source_terms_mhd_convergence_test_3d(u, x, t, equations)
    # 3d Alfven wave, rho gets perturbed
    r_1 = 0.02 * sqrt(5) * pi * sin(-sqrt(5) * pi * t + pi * (x[1] + 2 * x[2] - 3.0) + 1)

    r_2 = -mu_const * (0.04 * sqrt(5) * pi^2 *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
           0.0016 * sqrt(5) * pi^2 *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
           0.0008 * sqrt(5) * pi^2 *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
           cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
           3.2e-5 * sqrt(5) * pi^2 *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1)^2 *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3) -
          0.0016 * sqrt(5) * pi *
          (-0.01 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           sqrt(5)) *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) -
          0.008 * sqrt(5) * pi *
          (-0.008 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) -
           sqrt(5) / 5) *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) -
          0.016 * sqrt(5) * pi *
          (-0.004 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           2 * sqrt(5) / 5) *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) -
          0.0016 * sqrt(5) * pi *
          (0.04 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           sqrt(5)) *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) -
          0.0004 * pi *
          (-0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1)^2 *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
          cos(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
          8.0e-6 * pi *
          (-0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1)^2 *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
          sin(-sqrt(5) * pi * t +
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) +
              1) /
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3 +
          8.0e-6 * pi *
          (-0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
          sin(-sqrt(5) * pi * t +
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) +
              1) /
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
          0.04 * pi *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))

    r_3 = -mu_const * (-0.02 * sqrt(5) * pi^2 *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) +
           0.0008 * sqrt(5) * pi^2 *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
           0.0004 * sqrt(5) * pi^2 *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
           cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
           1.6e-5 * sqrt(5) * pi^2 *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1)^2 *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3) +
          0.0032 * sqrt(5) * pi *
          (-0.01 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           sqrt(5)) *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) -
          0.004 * sqrt(5) * pi *
          (-0.008 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) -
           sqrt(5) / 5) *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) -
          0.008 * sqrt(5) * pi *
          (-0.004 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           2 * sqrt(5) / 5) *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
          0.0032 * sqrt(5) * pi *
          (0.04 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           sqrt(5)) *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) -
          0.0008 * pi *
          (-0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1)^2 *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
          cos(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
          1.6e-5 * pi *
          (-0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1)^2 *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
          sin(-sqrt(5) * pi * t +
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) +
              1) /
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3 +
          1.6e-5 * pi *
          (-0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
          sin(-sqrt(5) * pi * t +
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) +
              1) /
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
          0.02 * pi *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))

    r_4 = -mu_const * (0.1 * pi^2 *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
           0.002 * pi^2 *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
           cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
           0.004 * pi^2 *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
           8.0e-5 * pi^2 *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1)^2 /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3) -
          0.04 * pi *
          (-0.004 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           2 * sqrt(5) / 5) *
          (-0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
          cos(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) +
          0.0008 * pi *
          (-0.004 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           2 * sqrt(5) / 5) *
          (-0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
          sin(-sqrt(5) * pi * t +
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) +
              1) /
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
          0.0008 * pi *
          (-0.004 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           2 * sqrt(5) / 5) *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
          sin(-sqrt(5) * pi * t +
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) +
              1) / (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
          0.02 * pi *
          (0.008 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           sqrt(5) / 5) *
          (-0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
          cos(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) +
          0.0004 * pi *
          (0.008 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           sqrt(5) / 5) *
          (-0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
          sin(-sqrt(5) * pi * t +
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) +
              1) /
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
          0.0004 * pi *
          (0.008 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           sqrt(5) / 5) *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
          sin(-sqrt(5) * pi * t +
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) +
              1) / (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
          0.02 * sqrt(5) * pi *
          cos(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))

    r_5 = -eta_const * (0.02 * sqrt(5) * pi^2 *
           (-0.004 * sqrt(5) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
            2 * sqrt(5) / 5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           0.02 * pi *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
           (0.02 * pi *
            (-0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
            0.0004 * pi *
            (-0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
            0.0004 * pi *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
           0.02 *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
           (0.02 * pi^2 *
            (-0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
            0.0004 * pi^2 *
            (-0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
            0.0008 * pi^2 *
            (-0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
            1.6e-5 * pi^2 *
            (-0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1)^2 /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3 -
            0.0004 * pi^2 *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) +
            0.0008 * pi^2 *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
            1.6e-5 * pi^2 *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1)^2 /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
           0.0004 * pi *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
           (0.02 * pi *
            (-0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
            0.0004 * pi *
            (-0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
            0.0004 * pi *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
           0.0004 * pi^2 *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 -
           0.0004 * pi *
           (0.02 * pi *
            (-0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
            0.0004 * pi *
            (-0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
            0.0004 * pi *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)) -
          eta_const * (-0.04 * sqrt(5) * pi^2 *
           (0.008 * sqrt(5) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
            sqrt(5) / 5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           0.04 * pi *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
           (0.04 * pi *
            (-0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
            0.0008 * pi *
            (-0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
            0.0008 * pi *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
           0.02 *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
           (0.08 * pi^2 *
            (-0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
            0.0016 * pi^2 *
            (-0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
            0.0032 * pi^2 *
            (-0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
            6.4e-5 * pi^2 *
            (-0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1)^2 /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3 -
            0.0016 * pi^2 *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) +
            0.0032 * pi^2 *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
            6.4e-5 * pi^2 *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1)^2 /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
           0.0008 * pi *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
           (0.04 * pi *
            (-0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
            0.0008 * pi *
            (-0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
            0.0008 * pi *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
           0.0016 * pi^2 *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 -
           0.0008 * pi *
           (0.04 * pi *
            (-0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
            0.0008 * pi *
            (-0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
            0.0008 * pi *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)) -
          mu_const * (0.00555555555555556 * sqrt(5) * pi^2 *
           (-0.01 * sqrt(5) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) + sqrt(5)) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) -
           0.00555555555555556 * sqrt(5) * pi^2 *
           (0.04 * sqrt(5) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) + sqrt(5)) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) -
           0.0694444444444444 * pi^2 *
           (0.0002 *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
            0.0002 *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2) *
           cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) +
           0.02 * pi *
           (0.02 * pi *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
            0.0004 * pi *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
           0.0004 * pi *
           (0.02 * pi *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
            0.0004 * pi *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
           0.008 * sqrt(5) * pi *
           (-0.016 * sqrt(5) * pi *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
            0.00032 * sqrt(5) * pi *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
           0.00016 * sqrt(5) * pi *
           (-0.016 * sqrt(5) * pi *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
            0.00032 * sqrt(5) * pi *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2) *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
           0.004 * sqrt(5) * pi *
           (-0.012 * sqrt(5) * pi *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
            0.00024 * sqrt(5) * pi *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) +
           8.0e-5 * sqrt(5) * pi *
           (-0.012 * sqrt(5) * pi *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
            0.00024 * sqrt(5) * pi *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2) *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
           0.00138888888888889 * pi^2 *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1)^2 *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
           0.00138888888888889 * pi^2 *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1)^2 *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
           2.77777777777778e-5 * pi^2 *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1)^2 *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
           cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3 -
           0.000111111111111111 * pi^2 *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1)^2 *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3 +
           1.66666666666667e-6 * pi^2 *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1)^2 *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1)^2 /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^4 +
           2.77777777777778e-5 * pi^2 *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
           cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
           0.000111111111111111 * pi^2 *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
           2.22222222222222e-6 * pi^2 *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1)^2 /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3 +
           (0.0694444444444444 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) +
            3.47222222222222) * (-5.42101086242752e-20 * pi^2 *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
            5.42101086242752e-20 * pi^2 *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
            8.0e-6 * pi^2 *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
            cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3 -
            5.55653613398821e-21 * pi^2 *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3 +
            8.0e-6 * pi^2 *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
            cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3 +
            4.8e-7 * pi^2 *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1)^2 /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^4 +
            4.8e-7 * pi^2 *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1)^2 *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^4) -
           0.138888888888889 * pi *
           (-5.42101086242752e-20 * pi *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
            8.0e-6 * pi *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3 +
            8.0e-6 * pi *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3) *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) +
           0.00138888888888889 * pi^2 *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 -
           0.02 *
           (0.02 * pi^2 *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
            0.0004 * pi^2 *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
            0.0008 * pi^2 *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
            1.6e-5 * pi^2 *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1)^2 /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) +
           0.004 * sqrt(5) *
           (0.012 * sqrt(5) * pi^2 *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
            0.00048 * sqrt(5) * pi^2 *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
            0.00024 * sqrt(5) * pi^2 *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
            9.6e-6 * sqrt(5) * pi^2 *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1)^2 *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
           0.008 * sqrt(5) *
           (0.016 * sqrt(5) * pi^2 *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
            0.00064 * sqrt(5) * pi^2 *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
            0.00032 * sqrt(5) * pi^2 *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
            1.28e-5 * sqrt(5) * pi^2 *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1)^2 *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) +
           5.55555555555556e-7 * pi^2 *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1)^2 /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2) -
          mu_const * (0.0222222222222222 * sqrt(5) * pi^2 *
           (-0.01 * sqrt(5) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) + sqrt(5)) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) -
           0.0222222222222222 * sqrt(5) * pi^2 *
           (0.04 * sqrt(5) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) + sqrt(5)) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) -
           0.277777777777778 * pi^2 *
           (0.0002 *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
            0.0002 *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2) *
           cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) +
           0.04 * pi *
           (0.04 * pi *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
            0.0008 * pi *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
           0.0008 * pi *
           (0.04 * pi *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
            0.0008 * pi *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
           0.016 * sqrt(5) * pi *
           (-0.012 * sqrt(5) * pi *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
            0.00024 * sqrt(5) * pi *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
           0.00032 * sqrt(5) * pi *
           (-0.012 * sqrt(5) * pi *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
            0.00024 * sqrt(5) * pi *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2) *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
           0.008 * sqrt(5) * pi *
           (0.016 * sqrt(5) * pi *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) +
            0.00032 * sqrt(5) * pi *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) +
           0.00016 * sqrt(5) * pi *
           (0.016 * sqrt(5) * pi *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) +
            0.00032 * sqrt(5) * pi *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2) *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
           0.00555555555555556 * pi^2 *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1)^2 *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
           0.00555555555555556 * pi^2 *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1)^2 *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
           0.000111111111111111 * pi^2 *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1)^2 *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
           cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3 -
           0.000444444444444444 * pi^2 *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1)^2 *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3 +
           6.66666666666667e-6 * pi^2 *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1)^2 *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1)^2 /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^4 +
           0.000111111111111111 * pi^2 *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
           cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
           0.000444444444444444 * pi^2 *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
           8.88888888888889e-6 * pi^2 *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1)^2 /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3 +
           (0.0694444444444444 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) +
            3.47222222222222) * (-2.16840434497101e-19 * pi^2 *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
            2.16840434497101e-19 * pi^2 *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
            3.2e-5 * pi^2 *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
            cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3 -
            2.22261445359528e-20 * pi^2 *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3 +
            3.2e-5 * pi^2 *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
            cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3 +
            1.92e-6 * pi^2 *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1)^2 /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^4 +
            1.92e-6 * pi^2 *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1)^2 *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^4) -
           0.277777777777778 * pi *
           (-1.0842021724855e-19 * pi *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
            1.6e-5 * pi *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3 +
            1.6e-5 * pi *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3) *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) +
           0.00555555555555556 * pi^2 *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 -
           0.02 *
           (0.08 * pi^2 *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
            0.0016 * pi^2 *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
            0.0032 * pi^2 *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
            6.4e-5 * pi^2 *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1)^2 /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) +
           0.004 * sqrt(5) *
           (-0.032 * sqrt(5) * pi^2 *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) +
            0.00128 * sqrt(5) * pi^2 *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
            0.00064 * sqrt(5) * pi^2 *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
            2.56e-5 * sqrt(5) * pi^2 *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1)^2 *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
           0.008 * sqrt(5) *
           (0.024 * sqrt(5) * pi^2 *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
            0.00096 * sqrt(5) * pi^2 *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
            0.00048 * sqrt(5) * pi^2 *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
            1.92e-5 * sqrt(5) * pi^2 *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1)^2 *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) +
           2.22222222222222e-6 * pi^2 *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1)^2 /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2) +
          0.008 * pi *
          (-0.01 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           sqrt(5)) *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
          (-0.008 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) -
           sqrt(5) / 5) * (0.004 * sqrt(5) * pi *
           (-0.004 * sqrt(5) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
            2 * sqrt(5) / 5) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) +
           8.0e-5 * sqrt(5) * pi *
           (-0.004 * sqrt(5) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
            2 * sqrt(5) / 5) *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
           0.008 * sqrt(5) * pi *
           (0.008 * sqrt(5) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
            sqrt(5) / 5) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
           0.00016 * sqrt(5) * pi *
           (0.008 * sqrt(5) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
            sqrt(5) / 5) *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
           0.0008 * pi *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
           1.6e-5 * pi *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3 -
           0.0004 * pi *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) +
           8.0e-6 * pi *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2) +
          (0.004 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) -
           2 * sqrt(5) / 5) * (0.008 * sqrt(5) * pi *
           (-0.004 * sqrt(5) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
            2 * sqrt(5) / 5) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) +
           0.00016 * sqrt(5) * pi *
           (-0.004 * sqrt(5) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
            2 * sqrt(5) / 5) *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
           0.016 * sqrt(5) * pi *
           (0.008 * sqrt(5) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
            sqrt(5) / 5) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
           0.00032 * sqrt(5) * pi *
           (0.008 * sqrt(5) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
            sqrt(5) / 5) *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
           0.0016 * pi *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
           3.2e-5 * pi *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3 -
           0.0008 * pi *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) +
           1.6e-5 * pi *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2) -
          0.008 * pi *
          (0.04 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           sqrt(5)) *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
          0.02 * sqrt(5) * pi *
          (0.0002 *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
           0.0002 *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2) *
          sin(-sqrt(5) * pi * t +
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) +
              1) +
          0.0004 * sqrt(5) * pi *
          (-0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1)^2 *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
          cos(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
          8.0e-6 * sqrt(5) * pi *
          (-0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1)^2 *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
          sin(-sqrt(5) * pi * t +
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) +
              1) /
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3 -
          8.0e-6 * sqrt(5) * pi *
          (-0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
          sin(-sqrt(5) * pi * t +
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) +
              1) /
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) *
          (5.42101086242752e-20 * sqrt(5) * pi *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
           8.0e-6 * sqrt(5) * pi *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3 -
           8.0e-6 * sqrt(5) * pi *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3) +
          0.004 * sqrt(5) *
          (-0.0064 * sqrt(5) * pi *
           (-0.01 * sqrt(5) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) + sqrt(5)) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           0.0064 * sqrt(5) * pi *
           (0.04 * sqrt(5) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) + sqrt(5)) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) -
           0.04 * pi *
           (0.0002 *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
            0.0002 *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2) *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) -
           0.0016 * pi *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1)^2 *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
           3.2e-5 * pi *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1)^2 *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3 +
           3.2e-5 * pi *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) *
           (-1.0842021724855e-19 * pi *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
            1.6e-5 * pi *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3 +
            1.6e-5 * pi *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3)) *
          cos(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
          0.008 * sqrt(5) *
          (-0.0032 * sqrt(5) * pi *
           (-0.01 * sqrt(5) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) + sqrt(5)) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           0.0032 * sqrt(5) * pi *
           (0.04 * sqrt(5) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) + sqrt(5)) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) -
           0.02 * pi *
           (0.0002 *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
            0.0002 *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2) *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) -
           0.0008 * pi *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1)^2 *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
           1.6e-5 * pi *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1)^2 *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3 +
           1.6e-5 * pi *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) *
           (-5.42101086242752e-20 * pi *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
            8.0e-6 * pi *
            sin(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3 +
            8.0e-6 * pi *
            sin(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
            cos(sqrt(5) * pi * t -
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))^2 /
            (0.02 * cos(-sqrt(5) * pi * t +
                 sqrt(5) * pi *
                 (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3)) *
          cos(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)

    r_6 = 0.04 * sqrt(5) * pi^2 * eta_const *
          cos(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
          0.016 * sqrt(5) * pi *
          (-0.004 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           2 * sqrt(5) / 5) *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) +
          0.00032 * sqrt(5) * pi *
          (-0.004 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           2 * sqrt(5) / 5) *
          sin(-sqrt(5) * pi * t +
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) +
              1) *
          cos(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
          0.008 * sqrt(5) * pi *
          (0.008 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           sqrt(5) / 5) *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) +
          0.00016 * sqrt(5) * pi *
          (0.008 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           sqrt(5) / 5) *
          sin(-sqrt(5) * pi * t +
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) +
              1) *
          cos(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
          0.04 * pi *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))

    r_7 = -0.02 * sqrt(5) * pi^2 * eta_const *
          cos(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) -
          0.008 * sqrt(5) * pi *
          (-0.004 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           2 * sqrt(5) / 5) *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
          0.00016 * sqrt(5) * pi *
          (-0.004 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           2 * sqrt(5) / 5) *
          sin(-sqrt(5) * pi * t +
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) +
              1) *
          cos(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
          0.004 * sqrt(5) * pi *
          (0.008 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           sqrt(5) / 5) *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
          8.0e-5 * sqrt(5) * pi *
          (0.008 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           sqrt(5) / 5) *
          sin(-sqrt(5) * pi * t +
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) +
              1) *
          cos(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
          0.02 * pi *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5))

    r_8 = -eta_const * (0.1 * pi^2 *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
           0.002 * pi^2 *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
           cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
           0.004 * pi^2 *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
           8.0e-5 * pi^2 *
           (-0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1)^2 /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^3 -
           0.002 * pi^2 *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
           cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) +
           0.004 * pi^2 *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) -
           8.0e-5 * pi^2 *
           sin(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
           sin(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1)^2 /
           (0.02 * cos(-sqrt(5) * pi * t +
                sqrt(5) * pi *
                (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2) -
          0.04 * pi *
          (-0.004 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           2 * sqrt(5) / 5) *
          cos(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) +
          0.0008 * pi *
          (-0.004 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           2 * sqrt(5) / 5) *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
          sin(-sqrt(5) * pi * t +
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) +
              1) /
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
          0.02 * pi *
          (0.008 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           sqrt(5) / 5) *
          cos(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) +
          0.0004 * pi *
          (0.008 * sqrt(5) *
           cos(sqrt(5) * pi * t -
               sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) +
           sqrt(5) / 5) *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
          sin(-sqrt(5) * pi * t +
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) +
              1) /
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 -
          0.02 * sqrt(5) * pi *
          (-0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
          cos(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) /
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1) +
          0.0004 * sqrt(5) * pi *
          (-0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) - 1) *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
          sin(-sqrt(5) * pi * t +
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) +
              1) /
          (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)^2 +
          0.0004 * sqrt(5) * pi *
          sin(sqrt(5) * pi * t -
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5)) *
          sin(-sqrt(5) * pi * t +
              sqrt(5) * pi * (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) +
              1) / (0.02 * cos(-sqrt(5) * pi * t +
               sqrt(5) * pi *
               (sqrt(5) * (x[1] - 1.0) / 5 + 2 * sqrt(5) * (x[2] - 1.0) / 5) + 1) + 1)

    r_9 = 0.0

    return SVector(r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8, r_9)
end

initial_condition = initial_condition_constant_alfven_3d
source_terms = source_terms_mhd_convergence_test_3d

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             source_terms = source_terms)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 1.5)
ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 100)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 200,
                                     solution_variables = cons2prim)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
cfl = 0.5
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
