using OrdinaryDiffEqLowStorageRK
using Trixi

# Why: This testcases adds uneven refinement in the taylor green vortex (TGV) to specifically test (MPI) mortar treatment in the parabolic part.

###############################################################################
# Physics

prandtl_number() = 0.72
mu = 6.25e-4  # Re ≈ 1600

equations = CompressibleEulerEquations3D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion3D(equations, mu = mu,
                                                          Prandtl = prandtl_number())

###############################################################################
# Initial condition (Taylor-Green vortex)

function initial_condition_taylor_green_vortex(x, t,
                                               equations::CompressibleEulerEquations3D)
    A = 1.0
    Ms = 0.1

    rho = 1.0
    v1 = A * sin(x[1]) * cos(x[2]) * cos(x[3])
    v2 = -A * cos(x[1]) * sin(x[2]) * cos(x[3])
    v3 = 0.0

    p = (A / Ms)^2 * rho / equations.gamma
    p += 1.0 / 16.0 * A^2 * rho *
         (cos(2x[1]) * cos(2x[3]) +
          2cos(2x[2]) + 2cos(2x[1]) +
          cos(2x[2]) * cos(2x[3]))

    return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end

initial_condition = initial_condition_taylor_green_vortex

###############################################################################
# DG setup

volume_flux = flux_ranocha

solver = DGSEM(polydeg = 3,
               surface_flux = flux_lax_friedrichs,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Mesh

coordinates_min = (-1.0, -1.0, -1.0) .* pi
coordinates_max = (1.0, 1.0, 1.0) .* pi

trees_per_dimension = (2, 2, 2)

mesh = P4estMesh(trees_per_dimension,
                 polydeg = 3,
                 coordinates_min = coordinates_min,
                 coordinates_max = coordinates_max,
                 periodicity = (true, true, true),
                 initial_refinement_level = 0)

###############################################################################
# Semidiscretization

semi = SemidiscretizationHyperbolicParabolic(mesh,
                                             (equations, equations_parabolic),
                                             initial_condition,
                                             solver;
                                             boundary_conditions = (boundary_condition_periodic,
                                                                    boundary_condition_periodic))

###############################################################################
# AMR: positional rule → creates 2 refinement bocks on the diagonal of the cube 
# Used to explicitly include mortars in the parabolic MPI testscases

function indicator_function(u, x, t)
    if ((x[1] < 0) && (x[2] < 0) && (x[3] < 0)) ||
       ((x[1] > 0) && (x[2] > 0) && (x[3] > 0))
        return 1.0   # refine
    else
        return 0.0   # keep coarse
    end
end

amr_indicator = IndicatorNodalFunction(indicator_function, semi)

amr_controller = ControllerThreeLevel(semi, amr_indicator;
                                      base_level = 0,
                                      med_level = 1, med_threshold = 0.5,
                                      max_level = 2, max_threshold = 0.9)

# refine only once to get a static mortar mesh
amr_callback = AMRCallback(semi,
                           amr_controller;
                           interval = typemax(Int),                 # no further AMR
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

###############################################################################
# Callbacks 
analysis_interval = 50
analysis_callback = AnalysisCallback(semi,
                                     interval = analysis_interval,  # effectively disabled for short runs
                                     save_analysis = true,
                                     extra_analysis_integrals = (energy_kinetic,
                                                                 energy_internal,
                                                                 enstrophy))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

callbacks = CallbackSet(analysis_callback,
                        alive_callback,
                        amr_callback)

###############################################################################
# Time integration

tspan = (0.0, 5)

ode = semidiscretize(semi, tspan)

sol = solve(ode,
            RDPK3SpFSAL49();
            abstol = 1e-8,
            reltol = 1e-8,
            ode_default_options()...,
            callback = callbacks)
