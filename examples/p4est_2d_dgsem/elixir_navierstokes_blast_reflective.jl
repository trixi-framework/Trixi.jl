using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the ideal compressible Navier-Stokes equations

prandtl_number() = 0.72
mu() = 1e-4

equations = CompressibleEulerEquations2D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number())

function initial_condition_weak_blast_wave(x, t, equations::CompressibleEulerEquations2D)
    # Set up polar coordinates
    inicenter = SVector(0.0, 0.0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)

    r0 = 0.2
    E = 1
    p0_inner = 3
    p0_outer = 1

    # Calculate primitive variables
    rho = 1.1
    v1 = 0.0
    v2 = 0.0
    p = r > r0 ? p0_outer : p0_inner

    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_weak_blast_wave

surface_flux = flux_hlle
volume_flux = flux_ranocha
polydeg = 4
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 1.0,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

# Realize reflective walls via slip walls which permit only tangential velocity
boundary_conditions = Dict(:x_neg => boundary_condition_slip_wall,
                           :y_neg => boundary_condition_slip_wall,
                           :y_pos => boundary_condition_slip_wall,
                           :x_pos => boundary_condition_slip_wall)

# The "Slip" boundary condition rotates all velocities into tangential direction
# and thus acts as a reflective wall here.
velocity_bc = Slip()
heat_bc = Adiabatic((x, t, equations_parabolic) -> zero(eltype(x)))
boundary_conditions_visc = BoundaryConditionNavierStokesWall(velocity_bc, heat_bc)

boundary_conditions_parabolic = Dict(:x_neg => boundary_conditions_visc,
                                     :x_pos => boundary_conditions_visc,
                                     :y_neg => boundary_conditions_visc,
                                     :y_pos => boundary_conditions_visc)

###############################################################################

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)

trees_per_dimension = (4, 4)
mesh = P4estMesh(trees_per_dimension,
                 polydeg = 1, initial_refinement_level = 3,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 periodicity = false)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################

tspan = (0.0, 0.7)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 300
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback)

###############################################################################

# 5th-order RKM optimized for compressible Navier-Stokes equations, see also
# https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/#Low-Storage-Methods
ode_alg = CKLLSRK65_4M_4R()

abstol = 1e-6
reltol = 1e-4
sol = solve(ode, ode_alg; abstol = abstol, reltol = reltol,
            ode_default_options()..., callback = callbacks);
