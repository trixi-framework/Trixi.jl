using OrdinaryDiffEqSSPRK
using Trixi

# CRM = Common Research Model
# https://doi.org/10.2514/6.2008-6919
###############################################################################

gamma = 1.4
prandtl_number() = 0.72

# Follows problem C3.5 of the 2015 Third International Workshop on High-Order CFD Methods
# https://www1.grc.nasa.gov/research-and-engineering/hiocfd/

#Re = 5 * 10^6 # C3.5 testcase
Re = 200 * 10^6 # Increase Reynolds number to 200 million (otherwise the simulation crashes immediately due to the very coarse mesh)

chord = 7.005 # m = 275.80 inches

c = 343.0 # m/s = 13504 inches/s
rho() = 1.293 # kg/m^3 = 2.1199e-5 kg/inches^3

p() = c^2 * rho() / gamma
M = 0.85
U() = M * c

mu() = rho() * chord * U() / Re

equations = CompressibleEulerEquations3D(gamma)
equations_parabolic = CompressibleNavierStokesDiffusion3D(equations, mu = mu(),
                                                          Prandtl = prandtl_number())

@inline function initial_condition(x, t, equations)
    # set the freestream flow parameters
    rho_freestream = 1.293

    v1 = 291.55 # = M * c
    v2 = 0.0
    v3 = 0.0

    p_freestream = 108657.255 # = c^2 * rho() / gamma

    prim = SVector(rho_freestream, v1, v2, v3, p_freestream)
    return prim2cons(prim, equations)
end

bc_farfield = BoundaryConditionDirichlet(initial_condition)

polydeg = 2
basis = LobattoLegendreBasis(polydeg)

shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max = 0.5,
                                            alpha_min = 0.001,
                                            alpha_smooth = true,
                                            variable = pressure)

surface_flux = flux_hll
volume_flux = flux_ranocha

volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

# This is an extremely coarse mesh with only ~79k cells, thus way too coarse for real simulations.
# The mesh is further truncated to linear elements from third-order elements.
mesh_file = Trixi.download("https://gist.githubusercontent.com/DanielDoehring/fbc9d785909263ffec76983c4d520fe3/raw/68741ba6c6965b2045af04323bf73df9dab6ed6d/CRM_HIOCFD_2015_meters.inp",
                           joinpath(@__DIR__, "CRM_HIOCFD_2015_meters.inp"))

boundary_symbols = [:SYMMETRY,
    :FARFIELD, :OUTFLOW,
    :WING, :FUSELAGE, :WING_UP, :WING_LO]

mesh = P4estMesh{3}(mesh_file, polydeg = polydeg, boundary_symbols = boundary_symbols)

boundary_conditions_hyp = Dict(:SYMMETRY => boundary_condition_slip_wall, # slip wall allows for tangential velocity => Sufficient for symmetry
                               :FARFIELD => bc_farfield,
                               :OUTFLOW => bc_farfield, # We also use farfield for "outflow" boundary
                               :WING => boundary_condition_slip_wall,
                               :FUSELAGE => boundary_condition_slip_wall,
                               :WING_UP => boundary_condition_slip_wall,
                               :WING_LO => boundary_condition_slip_wall)

velocity_bc_airfoil = NoSlip((x, t, equations) -> SVector(0.0, 0.0, 0.0))
heat_bc = Adiabatic((x, t, equations) -> 0.0)
bc_body = BoundaryConditionNavierStokesWall(velocity_bc_airfoil, heat_bc)

# The "SlipWall" boundary condition rotates all velocities into tangential direction
# and thus acts as a symmetry plane.
bc_symmetry_plane = BoundaryConditionNavierStokesWall(SlipWall(), heat_bc)

boundary_conditions_para = Dict(:SYMMETRY => bc_symmetry_plane,
                                :FARFIELD => bc_farfield,
                                :OUTFLOW => bc_farfield, # We also use farfield for "outflow" boundary
                                :WING => bc_body,
                                :FUSELAGE => bc_body,
                                :WING_UP => bc_body,
                                :WING_LO => bc_body)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions_hyp,
                                                                    boundary_conditions_para))

#tspan = (0.0, 1.5e-5)
tspan = (0.0, 1e-10)
ode = semidiscretize(semi, tspan)

###############################################################################

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = ())

alive_callback = AliveCallback(alive_interval = 10)

save_sol_interval = 5000
save_solution = SaveSolutionCallback(interval = save_sol_interval,
                                     save_initial_solution = false,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory = "out")

save_restart = SaveRestartCallback(interval = save_sol_interval,
                                   save_final_restart = true,
                                   output_directory = "out")

cfl = 0.4
stepsize_callback = StepsizeCallback(cfl = cfl, interval = 5)

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        analysis_callback,
                        save_solution,
                        save_restart,
                        stepsize_callback)

###############################################################################

sol = solve(ode, SSPRK33(), dt = 42.0,
            save_everystep = false, callback = callbacks);
