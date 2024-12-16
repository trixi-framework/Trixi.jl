using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

U_inf = 0.2
c_inf = 1.0

rho_inf = 1.4 # with gamma = 1.4 => p_inf = 1.0

Re = 10000.0
airfoil_cord_length = 1.0

t_c = airfoil_cord_length / U_inf

aoa = 4 * pi / 180
u_x = U_inf * cos(aoa)
u_y = U_inf * sin(aoa)

gamma = 1.4
prandtl_number() = 0.72
mu() = rho_inf * U_inf * airfoil_cord_length / Re

equations = CompressibleEulerEquations2D(gamma)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesPrimitive())

@inline function initial_condition_mach02_flow(x, t, equations)
    # set the freestream flow parameters
    rho_freestream = 1.4

    v1 = 0.19951281005196486 # 0.2 * cos(aoa)
    v2 = 0.01395129474882506 # 0.2 * sin(aoa)

    p_freestream = 1.0

    prim = SVector(rho_freestream, v1, v2, p_freestream)
    return prim2cons(prim, equations)
end

initial_condition = initial_condition_mach02_flow

# Boundary conditions for free-stream testing
boundary_condition_free_stream = BoundaryConditionDirichlet(initial_condition)

velocity_bc_airfoil = NoSlip((x, t, equations) -> SVector(0.0, 0.0))
heat_bc = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_airfoil = BoundaryConditionNavierStokesWall(velocity_bc_airfoil, heat_bc)

polydeg = 3

surf_flux = flux_hllc
vol_flux = flux_chandrashekar
solver = DGSEM(polydeg = polydeg, surface_flux = surf_flux,
               volume_integral = VolumeIntegralFluxDifferencing(vol_flux))

###############################################################################
# Get the uncurved mesh from a file (downloads the file if not available locally)

#path = "/storage/home/daniel/PERK4/SD7003/"

path = "/home/daniel/ownCloud - Döhring, Daniel (1MH1D4@rwth-aachen.de)@rwth-aachen.sciebo.de/Job/Doktorand/Content/Meshes/PERK_mesh/SD7003Laminar/"

#mesh_file = path * "sd7003_laminar_straight_sided_Trixi.inp"

mesh_file = path * "sd7003_laminar.inp"

boundary_symbols = [:Airfoil, :FarField]

mesh = P4estMesh{2}(mesh_file, boundary_symbols = boundary_symbols,
                    initial_refinement_level = 0)

boundary_conditions = Dict(:FarField => boundary_condition_free_stream,
                           :Airfoil => boundary_condition_slip_wall)

boundary_conditions_parabolic = Dict(:FarField => boundary_condition_free_stream,
                                     :Airfoil => boundary_condition_airfoil)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.005 * t_c) # Try to get into a state where initial pressure wave is gone
tspan = (0.0, 30 * t_c) # Try to get into a state where initial pressure wave is gone
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

# Choose analysis interval such that roughly every dt_c = 0.005 a record is taken
analysis_interval = 25 # PERK4_Multi, PERKSingle

f_aoa() = aoa
f_rho_inf() = rho_inf
f_U_inf() = U_inf
f_linf() = airfoil_cord_length

drag_coefficient = AnalysisSurfaceIntegral((:Airfoil,),
                                           DragCoefficientPressure(f_aoa(), f_rho_inf(),
                                                                   f_U_inf(), f_linf()))

drag_coefficient_shear_force = AnalysisSurfaceIntegral((:Airfoil,),
                                                       DragCoefficientShearStress(f_aoa(),
                                                                                  f_rho_inf(),
                                                                                  f_U_inf(),
                                                                                  f_linf()))

lift_coefficient = AnalysisSurfaceIntegral((:Airfoil,),
                                           LiftCoefficientPressure(f_aoa(), f_rho_inf(),
                                                                   f_U_inf(), f_linf()))

analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     output_directory = "out",
                                     save_analysis = true,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = (drag_coefficient,
                                                           drag_coefficient_shear_force,
                                                           lift_coefficient))

stepsize_callback = StepsizeCallback(cfl = 2.0) # PERK_4 Multi E = 5, ..., 14

# For plots etc
save_solution = SaveSolutionCallback(interval = 1_000_000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory = "out")

alive_callback = AliveCallback(alive_interval = 10)

save_restart = SaveRestartCallback(interval = Int(10^7), # Only at end
                                   save_final_restart = true)

callbacks = CallbackSet(stepsize_callback, # For measurements: Fixed timestep (do not use this)
                        alive_callback, # Not needed for measurement run
                        save_solution, # For plotting during measurement run
                        #save_restart, # For restart with measurements
                        summary_callback);

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false, thread = OrdinaryDiffEq.True());
            dt = 1.0, save_everystep = false, callback = callbacks)

summary_callback() # print the timer summary
