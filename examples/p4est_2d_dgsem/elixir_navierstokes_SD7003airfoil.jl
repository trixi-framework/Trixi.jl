using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

gamma = 1.4

U_inf = 0.2
aoa = 4 * pi / 180
rho_inf = 1.4 # with gamma = 1.4 => p_inf = 1.0

Re = 10000.0
airfoil_cord_length = 1.0

t_c = airfoil_cord_length / U_inf

prandtl_number() = 0.72
mu() = rho_inf * U_inf * airfoil_cord_length / Re

equations = CompressibleEulerEquations2D(gamma)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesPrimitive())

@inline function initial_condition_mach02_flow(x, t, equations)
    # set the freestream flow parameters such that c_inf = 1.0 => Mach 0.2
    rho_freestream = 1.4

    # Values correspond to `aoa = 4 * pi / 180`
    v1 = 0.19951281005196486 # 0.2 * cos(aoa)
    v2 = 0.01395129474882506 # 0.2 * sin(aoa)

    p_freestream = 1.0

    prim = SVector(rho_freestream, v1, v2, p_freestream)
    return prim2cons(prim, equations)
end
initial_condition = initial_condition_mach02_flow

surf_flux = flux_hllc
vol_flux = flux_chandrashekar
solver = DGSEM(polydeg = 3, surface_flux = surf_flux,
               volume_integral = VolumeIntegralFluxDifferencing(vol_flux))

###############################################################################
# Get the uncurved mesh from a file (downloads the file if not available locally)

# Get quadratic meshfile
mesh_file_name = "SD7003_2D_Quadratic.inp"

mesh_file = Trixi.download("https://gist.githubusercontent.com/DanielDoehring/bd2aa20f7e6839848476a0e87ede9f69/raw/1bc8078b4a57634819dc27010f716e68a225c9c6/SD7003_2D_Quadratic.inp",
                           joinpath(@__DIR__, mesh_file_name))

# There is also a linear mesh file available at
# https://gist.githubusercontent.com/DanielDoehring/375df933da8a2081f58588529bed21f0/raw/18592aa90f1c86287b4f742fd405baf55c3cf133/SD7003_2D_Linear.inp

boundary_symbols = [:Airfoil, :FarField]
mesh = P4estMesh{2}(mesh_file, boundary_symbols = boundary_symbols)

boundary_condition_free_stream = BoundaryConditionDirichlet(initial_condition)

velocity_bc_airfoil = NoSlip((x, t, equations) -> SVector(0.0, 0.0))
heat_bc = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_airfoil = BoundaryConditionNavierStokesWall(velocity_bc_airfoil, heat_bc)

boundary_conditions_hyp = Dict(:FarField => boundary_condition_free_stream,
                               :Airfoil => boundary_condition_slip_wall)

boundary_conditions_para = Dict(:FarField => boundary_condition_free_stream,
                                :Airfoil => boundary_condition_airfoil)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions_hyp,
                                                                    boundary_conditions_para))

###############################################################################
# ODE solvers, callbacks etc.

# Run simulation until initial pressure wave is gone.
# Note: This is a very long simulation!
tspan = (0.0, 30 * t_c)

# Drag/Lift coefficient measurements should then be done over the 30 to 35 t_c interval
# by restarting the simulation.

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

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

# For long simulation run, use a large interval.
# For measurements once the simulation has settled in, one should use a
# significantly smaller interval, e.g. 500 to record the drag/lift coefficients.
analysis_interval = 10_000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     output_directory = "out",
                                     save_analysis = true,
                                     analysis_errors = Symbol[], # Turn off standard errors
                                     analysis_integrals = (drag_coefficient,
                                                           drag_coefficient_shear_force,
                                                           lift_coefficient))

stepsize_callback = StepsizeCallback(cfl = 2.2)

alive_callback = AliveCallback(alive_interval = 50)

save_solution = SaveSolutionCallback(interval = analysis_interval,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory = "out")

save_restart = SaveRestartCallback(interval = analysis_interval,
                                   save_final_restart = true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback,
                        save_solution,
                        save_restart);

###############################################################################
# run the simulation

sol = solve(ode,
            CarpenterKennedy2N54(williamson_condition = false,
                                 thread = OrdinaryDiffEq.True());
            dt = 1.0, save_everystep = false, callback = callbacks)

summary_callback() # print the timer summary
