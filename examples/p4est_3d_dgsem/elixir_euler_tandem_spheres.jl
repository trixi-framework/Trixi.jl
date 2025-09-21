using Trixi
using OrdinaryDiffEqLowStorageRK

###############################################################################
# semidiscretization of the compressible Euler equations

gamma = 1.4
equations = CompressibleEulerEquations3D(gamma)

D = 1 # Sphere diameter, follows from mesh
U() = 0.1
rho_ref() = 1.4

@inline function initial_condition(x, t, equations)
    # set the freestream flow parameters
    rho_freestream = rho_ref()

    # v_total = 0.1 = Mach (for c = 1)
    v1 = U()
    v2 = 0.0
    v3 = 0.0

    p_freestream = 1.0

    prim = SVector(rho_freestream, v1, v2, v3, p_freestream)
    return prim2cons(prim, equations)
end

bc_farfield = BoundaryConditionDirichlet(initial_condition)

polydeg = 2
surface_flux = flux_hll

# Flux Differencing required to make this example run
volume_flux = flux_ranocha
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

case_path = "/home/daniel/Sciebo/Job/Doktorand/Content/Meshes/HighOrderCFDWorkshop/CS1/"
mesh_file = case_path * "Pointwise/TandemSpheresHexMesh1P2_fixed.inp"

# Boundary symbols follow from nodesets in the mesh file
boundary_symbols = [:FrontSphere, :BackSphere, :FarField]
mesh = P4estMesh{3}(mesh_file; boundary_symbols = boundary_symbols)

boundary_conditions = Dict(:FrontSphere => boundary_condition_slip_wall,
                           :BackSphere => boundary_condition_slip_wall,
                           :FarField => bc_farfield)

semi = SemidiscretizationHyperbolic(mesh, equations,
                                    initial_condition, solver;
                                    boundary_conditions = boundary_conditions)

t_star_end = 1.0 # 100 recommended in testcase description
t_end = t_star_end * D / U() # convert `t_star` to unit-equipped time 
tspan = (0.0, t_end)
ode = semidiscretize(semi, tspan)

###############################################################################

summary_callback = SummaryCallback()

analysis_interval = 500
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(alive_interval = 50)

save_sol_interval = 500
save_solution = SaveSolutionCallback(interval = save_sol_interval,
                                     save_initial_solution = false)

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        analysis_callback,
                        save_solution)

###############################################################################

sol = solve(ode, RDPK3SpFSAL35(thread = Trixi.True());
            abstol = 1.0e-5, reltol = 1.0e-5,
            ode_default_options()..., callback = callbacks);
