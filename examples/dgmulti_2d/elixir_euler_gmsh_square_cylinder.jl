using OrdinaryDiffEqSSPRK
using Trixi
using Trixi: StartUpDG

polydeg = 3
basis = DGMultiBasis(Tri(), polydeg, approximation_type = SBP())

mesh_file = Trixi.download("https://raw.githubusercontent.com/jlchan/StartUpDG.jl/041d270ac97f8a650fb3e7a32b35cc9314e0777f/test/testset_Gmsh_meshes/squareCylinder2D.msh",
                           joinpath(@__DIR__, "squareCylinder2D.msh"))
VXY, EToV = StartUpDG.read_Gmsh_2D(mesh_file)

# tag different boundary conditions
function freestream(x)
    (x[1] ≈ minimum(VXY[1])) || (x[2] ≈ minimum(VXY[2])) || (x[2] ≈ maximum(VXY[2]))
end
outflow(x) = (x[1] ≈ maximum(VXY[1]))
cylinder(x) = !freestream(x) && !outflow(x)
is_on_boundary = (; freestream = freestream, outflow = outflow, wall = cylinder)

equations = CompressibleEulerEquations2D(1.4)
@inline function initial_condition_mach2_flow(x, t, equations::CompressibleEulerEquations2D)
    # set the freestream flow parameters
    rho_freestream = 1.4
    v1 = 2.0
    v2 = 0.0
    p_freestream = 1.0

    prim = SVector(rho_freestream, v1, v2, p_freestream)
    return prim2cons(prim, equations)
end
initial_condition = initial_condition_mach2_flow

volume_flux = flux_ranocha
surface_flux = flux_lax_friedrichs
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

dg = DGMulti(basis,
             surface_integral = SurfaceIntegralWeakForm(surface_flux),
             volume_integral = volume_integral)
mesh = DGMultiMesh(dg, VXY, EToV; is_on_boundary)

boundary_conditions = (; freestream = BoundaryConditionDirichlet(initial_condition),
                       outflow = BoundaryConditionDirichlet(initial_condition),
                       wall = boundary_condition_slip_wall)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg,
                                    boundary_conditions = boundary_conditions)

tspan = (0.0, 25.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 50)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval, uEltype = real(dg))
save_solution = SaveSolutionCallback(interval = analysis_interval,
                                     solution_variables = cons2prim)

###############################################################################
# run the simulation

solver = SSPRK43()
callbacks = CallbackSet(summary_callback, alive_callback,
                        analysis_callback, save_solution)

sol = solve(ode, solver;
            dt = 1e-6, abstol = 1e-5, reltol = 1e-3,
            ode_default_options()...,
            callback = callbacks);
