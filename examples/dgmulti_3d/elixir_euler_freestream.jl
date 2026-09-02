using OrdinaryDiffEqLowStorageRK
using Trixi
using Trixi: StartUpDG

polydeg = 3
basis = DGMultiBasis(Tet(), polydeg, approximation_type = Polynomial())

# Import mesh consisting of tetrahedra
mesh_file = Trixi.download("https://raw.githubusercontent.com/jlchan/StartUpDG.jl/refs/heads/main/test/testset_Gmsh_meshes/cube1.msh",
                           joinpath(@__DIR__, "cube1.msh"))

VXY, EToV = StartUpDG.read_Gmsh_3D(mesh_file)

# tag all boundaries as freestream
function freestream(x)
    (x[1] ≈ minimum(VXY[1])) || (x[1] ≈ maximum(VXY[1])) ||
        (x[2] ≈ minimum(VXY[2])) || (x[2] ≈ maximum(VXY[2])) ||
        (x[3] ≈ minimum(VXY[3])) || (x[3] ≈ maximum(VXY[3]))
end

is_on_boundary = (; freestream = freestream)

equations = CompressibleEulerEquations3D(1.4)
initial_condition = initial_condition_constant

dg = DGMulti(basis,
             surface_integral = SurfaceIntegralWeakForm(flux_hllc),
             volume_integral = VolumeIntegralWeakForm())
mesh = DGMultiMesh(dg, VXY, EToV; is_on_boundary)

boundary_conditions = (; freestream = BoundaryConditionDirichlet(initial_condition))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg,
                                    boundary_conditions = boundary_conditions)

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 50)
analysis_interval = 200
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
save_solution = SaveSolutionCallback(interval = analysis_interval,
                                     solution_variables = cons2prim)
callbacks = CallbackSet(summary_callback, alive_callback,
                        analysis_callback, save_solution)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 0.5 * estimate_dt(mesh, dg), ode_default_options()...,
            callback = callbacks);
