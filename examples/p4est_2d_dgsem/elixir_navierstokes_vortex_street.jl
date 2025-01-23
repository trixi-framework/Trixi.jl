using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the compressible Euler equations

# Fluid parameters
gamma() = 5 / 3
prandtl_number() = 0.72

# Parameters for compressible von-Karman vortex street
Re() = 500
Ma() = 0.5f0
D() = 1 # Diameter of the cylinder as in the mesh file

# Parameters that can be freely chosen
v_in() = 1
p_in() = 1

# Parameters that follow from Reynolds and Mach number + adiabatic index gamma
mu() = v_in() * D() / Re()

c() = v_in() / Ma()
p_over_rho() = c()^2 / gamma()
rho_in() = p_in() / p_over_rho()

# Equations for this configuration
equations = CompressibleEulerEquations2D(gamma())
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesPrimitive())

# Freestream configuration
@inline function initial_condition(x, t, equations::CompressibleEulerEquations2D)
    rho = rho_in()
    v1 = v_in()
    v2 = 0.0
    p = p_in()

    prim = SVector(rho, v1, v2, p)
    return prim2cons(prim, equations)
end

# Mesh which is refined around the cylinder and in the wake region
mesh_file = Trixi.download("https://gist.githubusercontent.com/DanielDoehring/7312faba9a50ef506b13f01716b4ec26/raw/8e68f9006e634905544207ca322bc0a03a9313ad/cylinder_vortex_street.inp",
                           joinpath(@__DIR__, "cylinder_vortex_street.inp"))
mesh = P4estMesh{2}(mesh_file)

bc_freestream = BoundaryConditionDirichlet(initial_condition)

# Boundary names follow from the mesh file
boundary_conditions = Dict(:Bottom => bc_freestream,
                           :Circle => boundary_condition_slip_wall,
                           :Top => bc_freestream,
                           :Right => bc_freestream,
                           :Left => bc_freestream)

# Parabolic boundary conditions                            
velocity_bc_free = NoSlip((x, t, equations) -> SVector(v_in(), 0))
# Use adiabatic also on the boundaries to "copy" temperature from the domain
heat_bc_free = Adiabatic((x, t, equations) -> 0)
bc_freestream_parabolic = BoundaryConditionNavierStokesWall(velocity_bc_free, heat_bc_free)

velocity_bc_cylinder = NoSlip((x, t, equations) -> SVector(0, 0))
heat_bc_cylinder = Adiabatic((x, t, equations) -> 0)
bc_cylinder_parabolic = BoundaryConditionNavierStokesWall(velocity_bc_cylinder,
                                                          heat_bc_cylinder)

boundary_conditions_para = Dict(:Bottom => bc_freestream_parabolic,
                                :Circle => bc_cylinder_parabolic,
                                :Top => bc_freestream_parabolic,
                                :Right => bc_freestream_parabolic,
                                :Left => bc_freestream_parabolic)
# Standard DGSEM sufficient here
solver = DGSEM(polydeg = 3, surface_flux = flux_hll)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver,
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_para))

###############################################################################
# Setup an ODE problem
tspan = (0, 100)
ode = semidiscretize(semi, tspan)

# Callbacks
summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = analysis_interval,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution)

###############################################################################
# run the simulation

time_int_tol = 1e-6
sol = solve(ode,
            # Moderate number of threads (e.g. 4) advisable to speed things up
            RDPK3SpFSAL49(thread = OrdinaryDiffEq.True());
            abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)

summary_callback() # print the timer summary
