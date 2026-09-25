using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

# The off-center blast wave deliberately breaks the symmetry along mortar faces.
# Thus, reversing the nodes on one side of a mortar changes the numerical result.
function initial_condition_weak_blast_wave(x, t, equations::CompressibleEulerEquations2D)
    # Set up polar coordinates
    inicenter = SVector(-0.3, 0.2)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)

    r0 = 0.25
    p0_inner = 2.0
    p0_outer = 1.0

    # Calculate primitive variables
    rho = 1.1
    v1 = 0.0
    v2 = 0.0
    p = r > r0 ? p0_outer : p0_inner

    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_weak_blast_wave

# Get the DG approximation space

# The calculation of the time step with bar states uses `max_abs_speed_naive`. Therefore, use it as the surface flux here.
surface_flux = FluxLaxFriedrichs(max_abs_speed_naive)
volume_flux = flux_ranocha
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
limiter_idp = SubcellLimiterIDP(equations, basis;
                                positivity_variables_cons = ["rho"],
                                positivity_variables_nonlinear = [pressure],
                                local_twosided_variables_cons = ["rho"],
                                local_onesided_variables_nonlinear = [(entropy_guermond_etal,
                                                                       min)],
                                max_iterations_newton = 10,
                                bar_states = true)

volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)
mortar = MortarIDP(equations, basis, limiter_idp)
solver = DGSEM(basis, surface_flux, volume_integral, mortar)

###############################################################################
# unstructured nonconforming mesh containing rotated interfaces

mesh_file = Trixi.download("https://gist.githubusercontent.com/efaulhaber/a075f8ec39a67fa9fad8f6f84342cbca/raw/a7206a02ed3a5d3cadacd8d9694ac154f9151db7/square_unstructured_1.inp",
                           joinpath(@__DIR__, "square_unstructured_1.inp"))

mesh = P4estMesh{2}(mesh_file, polydeg = polydeg, initial_refinement_level = 0)

# Refine the bottom-left child of every tree recursively. This creates mortars
# across tree interfaces, including mortars whose large side is traversed backwards.
function refine_fn(p4est, which_tree, quadrant)
    quadrant_obj = unsafe_load(quadrant)
    if quadrant_obj.x == 0 && quadrant_obj.y == 0 && quadrant_obj.level < 3
        return Cint(1)
    else
        return Cint(0)
    end
end

refine_fn_c = @cfunction(refine_fn, Cint,
                         (Ptr{Trixi.p4est_t}, Ptr{Trixi.p4est_topidx_t},
                          Ptr{Trixi.p4est_quadrant_t}))
Trixi.refine_p4est!(mesh.p4est, true, refine_fn_c, C_NULL)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = (;
                                                           all = BoundaryConditionDirichlet(initial_condition)))

###############################################################################
# ODE solver and callbacks

tspan = (0.0, 0.01)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:conservation_error,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     extra_node_variables = (:limiting_coefficient,))

stepsize_callback = StepsizeCallback(cfl = 0.9, bar_states = true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

stage_callbacks = (SubcellLimiterIDPCorrection(), BoundsCheckCallback())

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
                  callback = callbacks)
