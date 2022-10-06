####################################################################################
# This example is equivalent to tree_2d_dgsem/elixir_shallowwater_source_terms.jl, #
# but instead of a function for the bottom topography, this version uses a bicubic #
# b spline interpolation with the not-a-knot boundary condition to approximate     #
# the bottom topography                                                            #
####################################################################################


using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = ShallowWaterEquations2D(gravity_constant=9.81)

# bottom topography function
bottom_topography(x,y) = 2.0 + 0.5 * sin(sqrt(2.0) * pi * x) + 0.5 * sin(sqrt(2.0) * pi * y)

# Setting
range_x        = [0.0, sqrt(2.0)]
range_y        = [0.0, sqrt(2.0)]
num_interp_val = 100
x_val          = Vector(LinRange(range_x[1], range_x[2], num_interp_val))
y_val          = Vector(LinRange(range_y[1], range_y[2], num_interp_val))
z_val          = zeros(num_interp_val, num_interp_val)

for i in 1:num_interp_val, j in 1:num_interp_val
  z_val[j, i] = bottom_topography(x_val[i], y_val[j])
end

# Spline interpolation
spline           = bicubic_b_spline(x_val, y_val, z_val; boundary = "not-a-knot")
spline_func(x,y) = spline_interpolation(spline, x, y )

function initial_condition_convergence_test_spline(x, t, equations::ShallowWaterEquations2D)
    # some constants are chosen such that the function is periodic on the domain [0,sqrt(2)]^2
    c  = 7.0
    omega_x = 2.0 * pi * sqrt(2.0)
    omega_t = 2.0 * pi
  
    x1, x2 = x
  
    H = c + cos(omega_x * x1) * sin(omega_x * x2) * cos(omega_t * t)
    v1 = 0.5
    v2 = 1.5
    b = spline_func(x1, x2)
    return prim2cons(SVector(H, v1, v2, b), equations)
  end

initial_condition = initial_condition_convergence_test_spline # MMS EOC test


###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
solver = DGSEM(polydeg=3, surface_flux=(flux_lax_friedrichs, flux_nonconservative_fjordholm_etal),
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Get the TreeMesh and setup a periodic mesh

coordinates_min = (0.0, 0.0)
coordinates_max = (sqrt(2.0), sqrt(2.0))
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=3,
                n_cells_max=10_000,
                periodicity=true)

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms=source_terms_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 500
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=200,
                                     save_initial_solution=true,
                                     save_final_solution=true)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution)

###############################################################################
# run the simulation

# use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(), abstol=1.0e-8, reltol=1.0e-8,
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
