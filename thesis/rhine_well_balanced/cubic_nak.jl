# Include packages
using Trixi
using OrdinaryDiffEq
using Downloads: download

# Downlowad data from gist
Rhine_data = download("https://gist.githubusercontent.com/maxbertrand1996/a30db4dc9f5427c78160321d75a08166/raw/fa53ceb39ac82a6966cbb14e1220656cf7f97c1b/Rhine_data_2D_40.txt")

# Reading the data fom the file
file = open(Rhine_data)
lines = readlines(file)
close(file)

n = parse(Int64, lines[2])
m = parse(Int64, lines[4])

x_file = [parse(Float64, val) for val in lines[6:(5+n)]]
y_file = [parse(Float64, val) for val in lines[(7+n):(6+n+m)]]
z_tmp  = [parse(Float64, val) for val in lines[(8+n+m):end]]

z_file = Matrix(transpose(reshape(z_tmp, (n, m))))

# Setting x and y from the file data to get the values from 
# an arbitrary cross section
x = x_file
y = z_file[22,:]

# Define B-spline structure
spline_struct = cubic_b_spline(x, y; end_condition="not-a-knot")
# Define B-spline interpolation function
spline_func(x) = spline_interpolation(spline_struct, x)

###############################################################################
# semidiscretization of the shallow water equations with a discontinuous
# bottom topography function

equations = ShallowWaterEquations1D(gravity_constant=9.81, H0=60.0)

function initial_condition_well_balancedness(x, t, equations::ShallowWaterEquations1D)
  # Set the background values
  H = equations.H0
  v = 0.0

  b = spline_func(x[1])

  return prim2cons(SVector(H, v, b), equations)
end

initial_condition = initial_condition_well_balancedness

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
surface_flux = (flux_fjordholm_etal, flux_nonconservative_fjordholm_etal)
solver = DGSEM(polydeg=3, surface_flux=surface_flux,
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Get the TreeMesh and setup a periodic mesh

coordinates_min = spline_struct.x[1]
coordinates_max = spline_struct.x[end]
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=3,
                n_cells_max=10_000)

# Create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solver

tspan = (0.0, 10000.0)
ode = semidiscretize(semi, tspan)

###############################################################################
# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_integrals=(lake_at_rest_error,))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1000,
                                     save_initial_solution=true,
                                     save_final_solution=true)

stepsize_callback = StepsizeCallback(cfl=1.0)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary