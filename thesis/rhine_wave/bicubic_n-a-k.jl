# Include packages
using OrdinaryDiffEq
using Trixi
using LinearAlgebra
using Downloads: download
using Plots

# Downlowad data from gist
Rhine_data = download("https://gist.githubusercontent.com/maxbertrand1996/87f57c6d1c3c99e1ace67f52d65a80f4/raw/fd29b9bda097532ce3d704cfe224c7d7f4efeab1/Rhine_data_2D_10.txt")

# Define B-spline structure
spline_struct = bicubic_b_spline(Rhine_data)
# Define B-spline interpolation function
spline_func(x,y) = spline_interpolation(spline_struct, x, y)

equations = ShallowWaterEquations2D(gravity_constant=9.81, H0=70.0)

function initial_condition_wave(x, t, equations::ShallowWaterEquations2D)
  
  x1, x2 = x
  center_x = 357500
  center_y = 5646500
  r = norm([x1,x2] - [center_x, center_y])

  if r < 100
    H = 90.0
  else
    H = equations.H0
  end

  v1 = 0.0
  v2 = 0.0

  b = spline_func(x1, x2)

  return prim2cons(SVector(H, v1, v2, b), equations)
end
  

function boundary_outflow(u_inner, orientation_or_normal, direction, x, t,
  surface_flux_function, equations::ShallowWaterEquations2D)

  g = equations.gravity
  h = u_inner[1]

  c = sqrt(g*h)

  v1, v2 = velocity(u_inner, equations)

  Fr = norm([v1, v2])/c

  if Fr > 1
    # Supercritical outflow
    u_outer = SVector(u_inner[1], u_inner[2], u_inner[3], u_inner[4])
    # calculate the boundary flux
    if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
      flux = surface_flux_function(u_inner, u_outer, orientation_or_normal, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
      flux = surface_flux_function(u_outer, u_inner, orientation_or_normal, equations)
    end

    return flux

  else
    # Subcritical outflow
    # Impulse from inside, height and bottom from outside
    u_outer = SVector(equations.H0, u_inner[2], u_inner[3], u_inner[4])
    # calculate the boundary flux
    if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
      flux = surface_flux_function(u_inner, u_outer, orientation_or_normal, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
      flux = surface_flux_function(u_outer, u_inner, orientation_or_normal, equations)
    end

    return flux
  end
end

initial_condition = initial_condition_wave

boundary_condition = boundary_outflow
###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
surface_flux =(flux_fjordholm_etal, flux_nonconservative_fjordholm_etal)
solver = DGSEM(polydeg=3, surface_flux=surface_flux,
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Get the TreeMesh and setup a periodic mesh

coordinates_min = (spline_struct.x[1], spline_struct.y[1])
coordinates_max = ( spline_struct.x[end], spline_struct.y[end])
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=3,
                n_cells_max=10_000,
                periodicity=false)

# Create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_condition)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 100.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=false,
                                     extra_analysis_integrals=(energy_kinetic,
                                                               energy_internal))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=10,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     output_directory="test_2d")

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution)

# visnodes = range(tspan[1], tspan[2], length=300)
###############################################################################
# run the simulation

# use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(), abstol=1.0e-8, reltol=1.0e-8,
            save_everystep=false, #saveat=visnodes,
            callback=callbacks);

summary_callback() # print the timer summary

# pd = PlotData2D(sol)

# pyplot()
# surface(pd.x[2:end-1], pd.y[2:end-1], pd.data[4][2:end-1, 2:end-1])
# wireframe!(pd.x[2:end-1], pd.y[2:end-1], pd.data[1][2:end-1, 2:end-1])

# display(hcat(pd.mesh_vertices_x[1:16], pd.mesh_vertices_y[1:16]))
# display(hcat(pd.mesh_vertices_x[17:32], pd.mesh_vertices_y[17:32]))
# display(hcat(pd.mesh_vertices_x[33:48], pd.mesh_vertices_y[33:48]))
# display(hcat(pd.mesh_vertices_x[49:64], pd.mesh_vertices_y[49:64]))
# display(hcat(pd.mesh_vertices_x[65:80], pd.mesh_vertices_y[65:80]))
# display(hcat(pd.mesh_vertices_x[81:96], pd.mesh_vertices_y[81:96]))

# display(pd.data)

# display(reshape(sol.u[1][1:4:end] + sol.u[1][4:4:end], (8,8)))

# display(pd.x)
# display(pd.y)
# display(pd.variable_names)
# display(pd.data[4])

# pyplot()
# surface(pd.x[2:end-1], pd.y[2:end-1], pd.data[4][2:end-1, 2:end-1])
# wireframe!(pd.x[2:end-1], pd.y[2:end-1], pd.data[1][2:end-1, 2:end-1])

# test = pd.data[1] #.+ pd.data[4]

# display(test)

# surface(pd.x, pd.y, test)

# H = sol.u[1][1:4:end]
# b = sol.u[1][4:4:end]
# x = collect(1:20)
# y = collect(1:20)
# reshape(H+b, (16,16))
# pyplot()
# surface(reshape(H+b, (20,20)))

# n = m = 40

# x_int = Vector(LinRange(spline_struct.x[1], spline_struct.x[end], n))
# y_int = Vector(LinRange(spline_struct.y[1], spline_struct.y[end], m))

# z_int = zeros(m,n)
# z_int2 = zeros(m,n)
# for i in 1:n, j in 1:m
#   z_int[j,i] = spline_func(x_int[i], y_int[j])
#   z_int2[j,i] = 70
# end

# pyplot()
# surface(x_int, y_int, z_int)
# wireframe!(x_int, y_int, z_int2)