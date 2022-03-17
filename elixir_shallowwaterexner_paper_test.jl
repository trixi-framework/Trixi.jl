
using Downloads: download
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the shallow water equations with a discontinuous
# bottom topography function

equations = ShallowWaterExnerEquations2D(gravity_constant=9.8, H0=10.0,
                                         porosity=0.4, Ag=0.001)

function initial_condition_cone_island(x, t, equations::ShallowWaterExnerEquations2D)
  # example taken from Section 4.3 of Benkhaldoun et al. (2010)
  # [DOI: 10.1002/fld.2129](https://doi.org/10.1002/fld.2129)
  x1, x2 = x
  b = 0.0
  if 300 <= x1 <= 500 && 400 <= x2 <= 600
    b += (sin(pi * (x1 - 300) / 200))^2 * (sin(pi * (x2 - 400) / 200))^2
  end

  # Set the background values
  h = equations.H0 - b
  h_v1 = 10.0
  h_v2 = 0.0

  return SVector(h, h_v1, h_v2, b)
end

initial_condition = initial_condition_cone_island

# OBS: just for a first test
function boundary_condition_inflow(u_inner, normal_direction::AbstractVector, x, t,
                                   surface_flux_function, equations::ShallowWaterExnerEquations2D)
  # u_outer = SVector(u_inner[1], 10.0, u_inner[3], u_inner[4])
  # # calculate the boundary flux
  # flux = surface_flux_function(u_inner, u_outer, normal_direction, equations)

  u_boundary = initial_condition_cone_island(x, t, equations)
  flux = Trixi.flux(u_boundary, normal_direction, equations)

  return flux
end

# OBS: just for a first test
function boundary_condition_do_nothing(u_inner, normal_direction::AbstractVector, x, t,
                                   surface_flux_function, equations::ShallowWaterExnerEquations2D)
  # calculate the boundary flux
  #flux = surface_flux_function(u_inner, u_inner, normal_direction, equations)
  flux = Trixi.flux(u_inner, normal_direction, equations)

  return flux
end

 boundary_condition_constant = BoundaryConditionDirichlet(initial_condition)

boundary_condition = Dict( :Bottom => boundary_condition_constant,
                           :Top    => boundary_condition_constant,
                           :Right  => boundary_condition_constant,
                           :Left   => boundary_condition_constant )

# boundary_condition = Dict( :Bottom => boundary_condition_constant,
#                            :Top    => boundary_condition_constant,
#                            :Right  => boundary_condition_constant,
#                            :Left   => boundary_condition_inflow )

# Currently not used but could be useful for other tests
@inline function source_terms_manning_friction(u, x, t, equations::ShallowWaterExnerEquations2D)
  # Manning friction source term

  n = 0.0196 # Manning friction coefficient

  h = Trixi.waterheight(u, equations)
  v_1, v_2 = Trixi.velocity(u, equations)

  Sf = -equations.gravity * h * n^2 * sqrt(v_1^2 + v_2^2) / h^(4.0/3.0)

  return SVector(0.0, Sf * v_1, Sf * v_2, 0.0)
end

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
solver = DGSEM(polydeg=3, surface_flux=(flux_lax_friedrichs, flux_nonconservative_fjordholm_etal),
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################

mesh_file = joinpath(@__DIR__, "box_mesh.mesh")
#mesh_file = joinpath(@__DIR__, "fine_box_mesh.mesh")

mesh = UnstructuredMesh2D(mesh_file)


# Create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_condition)#,
#                                    source_terms=source_terms_manning_friction)

###############################################################################
# ODE solver

# final time is for 100 hrs. Can be shortened if Ag is taken larger such that the bottom
# moves faster with the water flow
tspan = (0.0, 360000.0)
#ode = semidiscretize(semi, tspan, restart_filename);
ode = semidiscretize(semi, tspan);

###############################################################################
# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=500,
                                     save_initial_solution=true,
                                     save_final_solution=true)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution)

###############################################################################
# run the simulation

# use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL35(), abstol=1.0e-7, reltol=1.0e-7,
            save_everystep=false, callback=callbacks,
            maxiters=999999);
summary_callback() # print the timer summary
