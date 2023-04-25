
using Downloads: download
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the shallow water equations

equations = ShallowWaterEquations2D(gravity_constant=9.81, H0=1.875,
                                    threshold_limiter=1e-12, threshold_wet=1e-14)


"""
    initial_condition_three_mounds(x, t, equations::ShallowWaterEquations2D)

Initial condition simulating a dam break. The bottom topography is given by one large and two smaller
mounds.
The mounds are flooded by the water for t > 0. To smooth the discontinuity, a logistic function
is applied.

The initial conditions are based on section 6.3 from the paper:
  - Niklas Wintermeyer, Andrew R. Winters, Gregor J. Gassner and Timothy Warburton (2018)
    An entropy stable discontinuous Galerkin method for the shallow water equations on 
    curvilinear meshes with wet/dry fronts accelerated by GPUs\n
    [DOI: 10.1016/j.jcp.2018.08.038](https://doi.org/10.1016/j.jcp.2018.08.038)
"""
function initial_condition_three_mounds(x, t, equations::ShallowWaterEquations2D)
  
  # Set the background values
  v1 = 0.0
  v2 = 0.0
  
  x1, x2 = x
  M_1 = 1 - 0.1 * sqrt( (x1 - 30.0)^2 + (x2 - 22.5)^2 )
  M_2 = 1 - 0.1 * sqrt( (x1 - 30.0)^2 + (x2 - 7.5)^2 )
  M_3 = 2.8 - 0.28 * sqrt( (x1 - 47.5)^2 + (x2 - 15.0)^2 )
  
  b = max(0.0, M_1, M_2, M_3)
  
  # use a logistic function to tranfer water height value smoothly
  L  = equations.H0    # maximum of function
  x0 = 8  # center point of function
  k  = -75.0 # sharpness of transfer
  
  H = max(b, L / (1.0 + exp(-k * (x1 - x0))))

  H = max(H, b + equations.threshold_limiter)
  return prim2cons(SVector(H, v1, v2, b), equations)
end

initial_condition = initial_condition_three_mounds

function boundary_condition_outflow(u_inner, normal_direction::AbstractVector, x, t,
                                    surface_flux_function, equations::ShallowWaterEquations2D)
  # Impulse and bottom from inside, height from external state
  u_outer = SVector(equations.threshold_wet, u_inner[2], u_inner[3], u_inner[4])

  # calculate the boundary flux
  flux = surface_flux_function(u_inner, u_outer, normal_direction, equations)

  return flux
end

dirichlet_bc = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = Dict( :Bottom => boundary_condition_slip_wall,
                            :Top    => boundary_condition_slip_wall,
                            :Right  => boundary_condition_outflow,
                            :Left   => boundary_condition_slip_wall )

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
surface_flux = (FluxHydrostaticReconstruction(flux_hll_chen_noelle, hydrostatic_reconstruction_chen_noelle),
                flux_nonconservative_chen_noelle)

basis = LobattoLegendreBasis(4)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=waterheight_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

###############################################################################
# Get the unstructured quad mesh from a file (downloads the file if not available locally)

default_meshfile = joinpath(@__DIR__, "mesh_three_mound.mesh")

isfile(default_meshfile) || download("https://gist.githubusercontent.com/svengoldberg/c3c87fecb3fc6e46be7f0d1c7cb35f83/raw/e817ecd9e6c4686581d63c46128f9b6468d396d3/mesh_three_mound.mesh",
                                      default_meshfile)

meshfile = default_meshfile                          

mesh = UnstructuredMesh2D(meshfile)

# Create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions=boundary_conditions)

###############################################################################
# ODE solver

tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan)

###############################################################################
# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_integrals=(lake_at_rest_error,))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution)

###############################################################################
# run the simulation

stage_limiter! = PositivityPreservingLimiterShallowWater(thresholds=(equations.threshold_limiter,),
                                                         variables=(Trixi.waterheight,))                                       

sol = solve(ode, SSPRK43(stage_limiter!),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
