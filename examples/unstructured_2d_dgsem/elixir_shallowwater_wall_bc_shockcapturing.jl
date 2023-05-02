
using Downloads: download
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the shallow water equations with a continuous
# bottom topography function

equations = ShallowWaterEquations2D(gravity_constant=9.812, H0=2.0)

function initial_condition_stone_throw(x, t, equations::ShallowWaterEquations2D)
  # Set up polar coordinates
  inicenter = SVector(0.15, 0.15)
  x_norm = x[1] - inicenter[1]
  y_norm = x[2] - inicenter[2]
  r = sqrt(x_norm^2 + y_norm^2)

  # Calculate primitive variables
  H = equations.H0
  v1 = r < 0.6 ? 2.0 : 0.0
  v2 = r < 0.6 ? -2.0 : 0.0
  # bottom topography taken from Pond.control in [HOHQMesh](https://github.com/trixi-framework/HOHQMesh)
  x1, x2 = x
  b = (  1.5 / exp( 0.5 * ((x1 - 1.0)^2 + (x2 - 1.0)^2) )
       + 0.75 / exp( 0.5 * ((x1 + 1.0)^2 + (x2 + 1.0)^2) ) )

  return prim2cons(SVector(H, v1, v2, b), equations)
end

initial_condition = initial_condition_stone_throw

boundary_condition = Dict( :OuterCircle => boundary_condition_slip_wall )

###############################################################################
# Get the DG approximation space

surface_flux=(FluxHydrostaticReconstruction(flux_hll, hydrostatic_reconstruction_audusse_etal),
              flux_nonconservative_audusse_etal)
volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
polydeg = 6
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=Trixi.waterheight)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)

solver = DGSEM(polydeg=polydeg, surface_flux=surface_flux, volume_integral=volume_integral)

###############################################################################
# Get the unstructured quad mesh from a file (downloads the file if not available locally)

default_mesh_file = joinpath(@__DIR__, "mesh_outer_circle.mesh")
isfile(default_mesh_file) || download("https://gist.githubusercontent.com/andrewwinters5000/9beddd9cd00e2a0a15865129eeb24928/raw/be71e67fa48bc4e1e97f5f6cd77c3ed34c6ba9be/mesh_outer_circle.mesh",
                                      default_mesh_file)
mesh_file = default_mesh_file

mesh = UnstructuredMesh2D(mesh_file)

# Create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_condition)

###############################################################################
# ODE solvers, callbacks, etc.

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=false,
                                    extra_analysis_integrals=(energy_kinetic,
                                                              energy_internal))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution)

###############################################################################
# run the simulation

# use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(); abstol=1.0e-8, reltol=1.0e-8,
            ode_default_options()..., callback=callbacks);
summary_callback() # print the timer summary
