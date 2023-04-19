
using Downloads: download
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

"""
    initial_condition_double_mach_reflection(x, t, equations::CompressibleEulerEquations2D)

Compressible Euler setup for a double Mach reflection problem.
Involves strong shock interactions as well as steady / unsteady flow structures.
Also exercises special boundary conditions along the bottom of the domain that is a mixture of
Dirichlet and slip wall.
See Section IV c on the paper below for details.

- Paul Woodward and Phillip Colella (1984)
  The Numerical Simulation of Two-Dimensional Fluid Flows with Strong Shocks.
  [DOI: 10.1016/0021-9991(84)90142-6](https://doi.org/10.1016/0021-9991(84)90142-6)
"""
@inline function initial_condition_double_mach_reflection(x, t, equations::CompressibleEulerEquations2D)

  if x[1] < 1 / 6 + (x[2] + 20 * t) / sqrt(3)
    phi = pi / 6
    sin_phi, cos_phi = sincos(phi)

    rho =  8
    v1  =  8.25 * cos_phi
    v2  = -8.25 * sin_phi
    p   =  116.5
  else
    rho = 1.4
    v1  = 0
    v2  = 0
    p   = 1
  end

  prim = SVector(rho, v1, v2, p)
  return prim2cons(prim, equations)
end

initial_condition = initial_condition_double_mach_reflection


boundary_condition_inflow = BoundaryConditionDirichlet(initial_condition_double_mach_reflection)

# Supersonic outflow boundary condition. Solution is taken entirely from the internal state.
# See `examples/p4est_2d_dgsem/elixir_euler_forward_step_amr.jl` for complete documentation.
@inline function boundary_condition_outflow(u_inner, normal_direction::AbstractVector, x, t,
                                            surface_flux_function, equations::CompressibleEulerEquations2D)
  # NOTE: Only for the supersonic outflow is this strategy valid
  # Calculate the boundary flux entirely from the internal solution state
  return flux(u_inner, normal_direction, equations)
end

# Special mixed boundary condition type for the :Bottom of the domain.
# It is Dirichlet when x < 1/6 and a slip wall when x >= 1/6
@inline function boundary_condition_mixed_dirichlet_wall(u_inner, normal_direction::AbstractVector,
                                                         x, t, surface_flux_function,
                                                         equations::CompressibleEulerEquations2D)
  if x[1] < 1 / 6
    # From the BoundaryConditionDirichlet
    # get the external value of the solution
    u_boundary = initial_condition_double_mach_reflection(x, t, equations)
    # Calculate boundary flux
    flux = surface_flux_function(u_inner, u_boundary, normal_direction, equations)
  else # x[1] >= 1 / 6
    # Use the free slip wall BC otherwise
    flux = boundary_condition_slip_wall(u_inner, normal_direction, x, t, surface_flux_function, equations)
  end

  return flux
end

boundary_conditions = Dict( :Bottom => boundary_condition_mixed_dirichlet_wall,
                            :Top    => boundary_condition_inflow,
                            :Right  => boundary_condition_outflow,
                            :Left   => boundary_condition_inflow   )

volume_flux = flux_ranocha
surface_flux = flux_lax_friedrichs

polydeg = 4
basis = LobattoLegendreBasis(polydeg)
shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max=0.5,
                                            alpha_min=0.001,
                                            alpha_smooth=true,
                                            variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
solver = DGSEM(polydeg=polydeg, surface_flux=surface_flux, volume_integral=volume_integral)

# Get the unstructured quad mesh from a file (downloads the file if not available locally)
default_mesh_file = joinpath(@__DIR__, "abaqus_double_mach.inp")
isfile(default_mesh_file) || download("https://gist.githubusercontent.com/andrewwinters5000/a0806ef0d03cf5ea221af523167b6e32/raw/61ed0eb017eb432d996ed119a52fb041fe363e8c/abaqus_double_mach.inp",
                                      default_mesh_file)
mesh_file = default_mesh_file

mesh = P4estMesh{2}(mesh_file)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.2)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_integrals=(entropy,))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

amr_indicator = IndicatorLöhner(semi, variable=Trixi.density)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=0,
                                      med_level=3, med_threshold=0.05,
                                      max_level=6, max_threshold=0.1)

amr_callback = AMRCallback(semi, amr_controller,
                           interval=1,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        amr_callback)

# positivity limiter necessary for this example with strong shocks
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds=(5.0e-6, 5.0e-6),
                                                     variables=(Trixi.density, pressure))

###############################################################################
# run the simulation
sol = solve(ode, SSPRK43(stage_limiter!);
            ode_default_options()..., callback=callbacks);
summary_callback() # print the timer summary
