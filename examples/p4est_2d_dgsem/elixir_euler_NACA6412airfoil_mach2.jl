
using Trixi
using OrdinaryDiffEq

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

@inline function initial_condition_mach2_flow(x, t, equations::CompressibleEulerEquations2D)
    # set the freestream flow parameters
    rho_freestream = 1.4
    v1 = 2.0
    v2 = 0.0
    p_freestream = 1.0

    prim = SVector(rho_freestream, v1, v2, p_freestream)
    return prim2cons(prim, equations)
end

initial_condition = initial_condition_mach2_flow

# Supersonic inflow boundary condition.
# Calculate the boundary flux entirely from the external solution state, i.e., set
# external solution state values for everything entering the domain.
@inline function boundary_condition_supersonic_inflow(u_inner,
                                                      normal_direction::AbstractVector,
                                                      x, t, surface_flux_function,
                                                      equations::CompressibleEulerEquations2D)
    u_boundary = initial_condition_mach2_flow(x, t, equations)
    flux = Trixi.flux(u_boundary, normal_direction, equations)

    return flux
end

# Supersonic outflow boundary condition.
# Calculate the boundary flux entirely from the internal solution state. Analogous to supersonic inflow
# except all the solution state values are set from the internal solution as everything leaves the domain
@inline function boundary_condition_supersonic_outflow(u_inner,
                                                       normal_direction::AbstractVector, x,
                                                       t,
                                                       surface_flux_function,
                                                       equations::CompressibleEulerEquations2D)
    flux = Trixi.flux(u_inner, normal_direction, equations)

    return flux
end

polydeg = 3

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha

basis = LobattoLegendreBasis(polydeg)
shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max = 0.5,
                                            alpha_min = 0.001,
                                            alpha_smooth = true,
                                            variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

# DG Solver
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

# Mesh generated from the following gmsh geometry input file:
# https://gist.githubusercontent.com/DanielDoehring/5ade6d93629f0d8c23a598812dbee2a9/raw/d2bc904fe92146eae1a36156e7f5c535dc1a80f1/NACA6412.geo
mesh_file = joinpath(@__DIR__, "mesh_NACA6412.inp")
isfile(mesh_file) ||
    Trixi.download("https://gist.githubusercontent.com/DanielDoehring/e2a389f04f1e37b33819b9637e8ee4c3/raw/4bf7607a2ce4432fdb5cb87d5e264949b11bd5d7/mesh_NACA6412.inp",
                   mesh_file)

boundary_symbols = [:PhysicalLine1, :PhysicalLine2, :PhysicalLine3, :PhysicalLine4]

mesh = P4estMesh{2}(mesh_file, polydeg = polydeg, boundary_symbols = boundary_symbols)

boundary_conditions = Dict(:PhysicalLine1 => boundary_condition_supersonic_inflow, # Left boundary
                           :PhysicalLine2 => boundary_condition_supersonic_outflow, # Right boundary
                           :PhysicalLine3 => boundary_condition_supersonic_outflow, # Top and bottom boundary
                           :PhysicalLine4 => boundary_condition_slip_wall) # Airfoil

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 4.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        stepsize_callback)

# Run the simulation
###############################################################################
sol = solve(ode, SSPRK104(; thread = OrdinaryDiffEq.True());
            dt = 1.0, # overwritten by the `stepsize_callback`
            save_everystep = false,
            callback = callbacks);

summary_callback() # print the timer summary
