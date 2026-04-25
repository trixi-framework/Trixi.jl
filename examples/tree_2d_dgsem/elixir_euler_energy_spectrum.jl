using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

save_plot = false
plot_filename = joinpath("out", "energy_spectrum_2d_dgsem.png")
nvisnodes = 4

equations = CompressibleEulerEquations2D(1.4)

"""
    initial_condition_taylor_green_vortex(x, t, equations::CompressibleEulerEquations2D)

The classical Taylor-Green vortex in 2D. This setup is used here only as a simple periodic
flow field for demonstrating the PDE-agnostic spectral analysis utilities.
"""
function initial_condition_taylor_green_vortex(x, t,
                                               equations::CompressibleEulerEquations2D)
    RealT = eltype(x)
    A = one(RealT) # magnitude of speed
    Ms = convert(RealT, 0.1) # maximum Mach number

    rho = one(RealT)

    v1 = A * sin(x[1]) * cos(x[2])
    v2 = -A * cos(x[1]) * sin(x[2])
    p = (A / Ms)^2 * rho / equations.gamma
    p = p + convert(RealT, 0.25) * A^2 * rho * (cos(2 * x[1]) + cos(2 * x[2]))

    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_taylor_green_vortex

volume_flux = flux_ranocha
solver = DGSEM(polydeg = 3, surface_flux = flux_hllc,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-1.0, -1.0) .* pi
coordinates_max = (1.0, 1.0) .* pi
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 100_000, periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)
stepsize_callback = StepsizeCallback(cfl = 1.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0,
            ode_default_options()..., callback = callbacks);

###############################################################################
# interpolate to a Cartesian grid and compute kinetic energy spectrum

cartesian = interpolate_to_uniform_cartesian(sol; solution_variables = cons2prim,
                                             nvisnodes = nvisnodes)

rho_cartesian = cartesian["rho"]
v1_cartesian = cartesian["v1"]
v2_cartesian = cartesian["v2"]

energy_spectrum, wavenumbers = compute_energy_spectrum(rho_cartesian,
                                                       v1_cartesian,
                                                       v2_cartesian)

mean_kinetic_energy = sum(@. 0.5 * rho_cartesian * (v1_cartesian^2 + v2_cartesian^2)) /
                      length(rho_cartesian)
if !isapprox(sum(energy_spectrum), mean_kinetic_energy, rtol = sqrt(eps(real(solver))))
    error("energy spectrum does not satisfy Parseval consistency check: " *
          "sum(energy_spectrum) = $(sum(energy_spectrum)), " *
          "mean_kinetic_energy = $mean_kinetic_energy")
end

if save_plot
    @eval using Plots: plot, savefig

    plot_indices = (wavenumbers .> 0) .& (energy_spectrum .> 0)
    energy_spectrum_plot = plot(wavenumbers[plot_indices],
                                energy_spectrum[plot_indices];
                                xscale = :log10, yscale = :log10,
                                xlabel = "wavenumber k",
                                ylabel = "E(k)",
                                label = "energy spectrum",
                                linewidth = 2)

    mkpath(dirname(plot_filename))
    savefig(energy_spectrum_plot, plot_filename)
end
