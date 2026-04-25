using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations on a uniform Cartesian grid

save_plot = false
plot_filename = joinpath("out", "energy_spectrum_2d.png")
n_points_per_coordinate = 64

dg = DGMulti(element_type = Quad(),
             approximation_type = periodic_derivative_operator(derivative_order = 1,
                                                               accuracy_order = 4,
                                                               xmin = -pi, xmax = pi,
                                                               N = n_points_per_coordinate),
             surface_flux = flux_hll,
             volume_integral = VolumeIntegralWeakForm())

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

mesh = DGMultiMesh(dg, coordinates_min = (-pi, -pi),
                   coordinates_max = (pi, pi))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval, uEltype = real(dg))

alive_callback = AliveCallback(analysis_interval = analysis_interval)
stepsize_callback = StepsizeCallback(cfl = 1.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 0.5 * estimate_dt(mesh, dg),
            ode_default_options()..., callback = callbacks);

###############################################################################
# compute kinetic energy spectrum from the final numerical solution
#
# This example uses a finite-difference SBP discretization, whose solution values are already
# stored on a uniform Cartesian grid. DGSEM solutions on Legendre-Gauss-Lobatto nodes need to be
# interpolated to a uniform Cartesian grid before calling `compute_energy_spectrum`.

solution_components = StructArrays.components(Base.parent(sol.u[end]))
rho_cartesian_flat, rho_v1_cartesian_flat, rho_v2_cartesian_flat, _ = solution_components

# For this Cartesian example, reshape
# to a 2D grid before passing data to the spectral analysis utility.
n_points = length(rho_cartesian_flat)
n_points_per_dimension = round(Int, sqrt(n_points))
if n_points_per_dimension^2 != n_points
    error("expected a square Cartesian grid, got $n_points nodal points")
end

rho_cartesian = reshape(rho_cartesian_flat, n_points_per_dimension, n_points_per_dimension)
rho_v1_cartesian = reshape(rho_v1_cartesian_flat, n_points_per_dimension,
                           n_points_per_dimension)
rho_v2_cartesian = reshape(rho_v2_cartesian_flat, n_points_per_dimension,
                           n_points_per_dimension)

v1_cartesian = rho_v1_cartesian ./ rho_cartesian
v2_cartesian = rho_v2_cartesian ./ rho_cartesian

energy_spectrum, wavenumbers = compute_energy_spectrum(rho_cartesian,
                                                       v1_cartesian,
                                                       v2_cartesian)

mean_kinetic_energy = sum(@. 0.5 * rho_cartesian * (v1_cartesian^2 + v2_cartesian^2)) /
                      length(rho_cartesian)
if !isapprox(sum(energy_spectrum), mean_kinetic_energy, rtol = sqrt(eps(real(dg))))
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
