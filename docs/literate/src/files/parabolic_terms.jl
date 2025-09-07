#src # Parabolic terms (advection-diffusion).

# Experimental support for parabolic diffusion terms is available in Trixi.jl.
# This demo illustrates parabolic terms for the advection-diffusion equation.

using OrdinaryDiffEqLowStorageRK
using Trixi

# ## Splitting a system into hyperbolic and parabolic parts

# For a mixed hyperbolic-parabolic system, we represent the hyperbolic and parabolic
# parts of the system  separately. We first define the hyperbolic (advection) part of
# the advection-diffusion equation.

advection_velocity = (1.5, 1.0)
equations_hyperbolic = LinearScalarAdvectionEquation2D(advection_velocity);

# Next, we define the parabolic diffusion term. The constructor requires knowledge of
# `equations_hyperbolic` to be passed in because the [`LaplaceDiffusion2D`](@ref) applies
# diffusion to every variable of the hyperbolic system.

diffusivity = 5.0e-2
equations_parabolic = LaplaceDiffusion2D(diffusivity, equations_hyperbolic);

# ## Boundary conditions

# As with the equations, we define boundary conditions separately for the hyperbolic and
# parabolic part of the system. For this example, we impose inflow BCs for the hyperbolic
# system (no condition is imposed on the outflow), and we impose Dirichlet boundary conditions
# for the parabolic equations. Both `BoundaryConditionDirichlet` and `BoundaryConditionNeumann`
# are defined for `LaplaceDiffusion2D`.
#
# The hyperbolic and parabolic boundary conditions are assumed to be consistent with each other.

boundary_condition_zero_dirichlet = BoundaryConditionDirichlet((x, t, equations) -> SVector(0.0))

boundary_conditions_hyperbolic = (;
                                  x_neg = BoundaryConditionDirichlet((x, t, equations) -> SVector(1 +
                                                                                                  0.5 *
                                                                                                  x[2])),
                                  y_neg = boundary_condition_zero_dirichlet,
                                  y_pos = boundary_condition_do_nothing,
                                  x_pos = boundary_condition_do_nothing)

boundary_conditions_parabolic = (;
                                 x_neg = BoundaryConditionDirichlet((x, t, equations) -> SVector(1 +
                                                                                                 0.5 *
                                                                                                 x[2])),
                                 y_neg = boundary_condition_zero_dirichlet,
                                 y_pos = boundary_condition_zero_dirichlet,
                                 x_pos = boundary_condition_zero_dirichlet);

# ## Defining the solver and mesh

# The process of creating the DG solver and mesh is the same as for a purely
# hyperbolic system of equations.

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)
coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                periodicity = false, n_cells_max = 30_000) # set maximum capacity of tree data structure

initial_condition = (x, t, equations) -> SVector(0.0);

# ## Semidiscretizing

# To semidiscretize a hyperbolic-parabolic system, we create a [`SemidiscretizationHyperbolicParabolic`](@ref).
# This differs from a [`SemidiscretizationHyperbolic`](@ref) in that we pass in a `Tuple` containing both the
# hyperbolic and parabolic equation, as well as a `Tuple` containing the hyperbolic and parabolic
# boundary conditions.

semi = SemidiscretizationHyperbolicParabolic(mesh,
                                             (equations_hyperbolic, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions_hyperbolic,
                                                                    boundary_conditions_parabolic))

# The rest of the code is identical to the hyperbolic case. We create a system of ODEs through
# `semidiscretize`, defining callbacks, and then passing the system to OrdinaryDiffEq.jl.

tspan = (0.0, 1.5)
ode = semidiscretize(semi, tspan)
callbacks = CallbackSet(SummaryCallback())

# ## Time Integration

# ### Adaptive time integration

# We can run the simulation with an adaptive, i.e., automatic stepsize-selecting algorithm via

time_int_tol = 1.0e-6
sol = solve(ode, RDPK3SpFSAL49(); abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks);

# We can now visualize the solution, which develops a boundary layer at the outflow boundaries.

using Plots
plot(sol)

# ### CFL-based time integration

# If CFL-based time stepping is desired, it may be required to check for diffusive timestep restrictions.
# These become relevant if the mesh Péclet number ``\mathrm{Pe} = \frac{a \Delta x}{d (\Delta x)^2} = \frac{a}{d \Delta x}``
# becomes small.
# This is the case for overall diffusion dominated flows ``d \gg a`` or well-resolved boundary layers,
# where ``\Delta x`` becomes small.
#
# The `StepsizeCallback` selects the timestep according to the most restrictive CFL condition.
# In particular, if `cfl_diffusive` with a non-zero timestep is supplied,
# both the advective ``\delta t_d \sim \frac{\Delta x}{a}`` and the diffusive ``\Delta t_d \sim \frac{(\Delta x)^2}{d}``
# maximum timesteps are computed and the minimum is selected.
stepsize_callback = StepsizeCallback(cfl = 1.5,
                                     cfl_diffusive = 0.3)

# Supply `StepsizeCallback` and solve again.
callbacks = CallbackSet(SummaryCallback(), stepsize_callback)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

plot(sol)

# ## Package versions

# These results were obtained using the following versions.

using InteractiveUtils
versioninfo()

using Pkg
Pkg.status(["Trixi", "OrdinaryDiffEqLowStorageRK", "Plots"],
           mode = PKGMODE_MANIFEST)
