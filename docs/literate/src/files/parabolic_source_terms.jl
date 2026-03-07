#src # Parabolic source terms (advection-diffusion).

# Source terms which depend on the gradient of the solution are available by specifying 
# `source_terms_parabolic`. This demo illustrates parabolic source terms for the 
# advection-diffusion equation. 

using OrdinaryDiffEqLowStorageRK
using Trixi

const a = 0.1
const nu = 0.1
const beta = 0.3

equations = LinearScalarAdvectionEquation1D(a)
equations_parabolic = LaplaceDiffusion1D(nu, equations)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

# ## Gradient-dependent source terms

# For a mixed hyperbolic-parabolic system, one can specify source terms which depend
# on the gradient of the solution. Here, we solve a steady advection-diffusion 
# equation with both solution and gradient-dependent source terms. The exact solution 
# `u(x) = sin(x)` is given by `initial_condition`. 

initial_condition = function (x, t, equations::LinearScalarAdvectionEquation1D)
    return SVector(sin(x[1]))
end

# For standard hyperbolic source terms, we pass in the solution, the spatial coordinate, 
# the current time, and the hyperbolic equations. 

source_terms = function (u, x, t, equations::LinearScalarAdvectionEquation1D)
    f = a * cos(x[1]) + nu * sin(x[1]) - beta * (cos(x[1])^2)
    return SVector(f)
end

# For gradient-dependent source terms, we also pass the solution gradients and the 
# parabolic equations instead of the hyperbolic equations. Note that all parabolic 
# equations have `equations_hyperbolic` as a solution field. 
# 
# For advection-diffusion, the gradients are gradients of the solution `u`. However, 
# for systems such as `CompressibleNavierStokesDiffusion1D`, different gradient 
# variables can be selected through the `gradient_variables` keyword option. The 
# choice of `gradient_variables` will also determine the variables whose gradients 
# are passed into `source_terms_parabolic`.
#
# The `gradients` passed to the `source_terms_parabolic` are a tuple of vectors;
# `gradients[1]` are the gradients in the first coordinate direction,
# `gradients[1][1]` is the gradient of the first (and only in this case) variable
# in the first coordinate direction.

source_terms_parabolic = function (u, gradients, x, t, equations::LaplaceDiffusion1D)
    dudx = gradients[1][1]
    return SVector(beta * dudx^2)
end

# The parabolic source terms can then be supplied to `SemidiscretizationHyperbolicParabolic`
# by setting the optional keyword argument `source_terms_parabolic`.
# The rest of the code is identical to standard hyperbolic-parabolic cases. We create a 
# system of ODEs through `semidiscretize`, define callbacks, and then passing the system 
# to OrdinaryDiffEq.jl. 
# 
# Note that for this problem, since viscosity `nu` is relatively large, we utilize 
# `ViscousFormulationLocalDG` instead of the default `ViscousFormulationBassiRebay1` 
# parabolic solver, since the Bassi-Rebay 1 formulation is not accurate when the 
# diffusivity is large relative to the mesh size. 

mesh = TreeMesh(-Float64(pi), Float64(pi);
                initial_refinement_level = 4,
                n_cells_max = 30_000,
                periodicity = true)

boundary_conditions = boundary_condition_periodic
boundary_conditions_parabolic = boundary_condition_periodic

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             solver_parabolic = ViscousFormulationLocalDG(),
                                             source_terms = source_terms,
                                             source_terms_parabolic = source_terms_parabolic,
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

# Finally, we note that while we solve the ODE system using explicit time-stepping, the maximum 
# stable time-step is $O(h^2)$ due to the dominant parabolic term. We enforce this more stringent
# parabolic CFL condition using a diffusion-aware `StepsizeCallback`. 

cfl_advective = 0.5
cfl_diffusive = 0.05
stepsize_callback = StepsizeCallback(cfl = cfl_advective,
                                     cfl_diffusive = cfl_diffusive)
callbacks = CallbackSet(SummaryCallback(), stepsize_callback)
sol = solve(ode, RDPK3SpFSAL35(); adaptive = false, dt = stepsize_callback(ode),
            ode_default_options()..., callback = callbacks)
