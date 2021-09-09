#src # Differentiable programming

# [Julia and its ecosystem provide some tools for differentiable programming](https://sinews.siam.org/Details-Page/scientific-machine-learning-how-julia-employs-differentiable-programming-to-do-it-best).
# Trixi.jl is designed to be flexible, extendable, and composable with Julia's growing ecosystem for
# scientific computing and machine learning. Thus, the ultimate goal is to have fast implementations
# that allow automatic differentiation (AD) without too much hassle for users. If some parts do not
# meet these requirements, please feel free to open an issue or propose a fix in a PR.

# In the following, we will walk through some examples demonstrating how to differentiate through
# Trixi.jl.


# ## Forward mode automatic differentiation

# Trixi integrates well with [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
# for forward mode AD.


# ### Computing the Jacobian
# The high-level interface to compute the Jacobian this way is [`jacobian_ad_forward`](@trixi-docs:reference-trixi/#Trixi.jacobian_ad_forward-Tuple{Trixi.AbstractSemidiscretization}).

#nb using Trixi, LinearAlgebra, Plots
#nb #-
#nb equations = CompressibleEulerEquations2D(1.4);

#nb solver = DGSEM(3, flux_central);

#nb mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), initial_refinement_level=2, n_cells_max=10^5);

#nb semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_density_wave, solver);

#nb J = jacobian_ad_forward(semi);

#nb size(J)
#nb #-
#nb λ = eigvals(J);

#nb scatter(real.(λ), imag.(λ));

#nb 3.0e-10 < maximum(real, λ) / maximum(abs, λ) < 8.0e-10
#nb #-
#nb 1.0e-7 < maximum(real, λ) < 5.0e-7

#md # ```jldoctest euler_eigenvalues
#md # julia> using Trixi, LinearAlgebra, Plots

#md # julia> equations = CompressibleEulerEquations2D(1.4);

#md # julia> solver = DGSEM(3, flux_central);

#md # julia> mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), initial_refinement_level=2, n_cells_max=10^5);

#md # julia> semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_density_wave, solver);

#md # julia> J = jacobian_ad_forward(semi);

#md # julia> size(J)
#md # (1024, 1024)

#md # julia> λ = eigvals(J);

#md # julia> scatter(real.(λ), imag.(λ));

#md # julia> 3.0e-10 < maximum(real, λ) / maximum(abs, λ) < 8.0e-10
#md # true

#md # julia> 1.0e-7 < maximum(real, λ) < 5.0e-7
#md # true
#md # ```

# Interestingly, if we add dissipation by switching to the `flux_lax_friedrichs` at the interfaces,
# the maximal real part of the eigenvalues increases.

#nb solver = DGSEM(3, flux_lax_friedrichs);

#nb semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_density_wave, solver);

#nb J = jacobian_ad_forward(semi);

#nb λ = eigvals(J);

#nb scatter!(real.(λ), imag.(λ));

#nb λ = eigvals(J); round(maximum(real, λ) / maximum(abs, λ), sigdigits=2)
#nb #-
#nb round(maximum(real, λ), sigdigits=2)

#md # ```jldoctest euler_eigenvalues
#md # julia> solver = DGSEM(3, flux_lax_friedrichs);

#md # julia> semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_density_wave, solver);

#md # julia> J = jacobian_ad_forward(semi);

#md # julia> λ = eigvals(J);

#md # julia> scatter!(real.(λ), imag.(λ));

#md # julia> λ = eigvals(J); round(maximum(real, λ) / maximum(abs, λ), sigdigits=2)
#md # 2.1e-5

#md # julia> round(maximum(real, λ), sigdigits=2)
#md # 0.0057
#md # ```

# However, we should be careful when using this analysis, since the eigenvectors are not necessarily
# well-conditioned.

#nb λ, V = eigen(J);

#nb round(cond(V), sigdigits=2)

#md # ```jldoctest euler_eigenvalues
#md # julia> λ, V = eigen(J);

#md # julia> round(cond(V), sigdigits=2)
#md # 1.8e6
#md # ```

# In one space dimension, the situation is a bit different.

#nb equations = CompressibleEulerEquations1D(1.4);

#nb solver = DGSEM(3, flux_central);

#nb mesh = TreeMesh((-1.0,), (1.0,), initial_refinement_level=6, n_cells_max=10^5);

#nb semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_density_wave, solver);

#nb J = jacobian_ad_forward(semi);

#nb λ = eigvals(J);

#nb scatter(real.(λ), imag.(λ));

#nb 1.0e-17 < maximum(real, λ) / maximum(abs, λ) < 1.0e-15
#nb #-
#nb 1.0e-13 < maximum(real, λ) < 1.0e-11
#nb #-
#nb λ, V = eigen(J);

#nb 200 < cond(V) < 300

#md # ```jldoctest euler_eigenvalues
#md # julia> equations = CompressibleEulerEquations1D(1.4);

#md # julia> solver = DGSEM(3, flux_central);

#md # julia> mesh = TreeMesh((-1.0,), (1.0,), initial_refinement_level=6, n_cells_max=10^5);

#md # julia> semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_density_wave, solver);

#md # julia> J = jacobian_ad_forward(semi);

#md # julia> λ = eigvals(J);

#md # julia> scatter(real.(λ), imag.(λ));

#md # julia> 1.0e-17 < maximum(real, λ) / maximum(abs, λ) < 1.0e-15
#md # true

#md # julia> 1.0e-13 < maximum(real, λ) < 1.0e-11
#md # true

#md # julia> λ, V = eigen(J);

#md # julia> 200 < cond(V) < 300
#md # true
#md # ```

# If we add dissipation, the maximal real part is still approximately zero.

#nb solver = DGSEM(3, flux_lax_friedrichs);

#nb semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_density_wave, solver);

#nb J = jacobian_ad_forward(semi);

#nb λ = eigvals(J);

#nb scatter!(real.(λ), imag.(λ));

#nb λ = eigvals(J);

#nb -1.0e-15 < maximum(real, λ) / maximum(abs, λ) < 1.0e-15
#nb #-
#nb -9.0e-13 < maximum(real, λ) < 9.0e-13
#nb #-
#nb λ, V = eigen(J);

#nb 70_000 < cond(V) < 200_000

#md # ```jldoctest euler_eigenvalues
#md # julia> solver = DGSEM(3, flux_lax_friedrichs);

#md # julia> semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_density_wave, solver);

#md # julia> J = jacobian_ad_forward(semi);

#md # julia> λ = eigvals(J);

#md # julia> scatter!(real.(λ), imag.(λ));

#md # julia> λ = eigvals(J);

#md # julia> -1.0e-15 < maximum(real, λ) / maximum(abs, λ) < 1.0e-15
#md # true

#md # julia> -9.0e-13 < maximum(real, λ) < 9.0e-13
#md # true

#md # julia> λ, V = eigen(J);

#md # julia> 70_000 < cond(V) < 200_000
#md # true
#md # ```

# Note that the condition number of the eigenvector matrix increases but is still smaller than for the
# example in 2D.


# ### Computing other derivatives

# It is also possible to compute derivatives of other dependencies using AD in Trixi. For example,
# you can compute the gradient of an entropy-dissipative semidiscretization with respect to the
# ideal gas constant of the compressible Euler equations as described in the following. This example
# is also available as the elixir
# [examples/special\_elixirs/elixir\_euler\_ad.jl](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/special_elixirs/elixir_euler_ad.jl)

#nb using Trixi, LinearAlgebra, ForwardDiff
#nb #-
#nb equations = CompressibleEulerEquations2D(1.4);

#nb mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), initial_refinement_level=2, n_cells_max=10^5);

#nb solver = DGSEM(3, flux_lax_friedrichs, VolumeIntegralFluxDifferencing(flux_ranocha));

#nb semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_isentropic_vortex, solver);
#nb #-
#nb u0_ode = compute_coefficients(0.0, semi);

#nb size(u0_ode)
#nb #-
#nb J = ForwardDiff.jacobian((du_ode, γ) -> begin
#nb     equations_inner = CompressibleEulerEquations2D(first(γ))
#nb     semi_inner = Trixi.remake(semi, equations=equations_inner, uEltype=eltype(γ));
#nb     Trixi.rhs!(du_ode, u0_ode, semi_inner, 0.0)
#nb end, similar(u0_ode), [1.4]); # γ needs to be an `AbstractArray`

#nb round.(extrema(J), sigdigits=2)

#md # ```jldoctest euler_gamma_gradient
#md # julia> using Trixi, LinearAlgebra, ForwardDiff

#md # julia> equations = CompressibleEulerEquations2D(1.4);

#md # julia> mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), initial_refinement_level=2, n_cells_max=10^5);

#md # julia> solver = DGSEM(3, flux_lax_friedrichs, VolumeIntegralFluxDifferencing(flux_ranocha));

#md # julia> semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_isentropic_vortex, solver);

#md # julia> u0_ode = compute_coefficients(0.0, semi); size(u0_ode)
#md # (1024,)

#md # julia> J = ForwardDiff.jacobian((du_ode, γ) -> begin
#md #            equations_inner = CompressibleEulerEquations2D(first(γ))
#md #            semi_inner = Trixi.remake(semi, equations=equations_inner, uEltype=eltype(γ));
#md #            Trixi.rhs!(du_ode, u0_ode, semi_inner, 0.0)
#md #        end, similar(u0_ode), [1.4]); # γ needs to be an `AbstractArray`

#md # julia> round.(extrema(J), sigdigits=2)
#md # (-5.6, 5.6)
#md # ```

# Note that we create a semidiscretization `semi` at first to determine the state `u0_ode` around
# which we want to perform the linearization. Next, we wrap the RHS evaluation inside a closure
# and pass that to `ForwardDiff.jacobian`. There, we need to make sure that the internal caches
# are able to store dual numbers from ForwardDiff.jl by setting `uEltype` appropriately. A similar
# approach is used by [`jacobian_ad_forward`](@trixi-docs:reference-trixi/#Trixi.jacobian_ad_forward-Tuple{Trixi.AbstractSemidiscretization}).

# Note that the ideal gas constant does not influence the semidiscrete rate of change of the
# density, as demonstrated by

#nb norm(J[1:4:end])

#md # ```jldoctest euler_gamma_gradient
#md # julia> norm(J[1:4:end])
#md # 0.0
#md # ```

# Here, we used some knowledge about the internal memory layout of Trixi, an array of structs
# with the conserved variables as fastest-varying index in memory.


# ## Differentiating through a complete simulation

# It is also possible to differentiate through a complete simulation. As an example, let's differentiate
# the total energy of a simulation using the linear scalar advection equation with respect to the
# wave number (frequency) of the initial data.

#nb using Trixi, OrdinaryDiffEq, ForwardDiff, Plots

#nb function energy_at_final_time(k) # k is the wave number of the initial condition
#nb     equations = LinearScalarAdvectionEquation2D(1.0, -0.3)
#nb     mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), initial_refinement_level=3, n_cells_max=10^4)
#nb     solver = DGSEM(3, flux_lax_friedrichs)
#nb     initial_condition = (x, t, equation) -> begin
#nb             x_trans = Trixi.x_trans_periodic_2d(x - equation.advectionvelocity * t)
#nb             return SVector(sinpi(k * sum(x_trans)))
#nb     end
#nb     semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
#nb                                                uEltype=typeof(k))
#nb     ode = semidiscretize(semi, (0.0, 1.0))
#nb     sol = solve(ode, BS3(), save_everystep=false)
#nb     Trixi.integrate(energy_total, sol.u[end], semi)
#nb end
#nb #-
#nb k_values = range(0.9, 1.1, length=101)
#nb #-
#nb plot(k_values, energy_at_final_time.(k_values), label="Energy")

#md # ```jldoctest advection_differentiate_simulation
#md # julia> using Trixi, OrdinaryDiffEq, ForwardDiff, Plots

#md # julia> function energy_at_final_time(k) # k is the wave number of the initial condition
#md #            equations = LinearScalarAdvectionEquation2D(1.0, -0.3)
#md #            mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), initial_refinement_level=3, n_cells_max=10^4)
#md #            solver = DGSEM(3, flux_lax_friedrichs)
#md #            initial_condition = (x, t, equation) -> begin
#md #                x_trans = Trixi.x_trans_periodic_2d(x - equation.advectionvelocity * t)
#md #                return SVector(sinpi(k * sum(x_trans)))
#md #            end
#md #            semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
#md #                                                uEltype=typeof(k))
#md #            ode = semidiscretize(semi, (0.0, 1.0))
#md #            sol = solve(ode, BS3(), save_everystep=false)
#md #            Trixi.integrate(energy_total, sol.u[end], semi)
#md #        end
#md # energy_at_final_time (generic function with 1 method)

#md # julia> k_values = range(0.9, 1.1, length=101)
#md # 0.9:0.002:1.1

#md # julia> plot(k_values, energy_at_final_time.(k_values), label="Energy");
#md # ```
#md # ![tutorial_totalenergy](https://user-images.githubusercontent.com/74359358/126673315-15fc87a9-d236-4216-876f-5f1d3219595a.png)

# You see a plot of a curve that resembles a parabola with local maximum around `k = 1.0`.
# Why's that? Well, the domain is fixed but the wave number changes. Thus, if the wave number is
# not chosen as an integer, the initial condition will not be a smooth periodic function in the
# given domain. Hence, the dissipative surface flux (`flux_lax_friedrichs` in this example)
# will introduce more dissipation. In particular, it will introduce more dissipation for "less smooth"
# initial data, corresponding to wave numbers `k` further away from integers.

# We can compute the discrete derivative of the energy at the final time with respect to the wave
# number `k` as follows.

#nb round(ForwardDiff.derivative(energy_at_final_time, 1.0), sigdigits=2)

#md # ```jldoctest advection_differentiate_simulation
#md # julia> round(ForwardDiff.derivative(energy_at_final_time, 1.0), sigdigits=2)
#md # 1.4e-5
#md # ```

# This is rather small and we can treat it as zero in comparison to the value of this derivative at
# other wave numbers `k`.

#nb dk_values = ForwardDiff.derivative.((energy_at_final_time,), k_values);

#nb plot(k_values, dk_values, label="Derivative")

#md # ```jldoctest advection_differentiate_simulation
#md # julia> dk_values = ForwardDiff.derivative.((energy_at_final_time,), k_values);

#md # julia> plot(k_values, dk_values, label="Derivative");
#md # ```

# If you remember basic calculus, a sufficient condition for a local maximum is that the first derivative
# vanishes and the second derivative is negative. We can also check this discretely.

#nb round(ForwardDiff.derivative(
#nb     k -> Trixi.ForwardDiff.derivative(energy_at_final_time, k), 1.0),
#nb sigdigits=2)

#md # ```jldoctest advection_differentiate_simulation
#md # julia> round(ForwardDiff.derivative(
#md #           k -> Trixi.ForwardDiff.derivative(energy_at_final_time, k),
#md #        1.0), sigdigits=2)
#md # -0.9
#md # ```

# Having seen this application, let's break down what happens step by step.

#nb function energy_at_final_time(k) # k is the wave number of the initial condition
#nb     equations = LinearScalarAdvectionEquation2D(1.0, -0.3)
#nb     mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), initial_refinement_level=3, n_cells_max=10^4)
#nb     solver = DGSEM(3, flux_lax_friedrichs)
#nb     initial_condition = (x, t, equation) -> begin
#nb         x_trans = Trixi.x_trans_periodic_2d(x - equation.advectionvelocity * t)
#nb         return SVector(sinpi(k * sum(x_trans)))
#nb     end
#nb     semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
#nb                                                uEltype=typeof(k))
#nb     ode = semidiscretize(semi, (0.0, 1.0))
#nb     sol = solve(ode, BS3(), save_everystep=false)
#nb     Trixi.integrate(energy_total, sol.u[end], semi)
#nb end
#nb #-
#nb k = 1.0
#nb round(ForwardDiff.derivative(energy_at_final_time, k), sigdigits=2)

#md # ```julia
#md # julia> function energy_at_final_time(k) # k is the wave number of the initial condition
#md #           equations = LinearScalarAdvectionEquation2D(1.0, -0.3)
#md #           mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), initial_refinement_level=3, n_cells_max=10^4)
#md #           solver = DGSEM(3, flux_lax_friedrichs)
#md #           initial_condition = (x, t, equation) -> begin
#md #               x_trans = Trixi.x_trans_periodic_2d(x - equation.advectionvelocity * t)
#md #               return SVector(sinpi(k * sum(x_trans)))
#md #           end
#md #           semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
#md #                                               uEltype=typeof(k))
#md #           ode = semidiscretize(semi, (0.0, 1.0))
#md #           sol = solve(ode, BS3(), save_everystep=false)
#md #           Trixi.integrate(energy_total, sol.u[end], semi)
#md #        end;

#md # julia> round(ForwardDiff.derivative(energy_at_final_time, 1.0), sigdigits=2)
#md # 1.4e-5
#md # ```

# When calling `ForwardDiff.derivative(energy_at_final_time, k)` with `k=1.0`, ForwardDiff.jl
# will basically use the chain rule and known derivatives of existing basic functions
# to calculate the derivative of the energy at the final time with respect to the
# wave number `k` at `k0 = 1.0`. To do this, ForwardDiff.jl uses dual numbers, which
# basically store the result and its derivative w.r.t. a specified parameter at the
# same time. Thus, we need to make sure that we can treat these `ForwardDiff.Dual`
# numbers everywhere during the computation. Fortunately, generic Julia code usually
# supports these operations. The most basic problem for a developer is to ensure
# that all types are generic enough, in particular the ones of internal caches.

# The first step in this example creates some basic ingredients of our simulation.
equations = LinearScalarAdvectionEquation2D(1.0, -0.3)
mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), initial_refinement_level=3, n_cells_max=10^4)
solver = DGSEM(3, flux_lax_friedrichs)

# These do not have internal caches storing intermediate values of the numerical
# solution, so we do not need to adapt them. In fact, we could also define them
# outside of `energy_at_final_time` (but would need to take care of globals or
# wrap everything in another function).

# Next, we define the initial condition
initial_condition = (x, t, equation) -> begin
    x_trans = Trixi.x_trans_periodic_2d(x - equation.advectionvelocity * t)
    return SVector(sinpi(k * sum(x_trans)))
end
# as a closure capturing the wave number `k` passed to `energy_at_final_time`.
# If you call `energy_at_final_time(1.0)`, `k` will be a `Float64`. Thus, the
# return values of `initial_condition` will be `SVector`s of `Float64`s. When
# calculating the `ForwardDiff.derivative`, `k` will be a `ForwardDiff.Dual` number.
# Hence, the `initial_condition` will return `SVector`s of `ForwardDiff.Dual`
# numbers.

# The semidiscretization `semi` uses some internal caches to avoid repeated allocations
# and speed up the computations, e.g. for numerical fluxes at interfaces. Thus, we
# need to tell Trixi to allow `ForwardDiff.Dual` numbers in these caches. That's what
# the keyword argument `uEltype=typeof(k)` in
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    uEltype=typeof(k))

# does. This is basically the only part where you need to modify your standard Trixi
# code to enable automatic differentiation. From there on, the remaining steps
ode = semidiscretize(semi, (0.0, 1.0))
sol = solve(ode, BS3(), save_everystep=false)
round(Trixi.integrate(energy_total, sol.u[end], semi), sigdigits=5)

# do not need any modifications since they are sufficiently generic (and enough effort
# has been spend to allow general types inside these calls).


# ## Propagating errors using Measurements.jl

# [![Error bars by Randall Munroe](https://imgs.xkcd.com/comics/error_bars.png)](https://xkcd.com/2110/)

# Similar to AD, Trixi also allows propagating uncertainties using linear error propagation
# theory via [Measurements.jl](https://github.com/JuliaPhysics/Measurements.jl).
# As an example, let's create a system representing the linear advection equation
# in 1D with an uncertain velocity. Then, we create a semidiscretization using a
# sine wave as initial condition, solve the ODE, and plot the resulting uncertainties
# in the primitive variables.

#nb using Trixi, OrdinaryDiffEq, Measurements, Plots, LaTeXStrings

#nb equations = LinearScalarAdvectionEquation1D(1.0 ± 0.1);

#nb mesh = TreeMesh((-1.0,), (1.0,), n_cells_max=10^5, initial_refinement_level=5);

#nb solver = DGSEM(3);

#nb semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
#nb                                     solver, uEltype=Measurement{Float64});

#nb ode = semidiscretize(semi, (0.0, 1.5));

#nb sol = solve(ode, BS3(), save_everystep=false);

#nb plot(sol)

#md # ```jldoctest; output = false
#md # using Trixi, OrdinaryDiffEq, Measurements, Plots, LaTeXStrings

#md # equations = LinearScalarAdvectionEquation1D(1.0 ± 0.1);

#md # mesh = TreeMesh((-1.0,), (1.0,), n_cells_max=10^5, initial_refinement_level=5);

#md # solver = DGSEM(3);

#md # semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
#md #                                     solver, uEltype=Measurement{Float64});

#md # ode = semidiscretize(semi, (0.0, 1.5));

#md # sol = solve(ode, BS3(), save_everystep=false);

#md # plot(sol)

#md # # output

#md # Plot{Plots.GRBackend() n=1}
#md # ```

# You should see a plot like the following, where small error bars are shown around
# the extrema and larger error bars are shown in the remaining parts.
# This result is in accordance with expectations. Indeed, the uncertain propagation
# speed will affect the extrema less since the local variation of the solution is
# relatively small there. In contrast, the local variation of the solution is large
# around the turning points of the sine wave, so the uncertainties will be relatively
# large there.

# All this is possible due to allowing generic types and having good abstractions
# in Julia that allow packages to work together seamlessly.

# ![tutorial_measurements1](https://user-images.githubusercontent.com/12693098/114027260-78ca8300-9877-11eb-88d4-f93c9bc55d0b.png)

# ## Finite difference approximations

# Trixi provides the convenience function [`jacobian_fd`](@trixi-docs:reference-trixi/#Trixi.jacobian_fd-Tuple{Trixi.AbstractSemidiscretization}) to approximate the Jacobian
# via central finite differences.

#nb using Trixi, LinearAlgebra

#nb equations = CompressibleEulerEquations2D(1.4);

#nb solver = DGSEM(3, flux_central);

#nb mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), initial_refinement_level=2, n_cells_max=10^5);

#nb semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_density_wave, solver);

#nb J_fd = jacobian_fd(semi);

#nb J_ad = jacobian_ad_forward(semi);

#nb norm(J_fd - J_ad) / size(J_fd, 1) < 7.0e-7

#md # ```jldoctest
#md # julia> using Trixi, LinearAlgebra

#md # julia> equations = CompressibleEulerEquations2D(1.4);

#md # julia> solver = DGSEM(3, flux_central);

#md # julia> mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), initial_refinement_level=2, n_cells_max=10^5);

#md # julia> semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_density_wave, solver);

#md # julia> J_fd = jacobian_fd(semi);

#md # julia> J_ad = jacobian_ad_forward(semi);

#md # julia> norm(J_fd - J_ad) / size(J_fd, 1) < 7.0e-7
#md # true
#md # ```

# This discrepancy is of the expected order of magnitude for central finite difference approximations.


# ## Linear systems

# When a linear PDE is discretized using a linear scheme such as a standard DG method,
# the resulting semidiscretization yields an affine ODE of the form

# ```math
# \partial_t u(t) = A u(t) + b,
# ```

# where `A` is a linear operator ("matrix") and `b` is a vector. Trixi allows you
# to obtain this linear structure in a matrix-free way by using [`linear_structure`](@trixi-docs:reference-trixi/#Trixi.linear_structure-Tuple{Trixi.AbstractSemidiscretization}).
# The resulting operator `A` can be used in multiplication, e.g. `mul!` from
# LinearAlgebra, converted to a sparse matrix using `sparse` from SparseArrays,
# or converted to a dense matrix using `Matrix` for detailed eigenvalue analyses.
# For example,

#nb using Trixi, LinearAlgebra, Plots

#nb equations = LinearScalarAdvectionEquation2D(1.0, -0.3);

#nb solver = DGSEM(3, flux_lax_friedrichs);

#nb mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), initial_refinement_level=2, n_cells_max=10^5);

#nb semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver);

#nb A, b = linear_structure(semi);

#nb size(A), size(b)
#nb #-
#nb λ = eigvals(Matrix(A));

#nb scatter(real.(λ), imag.(λ));

#nb λ = eigvals(Matrix(A)); maximum(real, λ) / maximum(abs, λ) < 1.0e-15

#md # ```jldoctest
#md # julia> using Trixi, LinearAlgebra, Plots

#md # julia> equations = LinearScalarAdvectionEquation2D(1.0, -0.3);

#md # julia> solver = DGSEM(3, flux_lax_friedrichs);

#md # julia> mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), initial_refinement_level=2, n_cells_max=10^5);

#md # julia> semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver);

#md # julia> A, b = linear_structure(semi);

#md # julia> size(A), size(b)
#md # ((256, 256), (256,))

#md # julia> λ = eigvals(Matrix(A));

#md # julia> scatter(real.(λ), imag.(λ));

#md # julia> λ = eigvals(Matrix(A)); maximum(real, λ) / maximum(abs, λ) < 1.0e-15
#md # true
#md # ```
