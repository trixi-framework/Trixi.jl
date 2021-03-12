# Differentiable programming

[Julia and its ecosystem provide some tools for differentiable programming](https://sinews.siam.org/Details-Page/scientific-machine-learning-how-julia-employs-differentiable-programming-to-do-it-best).
Trixi.jl is designed to be flexible, extendable, and composable with Julia's growing ecosystem for
scientific computing and machine learning. Thus, the ultimate goal is to have fast implementations
that allow automatic differentiation (AD) without too much hassle for users. If some parts do not
meet these requirements, please feel free to open an issue or propose a fix in a PR.

In the following, we will walk through some examples demonstrating how to differentiate through
Trixi.jl.


## Linear systems

When a linear PDE is discretized using a linear scheme such as a standard DG method,
the resulting semidiscretization yields an affine ODE of the form
```math
\partial_t u(t) = A u(t) + b,
```
where `A` is a linear operator ("matrix") and `b` is a vector. Trixi allows you
to obtain this linear structure in a matrix-free way by using [`linear_structure`](@ref).
The resulting operator `A` can be used in multiplication, e.g. `mul!` from
`LinearAlgebra`, converted to a sparse matrix using `sparse` from `SparseArrays`,
or converted to a dense matrix using `Matrix` for detailed eigenvalue analyses.
For example,
```jldoctest
julia> using Trixi, LinearAlgebra, Plots

julia> equations = LinearScalarAdvectionEquation2D(1.0, -0.3);

julia> solver = DGSEM(3, flux_lax_friedrichs);

julia> mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), initial_refinement_level=2, n_cells_max=10^5);

julia> semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver);

julia> A, b = linear_structure(semi);

julia> size(A), size(b)
((256, 256), (256,))

julia> λ = eigvals(Matrix(A));

julia> scatter(real.(λ), imag.(λ));

julia> λ = eigvals(Matrix(A)); maximum(real, λ) / maximum(abs, λ) < 1.0e-15
true
```


## Forward mode automatic differentiation

Trixi integrates well with [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
for forward mode AD.


### Computing the Jacobian

The high-level interface to compute the Jacobian this way is [`jacobian_ad_forward`](@ref).

```jldoctest euler_eigenvalues
julia> using Trixi, LinearAlgebra, Plots

julia> equations = CompressibleEulerEquations2D(1.4);

julia> solver = DGSEM(3, flux_central);

julia> mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), initial_refinement_level=2, n_cells_max=10^5);

julia> semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_density_wave, solver);

julia> J = jacobian_ad_forward(semi);

julia> size(J)
(1024, 1024)

julia> λ = eigvals(J);

julia> scatter(real.(λ), imag.(λ));

julia> round(maximum(real, λ) / maximum(abs, λ), sigdigits=2)
6.7e-10

julia> round(maximum(real, λ), sigdigits=2)
2.1e-7
```

Interestingly, if we add dissipation by switching to the `flux_lax_friedrichs` at the interfaces,
the maximal real part of the eigenvalues increases.

```jldoctest euler_eigenvalues
julia> solver = DGSEM(3, flux_lax_friedrichs);

julia> semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_density_wave, solver);

julia> J = jacobian_ad_forward(semi);

julia> λ = eigvals(J);

julia> scatter!(real.(λ), imag.(λ));

julia> λ = eigvals(J); round(maximum(real, λ) / maximum(abs, λ), sigdigits=2)
2.1e-5

julia> round(maximum(real, λ), sigdigits=2)
0.0057
```

However, we should be careful when using this analysis, since the eigenvectors are not necessarily
well-conditioned.

```jldoctest euler_eigenvalues
julia> λ, V = eigen(J);

julia> round(cond(V), sigdigits=2)
1.8e6
```

In one space dimension, the situation is a bit different.

```jldoctest euler_eigenvalues
julia> equations = CompressibleEulerEquations1D(1.4);

julia> solver = DGSEM(3, flux_central);

julia> mesh = TreeMesh((-1.0,), (1.0,), initial_refinement_level=6, n_cells_max=10^5);

julia> semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_density_wave, solver);

julia> J = jacobian_ad_forward(semi);

julia> λ = eigvals(J);

julia> scatter(real.(λ), imag.(λ));

julia> round(maximum(real, λ) / maximum(abs, λ), sigdigits=2)
3.2e-16

julia> round(maximum(real, λ), sigdigits=2)
3.2e-12

julia> λ, V = eigen(J);

julia> round(cond(V), sigdigits=2)
250.0
```

If we add dissipation, the maximal real part is still approximately zero.

```jldoctest euler_eigenvalues
julia> solver = DGSEM(3, flux_lax_friedrichs);

julia> semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_density_wave, solver);

julia> J = jacobian_ad_forward(semi);

julia> λ = eigvals(J);

julia> scatter!(real.(λ), imag.(λ));

julia> λ = eigvals(J); round(maximum(real, λ) / maximum(abs, λ), sigdigits=2)
5.3e-18

julia> round(maximum(real, λ), sigdigits=2)
7.7e-14

julia> λ, V = eigen(J);

julia> round(cond(V), sigdigits=2)
93000.0
```
Note that the condition number of the eigenvector matrix increases but is still smaller than for the
example in 2D.


### Computing other derivatives

It is also possible to compute derivatives of other dependencies using AD in Trixi. For example,
you can compute the gradient of an entropy-dissipative semidiscretization with respect to the
ideal gas constant of the compressible Euler equations as described in the following. This example
is also available as the elixir
[examples/special_elixirs/elixir\_euler\_ad.jl](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/special_elixirs/elixir_euler_ad.jl)

```jldoctest euler_gamma_gradient
julia> using Trixi, LinearAlgebra, ForwardDiff

julia> equations = CompressibleEulerEquations2D(1.4);

julia> mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), initial_refinement_level=2, n_cells_max=10^5);

julia> solver = DGSEM(3, flux_lax_friedrichs, VolumeIntegralFluxDifferencing(flux_ranocha));

julia> semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_isentropic_vortex, solver);

julia> u0_ode = compute_coefficients(0.0, semi); size(u0_ode)
(1024,)

julia> J = ForwardDiff.jacobian((du_ode, γ) -> begin
           equations_inner = CompressibleEulerEquations2D(first(γ))
           semi_inner = Trixi.remake(semi, equations=equations_inner, uEltype=eltype(γ));
           Trixi.rhs!(du_ode, u0_ode, semi_inner, 0.0)
       end, similar(u0_ode), [1.4]); # γ needs to be an `AbstractArray`

julia> round.(extrema(J), sigdigits=2)
(-5.6, 5.6)
```

Note that we create a semidiscretization `semi` at first to determine the state `u0_ode` around
which we want to perform the linearization. Next, we wrap the RHS evaluation inside a closure
and pass that to `ForwardDiff.jacobian`. There, we need to make sure that the internal caches
are able to store dual numbers from ForwardDiff.jl bu setting `uEltype` appropriately. A similar
approach is used by [`jacobian_ad_forward`](@ref).

Note that the ideal gas constant does not influence the semidiscrete rate of change of the
density, as demonstrated by

```jldoctest euler_gamma_gradient
julia> norm(J[1:4:end])
0.0
```

Here, we used some knowledge about the internal memory layout of Trixi, an array of structs
with the conserved variables as fastest-varying index in memory.


### Differentiating through a complete simulation

It is also possible to differentiate through a complete simulation. As an example, let's differentiate
the total energy of a simulation using the linear scalar advection equation with respect to the
wave number (frequency) of the initial data.

```jldoctest advection_differentiate_simulation
julia> using Trixi, OrdinaryDiffEq, ForwardDiff, Plots

julia> function energy_at_final_time(k) # k is the wave number of the initial condition
           equations = LinearScalarAdvectionEquation2D(1.0, -0.3)
           mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), initial_refinement_level=3, n_cells_max=10^4);
           solver = DGSEM(3, flux_lax_friedrichs)
           initial_condition = (x, t, equation) -> begin
               x_trans = Trixi.x_trans_periodic_2d(x - equation.advectionvelocity * t)
               return SVector(sinpi(k * sum(x_trans)))
           end
           semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                               uEltype=typeof(k))
           ode = semidiscretize(semi, (0.0, 1.0))
           sol = solve(ode, BS3(), save_everystep=false)
           Trixi.integrate(energy_total, sol.u[end], semi)
       end
energy_at_final_time (generic function with 1 method)

julia> k_values = range(0.9, 1.1, length=101)
0.9:0.002:1.1

julia> plot(k_values, energy_at_final_time.(k_values), label="Energy");
```

You should see a plot of a curve that resembles a parabola with local maximum around `k = 1.0`.
Why's that? Well, the domain is fixed but the wave number changes. Thus, if the wave number is
not chosen as an integer, the initial condition will not be a smooth periodic function in the
given domain. Hence, the dissipative surface flux (`flux_lax_friedrichs` in this example)
will introduce more dissipation. In particular, it will introduce more dissipation for "less smooth"
initial data, corresponding to wave numbers `k` further away from integers.

We can compute the discrete derivative of the energy at the final time with respect to the wave
number `k` as follows.
```jldoctest advection_differentiate_simulation
julia> round(ForwardDiff.derivative(energy_at_final_time, 1.0), sigdigits=2)
1.4e-5
```

This is rather small and we can treat it as zero in comparison to the value of this derivative at
other wave numbers `k`.

```jldoctest advection_differentiate_simulation
julia> dk_values = ForwardDiff.derivative.((energy_at_final_time,), k_values);

julia> plot(k_values, dk_values, label="Derivative");
```

If you remember basic calculus, a sufficient condition for a local maximum is that the first derivative
vanishes and the second derivative is negative. We can also check this discretely.

```jldoctest advection_differentiate_simulation
julia> round(ForwardDiff.derivative(
           k -> Trixi.ForwardDiff.derivative(energy_at_final_time, k),
       1.0), sigdigits=2)
-0.9
```



## Finite difference approximations

Trixi provides the convenience function [`jacobian_fd`](@ref) to approximate the Jacobian
via central finite differences.

```jldoctest
julia> using Trixi, LinearAlgebra

julia> equations = CompressibleEulerEquations2D(1.4);

julia> solver = DGSEM(3, flux_central);

julia> mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0), initial_refinement_level=2, n_cells_max=10^5);

julia> semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_density_wave, solver);

julia> J_fd = jacobian_fd(semi);

julia> J_ad = jacobian_ad_forward(semi);

julia> round(norm(J_fd - J_ad) / size(J_fd, 1), sigdigits=2)
6.7e-7
```
This discrepancy is of the expected order of magnitude for central finite difference approximations.
