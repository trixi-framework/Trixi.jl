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

julia> equations = LinearScalarAdvectionEquation2D(1.0, 0.3);

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
julia> λ, V = eigen(J); round(cond(V), sigdigits=2)
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

julia> λ, V = eigen(J); round(cond(V), sigdigits=2)
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

julia> λ, V = eigen(J); round(cond(V), sigdigits=2)
93000.0
```
Note that the condition number of the eigenvector matrix increases but is still smaller than for the
example in 2D.


### Computing other derivatives

It is also possible to compute derivatives of other dependencies using AD in Trixi. For example,
you can compute the gradient of an entropy-dissipative semidiscretization with respect to the
ideal gas constant of the compressible Euler equations using

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

