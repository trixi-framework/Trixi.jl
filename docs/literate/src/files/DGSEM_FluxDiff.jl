#src # DGSEM with flux differencing

# This tutorial starts with a presentation of the weak formulation of the discontinuous Galerkin
# spectral element method (DGSEM) in order to fix the notation of the used operators.
# Then, the DGSEM formulation with flux differencing (split form DGSEM) and its implementation in
# [Trixi.jl](https://github.com/trixi-framework/Trixi.jl) is shown.

# We start with the one-dimensional conservation law
# ```math
# u_t + f(u)_x = 0, \qquad t\in \mathbb{R}^+, x\in\Omega
# ```
# with the physical flux $f$.

# We split the domain $\Omega$ into elements $K$ with center $x_K$ and size $\Delta x$. With the
# transformation mapping $x(\xi)=x_K + \frac{\Delta x}{2} \xi$ we can transform the reference element
# $[-1,1]$ to every physical element. So, the equation can be restricted to the reference element using the
# determinant of the Jacobian matrix of the transformation mapping
# $J=\frac{\partial x}{\partial \xi}=\frac{\Delta x}{2}$.
# ```math
# J u_t + f(u)_{\xi} = 0, \qquad t\in \mathbb{R}^+, \xi\in [-1,1]
# ```


# ## The weak form of the DGSEM
# We consider the so-called discontinuous Galerkin spectral element method (DGSEM) with collocation.
# It results from choosing a nodal DG ansatz using $N+1$ Gauss-Lobatto nodes $\xi_i$ in $[-1,1]$
# with matching interpolation weights $w_i$, which are used for numerical integration and interpolation with
# the Lagrange polynomial basis $l_i$ of degree $N$. The Lagrange functions are created with those nodes and
# hence fulfil a Kronecker property at the GL nodes.
# The weak formulation of the DGSEM for one element is
# ```math
# J \underline{\dot{u}}(t) = - M^{-1} B \underline{f}^* + M^{-1} D^T M \underline{f}
# ```
# where $\underline{u}=(u_0, u_1, \dots, u_N)^T\in\mathbb{R}^{N+1}$ is the collected pointwise evaluation
# of $u$ at the discretization nodes and $\dot{u} = \partial u / \partial t = u_t$ is the temporal derivative.
# The nodal values of the flux function $f$ results with collocation in $\underline{f}$, since
# $\underline{f}_j=f(\underline{u}_j)$. Moreover, we got the numerical flux $f^*=f^*(u^-, u^+)$.

# We will now have a short overview over the operators we used.

# The **derivative matrix** $D\in\mathbb{R}^{(N+1)\times (N+1)}$ mimics a spatial derivation on a
# discrete level with $\underline{f}_x \approx D \underline{f}$. It is defined by $D_{ij} = l_j'(\xi_i)$.

# The diagonal **mass matrix** $M$ is defined by $M_{ij}=\langle l_j, l_i\rangle_N$ with the
# numerical scalar product $\langle \cdot, \cdot\rangle_N$ defined for functions $f$ and $g$ by
# ```math
# \langle f, g\rangle_N := \int_{-1, N}^1 f(\xi) g(\xi) d\xi := \sum_{k=0}^N f(\xi_k) g(\xi_k) w_k.
# ```
# The multiplication by $M$ matches a discrete integration
# ```math
#   \int_{-1}^1 f(\xi) \underline{l}(\xi) d\xi \approx M \underline{f},
# ```

# The **boundary matrix** $B=\text{diag}([-1, 0,..., 0, 1])$ represents an evaluation of a
# function at the boundaries $\xi_0=-1$ and $\xi_N=1$.

# For these operators the following property holds:
# ```math
#   M D + (M D)^T = B.
# ```
# This is called the summation-by-parts (SBP) property since it mimics integration by parts on a
# discrete level ([Gassner (2013)](https://doi.org/10.1137/120890144)).

# The explicit definitions of the operators and the contruction of the 1D algorithm can be found
# for instance in the tutorial [introduction to DG methods](@ref scalar_linear_advection_1d)
# or in more detail in [Kopriva (2009)](https://link.springer.com/book/10.1007/978-90-481-2261-5).

# This property shows the equivalence between the weak form and the following strong formulation
# of the DGSEM.
# ```math
# \begin{align*}
# J \underline{\dot{u}}(t)
# &= - M^{-1} B \underline{f}^* + M^{-1} D^T M \underline{f}\\[5pt]
# &= - M^{-1} B \underline{f}^* + M^{-1} (B - MD) \underline{f}\\[5pt]
# &= - M^{-1} B (\underline{f}^* - \underline{f}) - D \underline{f}
# \end{align*}
# ```
# More information about the equivalence you can find in [Kopriva, Gassner (2010)](https://doi.org/10.1007/s10915-010-9372-3).


# ## DGSEM with flux differencing
# When using the diagonal SBP property it is possible to rewrite the application of the derivative
# operator $D$ in the calculation of the volume integral into a subcell based finite volume type
# differencing formulation ([Fisher, Carpenter (2013)](https://doi.org/10.1016/j.jcp.2013.06.014)).
# We replace $D \underline{f}$ in the strong form by $2D \underline{f}_{vol}(u^-, u^+)$ with
# the consistent two-point volume flux $f_{vol}$ and receive the DGSEM formulation with flux differencing
# (split form DGSEM) ([Gassner, Winters, Kopriva (2016)](https://doi.org/10.1016/j.jcp.2016.09.013)).

# ```math
# \begin{align*}
# J \underline{\dot{u}}(t) &= - M^{-1} B (\underline{f}^* - \underline{f}) - 2D \underline{f}_{vol}(u^-, u^+)\\[5pt]
# &= - M^{-1} B (\underline{f}^* - \underline{f}_{vol}(\underline{u}, \underline{u})) - 2D \underline{f}_{vol}(u^-, u^+)\\[5pt]
# &= - M^{-1} B \underline{f}_{sur}^* - (2D - M^{-1} B) \underline{f}_{vol}
# \end{align*}
# ```
# This formulation is in a weak form type formulation and can bei implemented by using the matrix $(2D-M^{-1}B)$
# and two different fluxes. We divide between the surface flux $f=f_{sur}$ used for the numerical flux
# $f_{sur}^*$ and the already mentioned volume flux $f_{vol}$ especially for this formulation.


# This formulation creates a more stable version of DGSEM, because it fulfils entropy stability.
# Moreover it allows the construction of entropy conserving discretizations without relying on
# exact integration. This is achieved when using a two-point entropy conserving flux function as
# volume flux in the volume flux differencing formulation.
# Then, the numerical surface flux can be used to control the dissipation of the discretization and to
# guarantee decreasing entropy, i.e. entropy stability.



# ## Implementation in Trixi
# Now, we have a look at the implementation of DGSEM with flux differencing with [Trixi.jl](https://github.com/trixi-framework/Trixi.jl).
using OrdinaryDiffEq, Trixi

# We implement a simulation for the compressible Euler equations in 2D
# ```math
# \partial_t \begin{pmatrix} \rho \\ \rho v_1 \\ \rho v_2 \\ \rho e \end{pmatrix}
# + \partial_x \begin{pmatrix} \rho v_1 \\ \rho v_1^2 + p \\ \rho v_1 v_2 \\ (\rho e +p) v_1 \end{pmatrix}
# + \partial_y \begin{pmatrix} \rho v_2 \\ \rho v_1 v_2 \\ \rho v_2^2 + p \\ (\rho e +p) v_2 \end{pmatrix}
# = \begin{pmatrix} 0 \\ 0 \\ 0 \\ 0 \end{pmatrix}
# ```
# for an ideal gas with ratio of specific heats $\gamma=1.4$.
# Here, $\rho$ is the density, $v_1$, $v_2$ the velocities, $e$ the specific total energy and
# ```math
# p = (\gamma - 1) \left( \rho e - \frac{1}{2} \rho (v_1^2+v_2^2) \right)
# ```
# the pressure.

gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

# As our initial condition we will use a weak blast wave from [Hennemann, Gassner (2020)](https://arxiv.org/abs/2008.12044).
# The primitive variables are defined by
# ```math
# \begin{pmatrix} \rho \\ v_1 \\ v_2 \\ p \end{pmatrix}
# = \begin{pmatrix} 1.0 \\ 0.0 \\ 0.0 \\ 1.0 \end{pmatrix} \text{if } \|x\|_2 > 0.5,\;
# \text{and } \begin{pmatrix} \rho \\ v_1 \\ v_2 \\ p \end{pmatrix}
# = \begin{pmatrix} 1.1691 \\ 0.1882 * \cos(\phi) \\ 0.1882 * \sin(\phi) \\ 1.245 \end{pmatrix} \text{else}
# ```
# with $\phi = \tan^{-1}(\frac{x_2}{x_1})$.

# This initial condition is implemented in Trixi under the name [`initial_condition_weak_blast_wave`](@ref).
initial_condition = initial_condition_weak_blast_wave

# In Trixi, flux differencing for the volume integral can be implemented with
# [`VolumeIntegralFluxDifferencing`](@ref) using symmetric two-point volume fluxes.
# Here, we are using two different fluxes. First, we set up the entropy conserving and kinetic energy preserving
# flux `flux_ranocha` by [Hendrik Ranocha (2018)](https://cuvillier.de/en/shop/publications/7743)
# and second, a modification of the original kinetic energy preserving two-point flux by
# [Shima et al (2020)](https://doi.org/10.1016/j.jcp.2020.110060).

# We will numerically confirm the difference in entropy conservation of the two fluxes.

volume_flux = flux_ranocha # = f_vol
solver_ranocha    = DGSEM(polydeg=3, surface_flux=flux_ranocha,
                          volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

volume_flux = flux_shima_etal # = f_vol
solver_shima_etal = DGSEM(polydeg=3, surface_flux=flux_shima_etal,
                          volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

# Now, we implement Trixi's `mesh`, `semi` and `ode` in a simple framework. For more information please
# have a look at the documentation, the basic tutorial [introduction to DG methods](@ref scalar_linear_advection_1d)
# or some basic elixirs.
coordinates_min = (-2.0, -2.0)
coordinates_max = ( 2.0,  2.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=5,
                n_cells_max=10_000,
                periodicity=true)

semi_ranocha    = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver_ranocha,
                                               boundary_conditions=boundary_condition_periodic)

semi_shima_etal = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver_shima_etal,
                                               boundary_conditions=boundary_condition_periodic)

## ODE solvers
tspan = (0.0, 0.4)
ode_ranocha    = semidiscretize(semi_ranocha,    tspan);
ode_shima_etal = semidiscretize(semi_shima_etal, tspan);

# To compare the entropy conservation of the different fluxes, we will use the analysis calllback
# implemented in Trixi. It provides some information about the approximation including the entropy change.
analysis_callback = AnalysisCallback(semi_ranocha, interval=100);

# We now run the simulation using `flux_ranocha`.
sol_ranocha = solve(ode_ranocha, RDPK3SpFSAL49(), abstol=1.0e-6, reltol=1.0e-6,
                    callback=analysis_callback, save_everystep=false);
# A look at the change in entropy $\sum \partial S/\partial U \cdot U_t$ in the analysis callback
# confirms that the flux is entropy conserving since the change is about machine precision.

sol_shima_etal = solve(ode_shima_etal, RDPK3SpFSAL49(), abstol=1.0e-6, reltol=1.0e-6,
                       callback=analysis_callback, save_everystep=false);
# On the other hand, this flux is not entropy conserving.

# We can plot the approximated solution at the time `t=0.4`.
using Plots
plot(sol_ranocha)

# Of course, you are not restricted to these two fluxes in Trixi. Here, we will give a short list
# of possible fluxes for the compressible Euler equations.
# For the volume flux Trixi provides for example [`flux_ranocha`](@ref), [`flux_shima_etal`](@ref),
# [`flux_chandrashekar`](@ref), [`flux_kennedy_gruber`](@ref).
# As surface flux you can use all volume fluxes and additionally for instance [`flux_lax_friedrichs`](@ref),
# [`flux_hll`](@ref), [`flux_hllc`](@ref).


# ## Improved stability by flux differencing
# Now, we want to give an example showing the stability advantage of flux differencing. We are
# implementing the isentropic vortex by [Shu (1997)](https://ntrs.nasa.gov/citations/19980007543)
# for the compressible Euler equation in 2D.
using OrdinaryDiffEq, Trixi

equations = CompressibleEulerEquations2D(1.4)

# The initial condition is the isentropic vortex.
function initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)
    ## needs appropriate mesh size, e.g. [-10,-10]x[10,10]
    ## make sure that the inicenter does not exit the domain, e.g. T=10.0
    ## initial center of the vortex
    inicenter = SVector(0.0, 0.0)
    ## size and strength of the vortex
    iniamplitude = 0.2
    ## base flow
    rho = 1.0
    v1 = 1.0
    v2 = 1.0
    vel = SVector(v1, v2)
    p = 10.0
    rt = p / rho                  # ideal gas equation
    cent = inicenter + vel*t      # advection of center
    cent = x - cent               # distance to centerpoint
    ## cent=cross(iniaxis,cent)   # distance to axis, tangent vector, length r
    ## cross product with iniaxis = [0,0,1]
    cent = SVector(-cent[2], cent[1])
    r2 = cent[1]^2 + cent[2]^2
    du = iniamplitude/(2*π)*exp(0.5*(1-r2))                     # vel. perturbation
    dtemp = -(equations.gamma-1)/(2*equations.gamma*rt)*du^2    # isentrop
    rho = rho * (1+dtemp)^(1\(equations.gamma-1))
    vel = vel + du*cent
    v1, v2 = vel
    p = p * (1+dtemp)^(equations.gamma/(equations.gamma-1))
    prim = SVector(rho, v1, v2, p)
    return prim2cons(prim, equations)
end
initial_condition = initial_condition_isentropic_vortex

# Defining the physical domain, discretized with Trixi's standard mesh [`TreeMesh`](@ref)
# and $2^4=16$ elements per dimension.
coordinates_min = (-10.0, -10.0)
coordinates_max = ( 10.0,  10.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=10_000,
                periodicity=true)

# Now, we define `solver`, `semi` and `ode` and some callbacks for the two different fluxes.
solver_LLF      = DGSEM(polydeg=5, surface_flux=flux_lax_friedrichs,)
solver_FluxDiff = DGSEM(polydeg=5, surface_flux=flux_shima_etal,
                        volume_integral=VolumeIntegralFluxDifferencing(flux_shima_etal))

semi_LLF      = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver_LLF,
                                             boundary_conditions=boundary_condition_periodic)
semi_FluxDiff = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver_FluxDiff,
                                             boundary_conditions=boundary_condition_periodic)

tspan = (0.0, 20.0)
ode_LLF      = semidiscretize(semi_LLF,      tspan)
ode_FluxDiff = semidiscretize(semi_FluxDiff, tspan)

analysis_callback = AnalysisCallback(semi_LLF, interval=400)

stepsize_callback = StepsizeCallback(cfl=1.0)

callbacks = CallbackSet(analysis_callback, stepsize_callback);

# We now run the simulation without flux differencing.
try
    sol_LLF = solve(ode_LLF, CarpenterKennedy2N54(williamson_condition=false),
                    dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                    save_everystep=false, callback=callbacks);
catch LoadError
    println("Simulation stops, because of a negative value for the density `rho`.")
end
# The simulation crashes shortly after `t=1.0` with a physically meaningless negative value for the density.

# The simulation with flux differencing is not crashing.
sol_FluxDiff = solve(ode_FluxDiff, CarpenterKennedy2N54(williamson_condition=false),
                     dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                     save_everystep=false, callback=callbacks);

using Plots
plot(sol_FluxDiff)

# Open questions/possibilities:
# - Use `RDPK3SpFSAL49()`, `CarpenterKennedy2N54(williamson_condition=false)` or another time integration method?
# - Use analysis, stepsize callback? (More extra code, but more maybe usefull output)
# - Give another example? (e.g. inviscid Taxlor-Green-Vortex)