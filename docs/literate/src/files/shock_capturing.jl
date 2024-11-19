#src # Shock capturing with flux differencing and stage limiter

# This tutorial contains a short summary of the idea of shock capturing for DGSEM with flux differencing
# and its implementation in [Trixi.jl](https://github.com/trixi-framework/Trixi.jl).
# In the second part, an implementation of a positivity preserving limiter is added to the simulation.

# # Shock capturing with flux differencing

# The following rough explanation is on a very basic level. More information about an entropy stable
# shock-capturing strategy for DGSEM discretizations of advection dominated problems, such as the
# compressible Euler equations or the compressible Navier-Stokes equations, can be found in
# [Hennemann et al. (2021)](https://doi.org/10.1016/j.jcp.2020.109935). In
# [Rueda-Ramírez et al. (2021)](https://doi.org/10.1016/j.jcp.2021.110580) you find the extension to
# the systems with non-conservative terms, such as the compressible magnetohydrodynamics (MHD) equations.

# The strategy for a shock-capturing method presented by Hennemann et al. is based on a hybrid blending
# of a high-order DG method with a low-order variant. The low-order subcell finite volume (FV) method is created
# directly with the Legendre-Gauss-Lobatto (LGL) nodes already used for the high-order DGSEM.
# Then, the final method is a convex combination with regulating indicator $\alpha$ of these two methods.

# Since the surface integral is equal for both the DG and the subcell FV method, only the volume integral divides
# between the two methods.

# This strategy for the volume integral is implemented in Trixi.jl under the name of
# [`VolumeIntegralShockCapturingHG`](@ref) with the three parameters of the indicator and the volume fluxes for
# the DG and the subcell FV method.

# Note, that the DG method is based on the flux differencing formulation. Hence, you have to use a
# two-point flux, such as [`flux_ranocha`](@ref), [`flux_shima_etal`](@ref), [`flux_chandrashekar`](@ref) or [`flux_kennedy_gruber`](@ref),
# for the DG volume flux. We would recommend to use the entropy conserving flux `flux_ranocha` by
# [Ranocha (2018)](https://cuvillier.de/en/shop/publications/7743) for the compressible Euler equations.
# ````julia
# volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
#                                                  volume_flux_dg=volume_flux_dg,
#                                                  volume_flux_fv=volume_flux_fv)
# ````

# We now focus on a choice of the shock capturing indicator `indicator_sc`.
# A possible indicator is $\alpha_{HG}$ presented by Hennemann et al. (p.10), which depends on the
# current approximation with modal coefficients $\{m_j\}_{j=0}^N$ of a given `variable`.

# The indicator is calculated for every DG element by itself. First, we calculate a smooth $\alpha$ by
# ```math
# \alpha = \frac{1}{1+\exp(-\frac{-s}{\mathbb{T}}(\mathbb{E}-\mathbb{T}))}
# ```
# with the total energy $\mathbb{E}=\max\big(\frac{m_N^2}{\sum_{j=0}^N m_j^2}, \frac{m_{N-1}^2}{\sum_{j=0}^{N-1} m_j^2}\big)$,
# threshold $\mathbb{T}= 0.5 * 10^{-1.8*(N+1)^{1/4}}$ and parameter $s=ln\big(\frac{1-0.0001}{0.0001}\big)\approx 9.21024$.

# For computational efficiency, $\alpha_{min}$ is introduced and used for
# ```math
# \tilde{\alpha} = \begin{cases}
# 0, & \text{if } \alpha<\alpha_{min}\\
# \alpha, & \text{if } \alpha_{min}\leq \alpha \leq 1- \alpha_{min}\\
# 1, & \text{if } 1-\alpha_{min}<\alpha.
# \end{cases}
# ```

# Moreover, the parameter $\alpha_{max}$ sets a maximal value for $\alpha$ by
# ```math
# \alpha = \min\{\tilde{\alpha}, \alpha_{max}\}.
# ```
# This allows to control the maximal dissipation.

# To remove numerical artifact the final indicator is smoothed with all the neighboring elements'
# indicators. This is activated with `alpha_smooth=true`.
# ```math
# \alpha_{HG} = \max_E \{ \alpha, 0.5 * \alpha_E\},
# ```
# where $E$ are all elements sharing a face with the current element.

# Furthermore, you can specify the variable used for the calculation. For instance you can choose
# `density`, `pressure` or both with `density_pressure` for the compressible Euler equations.
# For every equation there is also the option to use the first conservation variable with `first`.

# This indicator is implemented in Trixi.jl and called [`IndicatorHennemannGassner`](@ref) with the parameters
# `equations`, `basis`, `alpha_max`, `alpha_min`, `alpha_smooth` and `variable`.
# ````julia
# indicator_sc = IndicatorHennemannGassner(equations, basis,
#                                          alpha_max=0.5,
#                                          alpha_min=0.001,
#                                          alpha_smooth=true,
#                                          variable=variable)
# ````

# # Positivity preserving limiter

# Some numerical solutions are physically meaningless, for instance negative values of pressure
# or density for the compressible Euler equations. This often results in crashed simulations since
# the calculation of numerical fluxes or stable time steps uses mathematical operations like roots or
# logarithms. One option to avoid these cases are a-posteriori positivity preserving limiters.
# Trixi.jl provides the fully-discrete positivity-preserving limiter of
# [Zhang, Shu (2011)](https://doi.org/10.1098/rspa.2011.0153).

# It works the following way. For every passed (scalar) variable and for every DG element we calculate
# the minimal value $value_\text{min}$. If this value falls below the given threshold $\varepsilon$,
# the approximation is slightly adapted such that the minimal value of the relevant variable lies
# now above the threshold.
# ```math
# \underline{u}^\text{new} = \theta * \underline{u} + (1-\theta) * u_\text{mean}
# ```
# where $\underline{u}$ are the collected pointwise evaluation coefficients in element $e$ and
# $u_\text{mean}$ the integral mean of the quantity in $e$. The new coefficients are a convex combination
# of these two values with factor
# ```math
# \theta = \frac{value_\text{mean} - \varepsilon}{value_\text{mean} - value_\text{min}},
# ```
# where $value_\text{mean}$ is the relevant variable evaluated for the mean value $u_\text{mean}$.

# The adapted approximation keeps the exact same mean value, but the relevant variable is now greater
# or equal the threshold $\varepsilon$ at every node in every element.

# We specify the variables the way we did before for the shock capturing variables. For the
# compressible Euler equations `density`, `pressure` or the combined variable `density_pressure`
# are a reasonable choice.

# You can implement the limiter in Trixi.jl using [`PositivityPreservingLimiterZhangShu`](@ref) with parameters
# `threshold` and `variables`.
# ````julia
# stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds=thresholds,
#                                                      variables=variables)
# ````
# Then, the limiter is added to the time integration method in the `solve` function. For instance, like
# ````julia
# CarpenterKennedy2N54(stage_limiter!, williamson_condition=false)
# ````
# or
# ````julia
# SSPRK43(stage_limiter!).
# ````

# # Simulation with shock capturing and positivity preserving

# Now, we can run a simulation using the described methods of shock capturing and positivity
# preserving limiters. We want to give an example for the 2D compressible Euler equations.
using OrdinaryDiffEq, Trixi

equations = CompressibleEulerEquations2D(1.4)

# As our initial condition we use the Sedov blast wave setup.
function initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations2D)
    ## Set up polar coordinates
    inicenter = SVector(0.0, 0.0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)

    r0 = 0.21875 # = 3.5 * smallest dx (for domain length=4 and max-ref=6)
    ## r0 = 0.5 # = more reasonable setup
    E = 1.0
    p0_inner = 3 * (equations.gamma - 1) * E / (3 * pi * r0^2)
    p0_outer = 1.0e-5 # = true Sedov setup
    ## p0_outer = 1.0e-3 # = more reasonable setup

    ## Calculate primitive variables
    rho = 1.0
    v1 = 0.0
    v2 = 0.0
    p = r > r0 ? p0_outer : p0_inner

    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_sedov_blast_wave
#-
basis = LobattoLegendreBasis(3)

# We set the numerical fluxes and divide between the surface flux and the two volume fluxes for the DG
# and FV method. Here, we are using [`flux_lax_friedrichs`](@ref) and [`flux_ranocha`](@ref).
surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha

# Now, we specify the shock capturing indicator $\alpha$.

# We implement the described indicator of Hennemann, Gassner as explained above with parameters
# `equations`, `basis`, `alpha_max`, `alpha_min`, `alpha_smooth` and `variable`.
# Since density and pressure are the critical variables in this example, we use
# `density_pressure = density * pressure = rho * p` as indicator variable.
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)

# Now, we can use the defined fluxes and the indicator to implement the volume integral using shock
# capturing.
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

# We finalize the discretization by implementing Trixi.jl's `solver`, `mesh`, `semi` and `ode`,
# while `solver` now has the extra parameter `volume_integral`.
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-2.0, -2.0)
coordinates_max = (2.0, 2.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 6,
                n_cells_max = 10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan);

# We add some callbacks to get an solution analysis and use a CFL-based time step size calculation.
analysis_callback = AnalysisCallback(semi, interval = 100)

stepsize_callback = StepsizeCallback(cfl = 0.8)

callbacks = CallbackSet(analysis_callback, stepsize_callback);

# We now run the simulation using the positivity preserving limiter of Zhang and Shu for the variables
# density and pressure.
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (5.0e-6, 5.0e-6),
                                                     variables = (Trixi.density, pressure))

sol = solve(ode, CarpenterKennedy2N54(stage_limiter!, williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

using Plots
plot(sol)

# # Entropy bounded limiter

# As argued in the description of the positivity preserving limiter above it might sometimes be
# necessary to apply advanced techniques to ensure a physically meaningful solution.
# Apart from the positivity of pressure and density, the physical entropy of the system should increase 
# over the course of a simulation, see e.g. [this](https://doi.org/10.1016/0168-9274(86)90029-2) paper by Tadmor where this property is 
# shown for the compressible Euler equations.
# As this is not necessarily the case for the numerical approximation (especially for the high-order, non-diffusive DG discretizations), 
# Lv and Ihme devised an a-posteriori limiter in [this paper](https://doi.org/10.1016/j.jcp.2015.04.026) which can be applied after each Runge-Kutta stage.
# This limiter enforces a non-decrease in the physical, thermodynamic entropy $S$ 
# by bounding the entropy decrease (entropy increase is always tolerated) $\Delta S$ in each grid cell.
# 
# This translates into a requirement that the entropy of the limited approximation $S\Big(\mathcal{L}\big[\boldsymbol u(\Delta t) \big] \Big)$ should be 
# greater or equal than the previous iterates' entropy $S\big(\boldsymbol u(0) \big)$, enforced at each quadrature point:
# ```math
# S\Big(\mathcal{L}\big[\boldsymbol u(\Delta t, \boldsymbol{x}_i) \big] \Big) \overset{!}{\geq} S\big(\boldsymbol u(0, \boldsymbol{x}_i) \big), \quad i = 1, \dots, (k+1)^d
# ```
# where $k$ denotes the polynomial degree of the element-wise solution and $d$ is the spatial dimension.
# For an ideal gas, the thermodynamic entropy $S(\boldsymbol u) = S(p, \rho)$ is given by
# ```math
# S(p, \rho) = \ln \left( \frac{p}{\rho^\gamma} \right) \: .
# ```
# Thus, the non-decrease in entropy can be reformulated as
# ```math
# p(\boldsymbol{x}_i) - e^{ S\big(\boldsymbol u(0, \boldsymbol{x}_i) \big)} \cdot \rho(\boldsymbol{x}_i)^\gamma \overset{!}{\geq} 0, \quad i = 1, \dots, (k+1)^d \: .
# ```
# In a practical simulation, we might tolerate a maximum (exponentiated) entropy decrease per element, i.e., 
# ```math
# \Delta e^S \coloneqq \min_{i} \left\{ p(\boldsymbol{x}_i) - e^{ S\big(\boldsymbol u(0, \boldsymbol{x}_i) \big)} \cdot \rho(\boldsymbol{x}_i)^\gamma \right\} < c
# ```
# with hyper-parameter $c$ which is to be specified by the user.
# The default value for the corresponding parameter $c=$ `exp_entropy_decrease_max` is set to $-10^{-13}$, i.e., slightly less than zero to 
# avoid spurious limiter actions for cells in which the entropy remains effectively constant.
# Other values can be specified by setting the `exp_entropy_decrease_max` keyword in the constructor of the limiter:
# ```julia
# stage_limiter! = EntropyBoundedLimiter(exp_entropy_decrease_max=-1e-9)
# ```
# Smaller values (larger in absolute value) for `exp_entropy_decrease_max` relax the entropy increase requirement and are thus less diffusive.
# On the other hand, for larger values (smaller in absolute value) of `exp_entropy_decrease_max` the limiter acts more often and the solution becomes more diffusive.
#
# In particular, we compute again a limiting parameter $\vartheta \in [0, 1]$ which is then used to blend the 
# unlimited nodal values $\boldsymbol u$ with the mean value $\boldsymbol u_{\text{mean}}$ of the element:
# ```math
# \mathcal{L} [\boldsymbol u](\vartheta) \coloneqq (1 - \vartheta) \boldsymbol u + \vartheta \cdot \boldsymbol u_{\text{mean}}
# ```
# For the exact definition of $\vartheta$ the interested user is referred to section 4.4 of the paper by Lv and Ihme.
# Note that therein the limiting parameter is denoted by $\epsilon$, which is not to be confused with the threshold $\varepsilon$ of the Zhang-Shu limiter.

# As for the positivity preserving limiter, the entropy bounded limiter may be applied after every Runge-Kutta stage.
# Both fixed timestep methods such as [`CarpenterKennedy2N54`](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/) and 
# adaptive timestep methods such as [`SSPRK43`](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/) are supported.
# We would like to remark that of course every `stage_limiter!` can also be used as a `step_limiter!`, i.e., 
# acting only after the full time step has been taken.

# As an example, we consider a variant of the [1D medium blast wave example](https://github.com/trixi-framework/Trixi.jl/blob/main/examples/tree_1d_dgsem/elixir_euler_blast_wave.jl)
# wherein the shock capturing method discussed above is employed to handle the shock.
# Here, we use a standard DG solver with HLLC surface flux:
using Trixi

solver = DGSEM(polydeg = 3, surface_flux = flux_hllc)

# The remaining setup is the same as in the standard example:

using OrdinaryDiffEq

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations1D(1.4)

function initial_condition_blast_wave(x, t, equations::CompressibleEulerEquations1D)
    ## Modified From Hennemann & Gassner JCP paper 2020 (Sec. 6.3) -> "medium blast wave"
    ## Set up polar coordinates
    inicenter = SVector(0.0)
    x_norm = x[1] - inicenter[1]
    r = abs(x_norm)
    ## The following code is equivalent to
    ## phi = atan(0.0, x_norm)
    ## cos_phi = cos(phi)
    ## in 1D but faster
    cos_phi = x_norm > 0 ? one(x_norm) : -one(x_norm)

    ## Calculate primitive variables
    rho = r > 0.5 ? 1.0 : 1.1691
    v1 = r > 0.5 ? 0.0 : 0.1882 * cos_phi
    p = r > 0.5 ? 1.0E-3 : 1.245

    return prim2cons(SVector(rho, v1, p), equations)
end
initial_condition = initial_condition_blast_wave

coordinates_min = (-2.0,)
coordinates_max = (2.0,)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 6,
                n_cells_max = 10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 12.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100

analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

# We specify the `stage_limiter!` supplied to the classic SSPRK33 integrator
# with strict entropy increase enforcement
# (recall that the tolerated exponentiated entropy decrease is set to -1e-13).
stage_limiter! = EntropyBoundedLimiter()

# We run the simulation with the SSPRK33 method and the entropy bounded limiter:
sol = solve(ode, SSPRK33(stage_limiter!);
            dt = 1.0,
            callback = callbacks);

using Plots
plot(sol)

# ## Package versions

# These results were obtained using the following versions.

using InteractiveUtils
versioninfo()

using Pkg
Pkg.status(["Trixi", "OrdinaryDiffEq", "Plots"],
           mode = PKGMODE_MANIFEST)
