#src # Structured mesh with curvilinear mapping

# Here, we want to introduce another mesh type of [Trixi](https://github.com/trixi-framework/Trixi.jl).
# More precisely, this tutorial is about the curved mesh type [`StructuredMesh`](@ref) supporting
# curved meshes.

# # Creating a curved mesh
# There are two basic options to define a curved [`StructuredMesh`](@ref) in Trixi. You can
# implement curves for the domain boundaries, or alternatively, set up directly the complete
# transformation mapping. We now present one short example each.

# ## Mesh defined by domain boundary curves
# Both examples are based on a semdiscretization of the 2D compressible Euler equations.

using OrdinaryDiffEq
using Trixi

equations = CompressibleEulerEquations2D(1.4)

# We start with a pressure perturbation at `(xs, 0.0)` as initial condtition.
function initial_condition_pressure_perturbation(x, t, equations::CompressibleEulerEquations2D)
  xs = 1.5 # location of the initial disturbance on the x axis
  w = 1/8 # half width
  p = exp(-log(2) * ((x[1]-xs)^2 + x[2]^2)/w^2) + 1.0
  v1 = 0.0
  v2 = 0.0
  rho = 1.0

  return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_pressure_perturbation

# Initialize every boundary as a [`boundary_condition_slip_wall`](@ref).
boundary_conditions = boundary_condition_slip_wall

# The approximation setup is an entropy-stable split-form DG method with `polydeg=4`. We are using
# the two fluxes [`flux_ranocha`](@ref) and [`flux_lax_friedrichs`](@ref).
solver = DGSEM(polydeg=4, surface_flux=flux_lax_friedrichs,
               volume_integral=VolumeIntegralFluxDifferencing(flux_ranocha))

# We want to define a circular cylinder as physical domain. It contains an inner semicircle with
# radius `r0` and an outer semicircle of radius `r1`.

# ![](https://user-images.githubusercontent.com/74359358/158571800-db71f18a-b4b0-4ca1-b76d-13f68ebf002c.png)

#src # \documentclass{standalone}
#src # \usepackage{tikz}
#src # \begin{document}
#src #   \begin{tikzpicture}
#src #     % Circular cylinder
#src #     \draw[->] (0.5,0) -- node[above] {$\textbf{f1}$} (5,0);
#src #     \draw[<-] (-5,0)  -- node[above] {$\textbf{f2}$} (-0.5,0);
#src #     \draw[->] (0.5,0) arc (0:180:0.5); \node at (0, 0.8) {$\textbf{f3}$};
#src #     \draw[->] (5,0)   arc (0:180:5);   \node at (0, 4.7) {$\textbf{f4}$};
#src #     \draw[dashed] (-5, -0.5)   -- node[below] {$\textbf{r}_1$} (0, -0.5);
#src #     \draw[dashed] (-0.5, -0.7) -- node[below] {$\textbf{r}_0$} (0, -0.7);
#src #     % Arrow
#src #     \draw[line width=0.1cm, ->, bend angle=30, bend left] (4, 2) to (8,3);
#src #     % Right Square
#src #     \draw[->] (7,0)  -- node[below] {$\textbf{f1}$} (12,0);
#src #     \draw[->] (7,5)  -- node[above] {$\textbf{f2}$} (12,5);
#src #     \draw[->] (7,0)  -- node[left]  {$\textbf{f3}$} (7,5);
#src #     \draw[->] (12,0) -- node[right] {$\textbf{f4}$} (12,5);
#src #     % Pressure perturbation
#src #     \draw[fill] (1.6, 0) arc (0:180:0.1);
#src #     \draw (1.7, 0) arc (0:180:0.2);
#src #     \draw (1.8, 0) arc (0:180:0.3);
#src #   \end{tikzpicture}
#src # \end{document}


# The domain boundary curves with curve parameter in $[-1,1]$ are sorted as shown in the sketch.
# They always are orientated from negative to positive coordinate, such that the corners have to
# fit like this $f1(+1) = f4(-1)$ and $f3(+1) = f2(-1)$.

# In our case we can define the domain boundary curves as follows:
r0 = 0.5 # inner radius
r1 = 5.0 # outer radius
f1(xi)  = SVector( r0 + 0.5 * (r1 - r0) * (xi + 1), 0.0) # right line
f2(xi)  = SVector(-r0 - 0.5 * (r1 - r0) * (xi + 1), 0.0) # left line
f3(eta) = SVector(r0 * cos(0.5 * pi * (eta + 1)), r0 * sin(0.5 * pi * (eta + 1))) # inner circle
f4(eta) = SVector(r1 * cos(0.5 * pi * (eta + 1)), r1 * sin(0.5 * pi * (eta + 1))) # outer circle

# We create a curved mesh with 16 x 16 elements. The defined domain boundary curves are passed as a tuple.
cells_per_dimension = (16, 16)
mesh = StructuredMesh(cells_per_dimension, (f1, f2, f3, f4), periodicity=false)

# Then, we define the simulation with endtime `T=3` with `semi`, `ode` and `callbacks`.
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=0.9)

callbacks = CallbackSet(analysis_callback,
                        alive_callback,
                        stepsize_callback)

# Running the simulation
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

using Plots
plot(sol)
#-
pd = PlotData2D(sol)
plot(pd["p"])
plot!(getmesh(pd))


# ## Mesh directly defined by the transformation mapping
# As mentioned before, you can also define the domain for a `StructuredMesh` by directly setting up
# a transformation mapping. Here, we want to present a nice mapping, which is often used to test
# free-stream preservation. Exact free-stream preservation is a crucial property of any numerical
# method on curvilinear grids. The mapping is a reduced 2D version of the mapping described in
# [Rueda-Ramírez et al. (2021), p.18](https://arxiv.org/abs/2012.12040).

using OrdinaryDiffEq
using Trixi

equations = CompressibleEulerEquations2D(1.4)

# As mentioned, this mapping is used for testing free-stream preservation. So, we use a constant
# initial condition.
initial_condition = initial_condition_constant

solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

# We define the transformation mapping with variables in $[-1, 1]$ as described in
# Rueda-Ramírez et al. (2021), p.18 (reduced to 2D):
function mapping(xi_, eta_)
  ## Transform input variables between -1 and 1 onto [0,3]
  xi = 1.5 * xi_ + 1.5
  eta = 1.5 * eta_ + 1.5

  y = eta + 3/8 * (cos(1.5 * pi * (2 * xi - 3)/3) *
                   cos(0.5 * pi * (2 * eta - 3)/3))

  x = xi + 3/8 * (cos(0.5 * pi * (2 * xi - 3)/3) *
                  cos(2 * pi * (2 * y - 3)/3))

  return SVector(x, y)
end

cells_per_dimension = (16, 16)

# Instead of a tuple of boundary functions, the `mesh` now has the mapping as its parameter.
mesh = StructuredMesh(cells_per_dimension, mapping)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=0.9)

callbacks = CallbackSet(analysis_callback, alive_callback,
                        stepsize_callback)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

using Plots
pd = PlotData2D(sol)
plot(pd["rho"])
plot!(getmesh(pd))

#src # Plot lösung -1 in 10er log

# TODO: Revise next sentence when plot is changed
# Besides the expected constant solution for density, we see the nice mesh structure resulting from
# our transformation mapping.

# Of course, you can also use other mappings as for instance shifts by $(x, y)$
# ```julia
# mapping(xi, eta) = SVector(xi + x, eta + y)
# ```
# or rotations with a rotation matrix $T$
# ```julia
# mapping(xi, eta) = T * SVector(xi, eta).
# ```

# For more curved mesh mappings, please have a look at some
# [elixirs for `StructuredMesh`](https://github.com/trixi-framework/Trixi.jl/tree/main/examples).
# For another curved mesh type, there is a [tutorial](@ref hohqmesh_tutorial) about Trixi's
# unstructured mesh type [`UnstructuredMesh2D`] and its use of the
# [High-Order Hex-Quad Mesh (HOHQMesh) generator](https://github.com/trixi-framework/HOHQMesh),
# created and developed by David Kopriva.
