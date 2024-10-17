using Plots
plot(Shape([(-1.2, 4), (1.2, 4), (1.2, 3), (-1.2, 3)]), linecolor = "black",
     fillcolor = "white", label = false, linewidth = 2, size = (800, 600), showaxis = false,
     grid = false, xlim = (-2.6, 2.6), ylim = (-21.5, 4.5))
annotate!(0, 3.5, ("semidiscretize(semi, tspan; reset_threads)", 10, :black, :center))
annotate!(0, 2.5,
          ("creates and returns an ODEProblem object, initialized using rhs!, u0_ode, tspan and semi",
           9, :black, :center))
plot!([0, 0], [2, 0.5], arrow = true, color = :black, linewidth = 2, label = "")
plot!([0, -0.2], [2, 0.5], arrow = true, color = :black, linewidth = 2, label = "")
plot!([0, 0.2], [2, 0.5], arrow = true, color = :black, linewidth = 2, label = "")
annotate!(0.3, 1.25, ("specialized for semi type", 9, :black, :left))

plot!(Shape([(-1.6, 0.5), (1.6, 0.5), (1.6, -0.5), (-1.6, -0.5)]), linecolor = "black",
      fillcolor = "white", label = false, linewidth = 2)
annotate!(0, 0,
          ("compute_coefficients(t, semi::SemidiscretizationHyperbolic)", 10, :black,
           :center))
plot!([0, 0], [-0.5, -2], arrow = true, color = :black, linewidth = 2, label = "")

plot!(Shape([(-1.3, -2), (1.3, -2), (1.3, -3), (-1.3, -3)]), linecolor = "black",
      fillcolor = "white", label = false, linewidth = 2)
annotate!(0, -2.5,
          ("compute_coefficients(initial_conditions, t, semi)", 10, :black, :center))
plot!([0, 0.2], [-3, -4.5], arrow = true, color = :black, linewidth = 2, label = "")
plot!([0, -0.87], [-3, -9.5], arrow = true, color = :black, linewidth = 2, label = "")

plot!(Shape([(0.1, -4.5), (2.5, -4.5), (2.5, -6.5), (0.1, -6.5)]), linecolor = "black",
      fillcolor = "white", label = false, linewidth = 2)
annotate!(0.15, -5.5, ("allocate_coefficients(mesh, equations, solver,
                                 cache)", 10, :black, :left))
annotate!(0.1, -8,
          ("initializes u_ode as a 1D zero-vector with a length
depending on the number of variables, nodes,
dimensions of mesh and elements, this vector is used
by OrdinaryDiffEq.jl to keep a solution", 9, :black, :left))

plot!(Shape([(-2.5, -9.5), (0.05, -9.5), (0.05, -11.5), (-2.5, -11.5)]),
      linecolor = "black", fillcolor = "white", label = false, linewidth = 2)
annotate!(-2.45, -10.5,
          ("compute_coefficients!(u_ode, initial_conditions,
                      t, semi)", 10, :black, :left))
plot!([-2.4, -2.4], [-11.5, -18.5], arrow = false, color = :black, linewidth = 2,
      label = "")

plot!(Shape([(-2.2, -13), (-0.8, -13), (-0.8, -14), (-2.2, -14)]), linecolor = "black",
      fillcolor = "white", label = false, linewidth = 2)
annotate!(-1.5, -13.5, ("wrap_array(u_ode, semi)", 10, :black, :center))
annotate!(-2.2, -15.2,
          ("to simplify processing, it wraps the 1D array u_ode created in the allocate_coefficients function in
a multidimensional one, where dimensions are responsible for variables, nodes on each axis and
elements", 9, :black, :left))
plot!([-2.4, -2.2], [-13.5, -13.5], arrow = true, color = :black, linewidth = 2, label = "")

plot!(Shape([(-1.1, -17.5), (1.7, -17.5), (1.7, -19.5), (-1.1, -19.5)]),
      linecolor = "black", fillcolor = "white", label = false, linewidth = 2)
annotate!(-1.05, -18.5,
          ("compute_coefficients!(u, initial_conditions, t, mesh,
                      equations, solver, cache)", 10, :black, :left))
annotate!(-1.1, -20.5,
          ("applies an initial conditions to each node of each element for each variable,
saves in the wrapped u_ode", 9, :black, :left))
plot!([-2.4, -1.1], [-18.5, -18.5], arrow = true, color = :black, linewidth = 2, label = "")
plot!([-2.4, -1.1], [-18.5, -17.5], arrow = true, color = :black, linewidth = 2, label = "")
plot!([-2.4, -1.1], [-18.5, -19.5], arrow = true, color = :black, linewidth = 2, label = "")
annotate!(-2.4, -20.2, ("specialized for solver
type and dimensionality
of mesh", 9, :black, :left))

savefig("./semidiscretize")
