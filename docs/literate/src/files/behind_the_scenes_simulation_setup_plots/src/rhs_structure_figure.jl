using Plots
plot(Shape([(-0.4, 0.5), (0.4, 0.5), (0.4, -0.5), (-0.4, -0.5)]), linecolor = "black",
     fillcolor = "white", label = false, linewidth = 2, size = (800, 600), showaxis = false,
     grid = false, xlim = (-2, 2), ylim = (-17.5, 3))
annotate!(0, 0, ("solve(...) call", 12, :black, :center))
plot!([0, 0], [-0.5, -2.5], arrow = true, color = :black, linewidth = 2, label = "")
plot!(Shape([(-1.2, 1), (1.2, 1), (1.2, -1), (-1.2, -1)]), linecolor = :green,
      fillcolor = :transparent, label = false, linewidth = 1, linestyle = :dash)
annotate!(0.8, 0, ("Trixi.jl setup", 12, :green, :center))

plot!(Shape([(-1.25, -2.5), (1.25, -2.5), (1.25, -3.5), (-1.25, -3.5)]),
      linecolor = "black", fillcolor = "white", label = false, linewidth = 2)
annotate!(0, -3, ("DiffEqBase.__solve(prob, alg, args...; kwargs...)", 12, :black, :center))
plot!([0, 0], [-3.5, -5.5], arrow = true, color = :black, linewidth = 2, label = "")
plot!(Shape([(-1.9, -2), (1.9, -2), (1.9, -10), (-1.9, -10)]), linecolor = :red,
      fillcolor = :transparent, label = false, linewidth = 1, linestyle = :dash)
annotate!(1.4, -6, ("OrdinaryDiffEq.jl", 12, :red, :center))

plot!(Shape([(-0.85, -5.5), (0.85, -5.5), (0.85, -6.5), (-0.85, -6.5)]),
      linecolor = "black", fillcolor = "white", label = false, linewidth = 2)
annotate!(0, -6, ("DiffEqBase.solve!(integrator)", 12, :black, :center))
plot!([0, 0], [-6.5, -8.5], arrow = true, color = :black, linewidth = 2, label = "")
plot!([0, -0.5], [-6.5, -8.5], arrow = true, color = :black, linewidth = 2, label = "")
plot!([0, 0.5], [-6.5, -8.5], arrow = true, color = :black, linewidth = 2, label = "")
annotate!(0.8, -7.5, ("Specialized for time \nintegration method", 9, :black, :center))

plot!(Shape([(-1.2, -8.5), (1.2, -8.5), (1.2, -9.5), (-1.2, -9.5)]), linecolor = "black",
      fillcolor = "white", label = false, linewidth = 2)
annotate!(0, -9, ("perform_step!(integrator, integrator.cache)", 12, :black, :center))
plot!([0, 0], [-9.5, -11.5], arrow = true, color = :black, linewidth = 2, label = "")
plot!([0, -0.5], [-9.5, -11.5], arrow = true, color = :black, linewidth = 2, label = "")
plot!([0, 0.5], [-9.5, -11.5], arrow = true, color = :black, linewidth = 2, label = "")
annotate!(0.8, -10.5, ("Specialized for \nsemidiscretization", 9, :black, :center))

plot!(Shape([(-0.95, -11.5), (0.95, -11.5), (0.95, -12.5), (-0.95, -12.5)]),
      linecolor = "black", fillcolor = "white", label = false, linewidth = 2)
annotate!(0, -12, ("Trixi.rhs!(du_ode, u_ode, semi, t)", 12, :black, :center))
plot!([0, 0], [-12.5, -14.5], arrow = true, color = :black, linewidth = 2, label = "")
plot!([0, -0.5], [-12.5, -14.5], arrow = true, color = :black, linewidth = 2, label = "")
plot!([0, 0.5], [-12.5, -14.5], arrow = true, color = :black, linewidth = 2, label = "")
annotate!(0.8, -13.5, ("Specialized for \nmesh and dg", 9, :black, :center))
plot!(Shape([(-1.7, -11), (1.7, -11), (1.7, -17), (-1.7, -17)]), linecolor = :blue,
      fillcolor = :transparent, label = false, linewidth = 1, linestyle = :dash)
annotate!(1.5, -14, ("Trixi.jl", 12, :blue, :center))

plot!(Shape([(-1.4, -14.5), (1.4, -14.5), (1.4, -16.5), (-1.4, -16.5)]),
      linecolor = "black", fillcolor = "white", label = false, linewidth = 2)
annotate!(0, -15.5,
          ("Trixi.rhs!(du, u, t, mesh, equations, \nboundary_conditions, source_terms, dg, cache)",
           12, :black, :center))

plot!([-2, 2], [2, 2], linecolor = "white", fillcolor = "white", label = false,
      linewidth = 2)
savefig("./rhs!")
