using Plots
# level 0
plot(Shape([(2, 2), (2, -2), (-2, -2), (-2, 2)]), linecolor = "black", fillcolor = "white",
     label = "0th level, root", linewidth = 2, size = (600, 600))
# level 1
plot!(Shape([(1.97, 1.97), (1.97, 0), (0, 0), (0, 1.97)]), linecolor = "red",
      fillcolor = "white", label = "1st level", linewidth = 2)
plot!(Shape([(-1.97, 1.97), (0, 1.97), (0, 0), (-1.97, 0)]), linecolor = "red",
      fillcolor = "white", label = false, linewidth = 2)
plot!(Shape([(-1.97, 0), (0, 0), (0, -1.97), (-1.97, -1.97)]), linecolor = "red",
      fillcolor = "white", label = false, linewidth = 2)
plot!(Shape([(1.97, 0), (1.97, -1.97), (0, -1.97), (0, 0)]), linecolor = "red",
      fillcolor = "white", label = false, linewidth = 2)
# level 2
# upper right
plot!(Shape([(1.94, 1.94), (1.94, 1), (1, 1), (1, 1.94)]), linecolor = "blue",
      fillcolor = "white", label = "2nd level", linewidth = 2)
plot!(Shape([(0.03, 1.94), (1, 1.94), (1, 1), (0.03, 1)]), linecolor = "blue",
      fillcolor = "white", label = false, linewidth = 2)
plot!(Shape([(0.03, 0.03), (0.03, 1), (1, 1), (1, 0.03)]), linecolor = "blue",
      fillcolor = "white", label = false, linewidth = 2)
plot!(Shape([(1.94, 0.03), (1.94, 1), (1, 1), (1, 0.03)]), linecolor = "blue",
      fillcolor = "white", label = false, linewidth = 2)
# upper left
plot!(Shape([(-0.03, 1.94), (-0.03, 1), (-1, 1), (-1, 1.94)]), linecolor = "blue",
      fillcolor = "white", label = false, linewidth = 2)
plot!(Shape([(-1.94, 1.94), (-1, 1.94), (-1, 1), (-1.94, 1)]), linecolor = "blue",
      fillcolor = "white", label = false, linewidth = 2)
plot!(Shape([(-1.94, 0.03), (-1.94, 1), (-1, 1), (-1, 0.03)]), linecolor = "blue",
      fillcolor = "white", label = false, linewidth = 2)
plot!(Shape([(-0.03, 0.03), (-1, 0.03), (-1, 1), (-0.03, 1)]), linecolor = "blue",
      fillcolor = "white", label = false, linewidth = 2)
# lower left
plot!(Shape([(-0.03, -0.03), (-0.03, -1), (-1, -1), (-1, -0.03)]), linecolor = "blue",
      fillcolor = "white", label = false, linewidth = 2)
plot!(Shape([(-1.94, -1), (-1.94, -0.03), (-1, -0.03), (-1, -1)]), linecolor = "blue",
      fillcolor = "white", label = false, linewidth = 2)
plot!(Shape([(-1.94, -1.94), (-1.94, -1), (-1, -1), (-1, -1.94)]), linecolor = "blue",
      fillcolor = "white", label = false, linewidth = 2)
plot!(Shape([(-1, -1.94), (-1, -1), (-0.03, -1), (-0.03, -1.94)]), linecolor = "blue",
      fillcolor = "white", label = false, linewidth = 2)

savefig("./treemesh")
