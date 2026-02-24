using Plots

plot([-1, 1], [0, 0], linecolor = "black", label = "reference interval", size = (600, 200),
     ylim = (-0.5, 0.5), yaxis = false, grid = false)
scatter!([-1.0, -0.4472135954999579, 0.4472135954999579, 1.0], [0, 0, 0, 0], color = "red",
         label = "Gauss-Lobatto nodes")

savefig("./nodes")
