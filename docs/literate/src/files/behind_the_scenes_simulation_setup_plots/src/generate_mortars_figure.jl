using Plots

function min(coordinates::Vector{Tuple{Float64, Float64}}, i)
    min = coordinates[1][i]
    for j in coordinates
        if min > j[i]
            min = j[i]
        end
    end
    return min
end

function max(coordinates::Vector{Tuple{Float64, Float64}}, i)
    max = coordinates[1][i]
    for j in coordinates
        if max < j[i]
            max = j[i]
        end
    end
    return max
end

function nodes_project_four(coordinates::Vector{Tuple{Float64, Float64}})
    nodes_1d = [-1.0, -0.4472135954999579, 0.4472135954999579, 1.0]
    nodes_2d = Array{Float64, 2}(undef, 16, 2)
    for i in range(1, length(nodes_1d))
        for j in range(1, length(nodes_1d))
            nodes_2d[(i - 1) * 4 + j, :] = [nodes_1d[i], nodes_1d[j]]
        end
    end
    k_x = (max(coordinates, 1) - min(coordinates, 1)) / 2
    k_y = (max(coordinates, 2) - min(coordinates, 2)) / 2
    b_x = min(coordinates, 1)
    b_y = min(coordinates, 2)
    for i in range(1, Int(length(nodes_2d) / 2))
        nodes_2d[i, 1] = k_x * (nodes_2d[i, 1] + 1) + b_x
        nodes_2d[i, 2] = k_y * (nodes_2d[i, 2] + 1) + b_y
    end
    return nodes_2d
end

import Base.+
+(a::Tuple, b::Tuple) = a .+ b

# level 1
plot(Shape([(2, 0), (2, -2), (0, -2), (0, 0)] .+
           [(0.5, -0.75), (0.5, -0.75), (0.5, -0.75), (0.5, -0.75)]), linecolor = "black",
     fillcolor = "white", label = "elements", legend_position = :right, linewidth = 2,
     size = (800, 600), showaxis = false, grid = false, x_lims = (-3, 5.5),
     ylims = (-3.5, 3))

nodes_2d = nodes_project_four([(2, 0), (2, -2), (0, -2), (0, 0)] .+
                              [(0.5, -0.75), (0.5, -0.75), (0.5, -0.75), (0.5, -0.75)])
scatter!(nodes_2d[:, 1], nodes_2d[:, 2], color = "red", label = "Gauss-Lobatto nodes",
         markersize = 5)

plot!([-0.25, 0.125], [-1, -1], linecolor = "orange", fillcolor = "white",
      label = "mortars", linewidth = 3)
plot!([-0.25, 0.125], [-2.5, -2.5], linecolor = "orange", fillcolor = "white",
      label = false, linewidth = 3)
plot!([0.125, 0.125], [-1, -2.5], linecolor = "orange", fillcolor = "white", label = false,
      linewidth = 3)
plot!([0.125, 0.5], [-1.75, -1.75], linecolor = "orange", fillcolor = "white",
      label = false, linewidth = 3)
plot!([0.75, 0.75], [0, -0.325], linecolor = "orange", fillcolor = "white", label = false,
      linewidth = 3)
plot!([2.25, 2.25], [0, -0.325], linecolor = "orange", fillcolor = "white", label = false,
      linewidth = 3)
plot!([2.25, 0.75], [-0.325, -0.325], linecolor = "orange", fillcolor = "white",
      label = false, linewidth = 3)
plot!([1.5, 1.5], [-0.75, -0.325], linecolor = "orange", fillcolor = "white", label = false,
      linewidth = 3)

# level 2

# upper right
# 1
plot!(Shape([(2, 2), (2, 1), (1, 1), (1, 2)] .+
            [(0.75, 0.5), (0.75, 0.5), (0.75, 0.5), (0.75, 0.5)]), linecolor = "black",
      fillcolor = "white", label = false, linewidth = 2)

nodes_2d = nodes_project_four([(2, 2), (2, 1), (1, 1), (1, 2)] .+
                              [(0.75, 0.5), (0.75, 0.5), (0.75, 0.5), (0.75, 0.5)])
scatter!(nodes_2d[:, 1], nodes_2d[:, 2], color = "red", label = false, markersize = 5)

plot!([1.25, 1.75], [2, 2], linecolor = "blue", fillcolor = "white", label = "interfaces",
      linewidth = 3)
plot!([2.25, 2.25], [1.5, 1], linecolor = "blue", fillcolor = "white", label = false,
      linewidth = 3)

# 2
plot!(Shape([(0, 2), (1, 2), (1, 1), (0, 1)] .+
            [(0.25, 0.5), (0.25, 0.5), (0.25, 0.5), (0.25, 0.5)]), linecolor = "black",
      fillcolor = "white", label = false, linewidth = 2)

nodes_2d = nodes_project_four([(0, 2), (1, 2), (1, 1), (0, 1)] .+
                              [(0.25, 0.5), (0.25, 0.5), (0.25, 0.5), (0.25, 0.5)])
scatter!(nodes_2d[:, 1], nodes_2d[:, 2], color = "red", label = false, markersize = 5)

plot!([-0.25, 0.25], [2, 2], linecolor = "blue", fillcolor = "white", label = false,
      linewidth = 3)
plot!([0.75, 0.75], [1.5, 1], linecolor = "blue", fillcolor = "white", label = false,
      linewidth = 3)

# 3
plot!(Shape([(0, 0), (0, 1), (1, 1), (1, 0)] .+
            [(0.25, 0.0), (0.25, 0.0), (0.25, 0.0), (0.25, 0.0)]), linecolor = "black",
      fillcolor = "white", label = false, linewidth = 2)

nodes_2d = nodes_project_four([(0, 0), (0, 1), (1, 1), (1, 0)] .+
                              [(0.25, 0.0), (0.25, 0.0), (0.25, 0.0), (0.25, 0.0)])
scatter!(nodes_2d[:, 1], nodes_2d[:, 2], color = "red", label = false, markersize = 5)

plot!([1.25, 1.75], [0.5, 0.5], linecolor = "blue", fillcolor = "white", label = false,
      linewidth = 3)
plot!([-0.25, 0.25], [0.5, 0.5], linecolor = "blue", fillcolor = "white", label = false,
      linewidth = 3)

# 4
plot!(Shape([(2, 0), (2, 1), (1, 1), (1, 0)] .+
            [(0.75, 0.0), (0.75, 0.0), (0.75, 0.0), (0.75, 0.0)]), linecolor = "black",
      fillcolor = "white", label = false, linewidth = 2)

nodes_2d = nodes_project_four([(2, 0), (2, 1), (1, 1), (1, 0)] .+
                              [(0.75, 0.0), (0.75, 0.0), (0.75, 0.0), (0.75, 0.0)])
scatter!(nodes_2d[:, 1], nodes_2d[:, 2], color = "red", label = false, markersize = 5)

# upper left
# 1
plot!(Shape([(-0, 2), (-0, 1), (-1, 1), (-1, 2)] .+
            [(-0.25, 0.5), (-0.25, 0.5), (-0.25, 0.5), (-0.25, 0.5)]), linecolor = "black",
      fillcolor = "white", label = false, linewidth = 2)

nodes_2d = nodes_project_four([(-0, 2), (-0, 1), (-1, 1), (-1, 2)] .+
                              [(-0.25, 0.5), (-0.25, 0.5), (-0.25, 0.5), (-0.25, 0.5)])
scatter!(nodes_2d[:, 1], nodes_2d[:, 2], color = "red", label = false, markersize = 5)

plot!([-1.25, -1.75], [2, 2], linecolor = "blue", fillcolor = "white", label = false,
      linewidth = 3)
plot!([-0.75, -0.75], [1, 1.5], linecolor = "blue", fillcolor = "white", label = false,
      linewidth = 3)

# 2
plot!(Shape([(-2, 2), (-1, 2), (-1, 1), (-2, 1)] .+
            [(-0.75, 0.5), (-0.75, 0.5), (-0.75, 0.5), (-0.75, 0.5)]), linecolor = "black",
      fillcolor = "white", label = false, linewidth = 2)

nodes_2d = nodes_project_four([(-2, 2), (-1, 2), (-1, 1), (-2, 1)] .+
                              [(-0.75, 0.5), (-0.75, 0.5), (-0.75, 0.5), (-0.75, 0.5)])
scatter!(nodes_2d[:, 1], nodes_2d[:, 2], color = "red", label = false, markersize = 5)

plot!([-2.25, -2.25], [1, 1.5], linecolor = "blue", fillcolor = "white", label = false,
      linewidth = 3)

# 3
plot!(Shape([(-2, 0), (-2, 1), (-1, 1), (-1, 0)] .+
            [(-0.75, 0.0), (-0.75, 0.0), (-0.75, 0.0), (-0.75, 0.0)]), linecolor = "black",
      fillcolor = "white", label = false, linewidth = 2)

nodes_2d = nodes_project_four([(-2, 0), (-2, 1), (-1, 1), (-1, 0)] .+
                              [(-0.75, 0.0), (-0.75, 0.0), (-0.75, 0.0), (-0.75, 0.0)])
scatter!(nodes_2d[:, 1], nodes_2d[:, 2], color = "red", label = false, markersize = 5)

plot!([-1.75, -1.25], [0.5, 0.5], linecolor = "blue", fillcolor = "white", label = false,
      linewidth = 3)
plot!([-2.25, -2.25], [-0.5, 0], linecolor = "blue", fillcolor = "white", label = false,
      linewidth = 3)

# 4
plot!(Shape([(-0, 0), (-1, 0), (-1, 1), (-0, 1)] .+
            [(-0.25, 0.0), (-0.25, 0.0), (-0.25, 0.0), (-0.25, 0.0)]), linecolor = "black",
      fillcolor = "white", label = false, linewidth = 2)

nodes_2d = nodes_project_four([(-0, 0), (-1, 0), (-1, 1), (-0, 1)] .+
                              [(-0.25, 0.0), (-0.25, 0.0), (-0.25, 0.0), (-0.25, 0.0)])
scatter!(nodes_2d[:, 1], nodes_2d[:, 2], color = "red", label = false, markersize = 5)

plot!([-0.75, -0.75], [-0.5, 0], linecolor = "blue", fillcolor = "white", label = false,
      linewidth = 3)

# lower left
# 1
plot!(Shape([(-0, -0), (-0, -1), (-1, -1), (-1, -0)] .+
            [(-0.25, -0.5), (-0.25, -0.5), (-0.25, -0.5), (-0.25, -0.5)]),
      linecolor = "black", fillcolor = "white", label = false, linewidth = 2)

nodes_2d = nodes_project_four([(-0, -0), (-0, -1), (-1, -1), (-1, -0)] .+
                              [(-0.25, -0.5), (-0.25, -0.5), (-0.25, -0.5), (-0.25, -0.5)])
scatter!(nodes_2d[:, 1], nodes_2d[:, 2], color = "red", label = false, markersize = 5)

plot!([-1.75, -1.25], [-1, -1], linecolor = "blue", fillcolor = "white", label = false,
      linewidth = 3)
plot!([-0.75, -0.75], [-1.5, -2], linecolor = "blue", fillcolor = "white", label = false,
      linewidth = 3)

# 2
plot!(Shape([(-2, -1), (-2, 0), (-1, -0), (-1, -1)] .+
            [(-0.75, -0.5), (-0.75, -0.5), (-0.75, -0.5), (-0.75, -0.5)]),
      linecolor = "black", fillcolor = "white", label = false, linewidth = 2)

nodes_2d = nodes_project_four([(-2, -1), (-2, 0), (-1, -0), (-1, -1)] .+
                              [(-0.75, -0.5), (-0.75, -0.5), (-0.75, -0.5), (-0.75, -0.5)])
scatter!(nodes_2d[:, 1], nodes_2d[:, 2], color = "red", label = false, markersize = 5)

plot!([-2.25, -2.25], [-1.5, -2], linecolor = "blue", fillcolor = "white", label = false,
      linewidth = 3)

# 3
plot!(Shape([(-2, -2), (-2, -1), (-1, -1), (-1, -2)] .+
            [(-0.75, -1), (-0.75, -1), (-0.75, -1), (-0.75, -1)]), linecolor = "black",
      fillcolor = "white", label = false, linewidth = 2)

nodes_2d = nodes_project_four([(-2.0, -2.0), (-2.0, -1.0), (-1.0, -1.0), (-1.0, -2.0)] .+
                              [(-0.75, -1), (-0.75, -1), (-0.75, -1), (-0.75, -1)])
scatter!(nodes_2d[:, 1], nodes_2d[:, 2], color = "red", label = false, markersize = 5)

plot!([-1.75, -1.25], [-2.5, -2.5], linecolor = "blue", fillcolor = "white", label = false,
      linewidth = 3)

# 4
plot!(Shape([(-1, -2), (-1, -1), (-0, -1), (-0, -2)] .+
            [(-0.25, -1), (-0.25, -1), (-0.25, -1), (-0.25, -1)]), linecolor = "black",
      fillcolor = "white", label = false, linewidth = 2)

nodes_2d = nodes_project_four([(-1.0, -2.0), (-1.0, -1.0), (-0.0, -1.0), (-0.0, -2.0)] .+
                              [(-0.25, -1), (-0.25, -1), (-0.25, -1), (-0.25, -1)])
scatter!(nodes_2d[:, 1], nodes_2d[:, 2], color = "red", label = false, markersize = 5)

savefig("./mortars")
