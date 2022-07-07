################################################################################
# This elixir shows the influence of a smoothing factor λ when applying it     #
# to the cubic spline interpolation with not-a-knot boundary condition         #
# on a cross section of the Rhine river bed where we are given 100 data points #
################################################################################

using Trixi
using Plots

# smoothing factor
λ = 10.0

# Call the spline structure
spline        = cubic_spline(joinpath("examples","spline_plots","Rhine_data_1D_100.txt"); boundary = "not-a-knot")
spline_smooth = cubic_spline(joinpath("examples","spline_plots","Rhine_data_1D_100.txt"); boundary = "not-a-knot", smoothing_factor = λ)

# Call the spline functions
spline_func(x)        = spline_interpolation(spline       , x)
spline_func_smooth(x) = spline_interpolation(spline_smooth, x)

# Define calculation points
x_calc = Vector(LinRange(1, 100, 1000))

# Plot
pyplot()
scatter(spline.x, spline.y                   , label = "interpolation points")
plot!(x_calc    , spline_func.(x_calc)       , label = "cubic spline interpolation, not-a-knot boundary")
plot!(x_calc    , spline_func_smooth.(x_calc), label = "cubic spline interpolation, not-a-knot boundary with smoothing factor $λ")