################################################################################
# This elixir shows the influence of a smoothing factor λ when applying it     #
# to the cubic spline interpolation with not-a-knot end_condition condition    #
# on a cross section of the Rhine river bed where we are given 100 data points #
################################################################################

using Trixi
using Plots

# smoothing factor
λ = 10.0

# Download data
spline_data = download("https://gist.githubusercontent.com/maxbertrand1996/2fe7caf9b0bfbfd6474ce6aed5e90e34/raw/39a3708e2cf6a82081f93d5f8b02f2a1ff733087/Rhine_data_1D_100.txt")

# Call the spline structure
spline        = cubic_b_spline(spline_data; end_condition = "not-a-knot")
spline_smooth = cubic_b_spline(spline_data; end_condition = "not-a-knot", smoothing_factor = λ)

# Call the spline functions
spline_func(x)        = spline_interpolation(spline       , x)
spline_func_smooth(x) = spline_interpolation(spline_smooth, x)

# Define calculation points
x_calc = Vector(LinRange(1, 100, 1000))

# Plot
pyplot()
scatter(spline.x, spline.y, label = "interpolation points")
plot!(x_calc, spline_func.(x_calc), label = "cubic spline interpolation, not-a-knot end_condition")
plot!(x_calc, spline_func_smooth.(x_calc), 
      label = "cubic spline interpolation, not-a-knot end_condition with smoothing factor $λ")