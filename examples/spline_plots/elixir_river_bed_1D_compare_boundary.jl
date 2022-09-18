############################################################################################
# This elixir compares the differences of the natural boundary and the not-a-knot boundary #
# on a cross section of the Rhine river bed where we are given 100 data points             #
############################################################################################

using Trixi
using Plots

# Call the spline structure
spline_nat  = cubic_b_spline(joinpath("examples","spline_plots","Rhine_data_1D_10.txt"))
spline_knot = cubic_b_spline(joinpath("examples","spline_plots","Rhine_data_1D_10.txt"); boundary = "not-a-knot")

# Call the spline functions
spline_func_nat(x)  = spline_interpolation(spline_nat , x)
spline_func_knot(x) = spline_interpolation(spline_knot, x)

# Define calculation points
x_calc = Vector(LinRange(1, 10, 100))

# Plot
pyplot()
scatter(spline_nat.x, spline_nat.y                          , label = "interpolation points")
plot!(x_calc        , spline_func_nat.(x_calc)              , label = "cubic spline interpolation, natural boundary")
plot!(x_calc        , spline_func_knot.(x_calc), line =:dash, label = "cubic spline interpolation, not-a-knot boundary")