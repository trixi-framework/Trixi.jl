############################################################################################
# This elixir compares the differences of the natural end_condition and the not-a-knot     #
# end_condition on a cross section of the Rhine river bed where we are given 10 data       # 
# points                                                                                   #
############################################################################################

using Downloads: download
using Trixi
using Plots

# Download spline data
spline_data = download("https://gist.githubusercontent.com/maxbertrand1996/b05a90e66025ee1ebddf444a32c3fa01/raw/62d2dd1dcf4a2cafb0bfa556c464722d74a3304c/Rhine_data_1D_10.txt")

# Call the spline structure
spline_nat  = cubic_b_spline(spline_data)
spline_knot = cubic_b_spline(spline_data; end_condition = "not-a-knot")

# Call the spline functions
spline_func_nat(x)  = spline_interpolation(spline_nat , x)
spline_func_knot(x) = spline_interpolation(spline_knot, x)

# Define calculation points
x_calc = Vector(LinRange(1, 10, 100))

# Plot
pyplot()
scatter(spline_nat.x, spline_nat.y, label = "interpolation points")
plot!(x_calc, spline_func_nat.(x_calc), 
      label = "cubic spline interpolation, natural end_condition")
plot!(x_calc, spline_func_knot.(x_calc), line =:dash, 
      label = "cubic spline interpolation, not-a-knot end_condition")