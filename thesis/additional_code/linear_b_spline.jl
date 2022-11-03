# Not in use

using Trixi
using Plots

x_val = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
y_val = [5.0, 4.0, 2.0, 6.0, 8.0, 5.0, 6.0]

b_spline        = linear_b_spline(x_val, y_val)
b_spline_smooth = linear_b_spline(x_val, y_val; smoothing_factor = .05)

b_spline_func(x)        = spline_interpolation(b_spline       , x)
b_spline_func_smooth(x) = spline_interpolation(b_spline_smooth, x)

x = Vector(LinRange(x_val[1], x_val[end], 700))

pyplot()
scatter(x_val, y_val, label = "interpolation points")
plot!(x, b_spline_func_smooth.(x), label = "linear b spline, smooth")
plot!(x, b_spline_func.(x)       , label = "linear b spline", legend =:bottomright)