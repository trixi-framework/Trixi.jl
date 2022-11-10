#################################################################
# This script creates a plot of a graph using the               #
# cubic B-spline functionalities implemented in Trixi through   #
# arbitrary fit knots for the free end condition with no        #
# smoothing and smoothing factors 0.1 and 9999.                 #
#################################################################

# Including necessary packages
using Trixi
using Plots

# Fit knots for x and y position
x_fit = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
y_fit = [5.0, 4.0, 2.0, 6.0, 8.0, 5.0, 6.0]

# Definig B-spline objects with free end end_condition condition
# without smoothing, with smoothing and full smoothing
b_spline_no_smooth   = cubic_b_spline(x_fit, y_fit)
b_spline_smooth      = cubic_b_spline(x_fit, y_fit; 
                                      smoothing_factor = .1)
b_spline_full_smooth = cubic_b_spline(x_fit, y_fit; 
                                      smoothing_factor = 9999)

# Defining the corresponding B-spline interpolation functions
b_spline_no_smooth_func(t)   = 
  spline_interpolation(b_spline_no_smooth, t)
b_spline_smooth_func(t)      = 
  spline_interpolation(b_spline_smooth , t)
b_spline_full_smooth_func(t) = 
  spline_interpolation(b_spline_full_smooth, t)

# Defining points at whihc the B-spline functions 
# will be evaluated
t = Vector(LinRange(x_fit[1], x_fit[end], 700))

# Distance between fit knots
h = x_fit[2] - x_fit[1]

# x-axis position for fit knots
Q_x = vcat((x_fit[1]-h), x_fit, (x_fit[end]+h))

# Opening plotting environement
pgfplotsx()

# Scatterplotting the fit knots
scatter(x_fit, y_fit, markershape=:auto, label = "Fit knots")

# Plotting the cubic B-splines with free end condition
# and for different smoothing degrees
plot!(t, b_spline_no_smooth_func.(t), linestyle=:auto, 
      label = "No smoothing")
plot!(t, b_spline_smooth_func.(t), linestyle=:auto, 
      label = "Smoothing with λ = 0.1")
plot!(t, b_spline_full_smooth_func.(t), xlabel="x-axis", 
      ylabel="y-axis", linestyle=:auto, 
      label = "Smoothing with λ = 9999", legend =:topleft, 
      title="Comparison between different smoothing degrees")