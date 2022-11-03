#################################################################
# This script creates a plot of a graph using the               #
# cubic B-spline functionalities implemented in Trixi through   #
# arbitrary fit knots for the free end and not-a-knot end       #
# condition.                                                    #
#################################################################

# Including necessary packages
using Trixi
using Plots

# Fit knots for x and y position
x_fit = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
y_fit = [5.0, 4.0, 2.0, 6.0, 8.0, 5.0, 6.0]

# Definig B-spline object for free end and not-a-knot 
# boundary condition
b_spline_free  = cubic_b_spline(x_fit, y_fit; boundary="free")
b_spline_nak   = cubic_b_spline(x_fit, y_fit; 
                                boundary="not-a-knot")

# Defining the B-spline interpolation functions 
# for free end and not-a-knot boundary condition
b_spline_func_free(t) = spline_interpolation(b_spline_free, t)
b_spline_func_nak(t)  = spline_interpolation(b_spline_nak , t)

# Defining points at whihc the B-spline functions 
# will be evaluated
t = Vector(LinRange(x_fit[1], x_fit[end], 700))

# Distance between fit knots
h = x_fit[2] - x_fit[1]

# x-axis position for fit knots
Q_x = vcat((x_fit[1]-h), x_fit, (x_fit[end]+h))

# y-axis position of fit knots 
# for free end and not-a-knot boundary condition
Q_free_y = b_spline_free.Q 
Q_nak_y = b_spline_nak.Q

# Opening plotting environement
pgfplotsx()

# Scatterplotting the Fit knots, the Control points for the 
# free end and not-a-knot condition
scatter(x_fit, y_fit, markershape=:auto, label = "Fit knots")
scatter!(Q_x, Q_free_y, markershape=:auto, 
         label = "Control points, free end condition")
scatter!(Q_x, Q_nak_y, markershape=:auto, 
         label = "Control points, not-a-knot end condition")
         
# Plotting the cubic B-splines with 
# free end and not-a-knot condition
plot!(t, b_spline_func_free.(t), linestyle=:solid, 
      label = "Cubic b spline, free end condion")
plot!(t, b_spline_func_nak.(t), xlabel="x-axis", ylabel="y-axis", 
      linestyle=:dash, 
      label = "Cubic b spline, not-a-knot end condition", 
      legend =:topleft, 
      title="Comparison between different end conditions")