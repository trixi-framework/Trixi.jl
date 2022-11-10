#################################################################
# This script creates a plot of a parametric surface using the  #
# bicubic B-spline functionalities implemented in Trixi through #
# arbitrary fit knots for the free end and not-a-knot end       #
# condition.                                                    #
#################################################################

# Including necessary packages
using Trixi
using Plots

# Helperfunction to fill the solution matrix
# Input parameters:
#  - f: spline function
#  - x: vector of x values
#  - y: vector of y values
function fill_sol_mat(f, x, y)
    
  # Get dimensions for solution matrix
  n = length(x)
  m = length(y)

  # Create empty solution matrix
  z = zeros(n,m)

  # Fill solution matrix
  for i in 1:n, j in 1:m
    # Evaluate spline functions
    # at given x,y values
    z[j,i] = f(x[i], y[j])
  end

  # Return solution matrix
  return z
end

# x, y and z values of the fit knots
x_fit = [0.0, 1.0, 2.0, 3.0, 4.0,
         0.0, 1.0, 2.0, 3.0, 4.0,
         0.0, 1.0, 2.0, 3.0, 4.0,
         0.0, 1.0, 2.0, 3.0, 4.0,
         0.0, 1.0, 2.0, 3.0, 4.0]

y_fit = [0.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 1.0, 1.0, 1.0,
         2.0, 2.0, 2.0, 2.0, 2.0,
         3.0, 3.0, 3.0, 3.0, 3.0,
         4.0, 4.0, 4.0, 4.0, 4.0]

z_fit = [0.0, 1.0, 5.0, 7.0, 3.0,
         4.0, 6.0, 3.0, 7.0, 3.0,
         2.0, 7.0, 3.0, 7.0, 2.0,
         8.0, 4.0, 6.0, 7.0, 3.0,
         6.0, 7.0, 6.0, 7.0, 2.0]

# x, y and z in required format
x_spline = x_fit[1:5]
y_spline = y_fit[1:5:end]
z_spline = Matrix(transpose(reshape(z_fit,(5,5))))

# Definig bicubic B-spline object for free end 
# and not-a-knot end_condition condition
spline_bicub_free = bicubic_b_spline(x_spline, y_spline, z_spline)
spline_bicub_nak  = bicubic_b_spline(x_spline, y_spline, z_spline; 
                                     end_condition = "not-a-knot")

# Defining the B-spline interpolation functions for free end 
# and not-a-knot end_condition condition
func_bicub_free(x,y) = 
  spline_interpolation(spline_bicub_free, x, y)
func_bicub_nak(x,y)  = 
  spline_interpolation(spline_bicub_nak , x, y)

# Interpolation points
x_int = collect(0:0.1:4.0)
y_int = collect(0:0.1:4.0)

# Interpolated values
z_int_bicub_free = fill_sol_mat(func_bicub_free, x_int, y_int)
z_int_bicub_nak  = fill_sol_mat(func_bicub_nak , x_int, y_int)

# Opening plotting environement
pyplot()

# Plot the values in a single plot
plot(x_int, y_int, z_int_bicub_free, st =:surface, 
     label="Free end condition")
wireframe!(x_int, y_int, z_int_bicub_nak, 
           label = "Not-a-knot end condition")
scatter!(x_fit, y_fit, z_fit, label="Fit knots", 
         xlabel="x-axis", ylabel="y-axis", zlabel = "z-axis", 
         title="Bicubic B-spline interpolation with different 
                end conditions")