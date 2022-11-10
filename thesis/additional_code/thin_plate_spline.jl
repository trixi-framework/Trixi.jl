#################################################################
# This script creates a plot of a parametric surface using the  #
# bicubic B-spline functionalities implemented in Trixi through #
# arbitrary fit knots for the free end condition with smoothing #
# factors 1 and 9999.                                           #
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
  for i in 1:n
    for j in 1:m
      # Evaluate spline functions
      # at given x,y values
      z[j,i] = f(x[i], y[j])
    end
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
tps_smth      = bicubic_b_spline(x_spline, y_spline, z_spline;
                                 smoothing_factor = 1)
tps_full_smth = bicubic_b_spline(x_spline, y_spline, z_spline; 
                                 smoothing_factor = 9999)

# Defining the B-spline interpolation functions for free end 
# and not-a-knot end_condition condition
func_tps_smth(x,y) = 
  spline_interpolation(tps_smth, x, y)
func_tps_full_smth(x,y) = 
  spline_interpolation(tps_full_smth , x, y)

# Interpolation points
x_int = collect(0:0.1:4.0)
y_int = collect(0:0.1:4.0)

# Interpolated values
z_int_tps_smth      = fill_sol_mat(func_tps_smth, x_int, y_int)
z_int_tps_full_smth = fill_sol_mat(func_tps_full_smth , x_int, 
                                   y_int)

# Opening plotting environement
pgfplotsx()

# Plot the values in a single plot
plot(x_int, y_int, z_int_tps_smth, st =:surface, 
     label="Thin plate spline with λ = 1")
wireframe!(x_int, y_int, z_int_tps_full_smth, 
           label = "Thin plate spline with λ = 9999")
scatter!(x_fit, y_fit, z_fit, label="Fit knots", 
         xlabel="x-axis", ylabel="y-axis", zlabel = "z-axis", 
         title="Comparison between different 
                smoothing degrees")