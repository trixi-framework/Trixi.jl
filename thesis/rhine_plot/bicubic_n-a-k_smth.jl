# Include packages
using Trixi
using Plots
using Downloads: download

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

# Downlowad data from gist
Rhine_data = download("https://gist.githubusercontent.com/maxbertrand1996/a30db4dc9f5427c78160321d75a08166/raw/fa53ceb39ac82a6966cbb14e1220656cf7f97c1b/Rhine_data_2D_40.txt")

# Define B-spline structure
bicub_struct = bicubic_b_spline(Rhine_data; smoothing_factor=9999, end_condition="not-a-knot")
# Define B-spline interpolation function
bicub_func(x,y) = spline_interpolation(bicub_struct, x, y)

# Define interpolation points
n = 100
x_int_pts = Vector(LinRange(bicub_struct.x[1], bicub_struct.x[end], n))
y_int_pts = Vector(LinRange(bicub_struct.y[1], bicub_struct.y[end], n))

# Get interpolated matrix
z_int_pts = fill_sol_mat(bicub_func, x_int_pts, y_int_pts)

# Plotting
pgfplotsx()
plot(x_int_pts, y_int_pts, z_int_pts, st =:surface, camera=(-30,30),
     xlabel="ETRS89 East", ylabel="ETRS89 North", zlabel="DHHN2016 Height",
     label="Bottom topography", 
     title="Bicubic B-spline interpolation with not-a-knot end condition
            and smoothing")