#################################################################
# 
#################################################################

# Include packages
using Trixi
using Plots
using Downloads: download

# Downlowad data from gist
Rhine_data = download("https://gist.githubusercontent.com/maxbertrand1996/a30db4dc9f5427c78160321d75a08166/raw/fa53ceb39ac82a6966cbb14e1220656cf7f97c1b/Rhine_data_2D_40.txt")

# Reading the data fom the file
file = open(Rhine_data)
lines = readlines(file)
close(file)

n = parse(Int64, lines[2])
m = parse(Int64, lines[4])

x_file = [parse(Float64, val) for val in lines[6:(5+n)]]
y_file = [parse(Float64, val) for val in lines[(7+n):(6+n+m)]]
z_tmp  = [parse(Float64, val) for val in lines[(8+n+m):end]]

z_file = Matrix(transpose(reshape(z_tmp, (n, m))))

# Setting x and y from the file data to get the values from 
# an arbitrary cross section
x = x_file
y = z_file[22,:]

# Define B-spline structure
cub_struct = cubic_b_spline(x, y; end_condition="not-a-knot", 
                            smoothing_factor=9999)
# Define B-spline interpolation function
cub_func(x) = spline_interpolation(cub_struct, x)

# Define interpolation points
n = 100
x_int_pts = Vector(LinRange(cub_struct.x[1], cub_struct.x[end], n))

# Get interpolated values
y_int_pts = cub_func.(x_int_pts)

# Plotting
pgfplotsx()
plot(x_int_pts, y_int_pts,
     xlabel="ETRS89 East", ylabel="DHHN2016 Height",
     label="Bottom topography", 
     title="Cubic B-spline interpolation with not-a-knot end 
            condition and smoothing")