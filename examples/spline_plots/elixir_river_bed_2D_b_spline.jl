################################################################################
# This elixir compares the different b spline interpolation approaches on a    #
# 2D data set of the Rhine river bed.                                          #
# These approaches are:                                                        #
#   - b spline interpolation with free boundary condtition                     #
#   - b spline interpolation with free boundary condtition and smoothing       #
#   - b spline interpolation with not-a-knot boundary condtition               #
#   - b spline interpolation with not-a-knot boundary condtition and smoothing #
################################################################################

using Trixi
using Plots

# Function to get interpolated values
function calc_spline(b_spline_func, x_int, y_int)
    nx = length(x_int)
    ny = length(y_int)
        
    z_int = zeros(ny, nx)
        
    for i in 1:nx
        for j in 1:ny
        z_int[j,i] = b_spline_func(x_int[i], y_int[j])
        end
    end
        
    return z_int
end

# smoothing factor
λ = 0.1

# Fill structure
b_spline_free      = bicubic_b_spline(joinpath("examples","spline_plots","Rhine_data_2D_10.txt"); boundary = "free")
b_spline_free_smth = bicubic_b_spline(joinpath("examples","spline_plots","Rhine_data_2D_10.txt"); boundary = "free", smoothing_factor = λ)
b_spline_knot      = bicubic_b_spline(joinpath("examples","spline_plots","Rhine_data_2D_10.txt"); boundary = "not-a-knot")
b_spline_knot_smth = bicubic_b_spline(joinpath("examples","spline_plots","Rhine_data_2D_10.txt"); boundary = "not-a-knot", smoothing_factor = λ)

# Set up interpolation functions
b_spline_func_free(x,y)      = spline_interpolation(b_spline_free, x, y)
b_spline_func_free_smth(x,y) = spline_interpolation(b_spline_free_smth, x, y)
b_spline_func_knot(x,y)      = spline_interpolation(b_spline_knot, x, y)
b_spline_func_knot_smth(x,y) = spline_interpolation(b_spline_knot_smth, x, y)

# Calculation points
x_int = Vector(LinRange(b_spline_free.x[1], b_spline_free.x[end], 100))
y_int = Vector(LinRange(b_spline_free.y[1], b_spline_free.y[end], 100))

# Calculate interpolation values
z_free      = calc_spline(b_spline_func_free, x_int, y_int)
z_free_smth = calc_spline(b_spline_func_free_smth, x_int, y_int)
z_knot      = calc_spline(b_spline_func_knot, x_int, y_int)
z_knot_smth = calc_spline(b_spline_func_knot_smth, x_int, y_int)

pyplot()

p_free      = plot(x_int, y_int, z_free, st =:surface, title = "Free boundary")
p_free_smth = plot(x_int, y_int, z_free_smth, st =:surface, title = "Free boundary, smooth (λ = $λ)")
p_knot      = plot(x_int, y_int, z_knot, st =:surface, title = "Not-a-knot boundary")
p_knot_smth = plot(x_int, y_int, z_knot_smth, st =:surface, title = "Not-a-knot boundary, smooth (λ = $λ)")

plot(p_free, p_free_smth, p_knot, p_knot_smth, layout = (2,2), legend = false)