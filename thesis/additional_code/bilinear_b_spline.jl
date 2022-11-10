# Currently not in use

using Trixi
using Plots

function fill_sol_mat(f, x, y)
    
    n = length(x)
    m = length(y)

    z = zeros(n,m)

    for i in 1:n
        for j in 1:m
            z[j,i] = f(x[i], y[j])
        end
    end

    return z
end

# Base grid
x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
y = [0.0, 
     1.0,
     2.0,
     3.0,
     4.0,
     5.0,
     6.0,
     7.0,
     8.0,
     9.0,
     10.0]
z = [0.0 1 5 7 3 4 3 4 2 6 7;
     4 6 3 7 3 4 8 4 2 9 5;
     2 7 3 7 2 9 5 3 4 2 7;
     8 4 6 7 3 2 4 8 3 1 6;
     6 7 6 7 2 7 2 7 5 6 5;
     5 7 5 9 5 3 2 1 4 5 6;
     6 7 6 7 5 4 6 7 3 8 1;
     5 7 5 3 1 5 7 6 7 6 8;
     4 5 6 7 8 9 3 5 4 3 5;
     5 4 9 4 7 5 6 1 4 5 3;
     5 4 6 7 8 9 5 4 3 1 2]

# Call spline structures
spline_bilin      = bilinear_b_spline(x,y,z)
spline_bilin_smth = bilinear_b_spline(x,y,z; smoothing_factor = 1)

# Create spline functions
func_bilin(x,y)      = spline_interpolation(spline_bilin, x, y)
func_bilin_smth(x,y) = spline_interpolation(spline_bilin_smth, x, y)

# Interpolation points
x_int = collect(0:0.1:10.0)
y_int = collect(0:0.1:10.0)

# Interpolated values
z_int_bilin      = fill_sol_mat(func_bilin, x_int, y_int)
z_int_bilin_smth = fill_sol_mat(func_bilin_smth, x_int, y_int)

# Plot base grid as linear interpolation and the interpolated values (simplified & extended)
pyplot()
p_base = plot(x, y , z, 
    st =:surface, xlabel = "x", ylabel = "y", zlabel = "z", title = "base grid")
p_bilin = plot(x_int, y_int, z_int_bilin, 
    st =:surface, xlabel = "x", ylabel = "y", zlabel = "z", title = "bilinear interpolation") 
p_smth  = plot(x_int, y_int, z_int_bilin_smth , 
    st =:surface, xlabel = "x", ylabel = "y", zlabel = "z", title = "bilinear interpolation with smoothing")

plot(p_base, p_bilin, p_smth, layout = (1,3), legend = false)