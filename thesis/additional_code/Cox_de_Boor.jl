#################################################################
# This script creates a plot of the B-spline basis faunctions   #
# for degrees 0 to 3 using the Cox-de-Boor alogrithm            #
#################################################################

# Include the plotting package
using Plots

# Recursive function to calculate the B-spline functions
# as defined by the Cox-de-Boor algorithm
function B(i, d, t)
  
  t_vec = [0, 1, 2, 3, 4]

  if d == 0
    if t_vec[i] <= t < t_vec[i+1]
        return 1
    else
        return 0
    end

  else
    return (t-t_vec[i])/(t_vec[i+d]-t_vec[i])*B.(i, d-1, t)+
      (t_vec[i+d+1]-t)/(t_vec[i+d+1]-t_vec[i+1])*B.(i+1, d-1, t)
  end
  
end

# Interpolation points
t_int = Vector(LinRange(0,4,401))

# Open plotting environement
pgfplotsx()

# Plotting
plot(t_int[1:100], B.(1,0,t_int[1:100]), c=:red, label="Degree 0")
plot!(t_int[101:end], B.(1,0,t_int[101:end]), c=:red, label="")
plot!(t_int, B.(1,1,t_int), label="Degree 1")
plot!(t_int, B.(1,2,t_int), label="Degree 2")
plot!(t_int, B.(1,3,t_int), label="Degree 3", 
      xlabel="t" , ylabel="y", 
      title="B-spline functions of different degree")