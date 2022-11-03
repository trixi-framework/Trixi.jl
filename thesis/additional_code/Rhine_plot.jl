#################################################################
# This script creates a 3d plot of the DGM data of a section of #
# the Rhine river valley.                                       #
#################################################################

#################################################################
# This script creates a plot of a DGM data file using the 

# Include the plotting package
using Plots

# Read data from .xyz file
data = readlines("thesis\\additional_code\\dgm1_32_357_5646_1_nw.xyz")

# Get number of data points 
n = size(data)[1]

# Create empty Array to save values in
data_form = zeros((n,3))

# Iterate over all data points
for i in 1:n
  # Save formated data in data_form
  # Data has to be split by line and formated to Float64
  # from String
  data_form[i,:] = parse.(Float64, split(data[i]," ")[1:3])
end

# Open environment pyplot()
pyplot()

# Three dimensional plot of the data
plot(data_form[:,1], data_form[:,2], data_form[:,3], 
     st=:surface, camera=(-30,30), 
     xlabel="E", ylabel="N", zlabel="H")