# Plots for the tutorial "Behind the scenes of a simulation setup"

To create all the images from the directory of this README, one should execute the following 
command.
```julia
julia> include.(readdir("src"; join=true))
```
To create all images from a different directory, substitute "src" with the PATH to the "src" 
folder. The resulting images will be generated in your present directory as PNG files.

To generate a specific image, run the following command while replacing "path/to/src" and "file_name" with the appropriate values.
```julia
julia> include(joinpath("path/to/src", "file_name"))
```