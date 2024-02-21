# Plots for the tutorial "Behind the scenes of a simulation setup"

To create all the images for the tutorial, execute the following command from the directory of this `README.md`:
```julia
pkg> activate .
julia> include.(readdir("src"; join=true))
```
To create all images from a different directory, substitute `"src"` with the path to the `src` 
folder. The resulting images will be generated in your current directory as PNG files.

To generate a specific image, run the following command while replacing `"path/to/src"` and `"file_name"` with the appropriate values:
```julia
pkg> activate .
julia> include(joinpath("path/to/src", "file_name"))
```