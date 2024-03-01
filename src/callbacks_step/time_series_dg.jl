# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Store time series file for a DG solver
function save_time_series_file(time_series_callback,
                               mesh::Union{TreeMesh, UnstructuredMesh2D},
                               equations, dg::DG)
    @unpack (interval, solution_variables, variable_names,
    output_directory, filename, point_coordinates,
    point_data, time, step, time_series_cache) = time_series_callback
    n_points = length(point_data)

    h5open(joinpath(output_directory, filename), "w") do file
        # Add context information as attributes
        n_variables = length(variable_names)
        attributes(file)["ndims"] = ndims(mesh)
        attributes(file)["equations"] = get_name(equations)
        attributes(file)["polydeg"] = polydeg(dg)
        attributes(file)["n_vars"] = n_variables
        attributes(file)["n_points"] = n_points
        attributes(file)["interval"] = interval
        attributes(file)["variable_names"] = collect(variable_names)

        file["time"] = time
        file["timestep"] = step
        file["point_coordinates"] = point_coordinates
        for p in 1:n_points
            # Store data as 2D array for convenience
            file["point_data_$p"] = reshape(point_data[p], n_variables, length(time))
        end
    end
end
end # @muladd
