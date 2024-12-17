# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# This file contains callbacks that are performed on the surface like computation of
# pointwise surface forces.

"""
    AnalysisSurfacePointwise{Variable, NBoundaries}(boundary_symbol_or_boundary_symbols,
                                                    variable, output_directory = "out")

This struct is used to compute pointwise surface values of a quantity of interest `variable`
alongside the boundary/boundaries associated with particular name(s) given in `boundary_symbol`
or `boundary_symbols`.
For instance, this can be used to compute the surface pressure coefficient [`SurfacePressureCoefficient`](@ref) or
surface friction coefficient [`SurfaceFrictionCoefficient`](@ref) of e.g. an airfoil with the boundary
symbol `:Airfoil` in 2D.

- `boundary_symbols::NTuple{NBoundaries, Symbol}`: Name(s) of the boundary/boundaries
  where the quantity of interest is computed
- `variable::Variable`: Quantity of interest, like lift or drag
- `output_directory = "out"`: Directory where the pointwise value files are stored.
"""
struct AnalysisSurfacePointwise{Variable, NBoundaries}
    variable::Variable # Quantity of interest, like lift or drag
    boundary_symbols::NTuple{NBoundaries, Symbol} # Name(s) of the boundary/boundaries
    output_directory::String

    function AnalysisSurfacePointwise(boundary_symbols::NTuple{NBoundaries, Symbol},
                                      variable,
                                      output_directory = "out") where {NBoundaries}
        return new{typeof(variable), NBoundaries}(variable, boundary_symbols,
                                                  output_directory)
    end
end

struct FlowState{RealT <: Real}
    rhoinf::RealT
    uinf::RealT
    linf::RealT
end

struct SurfacePressureCoefficient{RealT <: Real}
    pinf::RealT # Free stream pressure
    flow_state::FlowState{RealT}
end

struct SurfaceFrictionCoefficient{RealT <: Real} <: VariableViscous
    flow_state::FlowState{RealT}
end

"""
    SurfacePressureCoefficient(pinf, rhoinf, uinf, linf)

Compute the surface pressure coefficient
```math
C_p \\coloneqq \\frac{p - p_{p_\\infty}}
                     {0.5 \\rho_{\\infty} U_{\\infty}^2 L_{\\infty}}
```
based on the pressure distribution along a boundary.
Supposed to be used in conjunction with [`AnalysisSurfacePointwise`](@ref)
which stores the boundary information and semidiscretization.

- `pinf::Real`: Free-stream pressure
- `rhoinf::Real`: Free-stream density
- `uinf::Real`: Free-stream velocity
- `linf::Real`: Reference length of geometry (e.g. airfoil chord length)
"""
function SurfacePressureCoefficient(pinf, rhoinf, uinf, linf)
    return SurfacePressureCoefficient(pinf, FlowState(rhoinf, uinf, linf))
end

"""
SurfaceFrictionCoefficient(rhoinf, uinf, linf)

Compute the surface skin friction coefficient
```math
C_f \\coloneqq \\frac{\\boldsymbol \\tau_w  \\boldsymbol n^\\perp}
                     {0.5 \\rho_{\\infty} U_{\\infty}^2 L_{\\infty}}
```
based on the wall shear stress vector ``\\tau_w`` along a boundary.
Supposed to be used in conjunction with [`AnalysisSurfacePointwise`](@ref)
which stores the boundary information and semidiscretization.

- `rhoinf::Real`: Free-stream density
- `uinf::Real`: Free-stream velocity
- `linf::Real`: Reference length of geometry (e.g. airfoil chord length)
"""
function SurfaceFrictionCoefficient(rhoinf, uinf, linf)
    return SurfaceFrictionCoefficient(FlowState(rhoinf, uinf, linf))
end

# Compute local pressure coefficient.
# Works for both purely hyperbolic and hyperbolic-parabolic systems.
# C_p(x) = (p(x) - p_inf) / (0.5 * rho_inf * u_inf^2 * l_inf)
function (pressure_coefficient::SurfacePressureCoefficient)(u, equations)
    p = pressure(u, equations)
    @unpack pinf = pressure_coefficient
    @unpack rhoinf, uinf, linf = pressure_coefficient.flow_state
    return (p - pinf) / (0.5 * rhoinf * uinf^2 * linf)
end

# Compute local friction coefficient.
# Works only in conjunction with a hyperbolic-parabolic system.
# C_f(x) = (tau_w(x) * n_perp(x)) / (0.5 * rho_inf * u_inf^2 * l_inf)
function (surface_friction::SurfaceFrictionCoefficient)(u, normal_direction, x, t,
                                                        equations_parabolic,
                                                        gradients_1, gradients_2)
    viscous_stress_vector_ = viscous_stress_vector(u, normal_direction,
                                                   equations_parabolic,
                                                   gradients_1, gradients_2)
    @unpack rhoinf, uinf, linf = surface_friction.flow_state

    # Normalize as `normal_direction` is not necessarily a unit vector
    n = normal_direction / norm(normal_direction)
    # Tangent vector = perpendicular vector to normal vector
    t = (-n[2], n[1])
    return (viscous_stress_vector_[1] * t[1] +
            viscous_stress_vector_[2] * t[2]) /
           (0.5 * rhoinf * uinf^2 * linf)
end

# Compute and save to disk a space-dependent `surface_variable`.
# For the purely hyperbolic, i.e., non-parabolic case, this is for instance 
# the pressure coefficient `SurfacePressureCoefficient`.
# The boundary/boundaries along which this quantity is to be integrated is determined by
# `boundary_symbols`, which is retrieved from `surface_variable`.
function analyze(surface_variable::AnalysisSurfacePointwise, du, u, t,
                 mesh::P4estMesh{2},
                 equations, dg::DGSEM, cache, semi, iter)
    @unpack boundaries = cache
    @unpack surface_flux_values, node_coordinates, contravariant_vectors = cache.elements
    @unpack weights = dg.basis

    @unpack variable, boundary_symbols = surface_variable
    @unpack boundary_symbol_indices = semi.boundary_conditions
    indices = get_boundary_indices(boundary_symbols, boundary_symbol_indices)

    dim = ndims(mesh)
    n_nodes = nnodes(dg)
    n_elements = length(indices)

    coordinates = Matrix{real(dg)}(undef, n_elements * n_nodes, dim) # physical coordinates of indices
    values = Vector{real(dg)}(undef, n_elements * n_nodes) # variable values at indices

    index_range = eachnode(dg)

    global_node_counter = 1 # Keeps track of solution point number on the surface
    for boundary in indices
        element = boundaries.neighbor_ids[boundary]
        node_indices = boundaries.node_indices[boundary]

        i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
        j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

        i_node = i_node_start
        j_node = j_node_start
        for node_index in index_range
            u_node = Trixi.get_node_vars(cache.boundaries.u, equations, dg, node_index,
                                         boundary)

            x = get_node_coords(node_coordinates, equations, dg, i_node, j_node,
                                element)
            value = variable(u_node, equations)

            coordinates[global_node_counter, 1] = x[1]
            coordinates[global_node_counter, 2] = x[2]
            values[global_node_counter] = value

            i_node += i_node_step
            j_node += j_node_step
            global_node_counter += 1
        end
    end

    # Save to disk
    save_pointwise_file(surface_variable.output_directory, varname(variable),
                        coordinates, values, t, iter)
end

# Compute and save to disk a space-dependent `surface_variable`.
# For the purely hyperbolic-parabolic case, this is may be for instance 
# the surface skin fricition coefficient `SurfaceFrictionCoefficient`.
# The boundary/boundaries along which this quantity is to be integrated is determined by
# `boundary_symbols`, which is retrieved from `surface_variable`.
function analyze(surface_variable::AnalysisSurfacePointwise{Variable},
                 du, u, t, mesh::P4estMesh{2},
                 equations, equations_parabolic,
                 dg::DGSEM, cache, semi,
                 cache_parabolic, iter) where {Variable <: VariableViscous}
    @unpack boundaries = cache
    @unpack surface_flux_values, node_coordinates, contravariant_vectors = cache.elements
    @unpack weights = dg.basis

    @unpack variable, boundary_symbols = surface_variable
    @unpack boundary_symbol_indices = semi.boundary_conditions
    indices = get_boundary_indices(boundary_symbols, boundary_symbol_indices)

    dim = ndims(mesh)
    n_nodes = nnodes(dg)
    n_elements = length(indices)

    coordinates = Matrix{real(dg)}(undef, n_elements * n_nodes, dim) # physical coordinates of indices
    values = Vector{real(dg)}(undef, n_elements * n_nodes) # variable values at indices

    # Additions for parabolic
    @unpack viscous_container = cache_parabolic
    @unpack gradients = viscous_container

    gradients_x, gradients_y = gradients

    index_range = eachnode(dg)
    global_node_counter = 1 # Keeps track of solution point number on the surface
    for boundary in indices
        element = boundaries.neighbor_ids[boundary]
        node_indices = boundaries.node_indices[boundary]
        direction = indices2direction(node_indices)

        i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
        j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

        i_node = i_node_start
        j_node = j_node_start
        for node_index in index_range
            u_node = Trixi.get_node_vars(cache.boundaries.u, equations, dg, node_index,
                                         boundary)

            x = get_node_coords(node_coordinates, equations, dg, i_node, j_node,
                                element)

            gradients_1 = get_node_vars(gradients_x, equations_parabolic, dg,
                                        i_node, j_node, element)
            gradients_2 = get_node_vars(gradients_y, equations_parabolic, dg,
                                        i_node, j_node, element)

            # Extract normal direction at nodes which points from the 
            # fluid cells *outwards*, i.e., *into* the structure.
            normal_direction = get_normal_direction(direction, contravariant_vectors,
                                                    i_node, j_node, element)

            # Integral over whole boundary surface
            value = variable(u_node, normal_direction, x, t, equations_parabolic,
                             gradients_1, gradients_2)

            coordinates[global_node_counter, 1] = x[1]
            coordinates[global_node_counter, 2] = x[2]
            values[global_node_counter] = value

            i_node += i_node_step
            j_node += j_node_step
            global_node_counter += 1
        end
    end

    # Save to disk
    save_pointwise_file(surface_variable.output_directory, varname(variable),
                        coordinates, values, t, iter)
end

varname(::Any) = @assert false "Surface variable name not assigned" # This makes sure default behaviour is not overwriting
varname(pressure_coefficient::SurfacePressureCoefficient) = "CP_x"
varname(friction_coefficient::SurfaceFrictionCoefficient) = "CF_x"

# Helper function that saves a space-dependent quantity `values`
# at every solution/quadrature point `coordinates` at 
# time `t` and iteration `iter` to disk.
# The file is written to the `output_directory` with name `varname` in HDF5 (.h5) format.
# The latter two are retrieved from the `surface_variable`,
# an instantiation of `AnalysisSurfacePointwise`.
function save_pointwise_file(output_directory, varname, coordinates, values, t, iter)
    n_points = length(values)

    filename = joinpath(output_directory, varname) * @sprintf("_%06d.h5", iter)

    h5open(filename, "w") do file
        # Add context information as attributes
        attributes(file)["n_points"] = n_points
        attributes(file)["variable_name"] = varname

        file["time"] = t
        file["timestep"] = iter
        file["point_coordinates"] = coordinates
        file["point_data"] = values
    end
end

function pretty_form_ascii(::AnalysisSurfacePointwise{<:SurfacePressureCoefficient{<:Any}})
    "CP(x)"
end
function pretty_form_utf(::AnalysisSurfacePointwise{<:SurfacePressureCoefficient{<:Any}})
    "CP(x)"
end

function pretty_form_ascii(::AnalysisSurfacePointwise{<:SurfaceFrictionCoefficient{<:Any}})
    "CF(x)"
end
function pretty_form_utf(::AnalysisSurfacePointwise{<:SurfaceFrictionCoefficient{<:Any}})
    "CF(x)"
end
end # muladd
