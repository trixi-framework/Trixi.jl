# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# This file contains callbacks that are performed on the surface like computation of
# pointwise surface forces.

"""
    AnalysisSurface{Semidiscretization, Variable}(semi,
                                                          boundary_symbol_or_boundary_symbols,
                                                          variable)

This struct is used to compute pointwise surface values of a quantity of interest `variable`
alongside the boundary/boundaries associated with particular name(s) given in `boundary_symbol`
or `boundary_symbols`.
For instance, this can be used to compute the surface pressure coefficient [`SurfacePressureCoefficient`](@ref) or
surface friction coefficient [`SurfaceFrictionCoefficient`](@ref) of e.g. an airfoil with the boundary
name `:Airfoil` in 2D.

- `semi::Semidiscretization`: Passed in to retrieve boundary condition information
- `boundary_symbol_or_boundary_symbols::Symbol|Vector{Symbol}`: Name(s) of the boundary/boundaries
  where the quantity of interest is computed
- `variable::Variable`: Quantity of interest, like lift or drag
"""
struct AnalysisSurface{Variable}
    indices::Vector{Int} # Indices in `boundary_condition_indices` where quantity of interest is computed
    variable::Variable # Quantity of interest, like lift or drag

    function AnalysisSurface(semi, boundary_symbol, variable)
        @unpack boundary_symbol_indices = semi.boundary_conditions
        indices = boundary_symbol_indices[boundary_symbol]

        return new{typeof(variable)}(indices, variable)
    end

    function AnalysisSurface(semi, boundary_symbols::Vector{Symbol}, variable)
        @unpack boundary_symbol_indices = semi.boundary_conditions
        indices = Vector{Int}()
        for name in boundary_symbols
            append!(indices, boundary_symbol_indices[name])
        end
        sort!(indices)

        return new{typeof(variable)}(indices, variable)
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
C_p \\coloneqq \\frac{ p - p_{p_\\infty}}
                        {0.5 \\rho_{\\infty} U_{\\infty}^2 L_{\\infty}}
```
based on the pressure distribution along a boundary.
Supposed to be used in conjunction with [`AnalysisSurface`](@ref)
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
C_f \\coloneqq \\frac{\\boldsymbol (\\tau_w  \\boldsymbol n) \\cdot \\boldsymbol b^\\perp}
                        {0.5 \\rho_{\\infty} U_{\\infty}^2 L_{\\infty}}
```
based on the wall shear stress vector ``\\tau_w`` along a boundary.
Supposed to be used in conjunction with [`AnalysisSurface`](@ref)
which stores the boundary information and semidiscretization.

- `rhoinf::Real`: Free-stream density
- `uinf::Real`: Free-stream velocity
- `linf::Real`: Reference length of geometry (e.g. airfoil chord length)
"""
function SurfaceFrictionCoefficient(rhoinf, uinf, linf)
    return SurfaceFrictionCoefficient(FlowState(rhoinf, uinf, linf))
end

function (pressure_coefficient::SurfacePressureCoefficient)(u, equations)
    p = pressure(u, equations)
    @unpack pinf = pressure_coefficient
    @unpack rhoinf, uinf, linf = pressure_coefficient.flow_state
    return (p - pinf) / (0.5 * rhoinf * uinf^2 * linf)
end

function (surface_friction::SurfaceFrictionCoefficient)(u, normal_direction, x, t,
                                                        equations_parabolic,
                                                        gradients_1, gradients_2)
    visc_stress_vector = viscous_stress_vector(u, normal_direction, equations_parabolic,
                                               gradients_1, gradients_2)
    @unpack rhoinf, uinf, linf = surface_friction.flow_state

    # Normalize as `normal_direction` is not necessarily a unit vector
    n = normal_direction / norm(normal_direction)
    n_perp = (-n[2], n[1])
    return (visc_stress_vector[1] * n_perp[1] + visc_stress_vector[2] * n_perp[2]) /
           (0.5 * rhoinf * uinf^2 * linf)
end

function analyze(surface_variable::AnalysisSurface, du, u, t,
                 mesh::P4estMesh{2},
                 equations, dg::DGSEM, cache)
    @unpack boundaries = cache
    @unpack surface_flux_values, node_coordinates, contravariant_vectors = cache.elements
    @unpack weights = dg.basis

    @unpack indices, variable = surface_variable

    dim = ndims(mesh)
    n_nodes = nnodes(dg)
    n_elements = length(indices)

    coords = Matrix{real(dg)}(undef, n_elements * n_nodes, dim) # physical coordinates of indices
    variables = Vector{real(dg)}(undef, n_elements * n_nodes) # variable values at indices

    # TODO - Decide whether to save element mean values too

    index_range = eachnode(dg)

    global_node_index = 1 # Keeps track of solution point number on the surface
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
            var = variable(u_node, equations)

            coords[global_node_index, 1] = x[1]
            coords[global_node_index, 2] = x[2]
            variables[global_node_index] = var
            i_node += i_node_step
            j_node += j_node_step
            global_node_index += 1
        end
    end
    # TODO - Sort coords, variables increasing x?
    mkpath("out")
    t_trunc = @sprintf("%.3f", t)
    filename = varname(variable) * "_" * t_trunc * ".txt"
    # TODO - Should we start with a bigger array and avoid hcat?
    writedlm(joinpath("out", filename), hcat(coords, variables))
end

function analyze(surface_variable::AnalysisSurface{Variable},
                 du, u, t, mesh::P4estMesh{2},
                 equations, equations_parabolic,
                 dg::DGSEM, cache,
                 cache_parabolic) where {Variable <: VariableViscous}
    @unpack boundaries = cache
    @unpack surface_flux_values, node_coordinates, contravariant_vectors = cache.elements
    @unpack weights = dg.basis

    @unpack indices, variable = surface_variable

    dim = ndims(mesh)
    n_nodes = nnodes(dg)
    n_elements = length(indices)

    coords = Matrix{real(dg)}(undef, n_elements * n_nodes, dim) # physical coordinates of indices
    variables = Vector{real(dg)}(undef, n_elements * n_nodes) # variable values at indices

    # TODO - Decide whether to save element mean values too

    # Additions for parabolic
    @unpack viscous_container = cache_parabolic
    @unpack gradients = viscous_container

    gradients_x, gradients_y = gradients

    surface_integral = zero(eltype(u))
    index_range = eachnode(dg)
    global_node_index = 1 # Keeps track of solution point number on the surface
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
            # Extract normal direction at nodes which points from the elements outwards,
            # i.e., *into* the structure.
            normal_direction = get_normal_direction(direction, contravariant_vectors,
                                                    i_node, j_node,
                                                    element)

            gradients_1 = get_node_vars(gradients_x, equations_parabolic, dg, i_node,
                                        j_node, element)
            gradients_2 = get_node_vars(gradients_y, equations_parabolic, dg, i_node,
                                        j_node, element)

            # Integral over whole boundary surface
            var = variable(u_node, normal_direction, x, t, equations_parabolic,
                           gradients_1, gradients_2)

            coords[global_node_index, 1] = x[1]
            coords[global_node_index, 2] = x[2]
            variables[global_node_index] = var
            i_node += i_node_step
            j_node += j_node_step
            global_node_index += 1
        end
    end
    # TODO - Sort coords, variables increasing x?
    mkpath("out")
    t_trunc = @sprintf("%.3f", t)
    filename = varname(variable) * "_" * t_trunc * ".txt"
    # TODO - Should we start with a bigger array and avoid hcat?
    writedlm(joinpath("out", filename), hcat(coords, variables))
end

varname(::Any) = @assert false "Surface variable name not assigned" # This makes sure default behaviour is not overwriting
varname(pressure_coefficient::SurfacePressureCoefficient) = "CP_x"
varname(friction_coefficient::SurfaceFrictionCoefficient) = "CF_x"

function pretty_form_ascii(::AnalysisSurface{<:SurfacePressureCoefficient{<:Any}})
    "CP(x)"
end
function pretty_form_utf(::AnalysisSurface{<:SurfacePressureCoefficient{<:Any}})
    "CP(x)"
end

function pretty_form_ascii(::AnalysisSurface{<:SurfaceFrictionCoefficient{<:Any}})
    "CF(x)"
end
function pretty_form_utf(::AnalysisSurface{<:SurfaceFrictionCoefficient{<:Any}})
    "CF(x)"
end
end # muladd
