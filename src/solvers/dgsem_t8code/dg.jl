@muladd begin
#! format: noindent

# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache_hyperbolic(mesh::T8codeMesh, equations::AbstractEquations,
                                 dg::DG, ::Any, ::Type{uEltype}) where {uEltype <: Real}
    count_required_surfaces!(mesh)

    elements = init_elements(mesh, equations, dg.basis, uEltype)
    interfaces = init_interfaces(mesh, equations, dg.basis, elements)
    boundaries = init_boundaries(mesh, equations, dg.basis, elements)
    mortars = init_mortars(mesh, equations, dg.basis, elements)

    fill_mesh_info!(mesh, interfaces, mortars, boundaries,
                    mesh.boundary_names)

    # Container cache
    cache = (; elements, interfaces, boundaries, mortars)

    # Add Volume-Integral cache
    cache = (; cache...,
             create_cache_all_volume_integrals(mesh, equations, dg.volume_integral,
                                               dg, cache, uEltype)...)
    # Add Mortar cache
    cache = (; cache..., create_cache_mortar(mesh, equations, dg.mortar, uEltype)...)
    # Add the performance settings of the GPU and KernelAbstractions.jl kernels, which
    # can be changed via `trixi_adapt`. Creating them together with the `cache` ensures
    # that they are available for all backends, e.g., the KernelAbstractions.jl CPU
    # backend without calling `trixi_adapt`.
    cache = (; cache..., tuning = default_tuning())

    return cache
end

include("containers.jl")
include("containers_2d.jl")
include("containers_3d.jl")

include("containers_parallel.jl")

include("dg_parallel.jl")
end # @muladd
