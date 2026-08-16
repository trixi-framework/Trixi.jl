# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Container data structure (structure-of-arrays style) for CG interfaces.
# In contrast to the DG interfaces, no solution values are stored since the
# elements are coupled by a direct stiffness summation, which only requires the
# connectivity of the elements.
struct CGInterfaceContainer2D
    neighbor_ids::Matrix{Int} # [leftright, interfaces]
    orientations::Vector{Int} # [interfaces]
end

@inline function ninterfaces(interfaces::CGInterfaceContainer2D)
    return length(interfaces.orientations)
end

# Create interface container and connect the elements. Since the mesh is
# conforming, `init_interfaces!` of the DG implementation can be reused.
function init_interfaces(cell_ids, mesh::TreeMesh2D,
                         elements::TreeElementContainer2D, cg::CGSEM)
    n_interfaces = count_required_interfaces(mesh, cell_ids)
    interfaces = CGInterfaceContainer2D(Matrix{Int}(undef, 2, n_interfaces),
                                        Vector{Int}(undef, n_interfaces))

    init_interfaces!(interfaces, elements, mesh)
    return interfaces
end
end # @muladd
