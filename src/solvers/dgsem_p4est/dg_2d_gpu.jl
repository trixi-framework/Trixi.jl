# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function apply_jacobian!(backend::Backend, du,
                         mesh::Union{StructuredMesh{2}, StructuredMeshView{2},
                                     UnstructuredMesh2D, P4estMesh{2}, P4estMeshView{2},
                                     T8codeMesh{2}},
                         equations, dg::DG, cache)
    nelements(dg, cache) == 0 && return nothing
    @unpack inverse_jacobian = cache.elements
    kernel! = apply_jacobian_KAkernel!(backend)
    kernel!(du, typeof(mesh), equations, dg, inverse_jacobian,
            ndrange = (nnodes(dg), nnodes(dg), nelements(dg, cache)))
end

@kernel function apply_jacobian_KAkernel!(du,
                                          MeshT::Type{<:Union{StructuredMesh{2},
                                                              StructuredMeshView{2},
                                                              UnstructuredMesh2D,
                                                              P4estMesh{2},
                                                              P4estMeshView{2},
                                                              T8codeMesh{2}}},
                                          equations, dg::DG, inverse_jacobian)
    i, j, element = @index(Global, NTuple)
    apply_jacobian_per_quadrature_node!(du, MeshT, equations, dg, inverse_jacobian,
                                        i, j, element)
end
end
