
#@muladd begin

"""
    finite_volume_nodes_weights(n_nodes::Integer, RealT = Float64)

Computes equidistant, cell-centered nodes ``x_j`` and weights ``w_j`` for the 
finite volume method on the reference interval ``[-1,1]``.
"""

struct UniformFiniteVolumeBasis{RealT, n_nodes, VectorT}
    nodes::VectorT
    weights::VectorT 
end

function UniformFiniteVolumeBasis(n_nodes::Integer, RealT = Float64)

    nodes = SVector{n_nodes, RealT}(-1+(2*i-1)/(n_nodes) for i in 1:n_nodes)
    weights = SVector{n_nodes, RealT}(2 /n_nodes for i in 1:n_nodes)

    return UniformFiniteVolumeBasis{RealT, n_nodes, typeof(nodes)}(nodes, weights)
end

const BlockFV = DG{Basis} where {Basis <: UniformFiniteVolumeBasis}

function BlockFV(n_nodes::Integer; 
               RealT = Float64,
               surface_flux = flux_lax_friedrichs)
    basis = UniformFiniteVolumeBasis(n_nodes, RealT)
     volume_integral = VolumeIntegralPureLGLFiniteVolume(surface_flux)
     surface_integral = SurfaceIntegralWeakForm(surface_flux)
    # `nothing` is passed as `mortar`
    return DG(basis, nothing, surface_integral, volume_integral)
end


#end # @muladd
