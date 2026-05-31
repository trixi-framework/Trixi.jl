
#@muladd begin

"""
    finite_volume_nodes_weights(n_nodes::Integer, RealT = Float64)

Computes equidistant, cell-centered nodes ``x_j`` and weights ``w_j`` for the 
finite volume method on the reference interval ``[0,1]``.
"""

struct UniformFiniteVolumebasis{RealT, n_nodes, VectorT} <: AbstractFiniteVolumeNodesWeights
    nodes::VectorT
    weights::VectorT 
end

function UniformFiniteVolumebasis(n_nodes::Integer, RealT = Float64)

    nodes = SVector{n_nodes, RealT}((2*i-1)/(2*n_nodes) for i in 1:n_nodes)
    weights = SVector{n_nodes, RealT}(1 /n_nodes for i in 1:n_nodes)

    return UniformFiniteVolumebasis{RealT, n_nodes, typeof(nodes)}(nodes, weights)
end

#end # @muladd
