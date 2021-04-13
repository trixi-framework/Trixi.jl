# construct the values of 1/Jacobian at each LGL point for a given element
# Note: This is agnostic as to whether or not the elemnt is curved
# TODO: this could be adapted to reuse the routine already persent in CurvedMesh
function calc_inverse_jacobian!(inverse_jacobian::AbstractArray{<:Any, 3}, element, X_xi, X_eta,
                                Y_xi, Y_eta)
  @. @views inverse_jacobian[:, :, element] = inv(X_xi[:, :, element] * Y_eta[:, :, element] -
                                                  X_eta[:, :, element] * Y_xi[:, :, element])
  return inverse_jacobian
end

include("curved_map_and_geom.jl")
include("straight_map_and_geom.jl")
