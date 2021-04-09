# construct the inverse Jacobian for a given element (this is agnostic as to the curvature)
# TODO: this could be adapted to reuse the routine already persent in CurveMesh
function calc_inverse_jacobian!(inverse_jacobian::AbstractArray{<:Any, 3}, element_id, X_xi, X_eta,
                                Y_xi, Y_eta)
  @. @views inverse_jacobian[:, :, element_id] = inv(X_xi[:, :, element_id] * Y_eta[:, :, element_id] -
                                                     X_eta[:, :, element_id] * Y_xi[:, :, element_id])
  return inverse_jacobian
end

include("curved_map_and_geom.jl")
include("straight_map_and_geom.jl")
