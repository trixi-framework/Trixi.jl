# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Magnitude |S| = sqrt(2 S_ij S_ij) of the strain rate tensor S from the velocity gradients
# `grad_x = (dv1/dx, dv2/dx, dv3/dx)` etc.
@inline function strain_rate_magnitude(grad_x, grad_y, grad_z)
    S12 = 0.5f0 * (grad_y[1] + grad_x[2])
    S13 = 0.5f0 * (grad_z[1] + grad_x[3])
    S23 = 0.5f0 * (grad_z[2] + grad_y[3])
    return sqrt(2 * (grad_x[1]^2 + grad_y[2]^2 + grad_z[3]^2 +
                 2 * (S12^2 + S13^2 + S23^2)))
end

# (C_s Delta)^2 in `element` with the filter width Delta = V^(1/3) / (N + 1) and the volume
# V of the element
Base.@propagate_inbounds function calc_subgrid_model_coefficients(u_transformed,
                                                                  gradients, element,
                                                                  subgrid_model::Smagorinsky,
                                                                  MeshT::Type{<:P4estMesh{3}},
                                                                  equations_parabolic::AbstractLESDiffusion{3},
                                                                  dg::DGSEM, cache)
    (; weights) = dg.basis
    (; inverse_jacobian) = cache.elements

    RealT = eltype(inverse_jacobian)
    volume = zero(RealT)
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        volume += weights[i] * weights[j] * weights[k] /
                  inverse_jacobian[i, j, k, element]
    end
    # `cbrt` is not used since it computes in `Float64` for `Float32`, which is not
    # supported on all GPUs
    delta = volume^(one(RealT) / 3) / nnodes(dg)

    return (subgrid_model.C_s * delta)^2
end

# Eddy viscosity mu_t = rho (C_s Delta)^2 |S|
@inline function eddy_viscosity(u, gradients, coefficient, subgrid_model::Smagorinsky,
                                equations_parabolic::AbstractLESDiffusion)
    S_abs = strain_rate_magnitude(convert_derivative_to_velocity_gradients(u, gradients,
                                                                           equations_parabolic)...)
    return convert_transformed_to_density(u, equations_parabolic) * coefficient * S_abs
end
end # @muladd
