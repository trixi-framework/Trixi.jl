# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# du .= zero(eltype(du)) doesn't scale when using multiple threads.
# See https://github.com/trixi-framework/Trixi.jl/pull/924 for a performance comparison.
function reset_du!(du, dg, cache)
    @threaded for element in eachelement(dg, cache)
        du[.., element] .= zero(eltype(du))
    end

    return nothing
end

function volume_jacobian(element, mesh::TreeMesh, cache)
    return inv(cache.elements.inverse_jacobian[element])^ndims(mesh)
end

@inline function get_inverse_jacobian(inverse_jacobian, mesh::TreeMesh,
                                      indices...)
    element = last(indices)
    return inverse_jacobian[element]
end

# Dimension agnostic, i.e., valid for all 1D, 2D, and 3D meshes
function create_cache(mesh, equations,
                      volume_integral::VolumeIntegralFluxDifferencing, dg::DG, uEltype)
    return NamedTuple()
end

# Dimension agnostic, i.e., valid for all 1D, 2D, and 3D meshes
function calc_volume_integral!(du, u, mesh,
                               nonconservative_terms, equations,
                               volume_integral::VolumeIntegralWeakForm,
                               dg::DGSEM, cache)
    @threaded for element in eachelement(dg, cache)
        weak_form_kernel!(du, u, element, mesh,
                          nonconservative_terms, equations,
                          dg, cache)
    end

    return nothing
end

# Dimension agnostic, i.e., valid for all 1D, 2D, and 3D meshes.
# For curved meshes averaging of the mapping terms, stored in `cache.elements.contravariant_vectors`, 
# is peeled apart from the evaluation of the physical fluxes in each Cartesian direction.
function calc_volume_integral!(du, u, mesh,
                               nonconservative_terms, equations,
                               volume_integral::VolumeIntegralFluxDifferencing,
                               dg::DGSEM, cache)
    @threaded for element in eachelement(dg, cache)
        flux_differencing_kernel!(du, u, element, mesh, nonconservative_terms,
                                  equations,
                                  volume_integral.volume_flux, dg, cache)
    end

    return nothing
end

# Dimension agnostic, i.e., valid for all 1D, 2D, and 3D meshes
function calc_volume_integral!(du, u, mesh,
                               nonconservative_terms, equations,
                               volume_integral::VolumeIntegralShockCapturingHG,
                               dg::DGSEM, cache)
    @unpack volume_flux_dg, volume_flux_fv, indicator = volume_integral

    # Calculate blending factors α: u = u_DG * (1 - α) + u_FV * α
    alpha = @trixi_timeit timer() "blending factors" indicator(u, mesh, equations, dg,
                                                               cache)

    # For `Float64`, this gives 1.8189894035458565e-12
    # For `Float32`, this gives 1.1920929f-5
    RealT = eltype(alpha)
    atol = max(100 * eps(RealT), eps(RealT)^convert(RealT, 0.75f0))
    @threaded for element in eachelement(dg, cache)
        alpha_element = alpha[element]
        # Clip blending factor for values close to zero (-> pure DG)
        dg_only = isapprox(alpha_element, 0, atol = atol)

        if dg_only
            flux_differencing_kernel!(du, u, element, mesh,
                                      nonconservative_terms, equations,
                                      volume_flux_dg, dg, cache)
        else
            # Calculate DG volume integral contribution
            flux_differencing_kernel!(du, u, element, mesh,
                                      nonconservative_terms, equations,
                                      volume_flux_dg, dg, cache, 1 - alpha_element)

            # Calculate FV volume integral contribution
            fv_kernel!(du, u, mesh, nonconservative_terms, equations, volume_flux_fv,
                       dg, cache, element, alpha_element)
        end
    end

    return nothing
end

# Dimension agnostic, i.e., valid for all 1D, 2D, and 3D meshes
function calc_volume_integral!(du, u, mesh,
                               nonconservative_terms, equations,
                               volume_integral::VolumeIntegralPureLGLFiniteVolume,
                               dg::DGSEM, cache)
    @unpack volume_flux_fv = volume_integral

    # Calculate LGL FV volume integral
    @threaded for element in eachelement(dg, cache)
        fv_kernel!(du, u, mesh, nonconservative_terms, equations, volume_flux_fv,
                   dg, cache, element, true)
    end

    return nothing
end

# Dimension agnostic, i.e., valid for all 1D, 2D, and 3D meshes
function calc_boundary_flux!(cache, t, boundary_condition::BoundaryConditionPeriodic,
                             mesh::TreeMesh, equations, surface_integral, dg::DG)
    @assert isempty(eachboundary(dg, cache))

    return nothing
end

# Indicators used for shock-capturing and AMR
include("indicators.jl")
include("indicators_1d.jl")
include("indicators_2d.jl")
include("indicators_3d.jl")

# Container data structures
include("containers.jl")

# Dimension-agnostic parallel setup
include("dg_parallel.jl")

# Helper structs for parabolic AMR
include("containers_viscous.jl")

# 1D DG implementation
include("dg_1d.jl")
include("dg_1d_parabolic.jl")

# 2D DG implementation
include("dg_2d.jl")
include("dg_2d_parallel.jl")
include("dg_2d_parabolic.jl")

# 3D DG implementation
include("dg_3d.jl")
include("dg_3d_parabolic.jl")

# Auxiliary functions that are specialized on this solver
# as well as specialized implementations used to improve performance
include("dg_2d_compressible_euler.jl")
include("dg_3d_compressible_euler.jl")

# Subcell limiters
include("subcell_limiters.jl")
include("subcell_limiters_2d.jl")
include("dg_2d_subcell_limiters.jl")
end # @muladd
