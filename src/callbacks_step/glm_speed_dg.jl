# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function calc_dt_for_cleaning_speed(cfl::Real, mesh,
                                    equations::Union{AbstractIdealGlmMhdEquations,
                                                     AbstractIdealGlmMhdMulticomponentEquations},
                                    dg::DG, cache)
    # compute time step for GLM linear advection equation with c_h=1 for the DG discretization on
    # Cartesian meshes
    max_scaled_speed_for_c_h = maximum(cache.elements.inverse_jacobian) *
                               ndims(equations)
    # OBS! This depends on the implementation details of the StepsizeCallback and needs to be adapted
    #      as well if that callback changes.
    return cfl * 2 / (nnodes(dg) * max_scaled_speed_for_c_h)
end

function calc_dt_for_cleaning_speed(cfl::Real, mesh,
                                    equations::Union{AbstractIdealGlmMhdEquations,
                                                     AbstractIdealGlmMhdMulticomponentEquations},
                                    dg::DGMulti, cache)
    rd = dg.basis
    md = mesh.md

    # Compute time step for GLM linear advection equation with c_h=1 for a DGMulti discretization.
    # Copies implementation behavior of `calc_dt_for_cleaning_speed` for DGSEM discretizations.
    max_scaled_speed_for_c_h = inv(minimum(md.J)) * ndims(equations)

    # This mimics `max_dt` for `TreeMesh`, except that `nnodes(dg)` is replaced by
    # `polydeg+1`. This is because `nnodes(dg)` returns the total number of
    # multi-dimensional nodes for DGMulti solver types, while `nnodes(dg)` returns
    # the number of 1D nodes for `DGSEM` solvers.
    polydeg = rd.N
    return cfl * 2 / ((polydeg + 1) * max_scaled_speed_for_c_h)
end
end # @muladd
