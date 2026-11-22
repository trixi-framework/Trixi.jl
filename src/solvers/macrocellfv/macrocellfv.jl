# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Macro-cell finite volume (MacroCellFV) solver: finite volumes with high-order face
# reconstruction on blocks of subcells.
#
# Each mesh element (macro-cell) holds an equidistant block of `n_cells` finite volume
# subcells per coordinate direction, stored with a `UniformFiniteVolumeBasis` as for
# `BlockFV`, whose containers, analysis, and visualization are reused. In contrast to
# `BlockFV`, which uses the subcell averages as face states and the volume and surface
# integrals of the DG solvers, the faces of each macro-cell are evaluated on a padded
# work array (pad) that holds the subcells of the macro-cell and `nhalo` halo subcells
# on each side. The pipeline of each RHS evaluation is, per macro-cell,
#   1. gather the subcells and the halos (from the neighbors or the ghost subcells of
#      physical boundaries) into the pad of the thread,
#   2. moment inversion (cell averages -> point values),
#   3. transformation to the reconstruction variables,
#   4. face reconstruction, numerical fluxes, and flux differences in one loop over the
#      subcell faces,
# followed by the completion of the flux differences of the first subcell of each
# macro-cell with the interface flux, which is computed only once, by the macro-cell on
# its left side.
#
# The pad holds the transformed state (reconstruction variables, or point values and
# averages) of a contiguous line of subcells across the macro-cell boundaries. The same
# kernel thus serves the faces inside a macro-cell and at its boundaries, for averages
# and point values alike, without index branching or repeated variable transformations
# in the stencils. Each thread owns one pad of `n_cells + 2 nhalo` subcells, so the
# memory of the pads does not grow with the number of elements.

"""
    MomentInversionIdentity()

Moment inversion that uses the cell averages directly as point values at the
cell centers. This is second-order accurate.

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
struct MomentInversionIdentity end

# Number of ghost subcells required by the moment inversion stencil
@inline nhalo(::MomentInversionIdentity) = 0

"""
    MomentInversionCentral()

Fourth-order moment inversion: the point value at the center of each subcell is
`ū - Σ_d Δ_d² ū / 24` with the second differences `Δ_d²` of the subcell averages `ū`
(conservative variables) along each coordinate direction `d`, exact for cubic
polynomials. Where the neighbors along a direction are missing (at the ends of the
padded work array), the one-sided second difference `2 ū_0 - 5 ū_1 + 4 ū_2 - ū_3` is
used, which keeps the order.

The point values are converted to the reconstruction variables (`cons2recon`)
pointwise, and the WENO reconstructions interpolate the point values to the face
midpoints with the weights for point values (see [`MomentInversionWENO`](@ref)).
This removes the second-order error of the reconstruction of nonlinear functions of
the averages (e.g., primitive variables).

If the central correction is applied across discontinuities, it can create
oscillations; at material interfaces, the oscillations of the conservative variables
perturb the pressure equilibrium. Use [`MomentInversionWENO`](@ref) for flows with
discontinuities. Point values outside of the admissible set (see
[`ReconstructionPositivity`](@ref)) are replaced by the averages.

Supported with the WENO reconstructions (also with the positivity limiter).

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
struct MomentInversionCentral end

"""
    MomentInversionWENO(; linear_weight = 1/6, epsilon = 1.0e-40)

Nonlinear (WENO) moment inversion: the point value at the center of each subcell is
the combination of the three third-order values from the subcell averages of the
stencils `(i-2, i-1, i)`, `(i-1, i, i+1)`, and `(i, i+1, i+2)` along each coordinate
direction, with the WENO-Z weights (Borges et al., 2008) for the symmetric positive
linear weights `(linear_weight, 1 - 2 linear_weight, linear_weight)` and the
smoothness indicators of Jiang and Shu (1996). The symmetric linear weights make the
combination exact for cubic polynomials (fourth order); the linear weights of the
fifth-order value would be negative. Next to discontinuities, the smooth one-sided
stencil is selected, so that the point values of a uniform state next to a jump (e.g.,
at a material interface at constant pressure and velocity) keep the uniform state.
Where the stencil does not fit into the available subcells (ends of the padded array),
the central inversion of [`MomentInversionCentral`](@ref) is used.

See [`MomentInversionCentral`](@ref) for the use of the point values.

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
struct MomentInversionWENO{RealT}
    linear_weight::RealT
    epsilon::RealT
end

function MomentInversionWENO(; linear_weight = 1 / 6, epsilon = 1.0e-40)
    0 < linear_weight < 0.5 ||
        throw(ArgumentError("`linear_weight` must be in (0, 1/2)"))
    linear_weight, epsilon = promote(linear_weight, epsilon)
    return MomentInversionWENO(linear_weight, epsilon)
end

const MomentInversionPoint = Union{MomentInversionCentral, MomentInversionWENO}

# The one-sided second differences at the ends of the available subcells keep the
# halo of the inversion at one subcell
@inline nhalo(::MomentInversionPoint) = 1

# `True()` if the moment inversion provides point values at the subcell centers
@inline has_point_values(::MomentInversionIdentity) = False()
@inline has_point_values(::MomentInversionPoint) = True()

# Correction `u_center - ū` of the point value at the center of the subcell `0` of
# `line` (conservative subcell averages `line(offset)` for `lo <= offset <= hi`, with
# `lo <= 0 <= hi`) along one coordinate direction
@inline function center_correction(::MomentInversionCentral, line, lo, hi)
    u_0 = line(0)
    if lo <= -1 && hi >= 1
        # (mirror-symmetric operations, see `weno5_smoothness`)
        return -((line(-1) + line(1)) - 2 * u_0) / 24
    elseif lo == 0
        # One-sided second difference, exact for cubic polynomials
        return -(2 * u_0 - 5 * line(1) + 4 * line(2) - line(3)) / 24
    else
        return -(2 * u_0 - 5 * line(-1) + 4 * line(-2) - line(-3)) / 24
    end
end

@inline function center_correction(inversion::MomentInversionWENO, line, lo, hi)
    if lo <= -2 && hi >= 2
        return map(@inline((a, b, c, d, e)->weno_center_correction(inversion, a, b, c,
                                                                   d, e)),
                   line(-2), line(-1), line(0), line(1), line(2))
    else
        return center_correction(MomentInversionCentral(), line, lo, hi)
    end
end

# WENO combination of the third-order center values `c - Δ²/24` of the stencils
# `(a, b, c)`, `(b, c, d)`, and `(c, d, e)`, minus the average `c`
@inline function weno_center_correction(inversion::MomentInversionWENO, a, b, c, d, e)
    RealT = typeof(c)
    beta0, beta1, beta2 = weno5_smoothness(a, b, c, d, e)
    epsilon = convert(RealT, inversion.epsilon)
    d_outer = convert(RealT, inversion.linear_weight)
    tau = abs(beta0 - beta2)
    alpha0 = d_outer * (1 + tau / (beta0 + epsilon))
    alpha1 = (1 - 2 * d_outer) * (1 + tau / (beta1 + epsilon))
    alpha2 = d_outer * (1 + tau / (beta2 + epsilon))
    # (mirror-symmetric operations, see `weno5_smoothness`)
    second_difference = ((alpha0 * (a - 2 * b + c) + alpha2 * (e - 2 * d + c)) +
                         alpha1 * ((b + d) - 2 * c)) / ((alpha0 + alpha2) + alpha1)
    return -second_difference / 24
end

# Point value at the subcell center from the conservative average `u_average` and the
# sum of the corrections of all directions; inadmissible point values (see
# `positivity_quantities`) are replaced by the average
@inline function center_point_value(u_average, correction, equations)
    u_center = u_average + correction
    bounds = convert(eltype(u_average), 1.0e-12) *
             positivity_quantities(u_average, equations)
    return positivity_admissible(u_center, bounds, equations) ? u_center : u_average
end

"""
    ReconstructionConstant()

Piecewise constant face reconstruction (first-order finite volume scheme).

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
struct ReconstructionConstant end

@inline nhalo(::ReconstructionConstant) = 1

"""
    ReconstructionMUSCL(slope_limiter = minmod)

Piecewise linear (MUSCL) face reconstruction on the equidistant subcells.
The slope of each subcell is computed by `slope_limiter(sl, sr)` from the
left and right one-sided differences `sl` and `sr`, applied componentwise. Any of
the limiter functions [`central_slope`](@ref) (unlimited), [`minmod`](@ref),
[`monotonized_central`](@ref), [`superbee`](@ref), [`vanleer`](@ref), or
[`koren_symmetric`](@ref) can be used.

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
struct ReconstructionMUSCL{SlopeLimiter}
    slope_limiter::SlopeLimiter
end

ReconstructionMUSCL() = ReconstructionMUSCL(minmod)

@inline nhalo(::ReconstructionMUSCL) = 2

"""
    ReconstructionWENO5JS(; epsilon = 1.0e-6)
    ReconstructionWENO5Z(; epsilon = 1.0e-40)

Fifth-order weighted essentially non-oscillatory (WENO) reconstruction of the face
states of each subcell from the five cell averages centered at the subcell, applied
componentwise to the reconstruction variables. `ReconstructionWENO5JS` uses the
nonlinear weights of

- Guang-Shan Jiang and Chi-Wang Shu (1996)
  Efficient implementation of weighted ENO schemes
  [DOI: 10.1006/jcph.1996.0130](https://doi.org/10.1006/jcph.1996.0130)

and `ReconstructionWENO5Z` the weights of

- Rafael Borges, Monique Carmona, Bruno Costa and Wai Sun Don (2008)
  An improved weighted essentially non-oscillatory scheme for hyperbolic
  conservation laws
  [DOI: 10.1016/j.jcp.2007.11.038](https://doi.org/10.1016/j.jcp.2007.11.038)

(with the exponent 1 of the global smoothness ratio). `epsilon` avoids divisions by
zero in the nonlinear weights.

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
struct ReconstructionWENO5JS{RealT}
    epsilon::RealT
end

ReconstructionWENO5JS(; epsilon = 1.0e-6) = ReconstructionWENO5JS(epsilon)

"""
    ReconstructionWENO5Z(; epsilon = 1.0e-40)

WENO5 reconstruction with the WENO-Z weights of Borges et al. (2008); see
[`ReconstructionWENO5JS`](@ref).

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
struct ReconstructionWENO5Z{RealT}
    epsilon::RealT
end

ReconstructionWENO5Z(; epsilon = 1.0e-40) = ReconstructionWENO5Z(epsilon)

const ReconstructionWENO5 = Union{ReconstructionWENO5JS, ReconstructionWENO5Z}

@inline nhalo(::ReconstructionWENO5) = 3

"""
    ReconstructionPositivity(reconstruction; epsilon = 1.0e-12, omega = 0.25)

Positivity limiter for the face states of a `reconstruction`, e.g.,
[`ReconstructionWENO5Z`](@ref) or [`ReconstructionMUSCL`](@ref), in the spirit of
- Xiangxiong Zhang and Chi-Wang Shu (2010)
  On positivity-preserving high order discontinuous Galerkin schemes for compressible
  Euler equations on rectangular meshes
  [DOI: 10.1016/j.jcp.2010.08.016](https://doi.org/10.1016/j.jcp.2010.08.016)
Each face state `w` of a subcell (in the reconstruction variables) is replaced by
`w̄ + θ (w - w̄)` with the subcell value `w̄` and the largest `θ ∈ [0, 1]` (found by
bisection) for which the quantities that must stay positive (for the compressible
Euler equations the density and the pressure, for the multicomponent Euler equations
the pressure and the partial densities) of the face state, converted to conservative variables, are at least
`epsilon` times those of the subcell value. Then, both face states of the subcell are
scaled together towards `w̄` such that the subcell value is a convex combination
`ū = omega (u_minus + u_plus) + (1 - 2 omega) u*` of the face states and an admissible
interior state `u*`, which bounds the outflow through the faces by the content of the
subcell for time steps with `λ α <= omega` per direction (Zhang and Shu, 2010). Since
the admissibility is checked on the conservative states, the limiter works with any
reconstruction variables; with `cons2recon = cons2cons` (conservative variables),
WENO keeps its order of accuracy for smooth solutions, while the reconstruction of
primitive variables from the subcell averages is only second order accurate (use a
moment inversion to point values, e.g., [`MomentInversionWENO`](@ref), for higher order
with primitive reconstruction variables).

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
struct ReconstructionPositivity{Reconstruction, RealT}
    reconstruction::Reconstruction
    epsilon::RealT
    omega::RealT
end

function ReconstructionPositivity(reconstruction; epsilon = 1.0e-12, omega = 0.25)
    0 < omega <= 0.5 || throw(ArgumentError("`omega` must be in (0, 1/2]"))
    epsilon, omega = promote(epsilon, omega)
    return ReconstructionPositivity(reconstruction, epsilon, omega)
end

function Base.show(io::IO, reconstruction::ReconstructionPositivity)
    print(io, "ReconstructionPositivity(", reconstruction.reconstruction, ")")
end

@inline nhalo(reconstruction::ReconstructionPositivity) = nhalo(reconstruction.reconstruction)

"""
    VolumeIntegralMacroCellFV(; moment_inversion, reconstruction,
                                             surface_flux, cons2recon, recon2cons,
                                             initial_condition_nodes)

Volume integral of the [`MacroCellFV`](@ref) solver. It collects the components of the
scheme; construct the solver with [`MacroCellFV`](@ref) instead of using this type
directly.

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
struct VolumeIntegralMacroCellFV{MomentInversion, Reconstruction,
                                 SurfaceFlux, Cons2Recon, Recon2Cons} <:
       AbstractVolumeIntegral
    moment_inversion::MomentInversion
    reconstruction::Reconstruction
    surface_flux::SurfaceFlux
    cons2recon::Cons2Recon
    recon2cons::Recon2Cons
    # Gauss-Legendre nodes per direction for the cell averages of the initial
    # condition (1: point value at the subcell center)
    initial_condition_nodes::Int
end

function VolumeIntegralMacroCellFV(;
                                   moment_inversion = MomentInversionIdentity(),
                                   reconstruction = ReconstructionMUSCL(),
                                   surface_flux,
                                   cons2recon = cons2cons,
                                   recon2cons = cons2cons,
                                   initial_condition_nodes = 1)
    initial_condition_nodes >= 1 ||
        throw(ArgumentError("`initial_condition_nodes` must be at least 1"))
    reconstruction = point_reconstruction(moment_inversion, reconstruction)
    return VolumeIntegralMacroCellFV(moment_inversion, reconstruction,
                                     surface_flux, cons2recon,
                                     recon2cons,
                                     Int(initial_condition_nodes))
end

function Base.show(io::IO, ::MIME"text/plain",
                   integral::VolumeIntegralMacroCellFV)
    @nospecialize integral
    setup = ["moment inversion" => integral.moment_inversion,
        "reconstruction" => integral.reconstruction,
        "surface flux" => integral.surface_flux,
        "reconstruction variables" => integral.cons2recon]
    summary_box(io, "VolumeIntegralMacroCellFV", setup)
end

# Halo width: the moment inversion is applied to the halo cells used by the face
# reconstruction, so the stencil widths add up.
@inline function nhalo(volume_integral::VolumeIntegralMacroCellFV)
    @unpack moment_inversion, reconstruction = volume_integral
    return nhalo(moment_inversion) + nhalo(reconstruction)
end

"""
    MacroCellFV(; n_cells::Integer,
                  surface_flux,
                  reconstruction = ReconstructionMUSCL(),
                  moment_inversion = MomentInversionIdentity(),
                  cons2recon = cons2cons,
                  recon2cons = cons2cons,
                  initial_condition_nodes = 1,
                  RealT = Float64)

Create a macro-cell finite volume solver with `n_cells` equidistant finite volume
subcells per coordinate direction in each element (macro-cell) of the mesh, with
high-order reconstruction of the face states. As for [`BlockFV`](@ref), the subcells
are stored with a [`UniformFiniteVolumeBasis`](@ref); for consistency with `BlockFV`,
the number of subcells can also be given as `n_nodes`. Currently supported on the 1D
[`TreeMesh`](@ref).

- `surface_flux`: numerical flux at the subcell faces, e.g.,
  [`flux_lax_friedrichs`](@ref) or [`flux_hllc`](@ref).
- `reconstruction`: [`ReconstructionConstant`](@ref), [`ReconstructionMUSCL`](@ref),
  [`ReconstructionWENO5JS`](@ref), [`ReconstructionWENO5Z`](@ref), or one of these
  with the positivity limiter [`ReconstructionPositivity`](@ref). The reconstruction
  reads the neighboring macro-cells through `nhalo` halo subcells per side, which
  requires `n_cells >= nhalo`.
- `moment_inversion`: [`MomentInversionIdentity`](@ref) (the averages are used as point
  values, second order for nonlinear `cons2recon`), [`MomentInversionCentral`](@ref), or
  [`MomentInversionWENO`](@ref) (fourth-order point values at the subcell centers,
  converted to the reconstruction variables pointwise, and WENO interpolation of the
  point values to the faces).
- `cons2recon`/`recon2cons`: transformation from conservative variables to the
  variables used for the reconstruction (e.g., [`cons2prim`](@ref)) and back.
- `initial_condition_nodes`: number of Gauss-Legendre nodes per coordinate direction
  for the subcell averages of the initial condition. The default `1` uses the value
  at the subcell center, which is second-order accurate; reconstructions of higher
  order need the exact averages.

With `reconstruction = ReconstructionConstant()`, the right-hand side is identical to
the one of `BlockFV(; n_nodes = n_cells, surface_flux)`.

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
const MacroCellFV = DG{Basis, Mortar, SurfaceIntegral,
                       VolumeIntegral} where {Basis <: UniformFiniteVolumeBasis,
                                              Mortar, SurfaceIntegral,
                                              VolumeIntegral <:
                                              VolumeIntegralMacroCellFV}

function MacroCellFV(; n_cells = nothing, n_nodes = nothing,
                     surface_flux,
                     reconstruction = ReconstructionMUSCL(),
                     moment_inversion = MomentInversionIdentity(),
                     cons2recon = cons2cons,
                     recon2cons = cons2cons,
                     initial_condition_nodes = 1,
                     RealT = Float64)
    if (n_cells === nothing) == (n_nodes === nothing)
        throw(ArgumentError("give the number of subcells either as `n_cells` or as " *
                            "`n_nodes`"))
    end
    n_cells = n_cells === nothing ? n_nodes : n_cells
    basis = UniformFiniteVolumeBasis(RealT, n_cells)
    volume_integral = VolumeIntegralMacroCellFV(; moment_inversion, reconstruction,
                                                surface_flux, cons2recon, recon2cons,
                                                initial_condition_nodes)
    # The surface integral is not used by the right-hand side, but it keeps the `DG`
    # interface (e.g., for callbacks) intact. As for `BlockFV`, the basis is passed as
    # mortar method.
    surface_integral = SurfaceIntegralWeakForm(surface_flux)
    return DG(basis, basis, surface_integral, volume_integral)
end

function Base.show(io::IO, mime::MIME"text/plain", dg::MacroCellFV)
    @nospecialize dg
    summary_header(io, "MacroCellFV")
    summary_line(io, "basis", dg.basis)
    summary_line(io, "volume integral", dg.volume_integral |> typeof |> nameof)
    summary_footer(io)
end

@inline nhalo(dg::MacroCellFV) = nhalo(dg.volume_integral)

# Number of variables of the pads: twice the number of variables for the point values
# and the averages
@inline function npadvariables(volume_integral::VolumeIntegralMacroCellFV,
                               equations)
    return npadvariables(has_point_values(volume_integral.moment_inversion), equations)
end
@inline npadvariables(::False, equations) = nvariables(equations)
@inline npadvariables(::True, equations) = 2 * nvariables(equations)

# Same as `wrap_array` for `DGSEM`: with Polyester.jl threading, use a `PtrArray`, which
# does not allocate (the general fallback for `DG` allocates an array header for each
# call). The number of subcells is a type parameter of the `UniformFiniteVolumeBasis`.
@inline function wrap_array(u_ode::AbstractVector, mesh::AbstractMesh, equations,
                            dg::MacroCellFV, cache)
    @boundscheck begin
        @assert length(u_ode) ==
                nvariables(equations) * nnodes(dg)^ndims(mesh) * nelements(dg, cache)
    end
    # See the comments on the `DGSEM` version, in particular on the Heisenbug with
    # `Threads.@threads`
    if _PREFERENCE_THREADING === :polyester && LoopVectorization.check_args(u_ode)
        PtrArray(pointer(u_ode),
                 (StaticInt(nvariables(equations)),
                  ntuple(_ -> StaticInt(nnodes(dg)), ndims(mesh))...,
                  nelements(dg, cache)))
    else
        wrap_array_native(u_ode, mesh, equations, dg, cache)
    end
end

# Conversion applied when the pads, ghost buffers, and halo layers are filled: the
# moment inversions to point values work on the conservative averages and convert to
# the reconstruction variables afterward
@inline function pad_cons2recon(volume_integral::VolumeIntegralMacroCellFV)
    return pad_cons2recon(has_point_values(volume_integral.moment_inversion),
                          volume_integral.cons2recon)
end
@inline pad_cons2recon(::False, cons2recon) = cons2recon
@inline pad_cons2recon(::True, cons2recon) = cons2cons

#####################################################################
# Pointwise building blocks of the reconstruction along one coordinate line

# Reconstructed states of one subcell at its two faces along a coordinate line,
# where `stencil(offset)` returns the state of the subcell `offset` cells away
# along the line. Returns `(w_minus, w_plus)`, the states at the face in negative
# and in positive line direction. Each subcell is reconstructed once per direction
# and RHS evaluation; the face states of all subcells are then combined by the
# numerical fluxes.
# Reconstruction with the equations and the conversion of the reconstruction variables
# to the conservative variables (needed by the positivity limiter only)
@inline function reconstruct_cell(reconstruction, stencil, equations, recon2cons)
    return reconstruct_cell(reconstruction, stencil)
end

# Quantities that must stay positive (see `ReconstructionPositivity`), as functions of
# the conservative variables (none for general equations)
@inline positivity_quantities(u, equations) = SVector{0, eltype(u)}()

@inline function positivity_quantities(u, equations::AbstractCompressibleEulerEquations)
    return SVector(density(u, equations), pressure(u, equations))
end

@inline function positivity_quantities(u,
                                       equations::AbstractCompressibleEulerMulticomponentEquations)
    # (the partial densities are the last variables)
    offset = nvariables(equations) - ncomponents(equations)
    return vcat(SVector(pressure(u, equations)),
                SVector{ncomponents(equations)}(u[i + offset]
                                                for i in eachcomponent(equations)))
end

# `true` if all quantities of `positivity_quantities` of the conservative state `u` are at
# least `bounds` (`false` for NaN values)
@inline function positivity_admissible(u, bounds, equations)
    q = positivity_quantities(u, equations)
    admissible = true
    for i in eachindex(bounds)
        admissible &= q[i] >= bounds[i]
    end
    return admissible
end

# Largest `θ ∈ [0, 1]` such that the quantities `quantities(θ)` (see
# `positivity_quantities`) are at least `bounds`, for admissible `quantities(0)`; 1
# without further evaluations if `quantities(1)` is admissible. First guess: the
# linear interpolation of the quantities between `θ = 0` and `θ = 1`, which is
# exact for quantities that are linear in `θ` (e.g., the partial densities) and
# admissible for concave ones (e.g., the pressure along a line in conservative
# variables). Otherwise (or for NaN values), 12 bisection steps below the guess; the
# result is always admissible.
@inline function positivity_theta(quantities::Quantities, bounds) where {Quantities}
    RealT = eltype(bounds)
    q_1 = quantities(one(RealT))
    all(q_1 .>= bounds) && return one(RealT)
    q_0 = quantities(zero(RealT))
    theta_hi = one(RealT)
    for i in eachindex(bounds)
        if q_1[i] < bounds[i]
            # `q_0[i] >= bounds[i] > q_1[i]` (the comparison skips NaN values)
            theta_i = (q_0[i] - bounds[i]) / (q_0[i] - q_1[i])
            theta_hi = theta_i < theta_hi ? theta_i : theta_hi
        end
    end
    # (the guess might not be admissible for quantities that are not concave in `θ`)
    all(quantities(theta_hi) .>= bounds) && return theta_hi
    theta_lo = zero(RealT)
    for _ in 1:12
        theta = (theta_lo + theta_hi) / 2
        if all(quantities(theta) .>= bounds)
            theta_lo = theta
        else
            theta_hi = theta
        end
    end
    return theta_lo
end

@inline function reconstruct_cell(reconstruction::ReconstructionPositivity, stencil,
                                  equations, recon2cons)
    w_minus, w_plus = reconstruct_cell(reconstruction.reconstruction, stencil)
    # (the subcell average, also for reconstructions from point values)
    w_center = reconstruction_center(reconstruction.reconstruction, stencil)
    RealT = eltype(w_center)
    epsilon = convert(RealT, reconstruction.epsilon)
    omega = convert(RealT, reconstruction.omega)
    u_center = recon2cons(w_center, equations)
    bounds = epsilon * positivity_quantities(u_center, equations)
    # Inadmissible subcell values cannot be helped here
    positivity_admissible(u_center, bounds, equations) || return w_center, w_center

    # Each face state separately
    theta_minus = positivity_theta_face(w_minus, w_center, bounds, equations,
                                        recon2cons)
    theta_plus = positivity_theta_face(w_plus, w_center, bounds, equations, recon2cons)
    w_minus_limited = w_center + theta_minus * (w_minus - w_center)
    w_plus_limited = w_center + theta_plus * (w_plus - w_center)

    # Both face states together: the subcell value must be a convex combination
    # `ū = ω (u_minus + u_plus) + (1 - 2ω) u*` with an admissible interior state `u*`
    # (Zhang and Shu, 2010), which bounds the outflow through both faces by the content
    # of the subcell
    theta = positivity_theta(PositivityInterior(w_minus_limited, w_plus_limited,
                                                w_center, u_center, omega, equations,
                                                recon2cons),
                             bounds)
    # (unlimited states are returned unchanged, without round-off errors)
    theta == 1 && theta_minus == 1 && theta_plus == 1 && return w_minus, w_plus
    return (w_center + theta * (w_minus_limited - w_center),
            w_center + theta * (w_plus_limited - w_center))
end

# Protected quantities of the face state `w_center + θ (w - w_center)` (function object
# instead of a closure, so that no captured variables are boxed)
struct PositivityFace{W, Equations, Recon2Cons}
    w::W
    w_center::W
    equations::Equations
    recon2cons::Recon2Cons
end

@inline function (f::PositivityFace)(theta)
    @unpack w, w_center, equations, recon2cons = f
    return positivity_quantities(recon2cons(w_center + theta * (w - w_center),
                                            equations),
                                 equations)
end

@inline function positivity_theta_face(w, w_center, bounds, equations, recon2cons)
    return positivity_theta(PositivityFace(w, w_center, equations, recon2cons),
                            bounds)
end

# Protected quantities of the interior state of the convex decomposition of the subcell
# value for the face states scaled by `θ` towards `w_center`
struct PositivityInterior{W, U, RealT, Equations, Recon2Cons}
    w_minus::W
    w_plus::W
    w_center::W
    u_center::U
    omega::RealT
    equations::Equations
    recon2cons::Recon2Cons
end

@inline function (f::PositivityInterior)(theta)
    @unpack w_minus, w_plus, w_center, u_center, omega, equations, recon2cons = f
    u_minus = recon2cons(w_center + theta * (w_minus - w_center), equations)
    u_plus = recon2cons(w_center + theta * (w_plus - w_center), equations)
    u_star = (u_center - omega * (u_minus + u_plus)) / (1 - 2 * omega)
    return positivity_quantities(u_star, equations)
end

@inline function reconstruct_cell(::ReconstructionConstant, stencil)
    u = stencil(0)
    return u, u
end

@inline function reconstruct_cell(reconstruction::ReconstructionMUSCL, stencil)
    @unpack slope_limiter = reconstruction
    u_m = stencil(-1)
    u_0 = stencil(0)
    u_p = stencil(1)

    # Equidistant subcells: the slope is expressed per subcell width, and the
    # faces are half a subcell away from the cell center.
    slope = slope_limiter.(u_0 - u_m, u_p - u_0)
    return u_0 - 0.5f0 * slope, u_0 + 0.5f0 * slope
end

@inline function reconstruct_cell(reconstruction::ReconstructionWENO5, stencil)
    u_m2 = stencil(-2)
    u_m1 = stencil(-1)
    u_0 = stencil(0)
    u_p1 = stencil(1)
    u_p2 = stencil(2)

    pairs = map(@inline((a, b, c, d, e)->weno5_cell(reconstruction, a, b, c, d, e)),
                u_m2, u_m1, u_0, u_p1, u_p2)
    return map(first, pairs), map(last, pairs)
end

# WENO5 values at the faces between the cells `b` and `c` (`w_minus`) and between `c`
# and `d` (`w_plus`) of the cell `c` from the cell averages `a, b, c, d, e` (Jiang and
# Shu, 1996). The face in negative direction is the mirrored reconstruction, whose
# smoothness indicators are those of the face in positive direction in reverse order.
@inline function weno5_cell(reconstruction, a, b, c, d, e)
    # Smoothness indicators of the three third-order substencils
    beta0, beta1, beta2 = weno5_smoothness(a, b, c, d, e)

    # Candidate values at the face in positive direction ...
    p0 = (2 * a - 7 * b + 11 * c) / 6
    p1 = (-b + 5 * c + 2 * d) / 6
    p2 = (2 * c + 5 * d - e) / 6
    alpha0, alpha1, alpha2 = weno5_alphas(reconstruction, beta0, beta1, beta2)
    w_plus = (alpha0 * p0 + alpha1 * p1 + alpha2 * p2) / (alpha0 + alpha1 + alpha2)

    # ... and in negative direction (mirrored stencils)
    q0 = (2 * e - 7 * d + 11 * c) / 6
    q1 = (-d + 5 * c + 2 * b) / 6
    q2 = (2 * c + 5 * b - a) / 6
    gamma0, gamma1, gamma2 = weno5_alphas(reconstruction, beta2, beta1, beta0)
    w_minus = (gamma0 * q0 + gamma1 * q1 + gamma2 * q2) / (gamma0 + gamma1 + gamma2)

    return w_minus, w_plus
end

# Smoothness indicators of Jiang and Shu (1996) of the substencils `(a, b, c)`,
# `(b, c, d)`, and `(c, d, e)`. The operations are mirror-symmetric: for the reversed
# data `(e, d, c, b, a)`, the indicators are `(beta2, beta1, beta0)` bitwise, so that
# mirrored data (e.g., at reflecting walls) give mirrored face values exactly, also
# through the decisions of the positivity limiter.
@inline function weno5_smoothness(a, b, c, d, e)
    RealT = typeof(c)
    c_13_12 = convert(RealT, 13 // 12)
    beta0 = c_13_12 * (a - 2 * b + c)^2 + 0.25f0 * (a - 4 * b + 3 * c)^2
    beta1 = c_13_12 * ((b + d) - 2 * c)^2 + 0.25f0 * (b - d)^2
    beta2 = c_13_12 * (e - 2 * d + c)^2 + 0.25f0 * (e - 4 * d + 3 * c)^2
    return beta0, beta1, beta2
end

# Unnormalized nonlinear weights with the linear weights (1, 6, 3) / 10
@inline function weno5_alphas(reconstruction::ReconstructionWENO5JS, beta0, beta1,
                              beta2)
    epsilon = convert(typeof(beta0), reconstruction.epsilon)
    return (1 / (10 * (epsilon + beta0)^2), 6 / (10 * (epsilon + beta1)^2),
            3 / (10 * (epsilon + beta2)^2))
end

@inline function weno5_alphas(reconstruction::ReconstructionWENO5Z, beta0, beta1,
                              beta2)
    epsilon = convert(typeof(beta0), reconstruction.epsilon)
    tau5 = abs(beta0 - beta2)
    return ((1 + tau5 / (beta0 + epsilon)) / 10,
            6 * (1 + tau5 / (beta1 + epsilon)) / 10,
            3 * (1 + tau5 / (beta2 + epsilon)) / 10)
end

# WENO reconstruction from the point values at the subcell centers (moment inversions
# `MomentInversionPoint`): the pad holds the point values (first half of the
# variables) and the averages (second half), both in reconstruction variables. The
# face values are the WENO interpolations of the point values to the face midpoints
# (as in the weighted compact nonlinear schemes of Deng and Zhang, 2000), with the
# candidates of the stencils `(i-2, i-1, i)`, `(i-1, i, i+1)`, `(i, i+1, i+2)` and the
# linear weights `(1, 10, 5) / 16`, exact for quartic polynomials.
struct ReconstructionPointValues{Reconstruction}
    reconstruction::Reconstruction
end

function Base.show(io::IO, reconstruction::ReconstructionPointValues)
    print(io, reconstruction.reconstruction, " (point values)")
end

@inline nhalo(reconstruction::ReconstructionPointValues) = nhalo(reconstruction.reconstruction)

@inline point_part(x::SVector{M}) where {M} = SVector(ntuple(@inline(i->x[i]),
                                                             Val(M ÷ 2)))
@inline average_part(x::SVector{M}) where {M} = SVector(ntuple(@inline(i->x[M ÷ 2 + i]),
                                                               Val(M ÷ 2)))

# The subcell value used as reference by the positivity limiter: the average
@inline reconstruction_center(reconstruction, stencil) = stencil(0)
@inline function reconstruction_center(::ReconstructionPointValues, stencil)
    return average_part(stencil(0))
end

@inline function reconstruct_cell(reconstruction::ReconstructionPointValues{<:ReconstructionWENO5},
                                  stencil)
    u_m2 = point_part(stencil(-2))
    u_m1 = point_part(stencil(-1))
    u_0 = point_part(stencil(0))
    u_p1 = point_part(stencil(1))
    u_p2 = point_part(stencil(2))
    weno = reconstruction.reconstruction
    pairs = map(@inline((a, b, c, d, e)->weno5_point_cell(weno, a, b, c, d, e)),
                u_m2, u_m1, u_0, u_p1, u_p2)
    return map(first, pairs), map(last, pairs)
end

# WENO interpolation of the point values `a, b, c, d, e` at the centers of the cells
# `i-2, ..., i+2` to the faces of the cell `c` (as `weno5_cell` for averages)
@inline function weno5_point_cell(reconstruction, a, b, c, d, e)
    beta0, beta1, beta2 = weno5_smoothness(a, b, c, d, e)

    p0 = (3 * a - 10 * b + 15 * c) / 8
    p1 = (-b + 6 * c + 3 * d) / 8
    p2 = (3 * c + 6 * d - e) / 8
    alpha0, alpha1, alpha2 = weno5_point_alphas(reconstruction, beta0, beta1, beta2)
    w_plus = (alpha0 * p0 + alpha1 * p1 + alpha2 * p2) / (alpha0 + alpha1 + alpha2)

    q0 = (3 * e - 10 * d + 15 * c) / 8
    q1 = (-d + 6 * c + 3 * b) / 8
    q2 = (3 * c + 6 * b - a) / 8
    gamma0, gamma1, gamma2 = weno5_point_alphas(reconstruction, beta2, beta1, beta0)
    w_minus = (gamma0 * q0 + gamma1 * q1 + gamma2 * q2) / (gamma0 + gamma1 + gamma2)

    return w_minus, w_plus
end

# Unnormalized nonlinear weights with the linear weights (1, 10, 5) / 16
@inline function weno5_point_alphas(reconstruction::ReconstructionWENO5JS, beta0,
                                    beta1, beta2)
    epsilon = convert(typeof(beta0), reconstruction.epsilon)
    return (1 / (16 * (epsilon + beta0)^2), 10 / (16 * (epsilon + beta1)^2),
            5 / (16 * (epsilon + beta2)^2))
end

@inline function weno5_point_alphas(reconstruction::ReconstructionWENO5Z, beta0,
                                    beta1, beta2)
    epsilon = convert(typeof(beta0), reconstruction.epsilon)
    tau5 = abs(beta0 - beta2)
    return ((1 + tau5 / (beta0 + epsilon)) / 16,
            10 * (1 + tau5 / (beta1 + epsilon)) / 16,
            5 * (1 + tau5 / (beta2 + epsilon)) / 16)
end

# The reconstruction used with the moment inversion
point_reconstruction(::MomentInversionIdentity, reconstruction) = reconstruction

function point_reconstruction(::MomentInversionPoint,
                              reconstruction::ReconstructionWENO5)
    return ReconstructionPointValues(reconstruction)
end

function point_reconstruction(inversion::MomentInversionPoint,
                              reconstruction::ReconstructionPositivity)
    return ReconstructionPositivity(point_reconstruction(inversion,
                                                         reconstruction.reconstruction),
                                    reconstruction.epsilon, reconstruction.omega)
end

function point_reconstruction(::MomentInversionPoint, reconstruction)
    throw(ArgumentError("the moment inversions to point values support only the WENO " *
                        "reconstructions (with or without `ReconstructionPositivity`)"))
end

# Reconstructed left and right states at the face between the pad cells `0` and
# `1` of the coordinate line `line(offset)`
@inline function reconstruct_face_states(reconstruction, line, equations, recon2cons)
    _, w_ll = reconstruct_cell(reconstruction, line, equations, recon2cons)
    w_rr, _ = reconstruct_cell(reconstruction, @inline(offset->line(offset + 1)),
                               equations, recon2cons)
    return w_ll, w_rr
end

# Numerical flux at the face between the pad cells `0` and `1` of `line` from the
# reconstructed states
@inline function macrocell_face_flux(line, orientation, equations,
                                     volume_integral::VolumeIntegralMacroCellFV)
    @unpack reconstruction, surface_flux, recon2cons = volume_integral

    w_ll, w_rr = reconstruct_face_states(reconstruction, line, equations, recon2cons)
    u_ll = recon2cons(w_ll, equations)
    u_rr = recon2cons(w_rr, equations)

    return surface_flux(u_ll, u_rr, orientation, equations)
end
end # @muladd
