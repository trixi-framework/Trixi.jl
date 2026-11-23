# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Macro-cell finite volume (MacroCellFV) solver: finite volumes with face reconstruction
# on blocks of subcells.
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
#   2. moment inversion (cell averages -> point values; `MomentInversionIdentity` uses
#      the averages),
#   3. transformation to the reconstruction variables,
#   4. face reconstruction, numerical fluxes, and flux differences in one loop over the
#      subcell faces,
# followed by the completion of the flux differences of the first subcell of each
# macro-cell with the interface flux, which is computed only once, by the macro-cell on
# its left side.
#
# The pad holds the transformed state (reconstruction variables) of a contiguous line of
# subcells across the macro-cell boundaries. The same kernel thus serves the faces
# inside a macro-cell and at its boundaries without index branching or repeated
# variable transformations in the stencils. Each thread owns one pad of `n_cells + 2 nhalo` subcells, so the
# memory of the pads does not grow with the number of elements.

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
reconstruction of the face states. As for [`BlockFV`](@ref), the subcells
are stored with a [`UniformFiniteVolumeBasis`](@ref); for consistency with `BlockFV`,
the number of subcells can also be given as `n_nodes`. Currently supported on the 1D
[`TreeMesh`](@ref).

- `surface_flux`: numerical flux at the subcell faces, e.g.,
  [`flux_lax_friedrichs`](@ref) or [`flux_hllc`](@ref).
- `reconstruction`: [`ReconstructionConstant`](@ref) or [`ReconstructionMUSCL`](@ref).
  The reconstruction reads the neighboring macro-cells through `nhalo` halo subcells
  per side, which requires `n_cells >= nhalo`.
- `moment_inversion`: conversion of the subcell averages to point values before the
  reconstruction; currently [`MomentInversionIdentity`](@ref) (the averages are used as
  point values).
- `cons2recon`/`recon2cons`: transformation from conservative variables to the
  variables used for the reconstruction (e.g., [`cons2prim`](@ref)) and back.
- `initial_condition_nodes`: number of Gauss-Legendre nodes per coordinate direction
  for the subcell averages of the initial condition. The default `1` uses the value
  at the subcell center, which is second-order accurate.

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

# Number of variables of the pads (the number of variables of the equations for the
# averages)
@inline function npadvariables(volume_integral::VolumeIntegralMacroCellFV,
                               equations)
    return npadvariables(has_point_values(volume_integral.moment_inversion), equations)
end
@inline npadvariables(::False, equations) = nvariables(equations)

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

# Conversion applied when the pads, ghost buffers, and halo layers are filled (the
# averages are converted to the reconstruction variables directly)
@inline function pad_cons2recon(volume_integral::VolumeIntegralMacroCellFV)
    return pad_cons2recon(has_point_values(volume_integral.moment_inversion),
                          volume_integral.cons2recon)
end
@inline pad_cons2recon(::False, cons2recon) = cons2recon

#####################################################################
# Coupling of the moment inversion, the reconstruction, and the numerical flux
# (the schemes are in `schemes/`)

# The reconstruction used with the moment inversion
point_reconstruction(::MomentInversionIdentity, reconstruction) = reconstruction

# Reconstructed left and right states at the face between the pad cells `0` and
# `1` of the coordinate line `line(offset)`
@inline function reconstruct_face_states(reconstruction, line)
    _, w_ll = reconstruct_cell(reconstruction, line)
    w_rr, _ = reconstruct_cell(reconstruction, @inline(offset->line(offset + 1)))
    return w_ll, w_rr
end

# Numerical flux at the face between the pad cells `0` and `1` of `line` from the
# reconstructed states
@inline function macrocell_face_flux(line, orientation, equations,
                                     volume_integral::VolumeIntegralMacroCellFV)
    @unpack reconstruction, surface_flux, recon2cons = volume_integral

    w_ll, w_rr = reconstruct_face_states(reconstruction, line)
    u_ll = recon2cons(w_ll, equations)
    u_rr = recon2cons(w_rr, equations)

    return surface_flux(u_ll, u_rr, orientation, equations)
end
end # @muladd
