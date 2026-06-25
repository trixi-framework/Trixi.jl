# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    UniformFiniteVolumeBasis([RealT = Float64,] n_nodes::Integer)

A finite volume basis with `n_nodes` equidistant, cell-centered nodes on the
reference interval `[-1, 1]`.

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
struct UniformFiniteVolumeBasis{RealT, NNODES, VectorT <: AbstractVector{RealT}}
    nodes::VectorT
    weights::VectorT
end

function UniformFiniteVolumeBasis(RealT, n_nodes::Integer)
    nodes = SVector{n_nodes, RealT}(-1 + (2 * i - 1) / RealT(n_nodes)
                                    for i in 1:n_nodes)
    weights = SVector{n_nodes, RealT}(2 / RealT(n_nodes) for _ in 1:n_nodes)
    return UniformFiniteVolumeBasis{RealT, n_nodes, typeof(nodes)}(nodes, weights)
end

function UniformFiniteVolumeBasis(n_nodes::Integer)
    return UniformFiniteVolumeBasis(Float64, n_nodes)
end

# Basis interface required by the DG infrastructure
@inline Base.real(::UniformFiniteVolumeBasis{RealT}) where {RealT} = RealT
@inline nnodes(::UniformFiniteVolumeBasis{RealT, NNODES}) where {RealT, NNODES} = NNODES
@inline eachnode(basis::UniformFiniteVolumeBasis) = Base.OneTo(nnodes(basis))
@inline polydeg(::UniformFiniteVolumeBasis) = 0
@inline get_nodes(basis::UniformFiniteVolumeBasis) = basis.nodes

function integrate(f, u, basis::UniformFiniteVolumeBasis)
    @unpack weights = basis
    res = zero(f(first(u)))
    for i in eachindex(u, weights)
        res = res + f(u[i]) * weights[i]
    end
    return res
end

# The basis itself serves as the solution analyzer (no polynomial interpolation needed)
SolutionAnalyzer(basis::UniformFiniteVolumeBasis; kwargs...) = basis

"""
    VolumeIntegralFiniteVolume(surface_flux)

Volume integral for the [`BlockFV`](@ref) solver. Computes numerical fluxes at internal
cell interfaces within each block element and applies the resulting flux differences.
"""
struct VolumeIntegralFiniteVolume{SurfaceFlux} <: AbstractVolumeIntegral
    surface_flux::SurfaceFlux
end

function Base.show(io::IO, ::MIME"text/plain",
                   integral::VolumeIntegralFiniteVolume)
    @nospecialize integral
    setup = ["surface flux" => integral.surface_flux]
    summary_box(io, "VolumeIntegralFiniteVolume", setup)
end

# Type alias: BlockFV is a DG solver whose basis is a UniformFiniteVolumeBasis
"""
    BlockFV(; n_nodes::Integer,
              surface_flux,
              RealT = Float64)

Create a block finite volume solver with `n_nodes` volumes per coordinate direction
in each cell of the mesh and the `surface_flux` as numerical flux.

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
const BlockFV = DG{Basis} where {Basis <: UniformFiniteVolumeBasis}

function BlockFV(; n_nodes::Integer,
                 surface_flux,
                 RealT = Float64)
    basis = UniformFiniteVolumeBasis(RealT, n_nodes)
    volume_integral = VolumeIntegralFiniteVolume(surface_flux)
    surface_integral = SurfaceIntegralWeakForm(surface_flux)
    #basis is passed as mortar method
    return DG(basis, basis, surface_integral, volume_integral)
end

function Base.show(io::IO, mime::MIME"text/plain", dg::BlockFV)
    @nospecialize dg
    summary_header(io, "BlockFV")
    summary_line(io, "basis", dg.basis)
    summary_line(io, "mortar method", dg.mortar)
    summary_line(io, "surface integral", dg.surface_integral |> typeof |> nameof)
    summary_line(io, "volume integral", dg.volume_integral |> typeof |> nameof)
    summary_footer(io)
end

@inline polydeg(dg::BlockFV) = polydeg(dg.basis)
end # @muladd
