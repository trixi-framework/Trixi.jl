# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

include("abstract_tree.jl")
include("serial_tree.jl")
include("parallel_tree.jl")

get_name(mesh::AbstractMesh) = mesh |> typeof |> nameof |> string

# Composite type to hold the actual tree in addition to other mesh-related data
# that is not strictly part of the tree.
# The mesh is really just about the connectivity, size, and location of the individual
# tree nodes. Neighbor information between interfaces or the large sides for mortars is
# something that is solver-specific and that might not be needed by all solvers (or in a
# different form). Also, these data values can be performance critical, so a mesh would
# have to store them for all solvers in an efficient way - OTOH, different solvers might
# use different cells of a shared mesh, so "efficient" is again solver dependent.
"""
    TreeMesh{NDIMS} <: AbstractMesh{NDIMS}

A Cartesian mesh based on trees of hypercubes to support adaptive mesh refinement.
"""
mutable struct TreeMesh{NDIMS, TreeType <: AbstractTree{NDIMS}} <: AbstractMesh{NDIMS}
    tree::TreeType
    current_filename::String
    unsaved_changes::Bool
    first_cell_by_rank::OffsetVector{Int, Vector{Int}}
    n_cells_by_rank::OffsetVector{Int, Vector{Int}}

    function TreeMesh{NDIMS, TreeType}(n_cells_max::Integer) where {NDIMS,
                                                                    TreeType <:
                                                                    AbstractTree{NDIMS}}
        # Create mesh
        m = new()
        m.tree = TreeType(n_cells_max)
        m.current_filename = ""
        m.unsaved_changes = true
        m.first_cell_by_rank = OffsetVector(Int[], 0)
        m.n_cells_by_rank = OffsetVector(Int[], 0)

        return m
    end

    # TODO: Taal refactor, order of important arguments, use of n_cells_max?
    # TODO: Taal refactor, allow other RealT for the mesh, not just Float64
    # TODO: Taal refactor, use NTuple instead of domain_center::AbstractArray{Float64}
    function TreeMesh{NDIMS, TreeType}(n_cells_max::Integer,
                                       domain_center::AbstractArray{Float64},
                                       domain_length,
                                       periodicity = true) where {NDIMS,
                                                                  TreeType <:
                                                                  AbstractTree{NDIMS}}
        @assert NDIMS isa Integer && NDIMS > 0

        # Create mesh
        m = new()
        m.tree = TreeType(n_cells_max, domain_center, domain_length, periodicity)
        m.current_filename = ""
        m.unsaved_changes = true
        m.first_cell_by_rank = OffsetVector(Int[], 0)
        m.n_cells_by_rank = OffsetVector(Int[], 0)

        return m
    end
end

const TreeMesh1D = TreeMesh{1, TreeType} where {TreeType <: AbstractTree{1}}
const TreeMesh2D = TreeMesh{2, TreeType} where {TreeType <: AbstractTree{2}}
const TreeMesh3D = TreeMesh{3, TreeType} where {TreeType <: AbstractTree{3}}

const SerialTreeMesh{NDIMS} = TreeMesh{NDIMS, <:SerialTree{NDIMS}}
const ParallelTreeMesh{NDIMS} = TreeMesh{NDIMS, <:ParallelTree{NDIMS}}

@inline mpi_parallel(mesh::SerialTreeMesh) = False()
@inline mpi_parallel(mesh::ParallelTreeMesh) = True()

partition!(mesh::SerialTreeMesh) = nothing

# Constructor for passing the dimension and mesh type as an argument
function TreeMesh(::Type{TreeType},
                  args...) where {NDIMS, TreeType <: AbstractTree{NDIMS}}
    TreeMesh{NDIMS, TreeType}(args...)
end

# Constructor accepting a single number as center (as opposed to an array) for 1D
function TreeMesh{1, TreeType}(n::Int, center::Real, len::Real,
                               periodicity = true) where {TreeType <: AbstractTree{1}}
    # TODO: Taal refactor, allow other RealT for the mesh, not just Float64
    return TreeMesh{1, TreeType}(n, SVector{1, Float64}(center), len, periodicity)
end

function TreeMesh{NDIMS, TreeType}(n_cells_max::Integer,
                                   domain_center::NTuple{NDIMS, Real},
                                   domain_length::Real,
                                   periodicity = true) where {NDIMS,
                                                              TreeType <:
                                                              AbstractTree{NDIMS}}
    # TODO: Taal refactor, allow other RealT for the mesh, not just Float64
    TreeMesh{NDIMS, TreeType}(n_cells_max, SVector{NDIMS, Float64}(domain_center),
                              convert(Float64, domain_length), periodicity)
end

function TreeMesh(coordinates_min::NTuple{NDIMS, Real},
                  coordinates_max::NTuple{NDIMS, Real};
                  n_cells_max,
                  periodicity = true,
                  initial_refinement_level,
                  refinement_patches = (),
                  coarsening_patches = ()) where {NDIMS}
    # check arguments
    if !(n_cells_max isa Integer && n_cells_max > 0)
        throw(ArgumentError("`n_cells_max` must be a positive integer (provided `n_cells_max = $n_cells_max`)"))
    end
    if !(initial_refinement_level isa Integer && initial_refinement_level >= 0)
        throw(ArgumentError("`initial_refinement_level` must be a non-negative integer (provided `initial_refinement_level = $initial_refinement_level`)"))
    end

    # TreeMesh requires equal domain lengths in all dimensions
    domain_center = @. (coordinates_min + coordinates_max) / 2
    domain_length = coordinates_max[1] - coordinates_min[1]
    if !all(coordinates_max[i] - coordinates_min[i] ≈ domain_length for i in 2:NDIMS)
        throw(ArgumentError("The TreeMesh domain must be a hypercube (provided `coordinates_max` .- `coordinates_min` = $(coordinates_max .- coordinates_min))"))
    end

    # TODO: MPI, create nice interface for a parallel tree/mesh
    if mpi_isparallel()
        if mpi_isroot() && NDIMS != 2
            println(stderr,
                    "ERROR: The TreeMesh supports parallel execution with MPI only in 2 dimensions")
            MPI.Abort(mpi_comm(), 1)
        end
        TreeType = ParallelTree{NDIMS}
    else
        TreeType = SerialTree{NDIMS}
    end

    # Create mesh
    mesh = @trixi_timeit timer() "creation" TreeMesh{NDIMS, TreeType}(n_cells_max,
                                                                      domain_center,
                                                                      domain_length,
                                                                      periodicity)

    # Initialize mesh
    initialize!(mesh, initial_refinement_level, refinement_patches, coarsening_patches)

    return mesh
end

function initialize!(mesh::TreeMesh, initial_refinement_level,
                     refinement_patches, coarsening_patches)
    # Create initial refinement
    @trixi_timeit timer() "initial refinement" refine_uniformly!(mesh.tree,
                                                                 initial_refinement_level)

    # Apply refinement patches
    @trixi_timeit timer() "refinement patches" for patch in refinement_patches
        # TODO: Taal refactor, use multiple dispatch?
        if patch.type == "box"
            refine_box!(mesh.tree, patch.coordinates_min, patch.coordinates_max)
        elseif patch.type == "sphere"
            refine_sphere!(mesh.tree, patch.center, patch.radius)
        else
            error("unknown refinement patch type '$(patch.type)'")
        end
    end

    # Apply coarsening patches
    @trixi_timeit timer() "coarsening patches" for patch in coarsening_patches
        # TODO: Taal refactor, use multiple dispatch
        if patch.type == "box"
            coarsen_box!(mesh.tree, patch.coordinates_min, patch.coordinates_max)
        else
            error("unknown coarsening patch type '$(patch.type)'")
        end
    end

    # Partition the mesh among multiple MPI ranks (does nothing if run in serial)
    partition!(mesh)

    return nothing
end

function TreeMesh(coordinates_min::Real, coordinates_max::Real; kwargs...)
    TreeMesh((coordinates_min,), (coordinates_max,); kwargs...)
end

function Base.show(io::IO, mesh::TreeMesh{NDIMS, TreeType}) where {NDIMS, TreeType}
    print(io, "TreeMesh{", NDIMS, ", ", TreeType, "} with length ", mesh.tree.length)
end

function Base.show(io::IO, ::MIME"text/plain",
                   mesh::TreeMesh{NDIMS, TreeType}) where {NDIMS, TreeType}
    if get(io, :compact, false)
        show(io, mesh)
    else
        setup = [
            "center" => mesh.tree.center_level_0,
            "length" => mesh.tree.length_level_0,
            "periodicity" => mesh.tree.periodicity,
            "current #cells" => mesh.tree.length,
            "#leaf-cells" => count_leaf_cells(mesh.tree),
            "maximum #cells" => mesh.tree.capacity
        ]
        summary_box(io, "TreeMesh{" * string(NDIMS) * ", " * string(TreeType) * "}",
                    setup)
    end
end

@inline Base.ndims(mesh::TreeMesh) = ndims(mesh.tree)

# Obtain the mesh filename from a restart file
function get_restart_mesh_filename(restart_filename, mpi_parallel::False)
    # Get directory name
    dirname, _ = splitdir(restart_filename)

    # Read mesh filename from restart file
    mesh_file = ""
    h5open(restart_filename, "r") do file
        mesh_file = read(attributes(file)["mesh_file"])
    end

    # Construct and return filename
    return joinpath(dirname, mesh_file)
end

function total_volume(mesh::TreeMesh)
    return mesh.tree.length_level_0^ndims(mesh)
end

isperiodic(mesh::TreeMesh) = isperiodic(mesh.tree)
isperiodic(mesh::TreeMesh, dimension) = isperiodic(mesh.tree, dimension)

include("parallel_tree_mesh.jl")
end # @muladd
