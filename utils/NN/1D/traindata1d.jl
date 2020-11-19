#using LinearAlgebra: dot
using Pkg.TOML: parsefile, parse
using Printf: @printf, @sprintf, println
using Plots; pyplot()
#using Profile: clear_malloc_data
#using Random: seed!

using EllipsisNotation
using HDF5: h5open, attrs
import MPI
using OffsetArrays: OffsetArray, OffsetVector
using StaticArrays: @MVector, @SVector, MVector, MMatrix, MArray, SVector, SMatrix, SArray
using TimerOutputs: @notimeit, @timeit, TimerOutput, print_timer, reset_timer!
using UnPack: @unpack

# Tullio.jl makes use of LoopVectorization.jl via Requires.jl.
# Hence, we need `using LoopVectorization` after loading Tullio and before using @tullio.
using Tullio: @tullio
using LoopVectorization

include("../../../src/parallel/parallel.jl")
include("../../../src/auxiliary/containers.jl")
include("../../../src/auxiliary/auxiliary.jl")
include("../../../src/equations/equations.jl")
include("../../../src/mesh/mesh.jl")
include("../../../src/run.jl")
include("../../../src/solvers/solvers.jl")
include("../../../src/solvers/dg/interpolation.jl")

include("approximation.jl")
include("function1d.jl")

init_mpi()
X = zeros(5,0)
Y = zeros(2,0)

n_traindata = zeros(26)
n_troubledcells = 0

#loop over meshs
for i in 1:4  
    println("Mesh $i")
    file = "utils/NN/1D/new/mesh$i.toml"
    parse_parameters_file(file)
    reset_timer!(timer())
    mesh = generate_mesh()

    leaf_cell_ids = leaf_cells(mesh.tree)
    n_elements = length(leaf_cell_ids)

    #loop over polydeg
    for r in 1:7
        #println("Polydeg $r")

        n_nodes = r + 1
        nodes, _ = gauss_lobatto_nodes_weights(n_nodes)
        _, inverse_vandermonde_legendre = vandermonde_legendre(nodes)
        node_coordinates = zeros(1,n_nodes, n_elements)
        modal = zeros(1, n_nodes, n_elements)
        c2e = zeros(Int, length(mesh.tree))

        for element_id in 1:n_elements 
            cell_id = leaf_cell_ids[element_id]
            dx = length_at_cell(mesh.tree, cell_id) 
            # Calculate node coordinates
            for i in 1:n_nodes
                node_coordinates[1, i, element_id] = (mesh.tree.coordinates[1, cell_id] + dx/2 * nodes[i])
            end
            c2e[leaf_cell_ids[element_id]] = element_id
        end
        
        #loop over functions
        for func in 1:26
            #println("Funktion $func")
            u(x) = trainfunction1d(func,x)
            for element_id in 1:n_elements 
                cell_id = leaf_cell_ids[element_id]
                neighbor_ids = Array{Int64}(undef, 2)
                n_traindata[func] += 1
                Xi = zeros(5,1)
                Yi = zeros(2,1)

                for direction in 1:n_directions(mesh.tree)
                    
                    if has_neighbor(mesh.tree, cell_id, direction)
                        neighbor_cell_id = mesh.tree.neighbor_ids[direction, cell_id]
                        if has_children(mesh.tree, neighbor_cell_id) # Cell has small neighbor
                            if direction == 1
                                neighbor_ids[direction] = c2e[mesh.tree.child_ids[2, neighbor_cell_id]]
                            else
                                neighbor_ids[direction] = c2e[mesh.tree.child_ids[1, neighbor_cell_id]]
                            end
                        else # Cell has same refinement level neighbor
                            neighbor_ids[direction] = c2e[neighbor_cell_id]
                        end
                    else # Cell is small and has large neighbor
                        parent_id = mesh.tree.parent_ids[cell_id]
                        neighbor_cell_id = mesh.tree.neighbor_ids[direction, parent_id]
                        neighbor_ids[direction] = c2e[neighbor_cell_id]
                    end
                end

                # Create Xi
                _, _, Xi[1] = legendreapprox(u, node_coordinates[1, 1, neighbor_ids[1]], node_coordinates[1, end, neighbor_ids[1]], r)
                Xi[4], Xi[5], Xi[2]= legendreapprox(u, node_coordinates[1, 1, element_id], node_coordinates[1, end, element_id], r)
                _, _, Xi[3] = legendreapprox(u, node_coordinates[1, 1, neighbor_ids[2]], node_coordinates[1, end, neighbor_ids[2]], r)


                h = length_at_cell(mesh.tree, cell_id)
                xi = mesh.tree.coordinates[1, cell_id]

                # Create label Yi
                Yi=Output(func,xi,h)

            
                # append to X and Y
                global X = cat(X, Xi, dims=2)
                global Y = cat(Y, Yi, dims=2)
            end

        end
        #troubled cells

        for t in 1:600
            ul = rand(Uniform(-1,1)) 
            ur = rand(Uniform(-1,1)) 
            x0 = rand(Uniform(-0.75, 0.75)) 
            u(x) = troubledcellfunctionstep(x, ul, ur, x0)

            for element_id in 1:n_elements 
                cell_id = leaf_cell_ids[element_id]
                # Get cell length
                dx = length_at_cell(mesh.tree, cell_id)
                #for last functions just troubled cells
                if good_cell(node_coordinates[:, 1, element_id], dx, x0)
                    continue
                end
                neighbor_ids = Array{Int64}(undef, 2)
                n_troubledcells += 1
                Xi = zeros(5,1)
                Yi = zeros(2,1)

                for direction in 1:n_directions(mesh.tree)
                    
                    if has_neighbor(mesh.tree, cell_id, direction)
                        neighbor_cell_id = mesh.tree.neighbor_ids[direction, cell_id]
                        if has_children(mesh.tree, neighbor_cell_id) # Cell has small neighbor
                            if direction == 1
                                neighbor_ids[direction] = c2e[mesh.tree.child_ids[2, neighbor_cell_id]]
                            else
                                neighbor_ids[direction] = c2e[mesh.tree.child_ids[1, neighbor_cell_id]]
                            end
                        else # Cell has same refinement level neighbor
                            neighbor_ids[direction] = c2e[neighbor_cell_id]
                        end
                    else # Cell is small and has large neighbor
                        parent_id = mesh.tree.parent_ids[cell_id]
                        neighbor_cell_id = mesh.tree.neighbor_ids[direction, parent_id]
                        neighbor_ids[direction] = c2e[neighbor_cell_id]
                    end
                end

                 # Create Xi
                 _, _, Xi[1] = legendreapprox(u, node_coordinates[1, 1, neighbor_ids[1]], node_coordinates[1, end, neighbor_ids[1]], r)
                 Xi[4], Xi[5], Xi[2]= legendreapprox(u, node_coordinates[1, 1, element_id], node_coordinates[1, end, element_id], r)
                 _, _, Xi[3] = legendreapprox(u, node_coordinates[1, 1, neighbor_ids[2]], node_coordinates[1, end, neighbor_ids[2]], r)
 
 
                 h = length_at_cell(mesh.tree, cell_id)
                 xi = mesh.tree.coordinates[1, cell_id]
 
                 # Create label Yi
                 Yi=Output(27,xi,h)

                 # append to X and Y
                 global X = cat(X, Xi, dims=2)
                 global Y = cat(Y, Yi, dims=2)


            end
        end
        println(size(X))
    end
    println(size(X))
end

println(size(X))

#scale data
println("Scale data")
for i in 1:size(X)[2]
    X[:,i]=X[:,i]./max(maximum(abs.(X[:,i])),1)
end

println("Safe data")
h5open("traindata1dnew.h5", "w") do file
    write(file, "X", X)
    write(file, "Y", Y)
end
println(n_traindata)
println(n_troubledcells)




