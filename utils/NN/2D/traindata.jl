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
include("functions.jl")

init_mpi()
X = zeros(15,0)
Y = zeros(2,0)
n_traindata = zeros(12)
n_troubledcells = zeros(2)

#loop over meshs
for i in 1:4  
    println("Mesh $i")
    file = "utils/NN/2D/mesh$i.toml"
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
        node_coordinates = zeros(2,n_nodes, n_nodes, n_elements)
        modal = zeros(1, n_nodes, n_nodes, n_elements)
        c2e = zeros(Int, length(mesh.tree))
        for element_id in 1:n_elements 
            cell_id = leaf_cell_ids[element_id]
            dx = length_at_cell(mesh.tree, cell_id) 
            # Calculate node coordinates
            for j in 1:n_nodes
                for i in 1:n_nodes
                    node_coordinates[1, i, j, element_id] = (mesh.tree.coordinates[1, cell_id] + dx/2 * nodes[i])
                    node_coordinates[2, i, j, element_id] = (mesh.tree.coordinates[2, cell_id] + dx/2 * nodes[j])
                end
            end
            c2e[leaf_cell_ids[element_id]] = element_id
        end
        
        #loop over functions
        for func in 1:12
            #println("Funktion $func")
            u(x,y) = trainfunction(func,x,y)

            for element_id in 1:n_elements 
                data = node_coordinates[:, :, :, element_id]
                indicator = zeros(1, n_nodes, n_nodes)
                indicator[1, :, :] = legendreapprox(u, data , r)
                modal[1,:,:, element_id] = multiply_dimensionwise_naive(inverse_vandermonde_legendre, indicator)
            end
            
            for element_id in 1:n_elements 
                cell_id = leaf_cell_ids[element_id]
                # Get cell length
                dx = length_at_cell(mesh.tree, cell_id)
                n_traindata[func] += 1
                neighbor_ids = 0
                Xi = zeros(15,1)
                Yi = zeros(2,1)

                for direction in 1:4
                    neighbor_ids = 0
                    # Ghost Layer einfügen 
                    # if no neighbor exists and current cell is not small --> Ghost Layer: TODO
                    if !has_any_neighbor(mesh.tree, cell_id, direction)
                        continue
                        #ToDo: Ghost Layer
                    end

                    #Get Input data from neighbors
                    if has_neighbor(mesh.tree, cell_id, direction)
                        neighbor_cell_id = mesh.tree.neighbor_ids[direction, cell_id]
                        if has_children(mesh.tree, neighbor_cell_id) # Cell has small neighbor 
                            #data2 for other neighbr and mean over both cells ?
                            if direction == 1
                                neighbor_ids = c2e[mesh.tree.child_ids[2, neighbor_cell_id]] 
                                neighbor_ids2 = c2e[mesh.tree.child_ids[4, neighbor_cell_id]] 
                                    
                                Xi[3*direction+1] = (modal[1,1,1,neighbor_ids] + modal[1,1,1,neighbor_ids2])/2
                                Xi[3*direction+2] = (modal[1,1,2,neighbor_ids] + modal[1,1,2,neighbor_ids2])/2
                                Xi[3*direction+3] = (modal[1,2,1,neighbor_ids] + modal[1,2,1,neighbor_ids2])/2
                            elseif direction == 2
                                neighbor_ids = c2e[mesh.tree.child_ids[1, neighbor_cell_id]]
                                neighbor_ids2 = c2e[mesh.tree.child_ids[3, neighbor_cell_id]]
                                    
                                Xi[3*direction+1] = (modal[1,1,1,neighbor_ids] + modal[1,1,1,neighbor_ids2])/2
                                Xi[3*direction+2] = (modal[1,1,2,neighbor_ids] + modal[1,1,2,neighbor_ids2])/2
                                Xi[3*direction+3] = (modal[1,2,1,neighbor_ids] + modal[1,2,1,neighbor_ids2])/2

                            elseif direction == 3
                                neighbor_ids = c2e[mesh.tree.child_ids[3, neighbor_cell_id]]
                                neighbor_ids2 = c2e[mesh.tree.child_ids[4, neighbor_cell_id]]
                                    
                                Xi[3*direction+1] = (modal[1,1,1,neighbor_ids] + modal[1,1,1,neighbor_ids2])/2
                                Xi[3*direction+2] = (modal[1,1,2,neighbor_ids] + modal[1,1,2,neighbor_ids2])/2
                                Xi[3*direction+3] = (modal[1,2,1,neighbor_ids] + modal[1,2,1,neighbor_ids2])/2

                            elseif direction == 4
                                neighbor_ids = c2e[mesh.tree.child_ids[1, neighbor_cell_id]]
                                neighbor_ids2 = c2e[mesh.tree.child_ids[2, neighbor_cell_id]]
                                
                                Xi[3*direction+1] = (modal[1,1,1,neighbor_ids] + modal[1,1,1,neighbor_ids2])/2
                                Xi[3*direction+2] = (modal[1,1,2,neighbor_ids] + modal[1,1,2,neighbor_ids2])/2
                                Xi[3*direction+3] = (modal[1,2,1,neighbor_ids] + modal[1,2,1,neighbor_ids2])/2

                            end
                        else # Cell has same refinement level neighbor
                            neighbor_ids = c2e[neighbor_cell_id]
                                
                            Xi[3*direction+1]=modal[1,1,1,neighbor_ids]
                            Xi[3*direction+2]=modal[1,2,1,neighbor_ids]
                            Xi[3*direction+3]=modal[1,1,2,neighbor_ids]
                        end
                    else # Cell is small and has large neighbor
                        parent_id = mesh.tree.parent_ids[cell_id]
                        neighbor_ids = c2e[mesh.tree.neighbor_ids[direction, parent_id]]
                            
                        Xi[3*direction+1]=modal[1,1,1,neighbor_ids]
                        Xi[3*direction+2]=modal[1,2,1,neighbor_ids]
                        Xi[3*direction+3]=modal[1,1,2,neighbor_ids]

                    end
                end
            
                Xi[1]=modal[1,1,1,element_id]
                Xi[2]=modal[1,2,1,element_id]
                Xi[3]=modal[1,1,2,element_id]

                # Create label Yi
                Yi=[0; 1] #good cell

                # append to X and Y
                global X = cat(X, Xi, dims=2)
                global Y = cat(Y, Yi, dims=2)
            end
            #println(size(X))
        end
        println(size(X))
        
        #troubled cells
        for t in 1:160
            if t < 140
                func = 1
                a = rand(Uniform(-100, 100)) 
                m = rand(Uniform(-1,1)) 
                x0 = rand(Uniform(-0.5, 0.5)) 
                y0 = rand(Uniform(-0.5, 0.5)) 
                u1(x,y) = troubledcellfunctionabs(x, y, a, m, x0, y0)
            elseif t >=140
                func = 2
                ui = rand(Uniform(-1,1),4)
                m = rand(Uniform(0,20))
                x0 = rand(Uniform(-0.5, 0.5))
                y0 = rand(Uniform(-0.5, 0.5))
                u2(x,y) = troubledcellfunctionstep(x, y, ui, m, x0, y0)
            end
            
            for element_id in 1:n_elements 
                data = node_coordinates[:, :, :, element_id]
                indicator = zeros(1, n_nodes, n_nodes)
                if func == 1
                    indicator[1, :, :] = legendreapprox(u1, data , r)
                elseif func == 2
                    indicator[1, :, :] = legendreapprox(u2, data , r)
                end
                modal[1,:,:, element_id] = multiply_dimensionwise_naive(inverse_vandermonde_legendre, indicator)
            end
            for element_id in 1:n_elements 
                cell_id = leaf_cell_ids[element_id]
                # Get cell length
                dx = length_at_cell(mesh.tree, cell_id)
                #for last functions just troubled cells
                if good_cell(node_coordinates[:, 1, 1, element_id], dx, func, m, x0, y0)
                    continue
                end
                n_troubledcells[func] += 1
                neighbor_ids = 0
                Xi = zeros(15,1)
                Yi = zeros(2,1)

                for direction in 1:4
                    neighbor_ids = 0
                    # Ghost Layer einfügen 
                    # if no neighbor exists and current cell is not small --> Ghost Layer: TODO
                    if !has_any_neighbor(mesh.tree, cell_id, direction)
                        continue
                        #ToDo: Ghost Layer
                    end

                    #Get Input data from neighbors
                    if has_neighbor(mesh.tree, cell_id, direction)
                        neighbor_cell_id = mesh.tree.neighbor_ids[direction, cell_id]
                        if has_children(mesh.tree, neighbor_cell_id) # Cell has small neighbor 
                            #data2 for other neighbr and mean over both cells ?
                            if direction == 1
                                neighbor_ids = c2e[mesh.tree.child_ids[2, neighbor_cell_id]] 
                                neighbor_ids2 = c2e[mesh.tree.child_ids[4, neighbor_cell_id]] 
                                    
                                Xi[3*direction+1] = (modal[1,1,1,neighbor_ids] + modal[1,1,1,neighbor_ids2])/2
                                Xi[3*direction+2] = (modal[1,1,2,neighbor_ids] + modal[1,1,2,neighbor_ids2])/2
                                Xi[3*direction+3] = (modal[1,2,1,neighbor_ids] + modal[1,2,1,neighbor_ids2])/2
                            elseif direction == 2
                                neighbor_ids = c2e[mesh.tree.child_ids[1, neighbor_cell_id]]
                                neighbor_ids2 = c2e[mesh.tree.child_ids[3, neighbor_cell_id]]
                                    
                                Xi[3*direction+1] = (modal[1,1,1,neighbor_ids] + modal[1,1,1,neighbor_ids2])/2
                                Xi[3*direction+2] = (modal[1,1,2,neighbor_ids] + modal[1,1,2,neighbor_ids2])/2
                                Xi[3*direction+3] = (modal[1,2,1,neighbor_ids] + modal[1,2,1,neighbor_ids2])/2

                            elseif direction == 3
                                neighbor_ids = c2e[mesh.tree.child_ids[3, neighbor_cell_id]]
                                neighbor_ids2 = c2e[mesh.tree.child_ids[4, neighbor_cell_id]]
                                    
                                Xi[3*direction+1] = (modal[1,1,1,neighbor_ids] + modal[1,1,1,neighbor_ids2])/2
                                Xi[3*direction+2] = (modal[1,1,2,neighbor_ids] + modal[1,1,2,neighbor_ids2])/2
                                Xi[3*direction+3] = (modal[1,2,1,neighbor_ids] + modal[1,2,1,neighbor_ids2])/2

                            elseif direction == 4
                                neighbor_ids = c2e[mesh.tree.child_ids[1, neighbor_cell_id]]
                                neighbor_ids2 = c2e[mesh.tree.child_ids[2, neighbor_cell_id]]
                                
                                Xi[3*direction+1] = (modal[1,1,1,neighbor_ids] + modal[1,1,1,neighbor_ids2])/2
                                Xi[3*direction+2] = (modal[1,1,2,neighbor_ids] + modal[1,1,2,neighbor_ids2])/2
                                Xi[3*direction+3] = (modal[1,2,1,neighbor_ids] + modal[1,2,1,neighbor_ids2])/2

                            end
                        else # Cell has same refinement level neighbor
                            neighbor_ids = c2e[neighbor_cell_id]
                                
                            Xi[3*direction+1]=modal[1,1,1,neighbor_ids]
                            Xi[3*direction+2]=modal[1,2,1,neighbor_ids]
                            Xi[3*direction+3]=modal[1,1,2,neighbor_ids]
                        end
                    else # Cell is small and has large neighbor
                        parent_id = mesh.tree.parent_ids[cell_id]
                        neighbor_ids = c2e[mesh.tree.neighbor_ids[direction, parent_id]]
                            
                        Xi[3*direction+1]=modal[1,1,1,neighbor_ids]
                        Xi[3*direction+2]=modal[1,2,1,neighbor_ids]
                        Xi[3*direction+3]=modal[1,1,2,neighbor_ids]

                    end
                end
            
                Xi[1]=modal[1,1,1,element_id]
                Xi[2]=modal[1,2,1,element_id]
                Xi[3]=modal[1,1,2,element_id]

                # Create label Yi
                Yi=[1; 0] #troubled cell

                # append to X and Y
                global X = cat(X, Xi, dims=2)
                global Y = cat(Y, Yi, dims=2)
            end
        end
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
h5open("traindata2d.h5", "w") do file
    write(file, "X", X)
    write(file, "Y", Y)
end
println(n_traindata)
println(n_troubledcells)




