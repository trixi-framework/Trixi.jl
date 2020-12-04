using Flux
using Flux: onecold, crossentropy, @epochs, Data.DataLoader, params, Dropout
using Statistics
using HDF5: h5open
using BSON: @save

η = 0.001                # learning rate
β = 0.001             # regularization parameter
ν = 0.001                # leakyrelu parameter
number_epochs = 100
Sb = 500                 # batch size
L = 10                   # early stopping parameter
R = 1

# Load data
x_train = h5open("utils/NN/2D/traindata/traindata2dlagrangemodal3.h5", "r") do file
    read(file, "X")
end
y_train = h5open("utils/NN/2D/traindata/traindata2dlagrangemodal3.h5", "r") do file
    read(file, "Y")
end
x_valid = h5open("utils/NN/2D/validdata2dlagrangemodal.h5", "r") do file
    read(file, "X")
end
y_valid = h5open("utils/NN/2D/validdata2dlagrangemodal.h5", "r") do file
    read(file, "Y")
end

println(size(x_train))

leakyreluv(x) = Flux.leakyrelu(x,ν)


function trainnetwork(d1,d2,d3,d4,d5)
    #opt_acc = 0
    #for r in 1:R
        @info("Building model...")
        model2d = Chain(
            Dense(15, d1, leakyrelu, initW=Flux.glorot_normal),         #initW=Flux.glorot_normal
            Dense(d1, d2, leakyrelu, initW=Flux.glorot_normal),
            Dense(d2, d3, leakyrelu, initW=Flux.glorot_normal),
            #Dropout(0.1),
            Dense(d3, d4, leakyrelu, initW=Flux.glorot_normal),
            Dense(d4, d5, leakyrelu, initW=Flux.glorot_normal),
            Dense(d5, 2, initW=Flux.glorot_normal),
            softmax) 

        # Getting predictions
        #ŷ = model2d(x_train)

        opt = ADAM(η)
        ps = params(model2d)

        accuracy(ŷ, y) = mean(onecold(ŷ) .== onecold(y))
        sqnorm(x) = sum(abs2, x)
        loss(x, y) = Flux.crossentropy(model2d(x), y) + β * sum(sqnorm, params(model2d)[1:2:11]) 

        @info("Beginning training loop...")
        best_acc = 0.0
        last_improvement = 0

        for epoch_idx in 1:number_epochs
            # Create the full dataset
            train_data = DataLoader((x_train, y_train) ; batchsize=Sb, shuffle=true)
            Flux.train!(loss, params(model2d), train_data, opt)
            acc = accuracy(model2d(x_valid), y_valid) 
            acc2 = accuracy(model2d(x_train), y_train) 
            loss_idx = loss(x_train, y_train)
            parasum = sum(sqnorm, params(model2d))
            @show epoch_idx, acc, loss_idx, parasum, acc2

            if acc > best_acc
                best_acc = acc
                last_improvement = epoch_idx
                #save model
                @save "utils/NN/2D/models/modellagmodal3-$(acc).bson" model2d
            end

            if epoch_idx - last_improvement >= L
                @warn("Early stopping")
                @show best_acc
                break
            end
        end
        #=
        if best_acc > opt_acc
            ps_opt = params(model2d)
            global opt_acc = best_acc
            #save model
            @save "utils/NN/2D/models/model-$(opt_acc).bson" model2d
        end
        =#
        println(accuracy(model2d(x_train), y_train))
        @show d1, d2, d3, d4, d5
        #println(accuracy(model2d(x_valid), y_valid))
    #end
    return best_acc
end


grid_best = 0
grid_acc = 0
for d1 in [20], d2 in [20], d3 in [20], d4 in [20], d5 in [20]
    grid_acc = trainnetwork(d1, d2, d3, d4, d5)
    if grid_acc > grid_best
        grid_best = grid_acc
        @show grid_acc , d1, d2, d3, d4, d5
    end
end