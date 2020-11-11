using Flux
using Flux: onecold, crossentropy, @epochs, Data.DataLoader, params
using Statistics
using HDF5: h5open
using BSON: @save

η = 0.001                # learning rate
β = 0.001                 # regularization paramter
ν = 0.001                # leakyeelu parameter
number_epochs = 100
Sb = 500                 # batch size
L = 10                   # early stopping parameter
R = 1

#Load data
x_train = h5open("utils/NN/1D/traindata.h5", "r") do file
    read(file, "X")
end
y_train = h5open("utils/NN/1D/traindata.h5", "r") do file
    read(file, "Y")
end
x_valid = h5open("utils/NN/1D/validdata.h5", "r") do file
    read(file, "X")
end
y_valid = h5open("utils/NN/1D/validdata.h5", "r") do file
    read(file, "Y")
end

#leakyreluv(x) = Flux.leakyrelu(x,ν)

opt_acc = 0
for r in 1:R
    @info("Building model...")
    model1d = Chain(
        Dense(5, 256, leakyrelu, initW=Flux.glorot_normal),
        Dense(256, 128, leakyrelu, initW=Flux.glorot_normal),
        Dense(128, 64, leakyrelu, initW=Flux.glorot_normal),
        Dense(64, 32, leakyrelu, initW=Flux.glorot_normal),
        Dense(32, 16, leakyrelu, initW=Flux.glorot_normal),
        Dense(16, 2),
        softmax) 

    # Getting predictions
    ŷ = model1d(x_train)

    opt = ADAM(η)
    ps = params(model1d)

    accuracy(ŷ, y) = mean(onecold(ŷ) .== onecold(y))
    sqnorm(x) = sum(abs2, x)
    loss(x, y) = Flux.crossentropy(model1d(x), y) + β * sum(sqnorm, params(model1d)[1:2:11])

    @info("Beginning training loop...")
    best_acc = 0.0
    last_improvement = 0 
    for epoch_idx in 1:number_epochs
        # Create the full dataset
        train_data = DataLoader((x_train, y_train) ; batchsize=Sb, shuffle=true)
        Flux.train!(loss, params(model1d), train_data, opt)
        acc = accuracy(model1d(x_valid), y_valid)
        acc2 = accuracy(model1d(x_train), y_train)
        loss_idx = loss(x_train, y_train)
        parasum = sum(sqnorm, params(model1d))
        @show epoch_idx, acc, loss_idx, parasum, acc2

        if acc > best_acc
            best_acc = acc
            last_improvement = epoch_idx
            #save model
            @save "utils/NN/1D/models/model-$(acc).bson" model1d
        end

        if epoch_idx - last_improvement >= L
            @warn("Early stopping")
            @show best_acc
            break
        end
    end
    if best_acc > opt_acc
        ps_opt = params(model1d)
        global opt_acc = best_acc
        #save model
        @save "utils/NN/1D/models/model-$(opt_acc).bson" model1d
    end
    println(accuracy(model1d(x_train), y_train))
    #println(accuracy(model1d(x_valid), y_valid))
end
