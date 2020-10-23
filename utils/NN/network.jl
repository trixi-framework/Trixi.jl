using Flux
using Flux: onecold, crossentropy, @epochs, Data.DataLoader, params
using Statistics
using HDF5: h5open
using BSON: @save

η = 0.001                # learning rate
β = 0.01                 # regularization paramter
ν = 0.001                # leakyeelu parameter
number_epochs = 100
Sb = 500                 # batch size
L = 10                   # early stopping parameter
R = 10

#Load data
x_train = h5open("utils/NN/traindata.h5", "r") do file
    read(file, "X")
end
y_train = h5open("utils/NN/traindata.h5", "r") do file
    read(file, "Y")
end
x_valid = h5open("utils/NN/validdata.h5", "r") do file
    read(file, "X")
end
y_valid = h5open("utils/NN/validdata.h5", "r") do file
    read(file, "Y")
end

#leakyreluv(x) = Flux.leakyrelu(x,ν)

opt_acc = 0
for r in 1:R
    @info("Building model...")
    model = Chain(
        Dense(5, 256, leakyrelu, initW=Flux.glorot_normal),
        Dense(256, 128, leakyrelu, initW=Flux.glorot_normal),
        Dense(128, 64, leakyrelu, initW=Flux.glorot_normal),
        Dense(64, 32, leakyrelu, initW=Flux.glorot_normal),
        Dense(32, 16, leakyrelu, initW=Flux.glorot_normal),
        Dense(16, 2),
        softmax) 

    # Getting predictions
    ŷ = model(x_train)

    opt = ADAM(η)
    ps = params(model)

    accuracy(ŷ, y) = mean(onecold(ŷ) .== onecold(y))
    sqnorm(x) = sum(abs2, x)
    loss(x, y) = Flux.crossentropy(model(x), y) #+ β * sum(sqnorm, params(model)[1:2:11])

    @info("Beginning training loop...")
    best_acc = 0.0
    last_improvement = 0 
    for epoch_idx in 1:number_epochs
        # Create the full dataset
        train_data = DataLoader((x_train, y_train) ; batchsize=Sb, shuffle=true)
        Flux.train!(loss, params(model), train_data, opt)
        acc = accuracy(model(x_valid), y_valid)
        loss_idx = loss(x_train, y_train)
        parasum = sum(sqnorm, params(model))
        @show epoch_idx, acc, loss_idx, parasum

        if acc > best_acc
            best_acc = acc
            last_improvement = epoch_idx
            #save model
            #@save "utils/NN/1D/model-$(acc).bson" model
        end

        if epoch_idx - last_improvement >= L
            @warn("Early stopping")
            @show best_acc
            break
        end
    end
    if best_acc > opt_acc
        ps_opt = params(model)
        global opt_acc = best_acc
        #save model
        #@save "utils/NN/1D/model-$(opt_acc).bson" model
    end
    #println(accuracy(model(x_train), y_train))
    #println(accuracy(model(x_valid), y_valid))
end
