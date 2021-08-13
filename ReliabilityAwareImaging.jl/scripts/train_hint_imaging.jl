# Train an normalizing flow for seismic imaging
# Authors: Ali Siahkoohi, alisk@gatech.edu
# Date: February 2021
"""
Example:
julia train_hint_imaging.jl
"""

using DrWatson
@quickactivate :ReliabilityAwareImaging

using InvertibleNetworks
using HDF5
using JLD2
using Random
using Statistics
using ArgParse
using ProgressMeter
using Flux: gpu, cpu
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, ExpDecay, update!, ADAM

# Random seed
Random.seed!(19)

s = ArgParseSettings()
@add_arg_table! s begin
    "--max_epoch"
        help = "Maximum # of epochs"
        arg_type = Int
        default = 100
    "--lr"
        help = "starting learning rate"
        arg_type = Float32
        default = 1f-4
    "--lr_step"
        help = "lr scheduler applied at evey lr_step epoch"
        arg_type = Int
        default = 2
    "--batchsize"
        help = "batch size"
        arg_type = Int
        default = 16
    "--n_hidden"
        help = "# hidden channels"
        arg_type = Int
        default = 128
    "--depth"
        help = "depth of the network"
        arg_type = Int
        default = 12
    "--sim_name"
        help = "simulation name"
        arg_type = String
        default = "seismic-imaging-posterior-learning"
end
parsed_args = parse_args(s)

max_epoch = parsed_args["max_epoch"]
lr = parsed_args["lr"]
lr_step = parsed_args["lr_step"]
batchsize = parsed_args["batchsize"]
n_hidden = parsed_args["n_hidden"]
depth = parsed_args["depth"]
sim_name = parsed_args["sim_name"]

# Define raw data directory
mkpath(datadir("training-data"))
data_path = datadir("training-data", "training-pairs.h5")

# Download the dataset into the data directory if it does not exist
if isfile(data_path) == false
    run(`wget https://www.dropbox.com/s/53u8ckb9aje8xv4/'
        'training-pairs.h5 -q -O $data_path`)
end

# Load seismic images and create training and testing data
file = h5open(data_path, "r")
X_train = file["dm"][:, :, :, :]
Y_train = file["rtm"][:, :, :, :]

Y_train[1:10, :, :, :] .= 0f0

nx, ny, nc, nsamples = size(X_train)

AN_x = ActNorm(nsamples)
AN_y = ActNorm(nsamples)
X_train = AN_x.forward(X_train)
Y_train = AN_y.forward(Y_train)

AN_params_x = get_params(AN_x)
AN_params_y = get_params(AN_y)

# Split in training/testing
ntrain = Int(floor((nsamples*.9)))
train_idx = randperm(nsamples)[1:ntrain]

# Dimensions after wavelet squeeze to increase no. of channels
nx = Int(nx/2)
ny = Int(ny/2)
n_in = Int(nc*4)


# Create network
CH = NetworkConditionalHINT(
    n_in, n_hidden, depth, k1=3, k2=3, p1=1, p2=1
) |> gpu

X = wavelet_squeeze(X_train[:, :, :, 1:64]) |> gpu
Y = wavelet_squeeze(Y_train[:, :, :, 1:64]) |> gpu
# CH.forward(X, Y)

# Training
# Batch extractor
train_loader = DataLoader(train_idx, batchsize=batchsize, shuffle=true)
num_batches = length(train_loader)

# Optimizer
opt = Optimiser(ExpDecay(lr, 9f-1, num_batches*lr_step, 1f-6), ADAM(lr))

# Training log keeper
fval = zeros(Float32, num_batches*max_epoch)
p = Progress(num_batches*max_epoch)

for epoch=1:max_epoch
    for (itr, idx) in enumerate(train_loader)
        Base.flush(Base.stdout)

        # Apply wavelet squeeze (change dimensions to -> n1/2 x n2/2 x nc*4 x ntrain)
        X = wavelet_squeeze(X_train[:, :, :, idx])
        Y = wavelet_squeeze(Y_train[:, :, :, idx])

        X = X |> gpu
        Y = Y |> gpu

        fval[(epoch-1)*num_batches + itr] = loss_supervised(CH, X, Y)[1]

        ProgressMeter.next!(
            p;
            showvalues = [
                (:Epoch, epoch),
                (:Itreration, itr),
                (:NLL, fval[(epoch-1)*num_batches + itr])
            ]
        )

        # Update params
        for p in get_params(CH)
            update!(opt, p.data, p.grad)
        end
        clear_grad!(CH)
    end

    # Saving parameters and logs
    Params = get_params(CH) |> cpu
    save_dict = @strdict epoch max_epoch lr lr_step batchsize n_hidden depth sim_name Params fval train_idx AN_params_x AN_params_y
    @tagsave(
        datadir(sim_name, savename(save_dict, "jld2"; digits=6)),
        save_dict;
        safe=true
    )

end
