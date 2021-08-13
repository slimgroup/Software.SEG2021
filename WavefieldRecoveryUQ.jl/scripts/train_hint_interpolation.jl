# Train an normalizing flow for seismic interpolation
# Authors: Rajiv Kumar
#          Ali Siahkoohi, alisk@gatech.edu
# Date: February 2021
"""
Example:
julia train_hint_interpolation.jl
"""

using DrWatson
@quickactivate :WavefieldRecoveryUQ

using InvertibleNetworks
using JLD2
using HDF5
using Random
using Statistics
using ArgParse
using ProgressMeter
using Flux: gpu, cpu
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, ExpDecay, update!, ADAM

# Random seed
Random.seed!(66)

s = ArgParseSettings()
@add_arg_table! s begin
    "--max_epoch"
        help = "Maximum # of epochs"
        arg_type = Int
        default = 12
    "--lr"
        help = "starting learning rate"
        arg_type = Float32
        default = 1f-3
    "--lr_step"
        help = "lr scheduler applied at evey lr_step epoch"
        arg_type = Int
        default = 1
    "--batchsize"
        help = "batch size"
        arg_type = Int
        default = 8
    "--n_hidden"
        help = "# hidden channels"
        arg_type = Int
        default = 64
    "--depth"
        help = "depth of the network"
        arg_type = Int
        default = 8
    "--subsample_fac"
        help = "subsampling factor"
        arg_type = Float32
        default = 7.5f-1
    "--sim_name"
        help = "simulation name"
        arg_type = String
        default = "wavefield_recovery_and_UQ"
end
parsed_args = parse_args(s)

max_epoch = parsed_args["max_epoch"]
lr = parsed_args["lr"]
lr_step = parsed_args["lr_step"]
batchsize = parsed_args["batchsize"]
n_hidden = parsed_args["n_hidden"]
depth = parsed_args["depth"]
subsample_fac = parsed_args["subsample_fac"]
sim_name = parsed_args["sim_name"]

# Define raw data directory
mkpath(datadir("dataset"))
data_path = datadir("dataset", "frequency_slices.h5")

# Download the dataset into the data directory if it does not exist
if isfile(data_path) == false
    run(`wget https://www.dropbox.com/s/wcqucrlye615fhi/'
        'frequency_slices.h5 -q -O $data_path`)
end

# Load seismic images and create training and testing data
h5open(data_path, "r") do file
    X_orig = file["dataset"][:, :, :, 1:2:end]

    X_orig = permutedims(X_orig, (2, 3, 1, 4))
    global nx, ny, nc, nsamples = size(X_orig)

    # Whiten data
    AN = ActNorm(nsamples)
    X_orig = AN.forward(X_orig)
    global AN_params = get_params(AN)

    # Forward operator
    global M = Mask([nx, ny], subsample_fac)

    # Split in training/testing
    ntrain = Int(floor((nsamples*.75)))
    ntest = nsamples - ntrain

    rand_per = randperm(nsamples)
    global train_idx = rand_per[1:ntrain]
    global test_idx = rand_per[ntrain+1:end]

    global X_train = X_orig[:, :, :, train_idx]
    global Y_train = M*X_train

end

# Create network
CH = NetworkConditionalHINT(
    nc, n_hidden, depth, k1=3, k2=3, p1=1, p2=1
) |> gpu

# Training
# Batch extractor
train_loader = DataLoader(
    range(1, length(train_idx), step=1),batchsize=batchsize, shuffle=true
)
num_batches = length(train_loader)

# Optimizer
opt = Optimiser(ExpDecay(lr, .9f0, num_batches*lr_step, 0f0), ADAM(lr))

# Training log keeper
fval = zeros(Float32, num_batches*max_epoch)
p = Progress(num_batches*max_epoch)

for epoch=1:max_epoch
    for (itr, idx) in enumerate(train_loader)
        Base.flush(Base.stdout)

        X = X_train[:, :, :, idx]
        Y = Y_train[:, :, :, idx]

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
    save_dict = @strdict epoch max_epoch lr lr_step batchsize n_hidden depth subsample_fac sim_name M.M AN_params Params fval train_idx test_idx
    @tagsave(
        datadir(sim_name, savename(save_dict, "jld2"; digits=6)),
        save_dict;
        safe=true
    )

end
