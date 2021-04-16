# Testing a pretrained normalizing flow for seismic imaging
# Authors: Ali Siahkoohi, alisk@gatech.edu
# Date: September 2020
"""
Example:
julia test_hint_imaging.jl
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
using PyPlot
using Seaborn
using LinearAlgebra
using Flux: gpu, cpu
using Flux.Data: DataLoader

set_style("whitegrid")
rc("font", family="serif", size=12)
font_prop = matplotlib.font_manager.FontProperties(
    family="serif",
    style="normal",
    size=13
)
sfmt=matplotlib.ticker.ScalarFormatter(useMathText=true)
sfmt.set_powerlimits((0, 0))
matplotlib.use("Agg")


s = ArgParseSettings()
@add_arg_table! s begin
    "--max_epoch"
        help = "Maximum # of epochs"
        arg_type = Int
        default = 100
    "--epoch"
        help = "Epoch to load net params"
        arg_type = Int
        default = -1
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

if parsed_args["epoch"] == -1
    parsed_args["epoch"] = parsed_args["max_epoch"]
end
epoch = parsed_args["epoch"]

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

nx, ny, nc, nsamples = size(X_train)

AN_x = ActNorm(nsamples)
AN_y = ActNorm(nsamples)
X_train = AN_x.forward(X_train)
Y_train = AN_y.forward(Y_train)


# Dimensions after wavelet squeeze to increase no. of channels
nx = Int(nx/2)
ny = Int(ny/2)
n_in = Int(nc*4)

# Create network
CH = NetworkConditionalHINT(
    nx, ny, n_in, batchsize, n_hidden, depth, k1=3, k2=3, p1=1, p2=1
)


# Loading the experiment—only network weights and training loss
Params, fval, exp_path = load_experiment(parsed_args; return_path=true)
train_idx = wload(exp_path)["train_idx"]
put_params!(CH, convert(Array{Any,1}, Params))

# test data pairs
idx = shuffle(setdiff(1:nsamples, train_idx))[1]
X_fixed = wavelet_squeeze(X_train[:, :, :, idx:idx])
Y_fixed = wavelet_squeeze(Y_train[:, :, :, idx:idx])


# Now select single fixed sample from all Ys
Zy_fixed = CH.forward_Y(Y_fixed)

# Draw new Zx, while keeping Zy fixed
n_samples = 1000
X_post = zeros(Float32, nx, ny, n_in, n_samples)
CH = CH |> gpu

test_batchsize = 16
test_loader = DataLoader(
    (randn(Float32, nx, ny, n_in, n_samples), repeat(Zy_fixed, 1, 1, 1, n_samples)),
    batchsize=test_batchsize, shuffle=false
)

p = Progress(length(test_loader))

for (itr, (X, Y)) in enumerate(test_loader)
    Base.flush(Base.stdout)
    counter = (itr - 1)*test_batchsize + 1

    X = X |> gpu
    Y = Y |> gpu

    X_post[:, :, :, counter:(counter+size(X)[4]-1)] = (CH.inverse(X, Y)[1] |> cpu)
    ProgressMeter.next!(p)
end

X_post = wavelet_unsqueeze(X_post)
X_post[1:10, :, :, :] .= 0f0
X_post = AN_x.inverse(X_post)


# Some stats
X_post_mean = mean(X_post; dims=4)
X_post_std = std(X_post; dims=4)

X_fixed = wavelet_unsqueeze(X_fixed)
Y_fixed = wavelet_unsqueeze(Y_fixed)
Zy_fixed = wavelet_unsqueeze(Zy_fixed)

X_fixed = AN_x.inverse(X_fixed)
Y_fixed = AN_y.inverse(Y_fixed)
Y_fixed[1:10, :, :, :] .= 0f0

save_dict = @strdict max_epoch epoch lr lr_step batchsize n_hidden depth sim_name
save_path = plotsdir(sim_name, savename(save_dict; digits=6))

spacing = [20.0, 12.5]
extent = [0., size(X_fixed, 1)*spacing[1], size(X_fixed, 2)*spacing[2], 0.]/1e3


# Training loss
fig = figure("training logs", figsize=(7, 2.5))
if epoch == parsed_args["max_epoch"]
    plot(range(0, epoch, length=length(fval)), fval, color="#d48955")
else
    plot(
        range(0, epoch, length=length(fval[1:findfirst(fval .== 0f0)-1])),
        fval[1:findfirst(fval .== 0f0)-1], color="#d48955"
    )
end
title("Negative log-likelihood")
ylabel("Training objective")
xlabel("Epochs")
wsave(joinpath(save_path, "log.png"), fig)
close(fig)


# Plot the true model
fig = figure("x", figsize=(7.68, 4.8))
imshow(
    X_fixed[:, :, 1, 1], vmin=-1.5e3, vmax=1.5e3, aspect=1, cmap="Greys", resample=true,
    interpolation="lanczos", filterrad=1, extent=extent
)
title(L"High-fidelity image, $\mathbf{x}$")
colorbar(fraction=0.03, pad=0.01, format=sfmt)
grid(false)
xlabel("Horizontal distance (km)")
ylabel("Depth (km)")
safesave(joinpath(save_path, string(idx),  "true_model.png"), fig)
close(fig)

# Plot the observed data
fig = figure("y", figsize=(7.68, 4.8))
imshow(
    Y_fixed[:, :, 1, 1], vmin=-1.5e6, vmax=1.5e6, aspect=1, cmap="Greys", resample=true,
    interpolation="lanczos", filterrad=1, extent=extent
)
title(L"Low-fidelity reverse-time migrated image, $\mathbf{y}$")
colorbar(fraction=0.03, pad=0.01, format=sfmt)
grid(false)
xlabel("Horizontal distance (km)")
ylabel("Depth (km)")
safesave(joinpath(save_path, string(idx), "observed_data.png"), fig)
close(fig)

# Plot the conditional mean estimate
fig = figure("x_cm", figsize=(7.68, 4.8))
imshow(
    X_post_mean[:, :, 1, 1], vmin=-1.5e3, vmax=1.5e3, aspect=1, cmap="Greys", resample=true,
    interpolation="lanczos", filterrad=1, extent=extent
)
title(L"Conditional mean, $\mu$")
colorbar(fraction=0.03, pad=0.01, format=sfmt)
grid(false)
xlabel("Horizontal distance (km)")
ylabel("Depth (km)")
safesave(joinpath(save_path, string(idx), "conditional_mean.png"), fig)
close(fig)

# Plot the pointwise standard deviation
fig = figure("x_std", figsize=(7.68, 4.8))
imshow(
    X_post_std[:, :, 1, 1], vmin=60,
    vmax=320.0, aspect=1, cmap="OrRd", resample=true,
    interpolation="lanczos", filterrad=1, extent=extent,
    norm=matplotlib.colors.LogNorm()
)
title(L"Pointwise standard deviation, $\sigma$")
cp = colorbar(fraction=0.03, pad=0.01)
grid(false)
xlabel("Horizontal distance (km)")
ylabel("Depth (km)")
safesave(joinpath(save_path, string(idx), "pointwise_std.png"), fig)
close(fig)

for ns= 1:10
    fig = figure("x_cm", figsize=(7.68, 4.8))
    imshow(
        X_post[:, :, 1, ns], vmin=-1.5e3, vmax=1.5e3, aspect=1, cmap="Greys",
        resample=true, interpolation="lanczos", filterrad=1, extent=extent
    )
    title(L"Posterior sample, $\mathbf{x} \sim p(\mathbf{x} \mid \mathbf{y})$")
    colorbar(fraction=0.03, pad=0.01, format=sfmt)
    grid(false)
    xlabel("Horizontal distance (km)")
    ylabel("Depth (km)")
    safesave(joinpath(save_path, string(idx), "sample.png"), fig)
    close(fig)
end
