# Testing a pretrained normalizing flow for seismic interpolation
# Authors: Rajiv Kumar
#          Ali Siahkoohi, alisk@gatech.edu
# Date: February 2021
"""
Example:
julia test_hint_interpolation.jl
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
using PyPlot
using Seaborn
using LinearAlgebra
using Flux: gpu, cpu
using Flux.Data: DataLoader

set_style("whitegrid")
rc("font", family="serif", size=13)
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
        default = 12
    "--epoch"
        help = "Epoch to load net params"
        arg_type = Int
        default = -1
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

if parsed_args["epoch"] == -1
    parsed_args["epoch"] = parsed_args["max_epoch"]
end
epoch = parsed_args["epoch"]

# Define raw data directory
mkpath(datadir("dataset"))
data_path = datadir("dataset", "frequency_slices.h5")

# Download the dataset into the data directory if it does not exist
if isfile(data_path) == false
    run(`wget https://www.dropbox.com/s/wcqucrlye615fhi/'
        'frequency_slices.h5 -q -O $data_path`)
end

# Loading the experiment—only network weights and training loss
Params, fval, exp_path = load_experiment(parsed_args; return_path=true)

mask = wload(exp_path)["M.M"]
AN_params = wload(exp_path)["AN_params"]
train_idx = wload(exp_path)["train_idx"]
test_idx = wload(exp_path)["test_idx"]

# Load seismic images and create training and testing data
h5open(data_path, "r") do file
    X_orig = file["dataset"][:, :, :, 1:2:end]

    X_orig = permutedims(X_orig, (2, 3, 1, 4))
    global nx, ny, nc, nsamples = size(X_orig)

    # Whiten data
    global AN = ActNorm(nsamples)
    put_params!(AN, convert(Array{Any,1}, AN_params))
    X_orig = AN.forward(X_orig)

    # Forward operator
    global M = Mask(mask, subsample_fac)

    global idx = shuffle(test_idx)[1]

    global X_fixed = X_orig[:, :, :, idx:idx]
    global Y_fixed = M*X_orig[:, :, :, idx:idx]

end

# Create network
CH = NetworkConditionalHINT(
    nx, ny, nc, batchsize, n_hidden, depth, k1=3, k2=3, p1=1, p2=1
)

# Loading the experiment—only network weights and training loss
put_params!(CH, convert(Array{Any,1}, Params))

# Now select single fixed sample from all Ys
Zy_fixed = CH.forward_Y(Y_fixed)

# Draw new Zx, while keeping Zy fixed
n_samples = 1000
X_post = zeros(Float32, nx, ny, nc, n_samples)
CH = CH |> gpu

test_loader = DataLoader(
    (randn(Float32, nx, ny, nc, n_samples), repeat(Zy_fixed, 1, 1, 1, n_samples)),
    batchsize=8, shuffle=false
)

p = Progress(length(test_loader))
for (itr, (X, Y)) in enumerate(test_loader)
    Base.flush(Base.stdout)
    counter = (itr - 1)*size(X)[4] + 1

    X = X |> gpu
    Y = Y |> gpu

    X_post[:, :, :, counter:(counter+size(X)[4]-1)] = (CH.inverse(X, Y)[1] |> cpu)
    ProgressMeter.next!(p)
end

X_post = AN.inverse(X_post)
X_fixed = AN.inverse(X_fixed)
Y_fixed = AN.inverse(Y_fixed)

# Some stats
X_post_mean = mean(X_post; dims=4)
X_post_std = std(X_post; dims=4)

min_std=1.5f0*minimum(X_post_std)
max_std=0.93f0*maximum(X_post_std)

X_post = (1 .- M.M) .* X_post .+ (M.M .* Y_fixed)
X_post_mean = (1 .- M.M) .* X_post_mean .+ (M.M .* Y_fixed)
X_post_std = (1 .- M.M) .* X_post_std

# Some stats
X_post_mean = mean(X_post; dims=4)
X_post_std = std(X_post; dims=4)

save_dict = @strdict max_epoch epoch lr lr_step batchsize n_hidden depth sim_name
save_path = plotsdir(sim_name, savename(save_dict; digits=6))

snr_list = []
signal_to_noise(x, xhat) = (-20.0 * log(norm(x - xhat)/norm(x))/log(10.0))

for j = 1:n_samples
    push!(snr_list, signal_to_noise(X_post[:, :, :, j], X_fixed[:, :, :, 1]))
end
X_post_mean_snr = signal_to_noise(X_post_mean[:, :, :, 1], X_fixed[:, :, :, 1])

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
safesave(joinpath(save_path, "log.png"), fig)
close(fig)

# Plot posterior samples, mean and standard deviation
fig = figure(figsize=(18, 8))
subplot(2,4,1)
imshow(
    X_fixed[:, :, 1, 1], vmin=-6.0, vmax=6.0, aspect=1, cmap="Greys", resample=true,
    interpolation="lanczos", filterrad=1
)
title(L"Fully sampled: $\mathbf{x} \sim p_x$")
colorbar(fraction=0.047, pad=0.01)
grid(false)

subplot(2,4,2)
imshow(
    X_post[:, :, 1, 1], vmin=-6.0, vmax=6.0, aspect=1, cmap="Greys", resample=true,
    interpolation="lanczos", filterrad=1
)
title(L"Conditional sample: $\widehat{\mathbf{x}} \sim \mathbf{x} \mid \mathbf{y}$")
colorbar(fraction=0.047, pad=0.01)
grid(false)

subplot(2,4,3)
imshow(
    X_post[:, :, 1, 2], vmin=-6.0, vmax=6.0, aspect=1, cmap="Greys", resample=true,
    interpolation="lanczos", filterrad=1
)
title(L"Conditional sample: $\widehat{\mathbf{x}} \sim \mathbf{x} \mid \mathbf{y}$")
colorbar(fraction=0.047, pad=0.01)
grid(false)

subplot(2,4,4)
imshow(
    X_post_mean[:, :, 1, 1], vmin=-6.0, vmax=6.0, aspect=1, cmap="Greys", resample=true,
    interpolation="lanczos", filterrad=1
)
title(L"Conditional mean: $\mathbb{E} \left [\mathbf{x} \mid \mathbf{y} \right ]$")
colorbar(fraction=0.047, pad=0.01)
grid(false)

subplot(2,4,5)
imshow(
    Y_fixed[:, :, 1, 1], vmin=-6.0, vmax=6.0, aspect=1, cmap="Greys", resample=true,
    interpolation="lanczos", filterrad=1
)
title(L"Subsampled $\mathbf{y} = \mathbf{M} \odot \mathbf{x}$ ")
colorbar(fraction=0.047, pad=0.01)
grid(false)

subplot(2,4,6)
imshow(
    X_post[:, :, 1, 4], vmin=-6.0, vmax=6.0, aspect=1, cmap="Greys", resample=true,
    interpolation="lanczos", filterrad=1
)
title(L"Conditional sample: $\widehat{\mathbf{x}} \sim \mathbf{x} \mid \mathbf{y}$")
colorbar(fraction=0.047, pad=0.01)
grid(false)

subplot(2,4,7)
imshow(
    X_post[:, :, 1, 5], vmin=-6.0, vmax=6.0, aspect=1, cmap="Greys", resample=true,
    interpolation="lanczos", filterrad=1
)
title(L"Conditional sample: $\widehat{\mathbf{x}} \sim \mathbf{x} \mid \mathbf{y}$")
colorbar(fraction=0.047, pad=0.01)
grid(false)

subplot(2,4,8)
imshow(
    X_post_std[:, :, 1,1], vmin=min_std, vmax=max_std,
    cmap="OrRd", resample=true, interpolation="lanczos", filterrad=1
)
title("Pointwise standard deviation");
colorbar(fraction=0.047, pad=0.01)
grid(false)
tight_layout()
safesave(joinpath(save_path, "posterior-"*string(test_idx[1])*".png"), fig)
close(fig)
