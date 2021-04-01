# WavefieldRecoveryUQ.jl

Experiments for "Enabling uncertainty quantification for seismic data pre-processing using normalizing flows (NF)—an interpolation example", submitted to the [SEG 2021 Annual Meeting](https://seg.org/AM/).

To start running the examples, clone the repository:

```bash
$ git clone https://github.com/slimgroup/Software.SEG2021
$ cd WavefieldRecoveryUQ.jl
```

Here, we heavily rely on [InvertibleNetworks.jl](https://github.com/slimgroup/InvertibleNetworks.jl), a recently-developed, memory-efficient framework for training invertible networks in Julia.

## Installation

This repository is based on [DrWatson.jl](https://github.com/JuliaDynamics/DrWatson.jl). Before running examples, install `DrWatson.jl` by:

```julia
pkg> add DrWatson
```

The only other manual installation is to make sure you have `matplotlib` and `seaborn` installed in your Python environment since we depend on `PyPlot.jl` and `Seaborn.jl` for creating figures.

The necessary dependencies will be installed upon running your first experiment. If you happen to have a CUDA-enabled GPU, the code will run on it. The training dataset will also download automatically into `data/dataset/` directory.

### Example

Run the script below for training the normalizing flow:

```bash
$ julia scripts/train_hint_interpolation.jl
```

To perform conditional (posterior) sampling via the pretrained normalizing flow (obtained by running the script above), run:

```bash
$ julia scripts/test_hint_interpolation.jl
```

## Author

Rajiv Kumar and Ali Siahkoohi (alisk@gatech.edu)
