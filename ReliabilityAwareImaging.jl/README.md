# ReliabilityAwareImaging.jl

Experiments for "Learning by example: fast reliability-aware seismic imaging with normalizing flows", submitted to the [SEG 2021 Annual Meeting](https://seg.org/AM/).

To start running the examples, clone the repository:

```bash
$ git clone https://github.com/slimgroup/Software.SEG2021
$ cd ReliabilityAwareImaging.jl/
```

Here, we heavily rely on [InvertibleNetworks.jl](https://github.com/slimgroup/InvertibleNetworks.jl), a recently-developed, memory-efficient framework for training invertible networks in Julia.

## Installation

This repository is based on [DrWatson.jl](https://github.com/JuliaDynamics/DrWatson.jl). Before running examples, install `DrWatson.jl` by:

```julia
pkg> add DrWatson
```

Also make sure you add the SLIM registry:

```julia
pkg> registry add https://github.com/slimgroup/SLIMregistryJL.git
```

The only other manual installation is to make sure you have `matplotlib` and `seaborn` installed in your Python environment since we depend on `PyPlot.jl` and `Seaborn.jl` for creating figures.

The necessary dependencies will be installed upon running your first experiment. If you happen to have a CUDA-enabled GPU, the code will run on it. The training dataset will also download automatically into `data/training-data/` directory.

### Example

Run the script below for training the normalizing flow:

```bash
$ julia scripts/train_hint_imaging.jl
```

To perform conditional (posterior) sampling via the pretrained normalizing flow (obtained by running the script above), run:

```bash
$ julia scripts/test_hint_imaging.jl
```

## Author

Ali Siahkoohi (alisk@gatech.edu)
