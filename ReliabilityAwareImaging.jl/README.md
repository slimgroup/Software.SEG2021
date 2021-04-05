# ReliabilityAwareImaging.jl

Experiments for "Learning by example: fast reliability-aware seismic imaging with normalizing flows", submitted to the [SEG 2021 Annual Meeting](https://seg.org/AM/).

To start running the examples, clone the repository:

```bash
$ git clone https://github.com/slimgroup/Software.SEG2021
$ cd Software.SEG2021/ReliabilityAwareImaging.jl/
```

Here, we heavily rely on [InvertibleNetworks.jl](https://github.com/slimgroup/InvertibleNetworks.jl), a recently-developed, memory-efficient framework for training invertible networks in Julia.

## Installation

Before starting installing the required packages in Julia, make sure you have `matplotlib` and `seaborn` installed in your Python environment since we depend on `PyPlot.jl` and `Seaborn.jl` for creating figures.

Next, run the following commands in the command line to install the necessary libraries and setup the Julia project:

```bash
julia -e 'using Pkg; Pkg.add("DrWatson")'
julia -e 'using Pkg; Pkg.Registry.add(RegistrySpec(url = "https://github.com/slimgroup/SLIMregistryJL.git"))'
julia --project -e 'using Pkg; Pkg.instantiate()'
```

After the last line, the necessary dependencies will be installed. If you happen to have a CUDA-enabled GPU, the code will run on it. The training dataset will also download automatically into `data/training-data/` directory upon running your first example describe below.

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
