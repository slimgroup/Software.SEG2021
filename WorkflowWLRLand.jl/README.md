# WorkflowWLRLand.jl

Codes for "A practical workflow for land seismic wavefield recovery with weighted matrix factorization", submitted to the [SEG 2021 Annual Meeting](https://seg.org/AM/).

The raw data for this work can not be provided. However, if you are interested in this work, you could follow these steps to apply these codes to your data. 

You could setup the environment with following dependencies:
## Dependencies

The minimum requirements for theis software, and tested version, are `Python 3.x` and `Julia 1.2.0`.
This software requires the following dependencies to be installed:
- [Distributed](https://docs.julialang.org/en/v1/stdlib/Distributed/) Support for distributed computing.
- [SharedArrays](https://docs.julialang.org/en/v1/stdlib/SharedArrays/) Support shared arrays across the processes. 
- [JLD](https://github.com/JuliaIO/JLD.jl). Save and load variables in Julia Data format (JLD).
- [GenSPGL](https://github.com/slimgroup/GenSPGL.jl). A Julia solver for large scale minimization problems using any provided norm.
- [SeisJOLI](https://github.com/slimgroup/SeisJOLI.jl). Collection of SLIM in-house operators based on JOLI package.
- [Arpack](https://github.com/JuliaLinearAlgebra/Arpack.jl). Julia wrapper for the arpack library designed to solve large scale eigenvalue problems.
- [LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/). Julia provides native implementations of many common and useful linear algebra operations which can be loaded with using LinearAlgebra. 
- [DSP](https://github.com/JuliaDSP/DSP.jl). DSP.jl provides a number of common digital signal processing routines in Julia.
- [Random](https://docs.julialang.org/en/v1/stdlib/Random/) Support for generating random numbers.
- [FFTW](https://github.com/JuliaMath/FFTW.jl) This package provides Julia bindings to the FFTW library for fast Fourier transforms (FFTs), as well as functionality useful for signal processing.

First, you need install the following packages from the stable master branch:
```
using Pkg; Pkg.add(PackageSpec(url="https://github.com/slimgroup/SeisJOLI.jl"))
using Pkg; Pkg.add(PackageSpec(url="https://github.com/slimgroup/GenSPGL.jl"))
```

then install dependencies:
```
Pkg.add("Distributed")
Pkg.add("SharedArrays")
Pkg.add("JLD")
Pkg.add("Arpack")
Pkg.add("LinearAlgebra")
Pkg.add("DSP")
Pkg.add("Random")
Pkg.add("FFTW")
```

## Software
This software is divided as follows:
 
*scripts/optimization*: 

 This directory contains codes to run the parallel weighted matrix factorization. 
 
 ```
 Weighted_LR.jl #The main function of  weighted matrix factorization.
 SubFunction #This directory contains necessary codes for weighted matrix factorization.
 ```

 *scripts/workflow*: 

 This directory contains codes to implement some important steps for this workflow. 
 
 ```
 ExtractFlattedGroundRoll.jl #This function is used to extract the shifted ground roll as mentioned in the paper. 
 UndoShift.jl #This function is used to remove the shift after extracting the groundroll estimate.
 WindowFKSpectrum.jl #This function is used to denoise the body waves after reconstruction. 
 ```


## Citation

If you find this software useful in your research, we would appreciate it if you cite:

```bibtex
@unpublished {zhang2021SEGapw,
	title = {A practical workflow for land seismic wavefield recovery with weighted matrix factorization},
	year = {2021},
	note = {Submitted to SEG 2021},
	month = {04},
	abstract = {While wavefield reconstruction through weighted low-rank matrix factorizations has been shown to perform well on marine data, out-of-the-box application of this technology to land data is hampered by ground roll. The presence of these strong surface waves tends to dominate the reconstruction at the expense of the weaker body waves. Because ground roll is slow, it also suffers more from aliasing. To overcome these challenges, we introduce a practical workflow where the ground roll and body wave components are recovered separately and combined. We test the proposed approach blindly on a subset of the 3D SEAM Barrett dataset. With our technique, we recover densely sampled data from 25 percent randomly subsampled receivers. Independent comparisons on a single shot demonstrate significant improvements achievable with the presented workflow.},
	keywords = {3D SEAM Barrett dataset, ground roll, wavefield reconstruction, weighted matrix factorization},
	url = {https://slim.gatech.edu/Publications/Public/Submitted/2021/zhang2021SEGapw/Yijun2021.html},
	software = {https://github.com/slimgroup/Software.SEG2021},
	author = {Yijun Zhang and Felix J. Herrmann}
}
```

## Author

Yijun (Yijun Zhang) (yzhang3198@gatech.edu)
