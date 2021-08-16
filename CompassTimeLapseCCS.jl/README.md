# CompassTimeLapseCCS.jl

Experiments for "[Compressive time-lapse seismic monitoring of carbon storage and sequestration with the joint recovery model](https://slim.gatech.edu/Publications/Public/Submitted/2021/yin2021SEGcts/yin2021SEGcts.html)", accepted by [International Meeting for Applied Geoscience & Energy (IMAGE) 2021](https://imageevent.org) as a 20-min talk.

Our examples are mainly based on 3 open-source software packages, [FwiFlow](https://github.com/lidongzh/FwiFlow.jl) for two-phase flow simulation (in Julia), and [JUDI](https://github.com/slimgroup/JUDI.jl) for wave-equation simulation (in Julia), which uses the highly optimized time-domain finite-difference propagators from [Devito](https://www.devitoproject.org) (in Python 3). The joint recovery framework is inplemented in JUDI. We greatly appreciate the support from developers of these packages.

## Installation

To start running the examples, clone the repository:

```bash
git clone https://github.com/slimgroup/Software.SEG2021
cd Software.SEG2021/CompassTimeLapseCCS.jl/
```

To reproduce the examples, first install [Julia](https://julialang.org/downloads/) and [Python](https://www.python.org/downloads/). A Julia version <= 1.3 is currently suggested by developers of [FwiFlow](https://github.com/lidongzh/FwiFlow.jl).

Next, install python packages [Devito](https://www.devitoproject.org) and [matplotlib](https://matplotlib.org) through

```bash
pip install --user git+https://github.com/devitocodes/devito.git
pip install matplotlib
```

Then, open a julia console and do

```julia
using Pkg
Pkg.activate("CompassTimeLapseCCS")
Pkg.instantiate()
```

This will install all necessary packages in julia.

Note: We use a 3rd party library, [CurveLab](http://www.curvelet.org), for forward and adjoint curvelet transforms, which has been incorporated into Julia by [JOLI](https://github.com/slimgroup/JOLI.jl). SLIM version of Curvelab 2.1.2 can be obtained at [http://www.curvelet.org/download.html](http://www.curvelet.org/download.html). You may also try other types of sparsifying transform (Fourier/wavelet/...) as you need.

## Reproduce the examples

Examples in the SEG abstract are in the `script` folder. A proper order to run the examples is: first run *GetCompassSlice.jl* to download the 3D Compass dataset and take out the 2D slice used in the numerical experiment, then run *MakePermPoro.jl* to convert acoustic velocity to permeability and porosity, then run *TwoPhase.jl* to generate CO2 concentration by two-phase flow simulation, then run *MakeTimeLapseV.jl* to convert CO2 concentration percentage to time-lapse velocity models through Patchy saturation model, next run *GenerateData.jl* to generate seismic data through wave-equation, and run *RemoveLongOffsetRec.jl* to cut the long-offset data, and finally, run *IndependentRecoveryX.jl* to recover the image for each vintage independently, OR run *JointRecovery.jl* to recover the images for each vintage jointly through joint recovery model. A script *PlotResults.jl* is also provided to plot the results through PyPlot.

## Citation

If you find the software useful in your research, we would appreciate it if you use the citation below.

```latex
@conference{yin2021SEGcts,
  author = {Ziyi Yin and Mathias Louboutin and Felix J. Herrmann},
  title = {Compressive time-lapse seismic monitoring of carbon storage and sequestration with the joint recovery model},
  booktitle = {Accepted by International Meeting for Applied Geoscience and Energy (IMAGE) 2021},
  year = {2021},
  month = {06},
  keywords = {compressive sensing, joint recovery method, imaging, CCS, marine, time-lapse},
  note = {(SEG, just accepted)},
  url = {https://slim.gatech.edu/Publications/Public/Conferences/SEG/2021/yin2021SEGcts/yin2021SEGcts.html},
  software = {https://github.com/slimgroup/Software.SEG2021}
}
```

Also we would appreciate it if you star our repository.

## LICENSE

Please refer to the MIT license in this repository. In addition, please also refer to the license of the Compass model:

The material provided is an synthetic and fictional dataset for testing purposes. Any similarity or resemblance to any location is entirely coincidental and unintentional. Accordingly, BG Group makes no representation or warranty, express or implied, in respect to the quality, accuracy or usefulness of such data.  The data is supplied with the explicit understanding and agreement of recipient that any action taken or expenditure made by recipient based on its examination, evaluation, interpretation or use of the data is at its own risk and responsibility. 

## Author

Ziyi (Francis) Yin (ziyi.yin@gatech.edu) and other contributors in [SLIM](https://slim.gatech.edu) group

## Acknowledgement

We would like to thank Charles Jones for the constructive discussion and thank BG Group for providing the Compass model. The CCS project information is taken from the Strategic UK CCS Storage Appraisal Project, funded by DECC, commissioned by the ETI and delivered by Pale Blue Dot Energy, Axis Well Technology and Costain. The information contains copyright information licensed under [ETI Open Licence](https://s3-eu-west-1.amazonaws.com/assets.eti.co.uk/legacyUploads/2016/04/ETI-licence-v2.1.pdf). This research was carried out with the support of Georgia Research Alliance and partners of the ML4Seismic Center.
