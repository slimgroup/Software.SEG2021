### Generate linear data through Born scattering modeling for each num_vintage
## author: Ziyi Yin (ziyi.yin@gatech.edu)

using DrWatson
@quickactivate "CompassTimeLapseCCS"

using PyPlot, JUDI
using JLD2, LinearAlgebra
using JOLI, FFTW
using Printf, Statistics

JLD2.@load "../model/CompassTimeLapse.jld2"
JLD2.@load "../model/MarineSourceLoc.jld2" xrec_stack zrec_stack
JLD2.@load "../model/OBN.jld2" OBN_loc

model0_stack = [Model(n,d,o,m0_stack[i]; nb=100) for i = 1:num_vintage]

### reciprocity

nsrc = length(OBN_loc)
nrec = length(xrec_stack[1])

dtS = dtR = 1f0
timeS = timeR = 2000f0

xsrc = convertToCell(OBN_loc)
ysrc = convertToCell(range(0f0,stop=0f0,length=nsrc))
zsrc = convertToCell(range(12*d[2],stop=12*d[2],length=nsrc))

srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)
wavelet = ricker_wavelet(timeS,dtS,0.025f0)

q = judiVector(srcGeometry, wavelet)

yrec = 0f0

recGeometry_stack = [Geometry(xrec_stack[i],yrec,zrec_stack[i]; dt=dtR, t=timeR, nsrc=nsrc) for i = 1:num_vintage]

ntComp = get_computational_nt(srcGeometry, recGeometry_stack[1], model0_stack[1])
info = Info(prod(n), nsrc, ntComp)

opt = Options(isic=true)

F0_stack = [judiProjection(info, recGeometry_stack[i])*judiModeling(info, model0_stack[i]; options=opt)*adjoint(judiProjection(info, srcGeometry)) for i = 1:num_vintage]
J_stack = [judiJacobian(F0_stack[i],q) for i = 1:num_vintage]
d_lin = [J_stack[i]*dm_stack[i] for i = 1:num_vintage]    # generate linear data through Born scattering

JLD2.@save "../data/d_lin.jld2" d_lin q
