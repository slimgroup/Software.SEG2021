### Read the 3D Compass slice from SLIM website and take out the 2D slice
## Please refer to the details in the MIT license of this repository and in the license of the Compass model
## author: Ziyi Yin (ziyi.yin@gatech.edu)

using DrWatson
@quickactivate "CompassTimeLapseCCS"

using PyPlot
using SegyIO
run(`wget -r ftp://slim.gatech.edu/data//synth/Compass`) # this might take a while

# get velocity
block = segy_read("slim.gatech.edu/final_velocity_model_ieee_6m.sgy")

# original compass model is in 25m*25m*6m
n1 = (1911,2730,341)
d = (25f0,25f0,6f0)

n = (637,910,341)

sx = get_header(block, "SourceX")
sy = get_header(block, "SourceY")

v_nogas = zeros(Float32,n)
v_gas = zeros(Float32,n)

for i = 1:n[1]
    x = d[1].*(i-1)
    inds = findall(sx.==x)
    slice = block.data[:,inds[sortperm(sy[inds])]]

    v_nogas[i,:,:] = transpose(slice[:,1:Int(end/3)])
end

for i = 1:n[1]
    x = d[1].*(i-1+n[1])
    inds = findall(sx.==x)
    slice = block.data[:,inds[sortperm(sy[inds])]]

    v_gas[n[1]-i+1,:,:] = transpose(slice[:,Int(end/3)+1:2*Int(end/3)][:,end:-1:1])
end

v = v_nogas[:,300,:]

# get density
block1 = segy_read("slim.gatech.edu/final_density_model_ieee_6m.sgy")

sx = get_header(block1, "SourceX")
sy = get_header(block1, "SourceY")

rho_nogas = zeros(Float32,n)
rho_gas = zeros(Float32,n)

for i = 1:n[1]
    x = d[1].*(i-1)
    inds = findall(sx.==x)
    slice = block1.data[:,inds[sortperm(sy[inds])]]

    rho_nogas[i,:,:] = transpose(slice[:,1:Int(end/3)])
end

for i = 1:n[1]
    x = d[1].*(i-1+n[1])
    inds = findall(sx.==x)
    slice = block1.data[:,inds[sortperm(sy[inds])]]

    rho_gas[n[1]-i+1,:,:] = transpose(slice[:,Int(end/3)+1:2*Int(end/3)][:,end:-1:1])
end

rho = rho_nogas[:,300,:]

d = (25f0,6f0)

JLD2.@save "../model/CompassSlice.jld2" v rho d
