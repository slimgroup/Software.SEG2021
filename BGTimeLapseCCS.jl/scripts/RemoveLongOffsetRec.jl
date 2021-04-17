### Remove the long-offset receiver traces, only use +-3km for imaging
## author: Ziyi Yin (ziyi.yin@gatech.edu)

using DrWatson
@quickactivate "BGTimeLapseCCS"

using PyPlot, JLD2, JUDI

JLD2.@load "../data/d_lin.jld2" d_lin q
JLD2.@load "../model/OBN.jld2"

d_lin_cut = Array{judiVector{Float32,Array{Float32,2}},1}(undef, 5)

for i = 1:5
    xloc_cut = Array{Array{Float32,1}}(undef, d_lin[i].nsrc)
    zloc_cut = Array{Array{Float32,1}}(undef, d_lin[i].nsrc)
    data_cut = Array{Array{Float32,2}}(undef, d_lin[i].nsrc)
    for j = 1:d_lin[i].nsrc
        obn = OBN_loc[j]
        xrec = d_lin[i][j].geometry.xloc[1]
        inds = findall(abs.(xrec.-obn).<=3000f0)
        data_cut[j] = d_lin[i].data[j][:,inds]
        xloc_cut[j] = d_lin[i].geometry.xloc[j][inds]
        zloc_cut[j] = d_lin[i].geometry.zloc[j][inds]
    end
    geometry_cut = Geometry(xloc_cut, d_lin[i].geometry.yloc, zloc_cut; dt=d_lin[i].geometry.dt[1], t=d_lin[i].geometry.t[1])
    d_lin_cut[i] = judiVector(geometry_cut,data_cut)
end

JLD2.@save "../data/d_lin_cut.jld2" d_lin_cut q
