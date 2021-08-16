### Convert CO2 concentration percentage to compressional wavespeed by Patchy saturation model

## Ziyi Yin followed the example in FwiFlow software, developed by Dongzhuo Li and Kailai Xu, from the publication https://doi.org/10.1029/2019WR027032

using DrWatson
@quickactivate "CompassTimeLapseCCS"

using PyPlot
using Random, Images, JLD2, LinearAlgebra
using JOLI, Statistics, FFTW
using JUDI

JLD2.@load "../model/CompassSlice.jld2" v rho d
JLD2.@load "../data/CompassCO2.jld2" conc

n = (1590, 205)
o = (0f0, 0f0)
d = (10f0, 10f0)

function down_sample(temp1)
    ## downsample the model to 10m grid spacing w/ low-pass filter
    fft1 = fftshift(fft(temp1))
    mask = zeros(size(fft1))
    taper_length = 8
    half_taper_length = Int(taper_length/2)
    taper = (1f0 .+ sin.((Float32(pi)*(0:taper_length-1))/(taper_length - 1) .- Float32(pi)/2f0))/2f0     
    mask[:,171-102:171+102] .= 1f0
    mask[:,171-102-taper_length+1+half_taper_length:171-102+half_taper_length] .= reshape(taper,1,taper_length)
    mask[:,171+102-half_taper_length:171+102+taper_length-1-half_taper_length] .= reshape(reverse(taper),1,taper_length)
    fft2 = mask .* fft1
    temp2 = real.(ifft(ifftshift(fft2)))
    temp2[:,1:35] .= temp1[:,1:35]  # fix water layer
    v = imresize(temp2,n)
    v[v.<=minimum(temp1)] .= minimum(temp1)
    v[v.>=maximum(temp1)] .= maximum(temp1)
    return v
end

v = down_sample(deepcopy(v))
rho = down_sample(deepcopy(rho))

idx_wb = find_water_bottom(v.-minimum(v))[1]

extentx = (n[1]-1)*d[1]
extentz = (n[2]-1)*d[2]

perm = 5f0*(v.-minimum(v))/(maximum(v)-minimum(v))

for i = 1:n[1]
  perm[i,1:idx_wb] .= 1f-4
  unconformaty = findall(v[i,:].>=3500f0)[1]
  perm[i,idx_wb+1:unconformaty] = perm[i,idx_wb+1:unconformaty] .+ 15f0
  perm[i,unconformaty+1:unconformaty+5] = perm[i,unconformaty+1:unconformaty+5]*1f-3
  perm[i,unconformaty+6:end] = perm[i,unconformaty+6:end] .+ 170f0
end

function poro_to_perm(phi)
    # Kozeny-carman equation
  c = (1.527/0.0314)^2f0
  perm = c*phi.^3f0./(1f0.-phi).^2f0
  return perm
end

phi = (perm*(1-0.22)^2/(1.527/0.0314)^2f0).^(1/3f0) # a proxy

vp = convert(Array{Float32},v)
vs = vp ./ sqrt(3f0)

num_vintage = 5
sw = zeros(Float32,num_vintage,n[1],n[2])
for i = 1:num_vintage
	sw[i,:,:] = imresize(conc[i,:,:]',n)
end

function Patchy(sw, vp, vs, rho, phi; bulk_min = 36.6f9, bulk_fl1 = 2.735f9, bulk_fl2 = 0.125f9, ρw = 7.766f2, ρo = 1.053f3)
    bulk_sat1 = rho .* (vp.^2f0 - 4f0/3f0 .* vs.^2f0) * 1f3
    shear_sat1 = rho .* (vs.^2f0) * 1f3

    bulk_min = zeros(Float32,size(bulk_sat1))

    bulk_min[findall(v.>=3350)] .= 50f9   # mineral bulk moduli
    bulk_min[findall(v.<3350)] .= 2.5f9   # mineral bulk moduli
    patch_temp = bulk_sat1 ./(bulk_min .- bulk_sat1) - 
    bulk_fl1 ./ phi ./ (bulk_min .- bulk_fl1) + 
    bulk_fl2 ./ phi ./ (bulk_min .- bulk_fl2)

    bulk_sat2 = bulk_min./(1f0./patch_temp .+ 1f0)

    bulk_new = 1f0./( (1f0.-sw)./(bulk_sat1+4f0/3f0*shear_sat1) 
    + sw./(bulk_sat2+4f0/3f0*shear_sat1) ) - 4f0/3f0*shear_sat1

	bulk_new[:,1:idx_wb] = bulk_sat1[:,1:idx_wb]

    rho_new = rho + phi .* sw * (ρw - ρo) / 1f3

    Vp_new = sqrt.((bulk_new+4f0/3f0*shear_sat1)./rho_new/1f3)
    Vs_new = sqrt.((shear_sat1)./rho_new/1f3)

    return Vp_new, Vs_new, rho_new
end

vp_stack = [Patchy(sw[i,:,:], vp, vs, rho, phi)[1] for i = 1:num_vintage]
m_stack = [(1f3./vp_stack[i]).^2f0 for i = 1:num_vintage]

m0_stack = [convert(Array{Float32,2},imfilter(m_stack[i], Kernel.gaussian(7))) for i=1:num_vintage]

for i = 1:num_vintage
    m0_stack[i][:,1:idx_wb] = m_stack[i][:,1:idx_wb]
end

dm_stack = [vec(m_stack[i]-m0_stack[i]) for i = 1:num_vintage]

JLD2.@save "../model/CompassTimeLapse.jld2" num_vintage n d o idx_wb m0_stack dm_stack m_stack
