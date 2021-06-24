## Ziyi Yin converted the BG velocity model to permeability and porosity

using DrWatson
@quickactivate "BGTimeLapseCCS"

using Random, PyPlot
using LinearAlgebra, JLD2, MAT, Images, FFTW

n = (637, 82)
d = (25.0f0, 25.0f0)

JLD2.@load "../model/bgSlice.jld2" v rho d # original Vp in 25m, 6m grid spacing

function find_water_bottom(m::AbstractArray{avDT,2};eps = 1e-4) where {avDT}
  #return the indices of the water bottom of a seismic image
  ### function from JUDI package
  n = size(m)
  idx = zeros(Integer, n[1])
  for j=1:n[1]
      k=1
      while true
          if abs(m[j,k]) > eps
              idx[j] = k
              break
          end
          k += 1
      end
  end
  return idx
end

function down_sample(temp1)
    ## downsample the model to 25m grid spacing w/ low-pass filter
    fft1 = fftshift(fft(temp1))
    mask = zeros(size(fft1))
    taper_length = 8
    half_taper_length = Int(taper_length/2)
    taper = (1f0 .+ sin.((Float32(pi)*(0:taper_length-1))/(taper_length - 1) .- Float32(pi)/2f0))/2f0     
    mask[:,171-41:171+41] .= 1f0
    mask[:,171-41-taper_length+1+half_taper_length:171-41+half_taper_length] .= reshape(taper,1,taper_length)
    mask[:,171+41-half_taper_length:171+41+taper_length-1-half_taper_length] .= reshape(reverse(taper),1,taper_length)
    fft2 = mask .* fft1
    temp2 = real.(ifft(ifftshift(fft2)))
    temp2[:,1:35] .= temp1[:,1:35]  # fix water layer
    v = imresize(temp2,n)
    v[v.<=minimum(temp1)] .= minimum(temp1)
    v[v.>=maximum(temp1)] .= maximum(temp1)
    return v
end

v = down_sample(deepcopy(v))

extentx = (n[1]-1)*d[1]
extentz = (n[2]-1)*d[2]

idx_wb = find_water_bottom(v.-minimum(v))[1]

perm = 5f0*(v.-minimum(v))/(maximum(v)-minimum(v))

for i = 1:n[1]
  perm[i,1:idx_wb] .= 1f-4
  unconformaty = findall(v[i,:].>=3500f0)[1]
  perm[i,idx_wb+1:unconformaty] = perm[i,idx_wb+1:unconformaty] .+ 15f0
  perm[i,unconformaty+1:unconformaty+2] = perm[i,unconformaty+1:unconformaty+2]*1f-3
  perm[i,unconformaty+3:end] = perm[i,unconformaty+3:end] .+ 170f0
end

function poro_to_perm(phi)
    # Kozeny-carman equation
  c = (1.527/0.0314)^2f0
  perm = c*phi.^3f0./(1f0.-phi).^2f0
  return perm
end

phi = (perm*(1-0.22)^2/(1.527/0.0314)^2f0).^(1/3f0) # a proxy

JLD2.@save "../model/PermPoro.jld2" perm phi d
