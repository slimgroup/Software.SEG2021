### Make figures
## author: Ziyi Yin (ziyi.yin@gatech.edu)

using DrWatson
@quickactivate "CompassTimeLapseCCS"

using PyPlot, JUDI, JLD2, LinearAlgebra, Statistics, Images, Flux

JLD2.@load "../model/CompassTimeLapse.jld2"

model0_stack = [Model(n,d,o,m0_stack[i]; nb = 100) for i = 1:num_vintage]

function plot_image(image, spacing; perc=98, cmap="Greys", o=(0, 0),
    interp="hanning", aspect=nothing, d_scale=1.5,
    name=nothing, units="km", new_fig=true, save=nothing, a=nothing,start_grid=nothing)
nx, nz = size(image)
dx, dz = spacing
ox, oz = o
depth = range(oz, oz + (nz - 1)*spacing[2], length=nz).^d_scale

scaled = image .* depth'

if isnothing(a)
    a = quantile(abs.(vec(scaled)), perc/100)
end
println(a)
start_x = start_z = 1
if !isnothing(start_grid)
    start_x = start_grid[1]
    start_z = start_grid[2]
end
extent = [(start_x-1)*dx/1f3+ox/1f3, (start_x-1)*dx/1f3+(ox+ (nx-1)*dx)/1f3, (start_z-1)*dz/1f3+(oz+(nz-1)*dz)/1f3, (start_z-1)*dz/1f3+oz/1f3]
isnothing(aspect) && (aspect = .5 * nx/nz)

if new_fig
figure()
end
imshow(scaled', vmin=-a, vmax=a, cmap=cmap, aspect=aspect,
interpolation=interp, extent=extent)
xlabel("X [$units]")
ylabel("Z [$units]")
if !isnothing(name)
    title("$name")
end

if ~isnothing(save)
save == True ? filename=name : filename=save
savefig(filename, bbox_inches="tight", dpi=150)
end
return a
end


function plot_data(data, dt; perc=98, cmap="seismic",
    interp="hanning", aspect=nothing,
    name="shot record", units="s", new_fig=true, save=nothing, a=nothing)
    nt, nrec = size(data)

if isnothing(a)
    a = quantile(abs.(vec(data)), perc/100)
end
extent = [1, nrec, (nt-1)*dt/1f3,0]
isnothing(aspect) && (aspect = 2*nrec/((nt-1)*dt/1f3))

if new_fig
figure()
end
imshow(data, vmin=-a, vmax=a, cmap=cmap, aspect=aspect,
interpolation=interp, extent=extent)
xlabel("Receiver no.")
ylabel("Time [$(units)]")
title("$name")

if ~isnothing(save)
save == True ? filename=name : filename=save
savefig(filename, bbox_inches="tight", dpi=150)
end
end

# Preconditioner

function dip(x,n;k=20)
    image = reshape(x,n)
    image_ext = zeros(Float32,n[1],2*n[2])
    image_ext[:,1:n[2]] = image
    image_ext[:,n[2]+1:end] = reverse(image,dims=2)
    image_f = fftshift(fft(image_ext))
    mask = ones(Float32,n[1],2*n[2])
    for i = 1:n[1]
        for j = 1:2*n[2]
            if (i-(n[1]+1)/2-k*j+k*(2*n[2]+1)/2)*(i-(n[1]+1)/2+k*j-k*(2*n[2]+1)/2)>0
                mask[i,j] = 0f0
            end
        end
    end
    mask = convert(Array{Float32},imfilter(mask,Kernel.gaussian(20)))
    image_f1 = mask .* image_f
    image_out = (vec(real.(ifft(ifftshift(image_f1)))[:,1:n[2]])+vec(real.(ifft(ifftshift(image_f1)))[:,end:-1:n[2]+1]))/2f0
    return image_out
end

D = joLinearFunctionFwd_T(prod(n), prod(n),
                                 v -> dip(v,n;k=10),
                                 w -> dip(w,n;k=10),
                                 Float32,Float32,name="dip filter")
Tm = judiTopmute(model0_stack[1].n, idx_wb, 1)  # Mute water column
S = judiDepthScaling(model0_stack[1])  # depth scaling
Mr = S*Tm*D

v_stack = [1f3./sqrt.(m_stack[i]) for i = 1:5]

extentx = (n[1]-1)*d[1]
extentz = (n[2]-1)*d[2]

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20,10))
PyPlot.rc("font", family="serif"); PyPlot.rc("xtick", labelsize=10); PyPlot.rc("ytick", labelsize=10)
ct=0
for row = 1:2
    for col = 1:2
        global ct = ct + 1
        global ax = axs[row,col]
        global pcm = ax.imshow((v_stack[ct+1]-v_stack[1])[18:1200,120:185]',cmap="jet",extent=(17*d[1]/1f3,1199*d[1]/1f3,184*d[2]/1f3,119*d[2]/1f3),interpolation="hanning",vmin=minimum(v_stack[5]'-v_stack[1]'),vmax=0,aspect=.5 * 1183/66);
        ax.set_xlabel("X [km]", fontsize=10)
        ax.set_ylabel("Z [km]", fontsize=10)
    end
end
cb = fig.colorbar(pcm,ax=axs[:,:],fraction=0.025, pad=0.04)
cb[:set_label]("[m/s]",fontsize=10)
savefig("../figures/plume.png", bbox_inches="tight", dpi=300)

JLD2.@load "../results/IndpRecovVintage1Iter23.jld2"
x1 = deepcopy(x)
JLD2.@load "../results/IndpRecovVintage2Iter23.jld2"
x2 = deepcopy(x)
JLD2.@load "../results/IndpRecovVintage3Iter23.jld2"
x3 = deepcopy(x)
JLD2.@load "../results/IndpRecovVintage4Iter23.jld2"
x4 = deepcopy(x)
JLD2.@load "../results/IndpRecovVintage5Iter23.jld2"
x5 = deepcopy(x)

dv1 = 1f3./(sqrt.(relu.(vec(m0_stack[1])+dip(Mr*x1,n;k=10))))-1f3./(sqrt.(vec(m0_stack[1])))
dv2 = 1f3./(sqrt.(relu.(vec(m0_stack[2])+dip(Mr*x2,n;k=10))))-1f3./(sqrt.(vec(m0_stack[2])))
dv3 = 1f3./(sqrt.(relu.(vec(m0_stack[3])+dip(Mr*x3,n;k=10))))-1f3./(sqrt.(vec(m0_stack[3])))
dv4 = 1f3./(sqrt.(relu.(vec(m0_stack[4])+dip(Mr*x4,n;k=10))))-1f3./(sqrt.(vec(m0_stack[4])))
dv5 = 1f3./(sqrt.(relu.(vec(m0_stack[5])+dip(Mr*x5,n;k=10))))-1f3./(sqrt.(vec(m0_stack[5])))

dv_indp = [dv1,dv2,dv3,dv4,dv5]
a = 120

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20,10))
PyPlot.rc("font", family="serif"); PyPlot.rc("xtick", labelsize=10); PyPlot.rc("ytick", labelsize=10)
ct=0
for row = 1:2
    for col = 1:2
        global ct = ct + 1
        global ax = axs[row,col]
        global pcm = ax.imshow(reshape(dv_indp[ct+1]-dv_indp[1],n)[18:1200,120:185]',cmap="Greys",extent=(17*d[1]/1f3,1199*d[1]/1f3,184*d[2]/1f3,119*d[2]/1f3),interpolation="hanning",vmin=-a,vmax=a,aspect=.5 * 1183/66);
        ax.set_xlabel("X [km]", fontsize=10)
        ax.set_ylabel("Z [km]", fontsize=10)
        region1 = reshape(dv_indp[1],n)[18:1200,120:185]
        regionnow = reshape(dv_indp[ct+1],n)[18:1200,120:185]
    end
end
cb = fig.colorbar(pcm,ax=axs[:,:],fraction=0.025, pad=0.04)
cb[:set_label]("[m/s]",fontsize=10)
savefig("../figures/IndpRecov.png", bbox_inches="tight", dpi=300)

JLD2.@load "../results/JointRecovIter23.jld2"

dv_jrm = [1f3./(sqrt.(relu.(vec(m0_stack[i])+dip(Mr*(x[1]/1f0+x[i+1]),n;k=10))))-1f3./(sqrt.(vec(m0_stack[i]))) for i = 1:5]

a = 120

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20,10))
PyPlot.rc("font", family="serif"); PyPlot.rc("xtick", labelsize=10); PyPlot.rc("ytick", labelsize=10)
ct=0
for row = 1:2
    for col = 1:2
        global ct = ct + 1
        global ax = axs[row,col]
        global pcm = ax.imshow(reshape(dv_jrm[ct+1]-dv_jrm[1],n)[18:1200,120:185]',cmap="Greys",extent=(17*d[1]/1f3,1199*d[1]/1f3,184*d[2]/1f3,119*d[2]/1f3),interpolation="hanning",vmin=-a,vmax=a,aspect=.5 * 1183/66);
        ax.set_xlabel("X [km]", fontsize=10)
        ax.set_ylabel("Z [km]", fontsize=10)
        region1 = reshape(dv_jrm[1],n)[18:1200,120:185]
        regionnow = reshape(dv_jrm[ct+1],n)[18:1200,120:185]
    end
end
cb = fig.colorbar(pcm,ax=axs[:,:],fraction=0.025, pad=0.04)
cb[:set_label]("[m/s]",fontsize=10)
savefig("../figures/JointRecov.png", bbox_inches="tight", dpi=300)