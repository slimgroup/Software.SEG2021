### Independent recovery on vintage 2
## author: Ziyi Yin (ziyi.yin@gatech.edu)

using DrWatson
@quickactivate "BGTimeLapseCCS"
using TimeProbeSeismic
using JLD2, FFTW, DSP, PyPlot, Images

JLD2.@load "../model/bgTimeLapse.jld2"
JLD2.@load "../data/d_lin_cut.jld2" d_lin_cut q

nv = 2

isdir("../results") || mkdir("../results")

Random.seed!(nv)

srcGeometry = q.geometry
recGeometry_stack = [d_lin_cut[i].geometry for i = 1:num_vintage]

model0_stack = [Model(n,d,o,m0_stack[i]; nb=100) for i = 1:num_vintage]

opt = Options(isic=true, limit_m=true)

ntComp = get_computational_nt(srcGeometry, recGeometry_stack[1], model0_stack[1])
info = Info(prod(n), d_lin_cut[1].nsrc, ntComp)

F0_stack = [judiProjection(info, recGeometry_stack[i])*judiModeling(info, model0_stack[i]; options=opt)*adjoint(judiProjection(info, srcGeometry)) for i = 1:num_vintage]

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

# Linearized Bregman parameters
nn = prod(model0_stack[1].n)
x = zeros(Float32, nn)
z = zeros(Float32, nn)

batchsize = 8
niter = 23

# Soft thresholding functions and Curvelet transform
soft_thresholding(x::Array{Float64}, lambda) = sign.(x) .* max.(abs.(x) .- convert(Float64, lambda), 0.0)
soft_thresholding(x::Array{Float32}, lambda) = sign.(x) .* max.(abs.(x) .- convert(Float32, lambda), 0f0)

n = model0_stack[1].n
C0 = joCurvelet2D(n[1], 2*n[2]; zero_finest = false, DDT = Float32, RDT = Float64)

function C_fwd(im, C, n)
	im = hcat(reshape(im, n), reshape(im, n)[:, end:-1:1])
	coeffs = C*vec(im)/sqrt(2f0)
	return coeffs
end

function C_adj(coeffs, C, n)
	im = reshape(C'*coeffs, n[1], 2*n[2])
	return vec(im[:, 1:n[2]] .+ im[:, end:-1:n[2]+1])/sqrt(2f0)
end

C = joLinearFunctionFwd_T(size(C0, 1), n[1]*n[2],
                          x -> C_fwd(x, C0, n),
                          b -> C_adj(b, C0, n),
                          Float32,Float64, name="Cmirrorext")       # mirror-extended curvelet

src_list = collect(1:d_lin_cut[1].nsrc)
lambda = 0f0

ps = 32     # number of probing vectors

# Main loop
for j = 1:niter
    # Select batch and set up left-hand preconditioner
    length(src_list) < batchsize && (global src_list = collect(1:d_lin_cut[nv].nsrc))
    src_list = src_list[randperm(MersenneTwister(nv+2000*j),length(src_list))]
    if j == 1
        global i = [pop!(src_list) for b=1:2*batchsize]
    else
        global i = [pop!(src_list) for b=1:batchsize]
    end
    println("Vintage $(nv) LS-RTM Iteration: $(j), imaging sources $(i)")
    flush(Base.stdout)

    J_probe = judiJacobian(F0_stack[nv][i],q[i],ps,d_lin_cut[nv][i])

    residual = J_probe*Mr*x-d_lin_cut[nv][i]
    phi = 0.5 * norm(residual)^2
    g = Mr'*J_probe'*residual

    # Step size and update variable
    t = Float32.(2*phi/norm(g)^2)

    # Update variables and save snapshot
    global z -= t*g
    C_z = C*z
    (j==1) && (global lambda = quantile(abs.(C_z), .9))   # estimate thresholding parameter in 1st iteration
    global x = adjoint(C)*soft_thresholding(C_z, lambda)

    @printf("At iteration %d function value is %2.2e and step length is %2.2e \n", j, phi, t)
    @printf("Lambda is %2.2e \n", lambda)

    JLD2.@save "../result/IndpRecovVintage$(nv)Iter$(j).jld2" x z g lambda phi
end