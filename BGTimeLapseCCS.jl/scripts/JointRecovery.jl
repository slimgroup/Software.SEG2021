### Joint recovery
## author: Ziyi Yin (ziyi.yin@gatech.edu)


using DrWatson
@quickactivate "BGTimeLapseCCS"
using JLD2, FFTW, DSP

import Base.*

Random.seed!(1234);

# Needed for A*z
*(A::joLinearFunction, v::Array{judiVector{Float32,Array{Float32,2}},1}) = A.fop(v)
*(A::joLinearFunction, v::Array{judiVector{Float32},1}) = A.fop(v)
# adjoint(A)*b
*(A::joLinearFunction, v::Array{Array{Float32,1},1}) = A.fop(v)

JLD2.@load "../model/bgTimeLapse.jld2"
JLD2.@load "../data/d_lin_cut.jld2" d_lin_cut q

model0_stack = [Model(n,d,o,m0_stack[i]; nb=100) for i = 1:num_vintage]

srcGeometry = q.geometry
recGeometry_stack = [d_lin_cut[i].geometry for i = 1:num_vintage]

ntComp = get_computational_nt(srcGeometry, recGeometry_stack[1], model0_stack[1])
info = Info(prod(n), d_lin_cut[1].nsrc, ntComp)

opt = Options(isic=true, limit_m=true)

F0_stack = [judiProjection(info, recGeometry_stack[i])*judiModeling(info, model0_stack[i]; options=opt)*adjoint(judiProjection(info, srcGeometry)) for i = 1:num_vintage]
J_stack = [judiJacobian(F0_stack[i],q) for i = 1:num_vintage]
data_ = d_lin_cut
###

L = num_vintage

function Setup4D_forward(As, z)
	# Sum z_i according to identity matrix
	vec = As[end] * z
	# Apply diag(ops) to summed z_i
	return [As[i]*vec[i] for i=1:length(As)-1]
end 

function Setup4D_adjoint(As, b)
	# Apply diag(adj(ops))  * dlin_i then apply IMatrix
    z = adjoint(As[end]) * [adjoint(As[i]) * b[i] for i=1:length(b)]
    return z
end


# Setup JRM
ps = 32

niter = 23
batchsize = 8 * ones(Int,L)

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
                          Float32,Float64, name="Cmirrorext")


x = [zeros(Float32, info.n) for i=1:L+1];
z = [zeros(Float32, info.n) for i=1:L+1];

src_list = [collect(1:d_lin_cut[1].nsrc) for i = 1:L]

γ   = 1f0 # hyperparameter to tune
Iz  = Array{Any}(undef, L, L+1)

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
    image_out = vec(real.(ifft(ifftshift(image_f1)))[:,1:n[2]])+vec(real.(ifft(ifftshift(image_f1)))[:,end:-1:n[2]+1])/2f0
    return image_out
end

D = joLinearFunctionFwd_T(prod(n), prod(n),
                                 v -> dip(v,n;k=10),
                                 w -> dip(w,n;k=10),
                                 Float32,Float32,name="dip filter")
Tm = judiTopmute(model0_stack[1].n, idx_wb, 1)  # Mute water column
S = judiDepthScaling(model0_stack[1])  # depth scaling
Mr = S*Tm*D

for i=1:L

	for j=1:L+1
		Iz[i,j] = 0
	end
	
	Iz[i,1]  = 1f0/γ*joEye(prod(info.n); RDT=Float32, DDT=Float32)
		
end
	
collect(Iz[i,i+1] = joEye(prod(info.n); RDT=Float32, DDT=Float32) for i=1:L)
J  = [[undef for i=1:L]; [Iz]]

b = Array{judiVector{Float32,Array{Float32,2}},1}(undef, L)
lambda = zeros(Float64,L+1)

for  j=1:niter

	# Main loop			   
	@printf("Iteration: %d \n", j)
    flush(Base.stdout)
				
	for i=1:L
		length(src_list[i]) < batchsize[i] && (global src_list[i] = collect(1:data_[i].nsrc))
		src_list[i] = src_list[i][randperm(MersenneTwister(i+2000*j),length(src_list[i]))]
		if j == 1
			global inds = [pop!(src_list[i]) for b=1:2*batchsize[i]]
		else
			global inds = [pop!(src_list[i]) for b=1:batchsize[i]]
		end
		global b[i]  = data_[i][inds]
		global J[i]  = judiJacobian(F0_stack[i][inds], q[inds],ps,data_[i][inds])*Mr
		println("Vintage $(i) Imaging source $(inds)")
			
  	end

	A = joLinearFunctionFwd_T(L, L+1,
		    z -> Setup4D_forward(J,z),
		    b -> Setup4D_adjoint(J,b),
		    Float32,Float32, name="JRM")
		
    r = A*x - b
    g = adjoint(A)*r

	phi = .5*norm(r)^2

	# Step size and update variable

	t = Float32.(2*phi/norm(g)^2)
	
	@printf("At iteration %d function value is %2.2e \n", j, phi)
	flush(Base.stdout)
	
	if j == 1
		global z[1] -= t*g[1]
		C_z1 = C*z[1]
		global lambda[1] = quantile(abs.(vec(C_z1)), .9)
		global x[1] = adjoint(C)*soft_thresholding(C_z1, lambda[1])
	else
		global z -= t*g
		C_z = [C*z[i] for i = 1:L+1]
		if j == 2
			for k = 2:L+1
				global lambda[k] = quantile(abs.(vec(C_z[k])), .9) # estimate thresholding parameter at 2nd iteration
			end
		end
		# Update variables and save snapshot
		global x = [adjoint(C)*soft_thresholding(C_z[i], lambda[i]) for i = 1:L+1]
	end

    JLD2.@save "../results/JointRecovIter$(j).jld2" x z g lambda phi t
end
