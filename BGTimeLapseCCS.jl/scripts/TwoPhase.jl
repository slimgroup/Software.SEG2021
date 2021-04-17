### Two-phase flow simulation on BG compass model

## Ziyi Yin converted the BG velocity model to permeability and porosity
## The simulation is done with FwiFlow software, developed by Dongzhuo Li and Kailai Xu, from the publication https://doi.org/10.1029/2019WR027032

using DrWatson
@quickactivate "BGTimeLapseCCS"

using Pkg
Pkg.build("ADCME")
using Random, PyCall, PyPlot
using LinearAlgebra, JLD2, MAT, Images, FFTW

n = (637, 82)
d = (25.0f0, 25.0f0)

JLD2.@load "../model/bgSlice.jld2" v rho d # original Vp in 25m, 6m grid spacing

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

## 2 phase flow simulation by FwiFlow
m, n = n[2],n[1]
h = 25.0 # m

const K_CONST =  9.869232667160130e-16 * 86400 * 1e3
const ALPHA = 1.0
mutable struct Ctx
  m; n; h; NT; Δt; Z; X; ρw; ρo;
  μw; μo; K; g; ϕ; qw; qo; sw0
end

function tfCtxGen(m,n,h,NT,Δt,Z,X,ρw,ρo,μw,μo,K,g,ϕ,qw,qo,sw0,ifTrue)
  tf_h = constant(h)
  # tf_NT = constant(NT)
  tf_Δt = constant(Δt)
  tf_Z = constant(Z)
  tf_X= constant(X)
  tf_ρw = constant(ρw)
  tf_ρo = constant(ρo)
  tf_μw = constant(μw)
  tf_μo = constant(μo)
  # tf_K = isa(K,Array) ? Variable(K) : K
  if ifTrue
    tf_K = constant(K)
  else
    tf_K = Variable(K)
  end
  tf_g = constant(g)
  # tf_ϕ = Variable(ϕ)
  tf_ϕ = constant(ϕ)
  tf_qw = constant(qw)
  tf_qo = constant(qo)
  tf_sw0 = constant(sw0)
  return Ctx(m,n,tf_h,NT,tf_Δt,tf_Z,tf_X,tf_ρw,tf_ρo,tf_μw,tf_μo,tf_K,tf_g,tf_ϕ,tf_qw,tf_qo,tf_sw0)
end

function Krw(Sw)
    return Sw ^ 1.5
end

function Kro(So)
    return So ^1.5
end

function ave_normal(quantity, m, n)
    aa = sum(quantity)
    return aa/(m*n)
end

# variables : sw, u, v, p
# (time dependent) parameters: qw, qo, ϕ
function onestep(sw, p, m, n, h, Δt, Z, ρw, ρo, μw, μo, K, g, ϕ, qw, qo)
    # step 1: update p
    # λw = Krw(sw)/μw
    # λo = Kro(1-sw)/μo
    λw = sw.*sw/μw
    λo = (1-sw).*(1-sw)/μo
    λ = λw + λo
    q = qw + qo + λw/(λo+1e-16).*qo
    # q = qw + qo
    potential_c = (ρw - ρo)*g .* Z

    # Step 1: implicit potential
    Θ = upwlap_op(K * K_CONST, λo, potential_c, h, constant(0.0))

    load_normal = (Θ+q/ALPHA) - ave_normal(Θ+q/ALPHA, m, n)

    # p = poisson_op(λ.*K* K_CONST, load_normal, h, constant(0.0), constant(1))
    p = upwps_op(K * K_CONST, λ, load_normal, p, h, constant(0.0), constant(0)) # potential p = pw - ρw*g*h 

    # step 2: implicit transport
    sw = sat_op(sw, p, K * K_CONST, ϕ, qw, qo, μw, μo, sw, Δt, h)
    return sw, p
end


function imseq(tf_ctx)
    ta_sw, ta_p = TensorArray(NT+1), TensorArray(NT+1)
    ta_sw = write(ta_sw, 1, tf_ctx.sw0)
    ta_p = write(ta_p, 1, constant(zeros(tf_ctx.m, tf_ctx.n)))
    i = constant(1, dtype=Int32)
    function condition(i, tas...)
        i <= tf_ctx.NT
    end
    function body(i, tas...)
        ta_sw, ta_p = tas
        sw, p = onestep(read(ta_sw, i), read(ta_p, i), tf_ctx.m, tf_ctx.n, tf_ctx.h, tf_ctx.Δt, tf_ctx.Z, tf_ctx.ρw, tf_ctx.ρo, tf_ctx.μw, tf_ctx.μo, tf_ctx.K, tf_ctx.g, tf_ctx.ϕ, tf_ctx.qw[i], tf_ctx.qo[i])
        ta_sw = write(ta_sw, i+1, sw)
        ta_p = write(ta_p, i+1, p)
        i+1, ta_sw, ta_p
    end

    _, ta_sw, ta_p = while_loop(condition, body, [i, ta_sw, ta_p])
    out_sw, out_p = stack(ta_sw), stack(ta_p)
end

using FwiFlow
np = pyimport("numpy")

const SRC_CONST = 86400.0 #
const GRAV_CONST = 9.8    # gravity constant

# Hyperparameter for flow simulation
NT  = 1826
dt_survey = 273
Δt = 20.0 # day
z = (1:m)*h|>collect
x = (1:n)*h|>collect
X, Z = np.meshgrid(x, z)

ρw = 776.6
ρo = 1053.0
μw = 0.1
μo = 1.0

eps_ = 1e-3 # numerical instability

g = GRAV_CONST
ϕ = convert(Array{Float64},phi')
qw_value = zeros(NT, m, n)
qw_value[1:1095,75,80] .= 0.3 * (1/h^2)/Float64(extentx)*10 * SRC_CONST
qo_value = zeros(NT, m, n)
qo_value[1:1095,60,480] .= -0.3 * (1/h^2)/Float64(extentx)*10 * SRC_CONST
sw0 = zeros(m, n)
survey_indices = collect(1:dt_survey:NT+1)[1:5]
n_survey = length(survey_indices)

K = convert(Array{Float64},perm')

qw = tf.placeholder(tf.float64)
qo = tf.placeholder(tf.float64)

tfCtxTrue = tfCtxGen(m,n,h,NT,Δt,Z,X,ρw,ρo,μw,μo,K,g,ϕ,qw,qo, sw0, true)

out_sw_true, out_p_true = imseq(tfCtxTrue)

sess = Session(); init(sess)
@time conc = sess.run(out_sw_true[survey_indices], Dict(qw=>qw_value,qo=>qo_value))

JLD2.@save "../data/bgCO2.jld2" conc
