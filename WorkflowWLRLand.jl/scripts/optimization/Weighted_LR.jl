using Distributed
addprocs(11)

@everywhere using SeisJOLI, GenSPGL, Arpack, MAT, LinearAlgebra, SharedArrays, JLD, Random
@everywhere include("SubFunction/minL2NormSPG.jl")
@everywhere include("SubFunction/minFrobNormspg.jl")
@everywhere include("SubFunction/minnorm.jl")
@everywhere include("SubFunction/RCAMspg.jl")
@everywhere include("SubFunction/l2Norm_project.jl")
@everywhere include("SubFunction/l2Norm_primaldual.jl")

#Set parameters
rank1 = 250;
sigma = 1e-6;
alt = 1;
iter = 1;
weight = 0.75;

obs = load("the observed data in the low-rank domain")
m,n = size(obs);

L = load("load the prior low-rank factor L")

R = load("load the prior low-rank factor R")

#get prior subspace from prior information
U_pre = svd(L);
V_pre = svd(R);

r1 = min(rank(L),rank(R),rank1);
println(r1)
U = U_pre.U[:,1:r1];
V = V_pre.U[:,1:r1];

eta = sigma*norm(obs[:])/(2*m-1);

#interpolate the data with weighted LR reconstruction
time1 = @elapsed L,R = RCAMspg(weight^2*obs, alt, rank1, weight^2*eta, U, V, L, weight, iter);

#obtain the final factors L and R
global L += (1/weight-1) * U * (U' * L);
global R += (1/weight-1) * V * (V' * R);
global L = convert(Array{Complex{Float32},2},L);
global R = convert(Array{Complex{Float32},2},R);

