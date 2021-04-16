export minL2NormSPG

function minL2NormSPG(A::Tx, b, rank1::Int, sigma, iter)where {ETx<:Number, Tx<:AbstractMatrix{ETx}}
xLS_jl = zeros(Complex{Float64}, rank1)
if length(b) != 0
    xinit   = A'*b;
    tau     = 0.0;
    #sigma   = sigmafact*norm(b[:],2);
    # Choose options for GenSPGL
        opts = spgOptions(  optTol = Float32(1e-5),
                        bpTol = Float32(1e-5),
                        decTol = Float32(1e-4),
                        project = l2Norm_project,
                        primal_norm = l2Norm_primaldual,
                        dual_norm = l2Norm_primaldual,
                        proxy = true,
                        ignorePErr = true,
                        iterations = iter,
                        funCompositeR = funCompR2,
                        verbosity = 1)
    # Create Params Dict
        afunT(x) = reshape(x,rank1,1);
        params = Dict("nr"=> rank1,
                    "Ind"=> vec(b) .== 0,
                    "numr"=> rank1,
                    "numc"=> 1,
                    "mode"=> 1,
                    "ls"=> 1,
                    "logical"=> 0,
                    "funPenalty"=> funLS)
    time1 = @elapsed xLS_jl, r, g, info = spgl1(A, vec(b), x = vec(xinit), tau = tau, sigma = sigma,  options = opts, params = params);
    println("LTime = ",string(time1))
    time1 = @elapsed xLS_jl, r, g, info = spgl1(A, vec(b), x = vec(xinit), tau = tau, sigma = sigma,  options = opts, params = params);
    println("LTime2 = ",string(time1))
end
xLS_jl = convert(Array{Float32,1},real(xLS_jl)) + (convert(Array{Float32,1},imag(xLS_jl)))im;
return xLS_jl
end