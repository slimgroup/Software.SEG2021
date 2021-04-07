export put_params!

"""
    P = put_params!(NL::NeuralNetLayer, Params::Array{Parameter,1})
 Inputs a initialized ResidualBlock and desired parameters values to load.
"""
function put_params!(RB::ResidualBlock, Params::Array{Any,1})
    RB.W1.data = Params[1].data
    RB.W2.data = Params[2].data
    RB.W3.data = Params[3].data
    RB.b1.data = Params[4].data
    RB.b2.data = Params[5].data
end


function put_params!(FB::FluxBlock, Params::Array{Any,1})
    Flux.loadparams!(FB, Params)
end


function put_params!(AL::AffineLayer, Params::Array{Any,1})
    AL.s.data = Params[1].data
    AL.b.data = Params[2].data
end


function put_params!(L::CouplingLayerIRIM, Params::Array{Any,1})
    put_params!(L.C, Params[1:3])
    put_params!(L.RB, Params[4:end])
end


function put_params!(HL::HyperbolicLayer, Params::Array{Any,1})
    HL.W.data = Params[1].data
    HL.b.data = Params[2].data
end


function put_params!(H::CouplingLayerHINT, Params::Array{Any,1})
    nlayers = length(H.CL)
    put_params!(H.CL[1], Params[1:5])
    if nlayers > 1
        for j=2:nlayers
            put_params!(H.CL[j], Params[5*(j-1)+1:5*j])
        end
    end
    ~isnothing(H.C) && put_params!(H.C, Params[5*nlayers+1:end])
end


function put_params!(L::CouplingLayerGlow, Params::Array{Any,1})
    put_params!(L.C, Params[1:3])
    put_params!(L.RB, Params[4:end])
end


function put_params!(C::Conv1x1, Params::Array{Any,1})
    C.v1.data = Params[1].data
    C.v2.data = Params[2].data
    C.v3.data = Params[3].data
end


put_params!(L::CouplingLayerBasic, Params::Array{Any,1}) = put_params!(L.RB, Params)


function put_params!(AN::ActNorm, Params::Array{Any,1})
    AN.s.data = Params[1].data
    AN.b.data = Params[2].data
end


function put_params!(CH::ConditionalLayerHINT, Params::Array{Any,1})
    idx_s = 1
    counter = 0

    nlayers_x = length(CH.CL_X.CL)
    isnothing(CH.CL_X.C) ? counter += 5*nlayers_x : counter += 5*nlayers_x + 3
    put_params!(CH.CL_X, Params[idx_s:idx_s+counter-1])

    idx_s += counter
    counter = 0
    nlayers_y = length(CH.CL_Y.CL)
    isnothing(CH.CL_Y.C) ? counter += 5*nlayers_y : counter += 5*nlayers_y + 3
    put_params!(CH.CL_Y, Params[idx_s:idx_s+counter-1])

    idx_s += counter
    counter = 5
    put_params!(CH.CL_YX, Params[idx_s:idx_s+counter-1])

    idx_s += counter
    counter = 3
    ~isnothing(CH.C_X) && put_params!(CH.C_X, Params[idx_s:idx_s+counter-1]); idx_s += counter
    ~isnothing(CH.C_Y) && put_params!(CH.C_Y, Params[idx_s:idx_s+counter-1]); idx_s += counter
end


function put_params!(RB::ConditionalResidualBlock, Params::Array{Any,1})
    RB.W0.data = Params[1].data
    RB.W1.data = Params[2].data
    RB.W2.data = Params[3].data
    RB.W3.data = Params[4].data
    RB.b0.data = Params[5].data
    RB.b1.data = Params[6].data
    RB.b2.data = Params[7].data
end


function put_params!(CH::NetworkConditionalHINT, Params::Array{Any,1})
    depth = length(CH.CL)
    idx_s = 1
    counter = 0

    for j = 1:depth
        idx_s += counter
        counter = 2
        put_params!(CH.AN_X[j], Params[idx_s:idx_s+counter-1])

        idx_s += counter
        counter = 2
        put_params!(CH.AN_Y[j], Params[idx_s:idx_s+counter-1])

        idx_s += counter
        counter = 0
        nlayers_x = length(CH.CL[j].CL_X.CL)
        isnothing(CH.CL[j].CL_X.C) ? counter += 5*nlayers_x : counter += 5*nlayers_x + 3
        nlayers_y = length(CH.CL[j].CL_Y.CL)
        isnothing(CH.CL[j].CL_Y.C) ? counter += 5*nlayers_y : counter += 5*nlayers_y + 3
        counter += 5
        ~isnothing(CH.CL[j].C_X) && (counter += 3)
        ~isnothing(CH.CL[j].C_Y) && (counter += 3)

        put_params!(CH.CL[j], Params[idx_s:idx_s+counter-1])
    end
end


function put_params!(G::NetworkGlow, Params::Array{Any,1})
    L, K = size(G.AN)
    idx_s = 1
    counter = 0

    for i=1:L
        for j=1:K
            idx_s += counter
            counter = 2
            put_params!(G.AN[i, j], Params[idx_s:idx_s+counter-1])

            idx_s += counter
            counter = 8
            put_params!(G.CL[i, j], Params[idx_s:idx_s+counter-1])
        end
    end
end


function put_params!(H::NetworkHyperbolic, Params::Array{Any,1})
    depth = length(H.CL)
    put_params!(H.AL, Params[1:2])
    for j = 1:depth
        put_params!(H.HL[j], Params[2*j+1:2*(j+1)])
    end
end


function put_params!(UL::NetworkLoop, Params::Array{Any,1})
    maxiter = length(UL.L)

    if UL.L[1] isa CouplingLayerHINT
        idx_s = 1
        counter = 0
        nlayers_x = length(UL.L[1].CL)
        isnothing(UL.L[1].C) ? counter += 5*nlayers_x : counter += 5*nlayers_x + 3
        put_params!(UL.L[1], Params[idx_s:idx_s+counter-1])

        if maxiter > 1
            for j=2:maxiter
                idx_s += counter
                counter = 0
                nlayers_x = length(UL.L[j].CL)
                isnothing(UL.L[j].C) ? counter += 5*nlayers_x : counter += 5*nlayers_x + 3
                put_params!(UL.L[j], Params[idx_s:idx_s+counter-1])
            end
        end

    elseif UL.L[1] isa CouplingLayerIRIM
        idx_s = 1
        counter = 8
        put_params!(UL.L[1], Params[idx_s:idx_s+counter-1])

        if maxiter > 1
            for j=2:maxiter
                idx_s += counter
                put_params!(UL.L[j], Params[idx_s:idx_s+counter-1])
            end
        end
    end
end

