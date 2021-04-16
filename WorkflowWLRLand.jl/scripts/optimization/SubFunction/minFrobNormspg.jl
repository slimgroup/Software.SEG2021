export minFrobNormspg

function minFrobNormspg(L::Tx, b, p, eta, indi, Lprev, weight, iter, fac)where {ETx<:Number, Tx<:AbstractMatrix{ETx}}
    ###### add for weighted part
    if weight!=1.0
        L*= weight
        L += (1/weight- 1) * Lprev * ( Lprev'*L)
    end
    

    ###### Matrix Dimension Setup
    undefine_1,m = size(b);
    undefine_2,r = size(L);
    R1 = Array{Any}(undef,m,r);

    ###### Solve for each row of R independently
        R1 = @distributed vcat for k = 1:length(indi)
         
            minnorm(L,b,k,eta, r, iter);
           
        end

        R = conj(permutedims(reshape(R1,r,m),[2,1]));

    return R
end