export minnorm

function minnorm(L::Tx, b, k, eta, rank1, iter)where {ETx<:Number, Tx<:AbstractMatrix{ETx}}
    ######Use closed form solution to solve for k-th row of R or L
    bsub = b[:,k];
    ind = findall(x->x!=0, bsub);
    v = minL2NormSPG(L[ind,:],bsub[ind,:],rank1, eta, iter);
    return v
end