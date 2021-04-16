export RCAMspg

function RCAMspg(b, alt, r, eta, Lprev, Rprev, L::Tx, weight, iter)where {ETx<:Number, Tx<:AbstractMatrix{ETx}}

    m,n = size(b);

     #change for different rows and cols 2021.01.28
    #indi = 1:1:(maximum[m,n]);
    indin = 1:1:n;
    indim = 1:1:m;

    R = zeros(Complex{Float32},size(Rprev));

        
        Wr = 1.;
        Ql = 1.;

    for k = 1:alt
       time1 = @elapsed R = minFrobNormspg(L,b,1, eta,indin,Lprev,weight, iter, Wr);
       println("Time1 = ", string(time1))
       time2 = @elapsed L = minFrobNormspg(R,b',1, eta,indim,Rprev,weight, iter, Ql);
       println("Time1 = ", string(time2))
    end
    return L,R
end