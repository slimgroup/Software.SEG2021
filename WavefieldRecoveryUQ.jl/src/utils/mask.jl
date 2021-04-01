# Author: Ali Siahkoohi, alisk@gatech.edu
# Date: February 2021
# Copyright: Georgia Institute of Technology, 2021

export Mask

mutable struct Mask
    M::AbstractArray{Float32,4}
    fac::Float32
end


function Mask(shape::Array{Int64,1}, fac::Float32)

    nzeros = Int(prod(shape[1:2]) * fac)

    M = ones(Float32, prod(shape[1:2]))
    M[sort(randperm(prod(shape[1:2]))[1:nzeros])] .= 0f0
    M = reshape(M, (shape[1], shape[2], 1, 1))

    return Mask(M, fac)
end

function *(Mask::Mask , X::AbstractArray{Float32,4})
    return Mask.M .* X
end
