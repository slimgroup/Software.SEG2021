# Author: Ali Siahkoohi, alisk@gatech.edu
# Date: February 2021

function getindex(J::HDF5.Dataset, ::Colon, ::Colon, ::Colon, a::Array{Int64,1})
    X = zeros(Float32, size(J, 1), size(J, 2), size(J, 3), length(a))

    for (i, idx) in enumerate(a)
        X[:, :, :, i] = J[:, :, :, idx]
    end
    return X
end
