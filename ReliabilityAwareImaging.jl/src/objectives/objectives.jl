# Author: Ali Siahkoohi, alisk@gatech.edu
# Date: September 2020
# Copyright: Georgia Institute of Technology, 2020

export loss_supervised


function loss_supervised(
    Net::NetworkConditionalHINT,
    X::AbstractArray{Float32,4},
    Y::AbstractArray{Float32,4}
)

    Zx, Zy, logdet = Net.forward(X, Y)
    CUDA.reclaim()
    z_size = size(Zx)

    f = sum(logpdf(0f0, 1f0, Zx))
    f = f + sum(logpdf(0f0, 1f0, Zy))
    f = f + logdet*z_size[4]

    ΔZx = -gradlogpdf(0f0, 1f0, Zx)/z_size[4]
    ΔZy = -gradlogpdf(0f0, 1f0, Zy)/z_size[4]

    ΔX, ΔY = Net.backward(ΔZx, ΔZy, Zx, Zy)[1:2]
    CUDA.reclaim()
    GC.gc()

    return -f/z_size[4], ΔX, ΔY
end
