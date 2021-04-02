# Author: Ali Siahkoohi, alisk@gatech.edu
# Date: September 2020
# Copyright: Georgia Institute of Technology, 2020

export loss_supervised

function loss_supervised(
    Net::NetworkConditionalHINT, X::AbstractArray{Float32,4}, Y::AbstractArray{Float32,4}
)

    Zx, Zy, logdet = Net.forward(X, Y)

    z_size = size(tensor_cat(Zx, Zy))
    f = sum(logpdf(0f0, 1f0, tensor_cat(Zx, Zy))) + logdet*z_size[4]

    ΔZ = -gradlogpdf(0f0, 1f0, tensor_cat(Zx, Zy))/z_size[4]
    ΔZx, ΔZy = tensor_split(ΔZ)
    ΔX, ΔY = Net.backward(ΔZx, ΔZy, Zx, Zy)[1:2]

    GC.gc()

    return -f/z_size[4], ΔX, ΔY
end
