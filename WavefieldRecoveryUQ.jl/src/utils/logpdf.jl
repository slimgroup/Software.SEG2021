# Author: Ali Siahkoohi, alisk@gatech.edu
# Date: September 2020
# Copyright: Georgia Institute of Technology, 2020

export logpdf, gradlogpdf

function logpdf(μ::Float32, σ::Float32, X)

    f = -.5f0*((X .- μ)/σ).^2
    f = f .+ -5f-1 * log(2f0π) .- log(σ)

    return f
end

function gradlogpdf(μ::Float32, σ::Float32, X)

    g = -(X .- μ)/σ^2

    return g
end

