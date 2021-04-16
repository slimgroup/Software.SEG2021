export l2Norm_project

"""
Force the rows of L and R to have norm at most B
"""
function l2Norm_project(x::Tx, B, weights, params::Dict{String,<:Any}) where {ETx<:Number, Tx<:AbstractVector{ETx}}

if isapprox(norm(x), 0.0)
    println("WARNING: norm(x) cannot be 0 in l2Norm_project")
end

c = B/norm(x)
xout = min(1,c).*x

#Dummy
itn = 1

return xout, itn

end