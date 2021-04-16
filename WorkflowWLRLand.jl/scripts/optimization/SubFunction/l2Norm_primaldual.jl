export l2Norm_primaldual

"""
Force the rows of L and R to have norm at most B
"""
function l2Norm_primaldual(x::Tx, weights, params::Dict{String,<:Any}) where {ETx<:Number, Tx<:AbstractVector{ETx}}

p = norm(x[:])

return p

end