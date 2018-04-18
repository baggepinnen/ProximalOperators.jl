# L-infinity norm

export NormLinf

"""
**``L_∞`` norm**

    NormLinf(λ=1.0)

Returns the function
```math
f(x) = λ⋅\\max\\{|x_1|, …, |x_n|\\},
```
for a nonnegative parameter `λ`.
"""
NormLinf(lambda::R=1.0) where {R <: Real} = Conjugate(IndBallL1(lambda))

function (f::Conjugate{IndBallL1{R}})(x::AbstractArray{S}) where {R <: Real, S <: RealOrComplex}
  return (f.f.r)*vecnorm(x, Inf)
end

function gradient!(y::AbstractArray{T}, f::Conjugate{IndBallL1{R}}, x::AbstractArray{T}) where {T <: RealOrComplex, R <: Real}
  # Find largest absolute value -- replaces:
  #   absxi, i = findmax(abs(xi) for xi in x)
  # which doesn't seem to work anymore in Julia 0.7
  imax = 1
  amax = abs(x[1])
  for i = 2:length(x)
    absxi = abs(x[i])
    if absxi > amax
      imax = i
      amax = absxi
    end
  end
  y .= 0
  y[imax] = f.f.r*sign(x[imax])
  return f.f.r*amax
end

fun_name(f::Postcompose{Conjugate{IndBallL1{R}}, R}) where {R <: Real} = "weighted L-infinity norm"
fun_expr(f::Postcompose{Conjugate{IndBallL1{R}}, R}) where {R <: Real} = "x ↦ λ||x||_∞ = λ⋅max(abs(x))"
fun_params(f::Postcompose{Conjugate{IndBallL1{R}}, R}) where {R <: Real} = "λ = $(f.a*(f.f.f.r))"
