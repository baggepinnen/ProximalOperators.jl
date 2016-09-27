# indicator of a generic box

"""
  IndBox(lb, ub)

Returns the function `g = ind{x : lb ⩽ x ⩽ ub}`. Parameters `lb` and `ub` can be
either scalars or arrays of the same dimension as `x`, and must satisfy `lb <= ub`.
Bounds are allowed to take values `-Inf` and `+Inf`.
"""

immutable IndBox{T <: Union{Real, AbstractArray}, S <: Union{Real, AbstractArray}} <: IndicatorConvex
  lb::T
  ub::S
  IndBox(lb::T, ub::S) =
    any(lb .> ub) ? error("arguments lb, ub must satisfy lb <= ub") : new(lb, ub)
end

IndBox{T <: Real}(lb::T, ub::T) = IndBox{Real, Real}(lb, ub)

IndBox{T <: Real}(lb::AbstractArray{T}, ub::T) = IndBox{AbstractArray{T}, T}(lb, ub)

IndBox{T <: Real}(lb::T, ub::AbstractArray{T}) = IndBox{T, AbstractArray{T}}(lb, ub)

IndBox{T <: Real}(lb::AbstractArray{T}, ub::AbstractArray{T}) =
  size(lb) != size(ub) ? error("bounds must have the same dimensions, or at least one of them be scalar") :
  IndBox{AbstractArray{T}, AbstractArray{T}}(lb, ub)

lb{T <: Real, S}(f::IndBox{T, S}, i) = f.lb
lb{T <: AbstractArray, S}(f::IndBox{T, S}, i) = f.lb[i]
ub{T, S <: Real}(f::IndBox{T, S}, i) = f.ub
ub{T, S <: AbstractArray}(f::IndBox{T, S}, i) = f.ub[i]

@compat function (f::IndBox){R <: Real}(x::AbstractArray{R})
  for k in eachindex(x)
    if x[k] < lb(f,k) || x[k] > ub(f,k)
      return +Inf
    end
  end
  return 0.0
end

function prox!{R <: Real}(f::IndBox, x::AbstractArray{R}, y::AbstractArray{R}, gamma::Real=1.0)
  for k in eachindex(x)
    if x[k] < lb(f,k)
      y[k] = lb(f,k)
    elseif x[k] > ub(f,k)
      y[k] = ub(f,k)
    else
      y[k] = x[k]
    end
  end
  return 0.0
end

"""
  IndBallInf(r::Real)

Returns the indicator function of an infinity-norm ball, that is function
`g(x) = ind{maximum(abs(x)) ⩽ r}` for `r ⩾ 0`.
"""

IndBallInf{R <: Real}(r::R) = IndBox(-r, r)

"""
  IndNonnegative()

Returns the indicator function the nonnegative orthant, that is

  `g(x) = 0 if x ⩾ 0, +∞ otherwise`
"""

IndNonnegative() = IndBox(0.0, +Inf)

"""
  IndNonpositive()

Returns the indicator function the nonpositive orthant, that is

  `g(x) = 0 if x ⩽ 0, +∞ otherwise`
"""

IndNonpositive() = IndBox(-Inf, 0.0)

fun_name(f::IndBox) = "indicator of a box"
fun_type(f::IndBox) = "Array{Real} → Real ∪ {+∞}"
fun_expr(f::IndBox) = "x ↦ 0 if all(lb ⩽ x ⩽ ub), +∞ otherwise"
fun_params(f::IndBox) =
  string( "lb = ", typeof(f.lb) <: Array ? string(typeof(f.lb), " of size ", size(f.lb)) : f.lb, ", ",
          "ub = ", typeof(f.ub) <: Array ? string(typeof(f.ub), " of size ", size(f.ub)) : f.ub)

function prox_naive{R <: Real}(f::IndBox, x::AbstractArray{R}, gamma::Real=1.0)
  y = min(f.ub, max(f.lb, x))
  return y, 0.0
end
