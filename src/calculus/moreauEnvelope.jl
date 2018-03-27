export MoreauEnvelope

"""
**Moreau envelope**

    MoreauEnvelope(f, γ=1.0)

Returns the Moreau envelope (also known as Moreau-Yosida regularization) of function `f` with parameter `γ` (positive), that is
```math
f^γ(x) = \\min_z \\left\\{ f(z) + \\tfrac{1}{2γ}\\|z-x\\|^2 \\right\\}.
```
If ``f`` is convex, then ``f^γ`` is a smooth, convex, lower approximation to ``f``, having the same minima as the original function.
"""
struct MoreauEnvelope{R <: Real, T <: ProximableFunction} <: ProximableFunction
	g::T
	lambda::R
end

function MoreauEnvelope{R, T}(g::T, lambda::R) where {R <: Real, T <: ProximableFunction}
	if lambda <= 0 error("parameter lambda must be positive") end
	MoreauEnvelope{R, T}(g, lambda)
end

# MoreauEnvelope(g::T, lambda::R=1.0) where {R <: Real, T <: ProximableFunction} = MoreauEnvelope{R, T}(g, lambda)

is_convex(f::MoreauEnvelope) = is_convex(f.g)
is_smooth(f::MoreauEnvelope) = is_convex(f.g)
is_quadratic(f::MoreauEnvelope) = is_generalized_quadratic(f.g)
is_strongly_convex(f::MoreauEnvelope) = is_strongly_convex(f.g)

function (f::MoreauEnvelope)(x::AbstractArray)
	if isnull(f.buf[1])
		f.buf[1] = Nullable{AbstractArray}(similar(x))
	end
	y, g_y = prox(f.g, x, f.lambda)
	return g_y + 1/(2*f.lambda)*deepvecnorm(y-x)^2
end

function gradient!(grad::AbstractArray, f::MoreauEnvelope, x::AbstractArray)
	g_prox = prox!(grad, f.g, x, f.lambda)
	grad .= (x - grad)/f.lambda
	fx = g_prox + (f.lambda/2)*deepvecnorm(grad)^2
	return fx
end

fun_name(f::MoreauEnvelope,i::Int64) =
"f$(i)(prox{λ$(i),f$(i)}(A$(i)x))+ 1/2 ‖x - prox{λ$(i),f$(i)}(A$(i)x)‖²"

fun_par( f::MoreauEnvelope,i::Int64)  = "λ$i = $(round(f.lambda,3))"
