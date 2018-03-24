### CONCRETE TYPE: DIRECT PROX EVALUATION
# prox! is computed using a QR factorization of A'.

struct IndAffineDirect{R <: Real, T <: RealOrComplex{R}, M <: AbstractMatrix{T}, V <: AbstractVector{T}, F <: Factorization} <: IndAffine
  A::M
  b::V
  fact::F
  res::V
  function IndAffineDirect{R, T, M, V, F}(A::M, b::V) where {R <: Real, T <: RealOrComplex{R}, M <: AbstractMatrix{T}, V <: AbstractVector{T}, F <: Factorization}
    if size(A,1) > size(A,2)
      error("A must be full row rank")
    end
    normrowsinv = 1.0 ./ vec(sqrt.(sum(abs2.(A), dims=2)))
    A = normrowsinv.*A # normalize rows of A
    b = normrowsinv.*b # and b accordingly
    Aadj = similar(A')
    adjoint!(Aadj, A)
    fact = LinearAlgebra.qrfact(Aadj)
    new(A, b, fact, similar(b))
  end
end

is_cone(f::IndAffineDirect) = norm(f.b) == 0.0

IndAffineDirect(A::M, b::V) where {R <: Real, T <: RealOrComplex{R}, M <: DenseMatrix{T}, V <: AbstractVector{T}} = IndAffineDirect{R, T, M, V, LinearAlgebra.QRCompactWY{T, M}}(A, b)

IndAffineDirect(A::M, b::V) where {R <: Real, T <: RealOrComplex{R}, I <: Integer, M <: SparseMatrixCSC{T, I}, V <: AbstractVector{T}} = IndAffineDirect{R, T, M, V, SuiteSparse.SPQR.QRSparse{T}}(A, b)

IndAffineDirect(a::V, b::T) where {R <: Real, T <: RealOrComplex{R}, V <: AbstractVector{T}} = IndAffineDirect(reshape(a,1,:), [b])

function (f::IndAffineDirect{R, T, M, V, F})(x::V) where {R, T, M, V, F}
  mul!(f.res, f.A, x)
  f.res .= f.b .- f.res
  # the tolerance in the following line should be customizable
  if norm(f.res, Inf) <= 1e-12
    return zero(R)
  end
  return typemax(R)
end

function prox!(y::V, f::IndAffineDirect{R, T, M, V, F}, x::V, gamma::R=one(R)) where {R, T, M, V, F <: LinearAlgebra.QRCompactWY}
  mul!(f.res, f.A, x)
  f.res .= f.b .- f.res
  Rfact = view(f.fact.factors, 1:length(f.b), 1:length(f.b))
  LinearAlgebra.LAPACK.trtrs!('U', 'C', 'N', Rfact, f.res)
  LinearAlgebra.LAPACK.trtrs!('U', 'N', 'N', Rfact, f.res)
  mul!(y, adjoint(f.A), f.res)
  y .+= x
  return zero(R)
end

function prox!(y::V, f::IndAffineDirect{R, T, M, V, F}, x::V, gamma::R=one(R)) where {R, T, M, V, F <: SuiteSparse.SPQR.QRSparse{T}}
  mul!(f.res, f.A, x)
  f.res .= f.b .- f.res
  # We have QR = PA'S, so A' = P'QRS', AA' = SR'Q'PP'QRS' = SR'RS'
  # So to solve AA'x = c, compute x = S(R\(R'\(S'c)))
  permute!(f.res, f.fact.pcol)
  ldiv!(LowerTriangular(f.fact.R'), f.res)
  ldiv!(UpperTriangular(f.fact.R), f.res)
  invpermute!(f.res, f.fact.pcol)
  mul!(y, adjoint(f.A), f.res)
  y .+= x
  return zero(R)
end

function prox_naive(f::IndAffineDirect, x::AbstractArray{T,1}, gamma::R=one(R)) where {R <: Real, T <: RealOrComplex{R}}
  y = x + f.A'*((f.A*f.A')\(f.b - f.A*x))
  return y, zero(R)
end
