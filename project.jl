# project.jl
#
# 21-241 Final Project, Fall 2019
# Eric Zheng, Section 4K
#
# This file provides numerical methods for computing the eigenvectors and
# singular vectors of matrices.

using LinearAlgebra;

" DominantEigen(A)

Compute the dominant eigenvector of the diagonalizable matrix A using
the power method. We can opt to use either the column method or the vector
method, although the vector method is used by default.
"
function DominantEigen(A; tol=0.00001, max_iter=1000, method=:vector)
  if method == :column
    return DominantEigen1(A, tol=tol, max_iter=1000)
  elseif method == :vector
    return DominantEigen2(A, tol=tol, max_iter=1000)
  else
    error("method not recognized")
  end
end

function DominantEigen1(A; tol=0.00001, max_iter=1000)

  function first_nonzero_col(B; ɛ=0.000001)
    for i in axes(B, 1)
      col = B[:,i]
      if norm(col) > ɛ
        return col
      end
    end
    return nothing
  end

  # squares a matrix and then makes the first nonzero column an unit vector
  function square_and_normalize(B; ɛ=0.000001)
    B ^= 2
    return B / norm(first_nonzero_col(B, ɛ=ɛ))
  end

  # continue computing powers of A until the Euclidean norm between
  # iterations is less than the tolerance
  A0 = A
  iters::Int = 0
  x = first_nonzero_col(A)
  A = square_and_normalize(A)
  while norm(x - first_nonzero_col(A)) > tol && iters < max_iter
    x = first_nonzero_col(A)
    A = square_and_normalize(A)
    iters += 1
  end
  x = first_nonzero_col(A)
  λ = dot(A0 * x, x) / dot(x, x)
  return (x = x, λ = λ)
end

function DominantEigen2(A; tol=0.00001, max_iter=1000)

  function normalize(x)
    return x / norm(x)
  end

  x = normalize(rand(size(A, 1)))
  iters = 0
  while norm(normalize(A * x) - x) > tol && iters < max_iter
    x = normalize(A * x)
    iters += 1
  end
  λ = dot(A * x, x) / dot(x, x)
  return (x = x, λ = λ)
end

" EigenPowerSymmetric(S)

Compute the eigenvectors of the symmetric matrix S using the power method.
This still works on non-symmetric matrices, but only returns the eigenvalues
and not the correct eigenvectors.
"
function EigenPowerSymmetric(S; tol=0.00001, max_iter=1000)
  xs = []
  λs = []
  for i in axes(S, 1)  # by Spectral Thm, symmetric S has n eigenvectors
                       # so we don't need to worry about zero, I think
    x, λ = DominantEigen2(S)
    S -= λ / (x' * x) * x * x'
    push!(xs, x)
    push!(λs, λ)
  end
  return (x = hcat(xs...), λ = hcat(λs...))
end

" EigenQR(A)

Compute the eigenvalues of the square matrix A using QR decomposition.
"
function EigenQR(A; tol=0.00001, max_iter=1000)
  Q, R = qr(A)
  B = R * Q
  iters = 1
  while norm(A - B) > tol && iters < max_iter
    Q, R = qr(B)
    A = B
    B = R * Q
    iters += 1
  end
  return diag(B)
end

" EigenSolver(A)

Ultimate subroutine to compute the eigenvectors and eigenvalues
of the square matrix A. If no method is given, default to the power method.

Returns: `(x, λ)` (for :power), `λ` (for :qr)

Examples:
```julia-repl
A = [1 3 ; 3 1];
EigenSolver(A, tol=0.001, method=:qr)
```

Notes:
* The `:power` method can be applied to non-symmetric matrices, as long as they
  are diagonalizable. In this case, the eigenvalues will be correct, but not
  the eigenvectors.
"
function EigenSolver(A; tol=0.00001, max_iter=1000, method=:power)
  if method == :power
    return EigenPowerSymmetric(A, tol=tol, max_iter=max_iter)
  elseif method == :qr
    return EigenQR(A, tol=tol)
  else
    error("method not recognized")
  end
end

" Singular(A)

Compute the singular vectors and values for the matrix A.

Returns: `(v, σ)`

Examples:
```julia-repl
A = [1 2 3 4 ; 5 6 7 8 ; 9 10 11 12];
Singular(A)
```

Notes:
* ε > 0 is the tolerance for comparing a singular value to zero.
  The default is ε = 0.0000001.
"
function Singular(A; tol=0.00001, max_iter=1000, method=:power, ε=0.0000001)
  v, λ = EigenSolver(A' * A, tol=tol, max_iter=max_iter, method=method)
  σ = map(sqrt, filter(x -> x > ε, λ))
  return (v = v[:,1:length(σ)], σ = σ)
end

