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
the power method.
"
function DominantEigen(A; tol=0.00001, max_iter=1000)

  function first_nonzero_col(B; ɛ=0.000001)
    for i in axes(B, 1)  # replace this with a matrix comprehension?
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
    return B / norm(first_nonzero_col(B))
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

# try to fix the previous version's problem with zero matrices
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
"
function EigenPowerSymmetric(S; tol=0.00001, max_iter=1000)
  # TODO: DominantEigen works better for [1 0 ; 0 -1]
  #       whereas DominantEigen2 handles zeros more gracefully
  xs = []
  λs = []
  for i in axes(S, 1)  # by Spectral Thm, symmetric S has n eigenvectors
                       # so we don't need to worry about zero, I think
    # x, λ = DominantEigen2(S - current)
    # current += λ * x * x'
    # attempt Hotelling deflation
    x, λ = DominantEigen2(S)
    S -= λ / (x' * x) * x * x'
    push!(xs, x)
    push!(λs, λ)
  end
  return (x = hcat(xs...), λ = hcat(λs...))
end

" EigenQR(A)

Compute the eigenvectors of the square matrix A using QR decomposition.
We require that A have full rank.
"
function EigenQR(A; tol=0.00001, max_iter=1000)

  function qr_step(B)
    Q, R = qr(B)
    return B, R * Q
  end

  A0, A = qr_step(A)
  iters = 1
  while norm(A - A0) > tol && iters < max_iter
    A0, A = qr_step(A)
    iters += 1
  end
  return A
end

" EigenSolver(A)

Ultimate subroutine to compute the eigenvectors and eigenvalues
of the square matrix A. If no method is given, default to the power method.

Returns: `(x, λ)`

Examples:
```julia-repl
A = [1 3 ; 3 1];
EigenSolver(A, tol=0.001, method=:qr)
```

Notes:
* The `:power` method only works on symmetric matrices
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
"
function Singular(A; tol=0.00001, max_iter=1000, method=:power)
  v, λ = EigenSolver(A' * A, tol=tol, max_iter=max_iter, method=method)
  σ = map(sqrt, filter(x -> x > tol, λ))  # should we be reusing tol?
  return (v = v[:,1:length(σ)], σ = σ)
end

# examples that break my current thing
ex1 = [0 -1 ; 1 0]  # rotation pi/2, no real eigenvectors
ex2 = [0 1 ; 0 0]   # only one eigenvector, wth eigenvalue 0
ex3 = [0 1 ; 1 0]   # neither eigenvalue is dominant
ex4 = [0 -1 0 ; 1 0 0 ; 0 0 1]  # dominant eigenvalues are actually complex

# DominantEigen1 is better on
ex3 = [1 0 ; 1 0]

# DominantEigen2 is better on
# ex4 =
