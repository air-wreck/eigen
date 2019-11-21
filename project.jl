# project.jl
#
# 21-241 Final Project, Fall 2019
# Eric Zheng, Section 4K
#
# This file provides numerical methods for computing the eigenvectors and
# singular vectors of matrices.

using LinearAlgebra;

" DominantEigen(A)

Compute the dominant eigenvector of the diagonalizable matrix A.
"
function DominantEigen(A; tol=0.00001, max_iter=1000)

  function first_nonzero_col(B; ɛ=0.000001)
    for i in axes(B, 1)  # replace this with a matrix comprehension!
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

" EigenPowerSymmetric(S)

Compute the eigenvectors of the symmetric matrix S using the power method.
"
function EigenPowerSymmetric(S; tol=0.00001, max_iter=1000)
  xs = []
  λs = []
  current = fill(0, size(S))
  for i in axes(S, 1)  # by Spectral Thm, symmetric A has n eigenvectors
                       # so we don't need to worry about zero, I think
    x, λ = DominantEigen(S - current)
    current += λ * x * x'
    push!(xs, x)
    push!(λs, λ)
  end
  return (x = hcat(xs...), λ = hcat(λs...))
end

" eigenvectors_qr(A)

Compute the eigenvectors of the square matrix A using QR decomposition.
"
function eigenvectors_qr(A; tol=0.00001)
  return 1
end

" eigenvectors(A)

Ultimate subroutine to compute the eigenvectors of the square matrix A.
If no method is given, default to the power method.

Examples:
```julia-repl
A = [1 3 ; 3 1];
eigenvectors(A, tol=0.001, method=:qr)
```

Notes:
* The `:power` method only works on symmetric matrices
"
function eigenvectors(A; tol=0.00001, max_iter=1000, method=:power)
  if method == :power
    return EigenPowerSymmetric(A, tol=tol, max_iter=max_iter)
  elseif method == :qr
    return eigenvectors_qr(A, tol=tol)
  else
    error("method not recognized")
  end
end
