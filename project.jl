# project.jl
#
# 21-241 Final Project, Fall 2019
# Eric Zheng, Section 4K
#
# This file provides numerical methods for computing the eigenvectors and
# singular vectors of matrices.

using LinearAlgebra;

" eigenvectors_power(A)

Compute the eigenvectors of the square matrix A using the power method.
"
function eigen_power_symmetric(A; tol=0.00001, max_iter=1000)
  function greatest_eigen(B)
    original = B
    # continue computing powers of A until the Euclidean norm between
    # iterations is less than the tolerance
    iters::Int = 0
    prev = B[:,1]
    B ^= 2
    B /= norm(B[:,1])
    while norm(prev - B[:,1]) > tol && iters < max_iter
      prev = B[:,1]
      B ^= 2
      B /= norm(B[:,1])
      iters += 1
    end
    x = B[:,1]
    λ = norm(original * x) * sign((original * x)[1] * x[1])
    return (x = x, λ = λ)
  end
  xs = []
  λs = []
  current = fill(0, size(A))
  for i in axes(A, 1)  # by Spectral Thm, symmetric A has n eigenvectors
    x, λ = greatest_eigen(A - current)
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
eigenvectors(A, tol=0.001, method=:qr)
```
"
function eigenvectors(A; tol=0.00001, max_iter=1000, method=:power)
  if method == :power
    return eigen_power_symmetric(A, tol=tol, max_iter=max_iter)
  elseif method == :qr
    return eigenvectors_qr(A, tol=tol)
  else
    error("method not recognized")
  end
end
