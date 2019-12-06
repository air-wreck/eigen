# project-graph.jl
#
# 21-241 Final Project, Fall 2019
# Eric Zheng, Section 4K
#
# This file makes the pretty plots for the project.

include("project.jl")
using Plots

# we reimlpement DominantEigen to keep
# track of prior values for plotting
function PlotPowerConvergence()
  A0 = [23 5 2 ; 5 23 2 ; 2 2 26]
  x  = [1,1,1] / norm([1,1,1])
  A  = A0 / norm(A0[:,1])

  xs = 1:20  # number of algorithm steps
  ys = Float64[]
  for _ in xs
    push!(ys, norm(A[:,1] - x))
    A *= A0
    A /= norm(A[:,1])
  end
  plot(xs, map(i -> (4/5)^i, xs),
       label="(4/5)^k", xlabel="steps", ylabel="eigenvector error")
  scatter!(xs, ys, label="power method")
  savefig("power-method.pdf")
end

function RandomDiag(n)
  A  = rand(n, n)
  λs = rand(n)
  λ  = sort(λs, rev=true)[1]
  A  = A * diagm(0 => λs) / A
  return A, λ
end

function PlotPowerCmp(A0, λ, n)
  A  = A0 / norm(A0[:,1])
  xs = 1:n
  ys = Float64[]
  for _ in xs
    x = A[:,1]
    push!(ys, abs(dot(A0 * x, x) / dot(x, x) - λ))
    A *= A0
    A /= norm(A[:,1])
  end
  scatter(xs, ys, label="power method",
          xlabel="steps", ylabel="eigenvalue error")
  savefig("power-cmp.pdf")
end

function PlotQRCmp(A0, λ, n)
  Q, R = qr(A0)
  A  = R * Q
  ys = Float64[]
  xs = 1:n
  for _ in xs
    guess = sort(diag(A), rev=true)[1]
    push!(ys, abs(guess - λ))
    Q, R = qr(A)
    A0 = A
    A  = R * Q
  end
  scatter(xs, ys, label="QR method",
          xlabel="steps", ylabel="eigenvalue error")
  savefig("qr-cmp.pdf")
end

function PlotPowerQRCmp()
  A, λ = RandomDiag(5)
  PlotPowerCmp(A, λ, 20)
  PlotQRCmp(A, λ, 20)
end
