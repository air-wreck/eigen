# project.jl
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
  for i in xs
    push!(ys, norm(A[:,1] - x))
    A *= A0
    A /= norm(A[:,1])
  end
  plot(xs, map(i -> (4/5)^i, xs),
       label="(4/5)^k", xlabel="steps", ylabel="eigenvector error")
  scatter!(xs, ys, label="power method")
  savefig("power-method.pdf")
end
