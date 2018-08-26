#import LowRankModels: prox, prox!, evaluate
#using LightGraphs
#export MatrixRegularizer, GraphQuadReg, matrix, prox, prox!, evaluate

abstract type AbstractGraphReg <: LowRankModels.Regularizer end

mutable struct GraphQuadReg <: AbstractGraphReg
  QL::AbstractMatrix{Float64}
  scale::Number
  quadamt::Number
  idxgraph::IndexGraph
end

#Retrieve the matrix component of the regularizer for use in initialization
matrix(g::GraphQuadReg) = g.QL

## Pass in a graph and a quadratic regularization amount
function GraphQuadReg(g::LightGraphs.Graph, scale::Number=1., quadamt::Number=1.)
  L = laplacian_matrix(g)
  QL = L + quadamt*I
  GraphQuadReg(QL, scale, quadamt, IndexGraph(g))
end

function GraphQuadReg(IG::IndexGraph, scale::Number=1., quadamt::Number=1.)
  QL = laplacian_matrix(IG.graph) + quadamt*I
  GraphQuadReg(QL, scale, quadamt, IG)
end

function prox(g::GraphQuadReg, Y::AbstractMatrix{Float64}, α::Number)
  #Y*(2α*g.scale*g.QL + eye(g.QL))⁻¹
  #g.QL is guaranteed to be sparse and symmetric positive definite
  #Factorize (2α*g.scale*g.QL + I)
  QL = Symmetric((2α*g.scale)*g.QL)
  A_ldiv_Bt(cholfact(QL, shift=1.), Y)'
end

function prox!(g::GraphQuadReg, Y::AbstractMatrix{Float64}, α::Number)
  # Y*(2α*g.scale*g.QL + eye(g.QL))⁻¹
  #g.QL is guaranteed to be sparse and symmetric positive definite
  #Factorize (2α*g.scale*g.QL + I)
  QL = Symmetric((2α*g.scale)*g.QL)
  chol_QL = cholfact(QL, shift=1.)
  #invQLpI = cholfact(QL, shift=1.) \ eye(QL)
  #Y*invQLpI
  transpose!(Y, A_ldiv_Bt(chol_QL, Y))
end

### All of these ideas work less well and are slower than the standard prox method, for sparse chordal (=most) graphs
# function prox_sparse!(g::GraphQuadReg, Y::AbstractMatrix{Float64}, α::Number)
#   #Y*(2α*g.scale*g.QL + eye(g.QL))⁻¹
#   QL = Symmetric((2α*g.scale)*g.QL)
#   # A = Symmetric((2α*g.scale)*g.QL) + I
#   # time for this scales quadratically (!) with size(Y,2)
#   # for i=1:size(Y,1)
#   #   lsqr!(view(Y,i,:), A, view(Y,i,:), maxiter=2)
#   # end
#
#   # so instead, here's one iter of cg, directly
#   # r = Y-Y*A
#   # n = diag(r*r') # [norm(r[i,:]) for i=1:size(Y,1)].^2
#   # n ./= diag(r*A*r') # [dot(r[i,:], Ap[i,:]) for i=1:size(Y,1)]
#   # Y += Diagonal(n)*r
#
#   # or 10 iters of cg
#   r = Y*QL # = Y - Y*A
#   rsq = diag(r*r')
#   p = copy(r)
#   Ap = zeros(size(r))
#   for iter=1:10
#     Ap = p*(QL + I)
#     # mul!(Ap, p, Ql)
#     # Ap += p
#     α = rsq ./ diag(p * Ap')
#     Y += Diagonal(α)*p
#     r -= Diagonal(α)*Ap
#     rsq, rsq_prev = diag(r*r'), rsq
#     if sum(rsq) < 1e-10
#       break
#     end
#     β = rsq ./ rsq_prev
#     p = r + Diagonal(β)*p
#     # lmul!(Diagonal(β), p)
#     # p += r
#   end
#
#   return Y
# end

function evaluate(g::GraphQuadReg, Y::AbstractMatrix{Float64})
  g.scale*sum((Y'*Y) .* g.QL)
end

function evaluate_sparse(g::GraphQuadReg, Y::AbstractMatrix{Float64})
  rows, cols, vals = findnz(g.QL)
  r = 0
  for (i,j) in zip(rows, cols)
    r += g.QL[i,j]*dot(Y[:,i], Y[:,j])
  end
  return r*g.scale
end

function embed(g::GraphQuadReg, yidxs::Array)
  GraphQuadReg(embed_graph(g.idxgraph, yidxs), g.scale, g.quadamt)
end
