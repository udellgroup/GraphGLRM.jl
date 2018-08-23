#import LowRankModels: prox, prox!, evaluate
#using LightGraphs
#export MatrixRegularizer, GraphQuadReg, matrix, prox, prox!, evaluate
import IterativeSolvers: lsqr!

abstract type AbstractGraphReg <: LowRankModels.Regularizer end

mutable struct GraphQuadReg <: AbstractGraphReg
  QL::AbstractMatrix{Float64}
  scale::Float64
  quadamt::Float64
  idxgraph::IndexGraph
end

#Retrieve the matrix component of the regularizer for use in initialization
matrix(g::GraphQuadReg) = g.QL

## Pass in a graph and a quadratic regularization amount
function GraphQuadReg(g::LightGraphs.Graph, scale::Float64=1., quadamt::Float64=1.)
  L = laplacian_matrix(g)
  QL = L + quadamt*I
  GraphQuadReg(QL, scale, quadamt, IndexGraph(g))
end

function GraphQuadReg(IG::IndexGraph, scale::Float64=1., quadamt::Float64=1.)
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
  #Y*(2α*g.scale*g.QL + eye(g.QL))⁻¹
  #g.QL is guaranteed to be sparse and symmetric positive definite
  #Factorize (2α*g.scale*g.QL + I)
  QL = Symmetric((2α*g.scale)*g.QL)
  #invQLpI = cholfact(QL, shift=1.) \ eye(QL)
  #Y*invQLpI
  transpose!(Y, A_ldiv_Bt(cholfact(QL, shift=1.), Y))
end

function prox_sparse(g::GraphQuadReg, Y::AbstractMatrix{Float64}, α::Number)
  #Y*(2α*g.scale*g.QL + eye(g.QL))⁻¹
  #g.QL is guaranteed to be sparse and symmetric positive definite
  QL = Symmetric((2α*g.scale)*g.QL)
  Yp = copy(Y)
  for i=1:size(Y,1)
    lsqr!(view(Yp,i,:), QL, view(Y,i,:), maxiter=2)
  end
  return Yp
end

function prox_sparse!(g::GraphQuadReg, Y::AbstractMatrix{Float64}, α::Number)
  #Y*(2α*g.scale*g.QL + eye(g.QL))⁻¹
  QL = Symmetric((2α*g.scale)*g.QL)
  for i=1:size(Y,1)
    lsqr!(view(Y,i,:), QL, view(Y,i,:), maxiter=2)
  end
  return Y
end

# function prox_sparse(g::GraphQuadReg, Y::AbstractMatrix{Float64}, α::Number)
#   #Y*(2α*g.scale*g.QL + eye(g.QL))⁻¹
#   #g.QL is guaranteed to be sparse and symmetric positive definite
#   QL = Symmetric((2α*g.scale)*g.QL)
#   Y / (QL + eye(QL))
# end
#
# function prox_sparse!(g::GraphQuadReg, Y::AbstractMatrix{Float64}, α::Number)
#   #Y*(2α*g.scale*g.QL + eye(g.QL))⁻¹
#   QL = Symmetric((2α*g.scale)*g.QL)
#   copy!(Y, Y / (QL + eye(QL))) # XXX can be more memory efficient
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
