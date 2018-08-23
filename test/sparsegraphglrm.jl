using LowRankModels, GraphGLRM

m,n,k = 20,20,3
X = randn(k,m)
Y = randn(k,n)
A = X'*Y

# form unweighted graph with edges connecting subsequent observations for each company
nodes = collect(1:m)
edges = [(i,i+1) for i in 1:(m-1)]
timegraph = IndexGraph(nodes, edges)

quadamt = .01
graphscale = .1
glrm = GGLRM(A, QuadLoss(), GraphQuadReg(timegraph, graphscale, quadamt),
                            QuadReg(quadamt), 5)
X,Y,ch = fit_sparse!(glrm)
