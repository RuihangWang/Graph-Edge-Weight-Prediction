class Triadic_Status():

    def __init__(self, G):
        self.G = G

    def cal_w_(self, u, v):
        iter_triad = 0
        w_ = 0
        for x in self.G.succ[u]:
            if self.G.has_edge(x, v):
                iter_triad += 1
                w_ += self.G[u][x]['weight'] + self.G[x][v]['weight']
            if self.G.has_edge(v, x):
                iter_triad += 1
                w_ += self.G[u][x]['weight'] - self.G[v][x]['weight']
        for x in self.G.pred[u]:
            if self.G.has_edge(x, v):
                iter_triad += 1
                w_ += self.G[x][v]['weight'] - self.G[x][u]['weight']
            if self.G.has_edge(v, x):
                iter_triad += 1
                w_ += -(self.G[v][x]['weight'] + self.G[x][u]['weight'])
        if iter_triad != 0:
            w_ /= iter_triad
        return w_