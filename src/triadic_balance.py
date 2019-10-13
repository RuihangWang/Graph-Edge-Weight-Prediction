class Triadic_Balance():
    def __init__(self,G):
        self.G = G

    def cal_w_(self, u, v):

        neigh_u = set(self.G.pred[u]) | set(self.G.succ[u])
        neigh_v = set(self.G.pred[v]) | set(self.G.succ[v])
        neigh = neigh_u & neigh_v
        w_u = []
        w_v = []
        if len(neigh) != 0:
            for nu in list(neigh):
                if self.G.has_edge(u, nu):
                    w_u.append(self.G[u][nu]['weight'])
                if self.G.has_edge(nu, u):
                    w_u.append(self.G[nu][u]['weight'])
            for nv in list(neigh):
                if self.G.has_edge(v, nv):
                    w_v.append(self.G[v][nv]['weight'])
                if self.G.has_edge(nv, v):
                    w_v.append(self.G[nv][v]['weight'])

            num = len(w_u + w_v)
            total = sum(w_u + w_v)
            w_ = total / num
        else:
            w_ = 0
        return w_