class Reciprocal():
    def __init__(self, G):
        self.G = G

    def cal_w_(self, u, v):
        if self.G.has_edge(v, u):
            w = self.G[v][u]['weight']
            w_ = w
        else:
            w_ = 0
        return w_