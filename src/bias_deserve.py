import math

class Bias_Deserve():

    def __init__(self, G):

        bias, des = self.initialize_scores(G)

        nodes = G.nodes()
        iter = 0
        while iter < 100:
            dB = 0
            dD = 0
            # print('-----------------')
            # print("Iteration number", iter)
            # print('Updating des')
            for node in nodes:
                inedges = G.in_edges(node, data='weight')
                D = 0
                for edge in inedges:
                    X = max(0, bias[edge[0]] * edge[2])
                    D += edge[2] * (1 - X)
                try:
                    dD += abs(D / len(inedges) - des[node])
                    des[node] = D / len(inedges)
                except:
                    pass

            # print('Updating bias')
            for node in nodes:
                outedges = G.out_edges(node, data='weight')
                B = 0
                for edge in outedges:
                    B += (edge[2] - des[edge[1]]) / 2.0
                try:
                    dB += abs(B / len(outedges) - bias[node])
                    bias[node] = B / len(outedges)
                except:
                    pass

            # print('Differences in bias score and des score = %.6f, %.6f' % (dB, dD))
            if dB < math.pow(10, -6) and dD < math.pow(10, -6):
                break
            iter += 1

        self.bias = bias
        self.des = des

    def initialize_scores(self, G):
        des = {}
        bias = {}
        nodes = G.nodes()
        for node in nodes:
            bias[node] = 1
            try:
                des[node] = G.in_degree(node, weight='weight') * (
                            1.0 - max(0, bias[node] * G.in_degree(node, weight='weight'))) / G.in_degree(node)
            except:
                des[node] = 0
        return bias, des
    
    def compute_bias_des(self, G):
        return self.bias, self.des

    def cal_w_(self, u, v):
        return self.des[v]
