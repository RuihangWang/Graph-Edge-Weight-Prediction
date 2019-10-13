import numpy as np
from sklearn.linear_model import LinearRegression
from utils import predict_weight

class Linear_Regression():
    def __init__(self, G, G_n, PR, FG, SH):
        self.G = G
        self.G_n = G_n
        self.PR = PR
        self.FG = FG
        self.SH = SH

        train_w_x = []
        train_w_y = []
        for (u, v, w) in G_n.edges(data='weight'):
            w_pr = predict_weight(PR, G, G_n, (u, v))
            w_fg = predict_weight(FG, G, G_n, (u, v))
            w_sh = predict_weight(SH, G, G_n, (u, v))
            w_x = [w_pr, w_fg, w_sh]
            train_w_x.append(w_x)
            train_w_y.append(w)

        # train Linear Regression
        train_w_x = np.array(train_w_x)
        train_w_y = np.array(train_w_y)
        reg = LinearRegression().fit(train_w_x, train_w_y)
        # print(reg.coef_, reg.intercept_)

        self.reg = reg

    def cal_w_(self, u, v):
        w_pr = predict_weight(self.PR, self.G, self.G_n, (u, v))
        w_fg = predict_weight(self.FG, self.G, self.G_n, (u, v))
        w_sh = predict_weight(self.SH, self.G, self.G_n, (u, v))
        w_ = float(self.reg.predict(np.array([[w_pr, w_fg, w_sh]])))
        return w_









