from page_rank import pagerank_predict_weight, pagerank_PR_graph
import fairness_goodness_computation as FG
from signed_hits import signed_hits, signed_G_hits
from triadic_status import triadic_status
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error

def train_reg(G, G_n,fairness, goodness, G_PR, G_hits):
    train_w_x = []
    train_w_y = []
    for (u, v, w) in G_n.edges(data='weight'):
        w_pr = pagerank_predict_weight(G, G_PR, (u, v))
        w_fg = FG.FG_predict_weight(G, G_n, fairness, goodness, (u, v))
        w_sh = signed_hits(G, G_hits, (u, v))
        w_x = [w_pr, w_fg, w_sh]
        train_w_x.append(w_x)
        train_w_y.append(w)

    # train Linear Regression
    train_w_x = np.array(train_w_x)
    train_w_y = np.array(train_w_y)
    reg = LinearRegression().fit(train_w_x, train_w_y)
    return reg

def cal_w_(G, G_n,fairness, goodness, G_PR, G_hits, reg, u, v):
    w_pr = pagerank_predict_weight(G, G_PR, (u, v))
    w_fg = FG.FG_predict_weight(G, G_n, fairness, goodness, (u, v))
    w_sh = signed_hits(G, G_hits, (u, v))
    w_ = float(reg.predict(np.array([[w_pr, w_fg, w_sh]])))
    return w_

def Multiple_Regression(G, G_n,fairness, goodness, G_PR, G_hits, u_v_edge = None):
    reg = train_reg(G, G_n,fairness, goodness, G_PR, G_hits)
    print(reg.coef_, reg.intercept_)

    if u_v_edge is not None:
        (u, v) = u_v_edge
        return cal_w_(G, G_n,fairness, goodness, G_PR, G_hits, reg, u, v)

    total_w = []
    total_w_ = []
    for (u, v, w) in G.edges(data='weight'):
        if G_n.has_edge(u, v):
            continue
        w_ = cal_w_(G, G_n, fairness, goodness, G_PR, G_hits, reg, u, v)

        total_w_.append(w_)
        total_w.append(w)
    RMSE = mean_squared_error(total_w, total_w_)
    PCC = pearsonr(total_w, total_w_)

    return RMSE, PCC[0]










