"""
    Experiment for
    Leave N out
"""
import sys
sys.path.append('../src')
from utils import *
from page_rank import Page_Rank
from bias_deserve import Bias_Deserve
from fairness_goodness import Fairness_Goodness
from reciprocal import Reciprocal
from signed_hits import Sighed_Hits
from status_theory import Status_Theory
from triadic_balance import Triadic_Balance
from triadic_status import Triadic_Status
from multiple_regression import Linear_Regression

<<<<<<< HEAD
print('\nSelect dataset for evaluation\n')
print('Avaliable datset: BTCAlphaNet.csv, OTCNet.csv, RFAnet.csv\n')

G = init_Graph(filename=input('Input dataset:'), path='../dataset/')

if __name__ == "__main__":
    percentages = list(range(10, 100, 10))

    for step, n in enumerate(percentages):
        G_n = leave_out_n(G, n)

        print('\nWaiting for the prediction of leaving {} out\n'.format(str(n)+'%'))
        PR = Page_Rank(G_n)
        BG = Bias_Deserve(G_n)
        FG = Fairness_Goodness(G_n)
        RP = Reciprocal(G_n)
        SH = Sighed_Hits(G_n)
        ST = Status_Theory(G_n)
        TB = Triadic_Balance(G_n)
        TS = Triadic_Status(G_n)
        LR = Linear_Regression(G,G_n,PR,FG,SH)

        print("PageRank:", predict_weight(PR, G, G_n))
        print("Bias_Deserve:", predict_weight(BG, G, G_n))
        print("Fairness_Goodness:", predict_weight(FG, G, G_n))
        print("Reciprocal:", predict_weight(RP, G, G_n))
        print("Signed HITS:", predict_weight(SH, G, G_n))
        print("Status_Theory:", predict_weight(ST, G, G_n))
        print("Triadic_Balance:", predict_weight(TB, G, G_n))
        print("Triadic_Status:", predict_weight(TS, G, G_n))
        print("Linear_Regression:", predict_weight(LR, G, G_n))
=======
G = init_Graph(filename='OTCNet.csv', path='../dataset/')

percentages = list(range(10, 100, 10))

for step, n in enumerate(percentages):
    G_n = leave_out_n(G, n)
    print(n)
    PR = Page_Rank(G_n)
    BD = Bias_Deserve(G_n)
    FG = Fairness_Goodness(G_n)
    RP = Reciprocal(G_n)
    SH = Sighed_Hits(G_n)
    ST = Status_Theory(G_n)
    TB = Triadic_Balance(G_n)
    TS = Triadic_Status(G_n)
    LR = Linear_Regression(G,G_n,PR,FG,SH)

    print(predict_weight(PR, G, G_n))
    print(predict_weight(BD, G, G_n))
    print(predict_weight(FG, G, G_n))
    print(predict_weight(RP, G, G_n))
    print(predict_weight(SH, G, G_n))
    print(predict_weight(ST, G, G_n))
    print(predict_weight(TB, G, G_n))
    print(predict_weight(TS, G, G_n))
    print(predict_weight(LR, G, G_n))
>>>>>>> 7c63b36ac1aa178c8ceffe61b1169ec1b53cb750














