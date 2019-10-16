"""
    experiment for
    Leave one out
"""
import numpy as np 
import pandas as pd
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

print('\nSelect dataset for evaluation\n')
print('Avaliable datset: BTCAlphaNet.csv, OTCNet.csv, RFAnet.csv\n')

filename = input('Input dataset:')
G = init_Graph(filename, path='../dataset/')

if __name__ == "__main__":

    ### remove edges
    remove_edges = leave_out_edges(G, 1000)
    ### G_1
    G_1 = G.copy()

    algorithm_type = ['PageRank', 'Bias_Deserve', 'Fairness_Goodness',
    'Reciprocal', 'Signed_HITS', 'Status_Theory',
    'Triadic_Balance', 'Triadic_Status', 'Linear_Regression']

    total_w = {'FG':[], 'PR':[], 'SH':[], 'TS':[], 'LR':[], 'BD':[], 'RP':[], 'ST':[], 'TB':[]}
    total_w_ = {'FG':[], 'PR':[], 'SH':[], 'TS':[], 'LR':[], 'BD':[], 'RP':[], 'ST':[], 'TB':[]}

    RMSE = {'FG':[], 'PR':[], 'SH':[], 'TS':[], 'LR':[], 'BD':[], 'RP':[], 'ST':[], 'TB':[]}
    PCC = {'FG':[], 'PR':[], 'SH':[], 'TS':[], 'LR':[], 'BD':[], 'RP':[], 'ST':[], 'TB':[]}
        
    print('\nWaiting for the prediction of leaving one edge out ...\n')
    PR = Page_Rank(G_1)
    BD = Bias_Deserve(G_1)
    FG = Fairness_Goodness(G_1)
    RP = Reciprocal(G_1)
    SH = Sighed_Hits(G_1)
    ST = Status_Theory(G_1)
    TB = Triadic_Balance(G_1)
    TS = Triadic_Status(G_1)
    LR = Linear_Regression(G,G_1,PR,FG,SH)

    for step, (u, v) in enumerate(remove_edges):
        G_1.remove_edge(u, v)
        
        total_w['PR'].append(predict_weight(PR, G, G_1, (u, v))[0])
        total_w['BD'].append(predict_weight(BD, G, G_1, (u, v))[0])
        total_w['FG'].append(predict_weight(FG, G, G_1, (u, v))[0])
        total_w['RP'].append(predict_weight(RP, G, G_1, (u, v))[0])
        total_w['SH'].append(predict_weight(SH, G, G_1, (u, v))[0])
        total_w['ST'].append(predict_weight(ST, G, G_1, (u, v))[0])
        total_w['TB'].append(predict_weight(TB, G, G_1, (u, v))[0])
        total_w['TS'].append(predict_weight(TS, G, G_1, (u, v))[0])
        total_w['LR'].append(predict_weight(LR, G, G_1, (u, v))[0])

        total_w_['PR'].append(predict_weight(PR, G, G_1, (u, v))[1])
        total_w_['BD'].append(predict_weight(BD, G, G_1, (u, v))[1])
        total_w_['FG'].append(predict_weight(FG, G, G_1, (u, v))[1])
        total_w_['RP'].append(predict_weight(RP, G, G_1, (u, v))[1])
        total_w_['SH'].append(predict_weight(SH, G, G_1, (u, v))[1])
        total_w_['ST'].append(predict_weight(ST, G, G_1, (u, v))[1])
        total_w_['TB'].append(predict_weight(TB, G, G_1, (u, v))[1])
        total_w_['TS'].append(predict_weight(TS, G, G_1, (u, v))[1])
        total_w_['LR'].append(predict_weight(LR, G, G_1, (u, v))[1])

        G_1.add_edge(u, v, weight=G[u][v]['weight'],
                    signed_weight=G[u][v]['signed_weight'],
                    positive=G[u][v]['positive'],
                    negative=G[u][v]['negative'])
                    
    RMSE['PR'] = mean_squared_error(total_w['PR'], total_w_['PR']) ** 0.5
    RMSE['BD'] = mean_squared_error(total_w['BD'], total_w_['BD']) ** 0.5
    RMSE['FG'] = mean_squared_error(total_w['FG'], total_w_['FG']) ** 0.5
    RMSE['RP'] = mean_squared_error(total_w['RP'], total_w_['RP']) ** 0.5
    RMSE['SH'] = mean_squared_error(total_w['SH'], total_w_['SH']) ** 0.5
    RMSE['ST'] = mean_squared_error(total_w['ST'], total_w_['ST']) ** 0.5
    RMSE['TB'] = mean_squared_error(total_w['TB'], total_w_['TB']) ** 0.5
    RMSE['TS'] = mean_squared_error(total_w['TS'], total_w_['TS']) ** 0.5
    RMSE['LR'] = mean_squared_error(total_w['LR'], total_w_['LR']) ** 0.5

    PCC['PR'] = pearsonr(total_w['PR'], total_w_['PR'])[0]
    PCC['BD'] = pearsonr(total_w['BD'], total_w_['BD'])[0]
    PCC['FG'] = pearsonr(total_w['FG'], total_w_['FG'])[0]
    PCC['RP'] = pearsonr(total_w['RP'], total_w_['RP'])[0]
    PCC['SH'] = pearsonr(total_w['SH'], total_w_['SH'])[0]
    PCC['ST'] = pearsonr(total_w['ST'], total_w_['ST'])[0]
    PCC['TB'] = pearsonr(total_w['TB'], total_w_['TB'])[0]
    PCC['TS'] = pearsonr(total_w['TS'], total_w_['TS'])[0]
    PCC['LR'] = pearsonr(total_w['LR'], total_w_['LR'])[0]

    RMSE_stack = np.vstack((RMSE['PR'], RMSE['BD'], RMSE['FG'], 
                            RMSE['RP'], RMSE['SH'], RMSE['ST'], 
                            RMSE['TB'], RMSE['TS'], RMSE['LR']))
    
    PCC_stack = np.vstack((PCC['PR'], PCC['BD'], PCC['FG'], 
                           PCC['RP'], PCC['SH'], PCC['ST'], 
                           PCC['TB'], PCC['TS'], PCC['LR']))

    df_RMSE = pd.DataFrame(RMSE_stack, index = algorithm_type, columns = ['1'])
    df_PCC = pd.DataFrame(PCC_stack, index = algorithm_type, columns = ['1'])

    df_RMSE.to_csv('../results/leave_one_rmse_{}'.format(filename))
    df_PCC.to_csv('../results/leave_one_pcc_{}'.format(filename))

    print('RMSE:',df_RMSE)
    print('\nPCC:',df_PCC)














