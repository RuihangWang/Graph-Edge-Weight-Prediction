import pandas as pd
import plotly.graph_objects as go 

print("Possible results: leave_N_pcc/rmse_+'dataset' \n e.g. leave_N_pcc_BTCAlphaNet.csv, leave_N_rmse_BTCAlphaNet.csv")
print('Avaliable datset: BTCAlphaNet.csv, OTCNet.csv, RFAnet.csv\n')
path = '../results/'
filename = input('Input results:')

try:
    df = pd.read_csv(path + filename, index_col=0)
except:
    print('None such results')

percentages = list(range(10, 100, 10))

PageRank = list(df.loc['PageRank'])
Bias_Deserve = list(df.loc['Bias_Deserve'])
Fairness_Goodness = list(df.loc['Fairness_Goodness'])
Reciprocal = list(df.loc['Reciprocal'])
Signed_HITS = list(df.loc['Signed_HITS'])
Status_Theory = list(df.loc['Status_Theory'])
Triadic_Balance = list(df.loc['Triadic_Balance'])
Triadic_Status = list(df.loc['Triadic_Status'])
Linear_Regression = list(df.loc['Linear_Regression'])

fig = go.Figure()
# Create and style traces
fig.add_trace(go.Scatter(x=percentages, y=PageRank, name='PageRank',
                         line=dict(color='rosybrown', width=4)))
fig.add_trace(go.Scatter(x=percentages, y=Bias_Deserve, name = 'Bias_Deserve',
                         line=dict(color='gainsboro', width=4)))
fig.add_trace(go.Scatter(x=percentages, y=Fairness_Goodness, name='Fairness_Goodness',
                         line=dict(color='firebrick', width=4, dash='dash')))
fig.add_trace(go.Scatter(x=percentages, y=Reciprocal, name='Reciprocal',
                         line = dict(color='darkorchid', width=4, dash='dash')))
fig.add_trace(go.Scatter(x=percentages, y=Signed_HITS, name='Signed_HITS',
                         line = dict(color='lightcoral', width=4, dash='dot')))
fig.add_trace(go.Scatter(x=percentages, y=Status_Theory, name='Status_Theory',
                         line=dict(color='royalblue', width=4, dash='dot')))
fig.add_trace(go.Scatter(x=percentages, y=Triadic_Balance, name='Triadic_Balance',
                         line=dict(color='olivedrab', width=4, dash='dashdot')))
fig.add_trace(go.Scatter(x=percentages, y=Triadic_Status, name='Triadic_Status',
                         line=dict(color='palevioletred', width=4, dash='dashdot')))
fig.add_trace(go.Scatter(x=percentages, y=Linear_Regression, name='Linear_Regression',
                         line=dict(color='mediumaquamarine', width=4, dash='dot')))
# dash options include 'dash', 'dot', and 'dashdot'

# Edit the layout
if 'rmse' in filename:
    fig.update_layout(title='RMSE variation of {}'.format(filename[13:-4]),
                    xaxis_title='Percentages of edges removed',
                    yaxis_title='Root Mean Square Error')

elif 'pcc' in filename:
    fig.update_layout(title='PCC variation{}'.format(filename[13:-4]),
                    xaxis_title='Percentages of edges removed',
                    yaxis_title='Pearson Correlation Coeff.')

fig.show()

fig.write_image("../results/fig2.png")