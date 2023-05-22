from model_uniform import *
import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import optuna
from optuna.samplers import TPESampler
import tqdm
from lifelines.utils import concordance_index
import sqlite3
import plotly.io as pio
pio.kaleido.scope.mathjax = None

seed = 77
#np.random.seed(seed)
#torch.manual_seed(seed)

n_pred = 20
f_ = 0
fig = make_subplots(1,5)
for f_idx in range(5):
    data = 'test'
    df = pd.read_csv('{}_seed_{}_fold_{}.csv'.format(data,seed,f_idx))
    df['PFI.time'] = df['PFI.time']+2
    t = df['PFI.time']
    e = df['PFI']
    sort_idx = np.argsort(t)
    t = torch.tensor(t.to_numpy().flatten()).reshape(-1,1).cuda()
    e = torch.tensor(e.to_numpy().flatten()).reshape(-1,1).cuda()
    t = t[sort_idx]
    e = e[sort_idx]
    print(df)
    con = sqlite3.connect('seed_{}_fold_{}.db'.format(seed,f_idx))

    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print(cursor.fetchall())

    df = pd.read_sql_query("SELECT * FROM trial_params", con)
    df = df.merge(pd.read_sql_query("SELECT * FROM trial_values", con), on = 'trial_id')
    df = df.sort_values(by = 'value', ascending = False)
    #df = dict(df.iloc[0,:])
    #print(dict(df))

    tau = df[df['param_name'] == 'tau']['param_value'].iloc[0]
    e_tau = df[df['param_name'] == 'e_tau']['param_value'].iloc[0]
    rho = df[df['param_name'] == 'rho']['param_value'].iloc[0]
    e_rho = df[df['param_name'] == 'e_rho']['param_value'].iloc[0]

    print(tau, e_tau, rho, e_rho)
    for _ in tqdm.tqdm(range(n_pred)):
        #print(tau, e_tau, rho, e_rho)
        if tau-e_tau > 0:
            _tau = np.random.uniform(tau-e_tau,tau+e_tau,t.size(0))
        else:
            _tau = np.random.uniform(0,tau+e_tau,t.size(0))
        _tau = torch.tensor(_tau).reshape(-1,1).cuda()
        # PATIENT INDICATOR
        if rho-e_rho > 0:
            _rho = np.random.uniform(rho-e_rho,rho+e_rho,t.size(0))
        else:
            _rho = np.random.uniform(0,rho+e_rho,t.size(0))
        _rho = torch.tensor(_rho).reshape(-1,1).cuda()
        S = []
        H = []
        for i in range(t.size(0)):
            s, h = surv_prob(t[i,:], _tau[i,:], _rho[i,:])
            S.append(s)
            H.append(-h)
        S, H = torch.cat(S, dim = 0), torch.cat(H, dim = 0)
        #print(t)
        print(S)
        fig.add_trace(go.Scatter(x=t.flatten().detach().cpu().numpy(),y = S.detach().cpu().numpy(),mode = 'markers',name = 'Fold {}|Test'.format(f_idx),
            marker = dict(color = px.colors.qualitative.Set2[f_idx], opacity = 0.5, size = 5)
            ), row = 1, col = 1+f_)
    f_ += 1



fig.update_layout(width = 1600, height = 500,yaxis_title = '<b>Progression-free Probability</b><br>S(t)',
    xaxis_title = '<b>Time in days</b>',font = dict(size = 24))

fig.update_layout(showlegend = False)
fig.update_layout({'plot_bgcolor': 'rgba(255,255,255,1)',
					'paper_bgcolor': 'rgba(255,255,255,1)'})
fig.update_xaxes(showline = True, linecolor = 'black', linewidth = 2)
fig.update_yaxes(showline = True, linecolor = 'black', linewidth = 2)
fig.update_yaxes(range = [0,0.4])
fig.update_xaxes(range = [1000,10000])
fig.update_layout(legend = dict(orientation = 'h', x = 0, y = -0.4, itemsizing='constant'))
fig.write_image('result_TCGA_CV5_SURV_{}.pdf'.format(n_pred))

fig.show()



















































#
