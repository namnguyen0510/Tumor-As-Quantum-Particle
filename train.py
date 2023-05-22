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

seed = 77
np.random.seed(seed)
torch.manual_seed(seed)


df = pd.read_csv('tcga.csv')[['bcr_patient_barcode','PFI','PFI.time']]
df = df.dropna(axis = 0).reset_index(drop = True)
print(df.describe())
print(df)
df = df.drop(columns = ['bcr_patient_barcode'])

# CALIBRATE TIME
df['PFI.time'] = df['PFI.time']+2

fold_idx = 0
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state=seed)
for  train_index, test_index in skf.split(df, df['PFI']):


    def objective(trial):
        tau = trial.suggest_float('tau',0,10)
        e_tau = trial.suggest_float('e_tau',0,1e-2)
        rho = trial.suggest_float('rho',0,1)
        e_rho = trial.suggest_float('e_rho',0,1e-2)
        t_train , e_train = df['PFI.time'][train_index], df['PFI'][train_index]
        t_test , e_test = df['PFI.time'][test_index], df['PFI'][test_index]
        df.iloc[train_index].to_csv('train_seed_{}_fold_{}.csv'.format(seed,fold_idx))
        df.iloc[test_index].to_csv('test_seed_{}_fold_{}.csv'.format(seed,fold_idx))
        print(t_train.shape, t_test.shape)



        t_train = torch.tensor(t_train.to_numpy().flatten()).reshape(-1,1).cuda()
        t_test = torch.tensor(t_test.to_numpy().flatten()).reshape(-1,1).cuda()
        e_train = torch.tensor(e_train.to_numpy().flatten()).reshape(-1,1).cuda()
        e_test = torch.tensor(e_test.to_numpy().flatten()).reshape(-1,1).cuda()
        chunks = 10
        #_t_train = torch.chunk(t_train, chunks)[0]
        LOSS = []
        print('Training: {} batches from {} samples'.format(chunks,t_train.size(0)))

        _t_train = torch.chunk(t_train, chunks)
        _e_train = torch.chunk(e_train, chunks)
        for ch in tqdm.tqdm(range(len(_t_train))):
            #print(_t_train.size())
            # TRAP STRENGTH - TREATMENT EFFECTS
            if tau-e_tau > 0:
                _tau = np.random.uniform(tau-e_tau,tau+e_tau,_t_train[ch].size(0))
            else:
                _tau = np.random.uniform(0,tau+e_tau,_t_train[ch].size(0))
            _tau = torch.tensor(_tau).reshape(-1,1).cuda()
            # PATIENT INDICATOR
            if rho-e_rho > 0:
                _rho = np.random.uniform(rho-e_rho,rho+e_rho,_t_train[ch].size(0))
            else:
                _rho = np.random.uniform(0,rho+e_rho,_t_train[ch].size(0))
            _rho = torch.tensor(_rho).reshape(-1,1).cuda()
            #print('_tau: {}'.format(_tau))
            #print('_rho: {}'.format(_rho))
            def cost(t, _tau, _rho):
                S = []
                H = []
                for i in range(t.size(0)):
                    s, h = surv_prob(_t_train[ch][i,:], _tau[i,:], _rho[i,:])
                    S.append(s)
                    H.append(-h)
                S, H = torch.cat(S, dim = 0), torch.cat(H, dim = 0)
                #print(H)
                #print(_t_train[ch].shape,H.shape,_e_train[ch].shape)

                c_index = concordance_index(_t_train[ch].detach().cpu().numpy(),
                    H.detach().cpu().numpy(),
                    _e_train[ch].detach().cpu().numpy())
                return c_index


            # PENALIZE NAN FROM INTEGRATOR
            try:
                loss = cost(_t_train[ch], _tau,_rho).item()
            except:
                loss = 0
            #print(loss)
            LOSS.append(loss)
        LOSS = np.array(LOSS)
        LOSS = LOSS.mean()

        return LOSS

    study_name = 'seed_{}_fold_{}'.format(seed,fold_idx)
    study_storage = "sqlite:///{}.db".format(study_name)
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(sampler=sampler, direction = 'maximize',
            study_name=study_name,
            storage=study_storage,
            load_if_exists=True)



    study.optimize(objective, n_trials=20)
    fold_idx += 1













#
