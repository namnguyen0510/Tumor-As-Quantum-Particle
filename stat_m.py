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

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)


df = pd.read_csv('tcga.csv')[['bcr_patient_barcode','PFI','PFI.time']]
df = df.dropna(axis = 0).reset_index(drop = True)
print(df.describe())
print(df)
df = df.drop(columns = ['bcr_patient_barcode'])
print(len(df))
