from model_uniform import *
import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

seed = np.random.randint(0,999)
np.random.seed(seed)

t = torch.tensor([2.], requires_grad=True)
gamma = 3
eta = 2

n = 500
# TIME
t = np.random.uniform(2,100,n)
t = torch.tensor(t).reshape(-1,1).cuda()


# EVENT INDICATOR
e = np.random.choice([0,1], p = [0.5,0.5])
e = torch.tensor(e)
# TRAP STRENGTH - TREATMENT EFFECTS
gamma = np.random.uniform(0,100,n)
gamma = torch.tensor(gamma).reshape(-1,1).cuda()
# PATIENT INDICATOR
rho = np.random.uniform(0,1,n)
rho = torch.tensor(rho).reshape(-1,1).cuda()




S = []
H = []
for i in range(t.size(0)):
    s, h = surv_prob(t[i,:], gamma[i,:], rho[i,:])
    S.append(s)
    H.append(-h)
S, H = torch.cat(S, dim = 0), torch.cat(H, dim = 0)
print(t)
print(S)
print(H)

fig = px.scatter_3d(x = t.flatten().detach().cpu().numpy(),
                    y = H.detach().cpu().numpy(),
                    z = S.detach().cpu().numpy(),
                    color = H.detach().cpu().numpy(),
                    )
fig.update_layout(scene = dict(
                    xaxis_title = '<b>Time</b> (t)',
                    yaxis_title = '<b>Response Score</b><br>(Patient)',
                    zaxis_title= '<b>Survival Probability</b><br>(Tumor)'),
                    margin=dict(t=0, r=0, l=0, b=0),
                    )
fig.update_layout(font = dict(size =14))
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1, y=2, z=1)
)
fig.update_layout(
    coloraxis_colorbar=dict(
        title="Response<br>Score",
    ),
)
fig.show()
fig.write_image('0_3d.pdf')
fig.write_html('0_3d.html')



fig = px.scatter_3d(x = gamma.flatten().detach().cpu().numpy(),
                    y = rho.flatten().detach().cpu().numpy(),
                    z = S.detach().cpu().numpy(),
                    color = H.detach().cpu().numpy(),
                    )
fig.update_layout(scene = dict(
                    xaxis_title = '<b>Trap Strength</b><br>(gamma)',
                    yaxis_title = '<b>Patient indicator</b><br>(a)',
                    zaxis_title= '<b>Survival Probability</b><br>(Tumor)'),
                    margin=dict(t=0, r=0, l=0, b=0),
                    )
fig.update_layout(font = dict(size =14))
camera = dict(
    eye=dict(x=-1, y= -2, z=0.7)
)
fig.update_layout(scene_camera=camera)
fig.update_layout(
    coloraxis_colorbar=dict(
        title="Response<br>Score",
    ),
)
fig.show()
fig.write_image('1_3d.pdf')
fig.write_html('1_3d.html')













































#
