import random
import scipy.io as sio
import matplotlib.pyplot as plt

from app import App
from g2p.plot import plot_fp

model_path = '../Interface/model/model.pth'
device='cuda'
dataset_path = '../Network/data/data_test.mat'

app = App(model_path,device)

dataset = sio.loadmat(dataset_path, squeeze_me=True, struct_as_record=False)['data']
data_test = dataset[0]

data = app.forward(data_test,network_data=True)
data = app.align(data)
data = app.decorate(data)

# visualize and save
ax = plot_fp(data.boundary,data.newBox[data.order],data.rType[data.order],data.doors,data.windows)
fig = plt.gcf()
fig.canvas.draw()
fig.canvas.print_figure('test.png')