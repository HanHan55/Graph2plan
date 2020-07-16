
import random
import pickle
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

from app import App
from g2p.plot import plot_fp

model_path = '../Interface/model/model.pth'
device='cuda'
train_path='../Interface/static/Data/data_train_converted.pkl'
tf_path='../Interface/retrieval/tf_train.npy'
centroid_path='../Interface/retrieval/centroids_train.npy'
cluster_path='../Interface/retrieval/clusters_train.npy'
dataset_path = '../Interface/static/Data/data_test_converted.pkl'

app = App(model_path,device,train_path,tf_path,centroid_path,cluster_path)

dataset = pickle.load(open(dataset_path,'rb'))['data']

# retrieve-> transfer -> predict -> align -> decorate
data_boundary = dataset[0]
data_graph = app.retrieve(data_boundary)[0]
data = app.transfer(data_boundary,data_graph)
data = app.forward(data,network_data=False)
data = app.align(data)
data = app.decorate(data)
# or just: 
# data = app.generate(data_boundary)

# visualize and save
ax = plot_fp(data.boundary,data.newBox[data.order],data.rType[data.order],data.doors,data.windows)
fig = plt.gcf()
fig.canvas.draw()
fig.canvas.print_figure('test_interface_data.png')