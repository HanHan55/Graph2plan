import os
import pickle
import numpy as np
import scipy.io as sio
from config import data_path
from tqdm.auto import tqdm

def compute_tf(boundary):
    '''
    input: boundary points array (x,y,dir,isNew)
    return: tf.x, tf.y
    '''
    if boundary.shape[1]>2:
        boundary=boundary[:,:2]
    boundary = np.concatenate((boundary,boundary[:1]))
    num_point = len(boundary)-1
    line_vector = boundary[1:]-boundary[:-1]
    line_length = np.linalg.norm(line_vector,axis=1)
    
    perimeter = line_length.sum()
    line_vector = line_vector/perimeter
    line_length = line_length/perimeter

    angles = np.zeros(num_point)
    for i in range(num_point):
        z = np.cross(line_vector[i],line_vector[(i+1)%num_point])
        sign = np.sign(z)
        angles[i] = np.arccos(np.dot(line_vector[i],line_vector[(i+1)%num_point]))*sign

    x = np.zeros(num_point+1)
    y = np.zeros(num_point+1)
    s = 0
    for i in range(1,num_point+1):
        x[i] = line_length[i-1]+x[i-1]
        y[i-1] = angles[i-1]+s
        s = y[i-1]
    y[-1] = s
    return x,y

def compute_tf_dist(tf1,tf2):
    x = np.unique(np.concatenate((tf1['x'],tf2['x'])))
    dist = 0
    idx1,idx2 =0,0
    for i in range(1,len(x)-1):
        idx1 = idx1+(x[i]>tf1['x'][idx1+1])
        idx2 = idx2+(x[i]>tf2['x'][idx2+1])
        seg = x[i]-x[i-1]
        d = abs(tf1['y'][idx1]-tf2['y'][idx2])
        dist = dist+seg*d
    seg = x[-1]-x[-2]
    d = abs(tf1['y'][-1]-tf2['y'][-1])
    dist = dist+seg*d
    return dist

def sample_tf(x,y,ndim=1000):
    '''
    input: tf.x,tf.y, ndim
    return: n-dim tf values
    '''
    t = np.linspace(0,1,ndim)
    return np.piecewise(t,[t>=xx for xx in x],y)

# load data
data = sio.loadmat(data_path, squeeze_me=True, struct_as_record=False)['data']
data_dict = {d.name:d for d in data}

names_train = open('./data/train.txt').read().split('\n')
names_test = open('./data/test.txt').read().split('\n')
n_train = len(names_train)
n_test = len(names_test)

# turning function: training data
trainTF = []
tf_train = []
for i in tqdm(range(n_train)):
    boundary = data_dict[names_train[i]].boundary
    x,y = compute_tf(boundary)
    trainTF.append({'x':x,'y':y})
pickle.dump(trainTF,open('./data/trainTF.pkl','wb'))

tf_train = []
for i in tqdm(range(n_train)):
    x,y = trainTF[i]['x'],trainTF[i]['y']
    tf_train.append(sample_tf(x,y))
tf_train = np.stack(tf_train,axis=0)
np.save('./data/tf_train.npy',tf_train)
      
# turning function: testing data                   
testTF = []
for i in tqdm(range(n_test)):
    boundary = data_dict[names_test[i]].boundary
    x,y = compute_tf(boundary)
    testTF.append({'x':x,'y':y})
pickle.dump(testTF,open('./data/testTF.pkl','wb'))

# turning function distance: test-train
print('Computing turning function distance ... it will take a long time.')
D_test_train = np.zeros((n_test,n_train),dtype='float32')
for i in tqdm(range(n_test)):
    for j in range(n_train):
        D_test_train[i,j] = compute_tf_dist(testTF[i],trainTF[j])
np.save('./data/D_test_train.npy',D_test_train)
