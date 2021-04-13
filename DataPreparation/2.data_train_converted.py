import numpy as np
import pickle
import scipy.io as sio
from config import data_path
from tqdm.auto import tqdm

# load data
data = sio.loadmat(data_path, squeeze_me=True, struct_as_record=False)['data']
data_dict = {d.name:d for d in data}

names_train = open('./data/train.txt').read().split('\n')
n_train = len(names_train)

trainTF = pickle.load(open('./data/trainTF.pkl','rb'))

data_converted = []

for i in tqdm(range(n_train)):
    d = data_dict[names_train[i]]
    d_converted = {}
    d_converted['name'] = d.name
    d_converted['boundary'] = d.boundary
    d_converted['box'] = np.concatenate([d.gtBoxNew,d.rType[:,None]],axis=-1)
    d_converted['order'] = d.order
    d_converted['edge'] = d.rEdge
    d_converted['rBoundary'] = d.rBoundary
    data_converted.append(d_converted)

sio.savemat('./data/data_train_converted.mat',{'data':data_converted,'nameList':names_train,'trainTF':trainTF})
data = sio.loadmat('./data/data_train_converted.mat', squeeze_me=True, struct_as_record=False)
pickle.dump(data,open('./data/data_train_converted.pkl','wb'))

