
import numpy as np
import scipy.io as sio
from config import data_path
from tqdm.auto import tqdm

data = sio.loadmat(data_path, squeeze_me=True, struct_as_record=False)['data']
data_dict = {d.name:d for d in data}

names_train = open('./data/train.txt').read().split('\n')
n_train = len(names_train)

rNum = np.zeros((n_train,14),dtype='uint8')
for i in tqdm(range(n_train)):
    rType = data_dict[names_train[i]].rType
    for j in range(13):
        rNum[i,j] = (rType==j).sum()
    rNum[i,13] = rNum[i,[1,5,6,7,8]].sum()

np.save('./data/rNum_train.npy',rNum)