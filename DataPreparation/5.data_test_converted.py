import numpy as np
import pickle
import scipy.io as sio
from config import data_path
from tqdm.auto import tqdm

data = sio.loadmat(data_path, squeeze_me=True, struct_as_record=False)['data']
data_dict = {d.name:d for d in data}
testTF = pickle.load(open('./data/testTF.pkl','rb'))
rNum = np.load('./data/rNum_train.npy')

names_train = open('./data/train.txt').read().split('\n')
names_test = open('./data/test.txt').read().split('\n')
n_train = len(names_train)
n_test = len(names_test)

D = np.load('./data/D_test_train.npy')
data_converted = []
for i in tqdm(range(n_test)):
    d = data_dict[names_test[i]]
    d_converted = {}
    d_converted['boundary'] = d.boundary
    d_converted['tf'] = testTF[i]
    topK = np.argsort(D[i])[:1000]
    d_converted['topK'] = topK
    d_converted['topK_rNum'] = rNum[topK]
    data_converted.append(d_converted)

sio.savemat('./data/data_test_converted.mat',{'data':data_converted,'testNameList':names_test,'trainNameList':names_train})
data = sio.loadmat('./data/data_test_converted.mat', squeeze_me=True, struct_as_record=False)
pickle.dump(data,open('./data/data_test_converted.pkl','wb'))
