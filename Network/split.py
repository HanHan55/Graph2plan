import scipy.io as sio
import numpy as np
import os
data = sio.loadmat('data.mat',squeeze_me=True,struct_as_record=False)['data']
np.random.shuffle(data)

len_train,len_valid = int(len(data)*0.70), int(len(data)*0.15)
# deep_layout
# len_train,len_valid = 75000, 3000

data_train = data[:len_train]
data_valid = data[len_train:len_train+len_valid]
data_test = data[len_train+len_valid:]

data_dir = './data'
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

sio.savemat(f'{data_dir}/data_train.mat',{'data':data_train})
sio.savemat(f'{data_dir}/data_valid.mat',{'data':data_valid})
sio.savemat(f'{data_dir}/data_test.mat',{'data':data_test})