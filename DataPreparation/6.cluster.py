import pickle
import numpy as np
import faiss
from tqdm.auto import tqdm

def sample_tf(x,y,ndim=1000):
    '''
    input: tf.x,tf.y, ndim
    return: n-dim tf values
    '''
    t = np.linspace(0,1,ndim)
    return np.piecewise(t,[t>=xx for xx in x],y)

tf_train = pickle.load(open('./data/trainTF.pkl','rb'))

tf = []
for i in tqdm(range(len(tf_train))):
    tf_i = tf_train[i]
    tf.append(sample_tf(tf_i['x'],tf_i['y']))

d = 1000
tf = np.array(tf).astype(np.float32)

ncentroids = 1000
niter = 200
verbose = True

kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose,gpu=True)
kmeans.train(tf)
centroids = kmeans.centroids

index = faiss.IndexFlatL2(d)
index.add(tf)
nNN = 1000
D, I = index.search (kmeans.centroids, nNN)

np.save(f'./data/centroids_train.npy',centroids)
np.save(f'./data/clusters_train.npy',I)