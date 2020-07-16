import numpy as np
import time
def compute_tf(b):
    '''
    input: boundary points array (x,y,dir,isNew)
    return: tf.x, tf.y
    '''
    if b.shape[1]>2:
        b=b[:,:2]
    b = np.concatenate((b,b[:1]))
    nPoint = len(b)-1
    lineVec = b[1:]-b[:-1]
    lineLength = np.linalg.norm(lineVec,axis=1)
    
    perimeter = lineLength.sum()
    lineVec = lineVec/perimeter
    lineLength = lineLength/perimeter

    angles = np.zeros(nPoint)
    for i in range(nPoint):
        z = np.cross(lineVec[i],lineVec[(i+1)%nPoint])
        sign = np.sign(z)
        angles[i] = np.arccos(np.dot(lineVec[i],lineVec[(i+1)%nPoint]))*sign

    x = np.zeros(nPoint+1)
    y = np.zeros(nPoint+1)
    s = 0
    for i in range(1,nPoint+1):
        x[i] = lineLength[i-1]+x[i-1]
        y[i-1] = angles[i-1]+s
        s = y[i-1]
    y[-1] = s
    return x,y

def sample_tf(x,y,ndim=1000):
    '''
    input: tf.x,tf.y, ndim
    return: n-dim tf values
    '''
    t = np.linspace(0,1,ndim)
    return np.piecewise(t,[t>=xx for xx in x],y)

class DataRetriever():
    def __init__(self,tf_train,centroids,clusters):
        '''
        tf_train: tf of training data
        centroids: tf cluster centroids of training data
        clusters: data index for each cluster of training data
        '''
        self.tf_train = tf_train
        self.centroids = centroids
        self.clusters = clusters
    
    def retrieve_bf(self,datum,k=20):
        # compute tf for the data boundary
        x,y = compute_tf(datum.boundary)
        y_sampled = sample_tf(x,y,1000)
        dist = np.linalg.norm(y_sampled-self.tf_train,axis=1)
        if k>np.log2(len(self.tf_train)):
            index = np.argsort(dist)[:k]
        else:
            index = np.argpartition(dist,k)[:k]
            index = index[np.argsort(dist[index])]
        return index

    def retrieve_cluster(self,datum,k=20,multi_clusters=False):
        '''
        datum: test data
        k: retrieval num
        return: index for training data 
        '''
        # compute tf for the data boundary
        x,y = compute_tf(datum.boundary)
        y_sampled = sample_tf(x,y,1000)
        # compute distance to cluster centers
        dist = np.linalg.norm(y_sampled-self.centroids,axis=1)

        if multi_clusters:
            # more candicates
            c = int(np.max(np.clip(np.log2(k),1,5)))
            cluster_idx = np.argsort(dist)[:c]
            cluster = np.unique(self.clusters[cluster_idx].reshape(-1))
        else:
            # only candicates
            cluster_idx = np.argmin(dist)
            cluster = self.clusters[cluster_idx]

        # compute distance to cluster samples
        dist = np.linalg.norm(y_sampled-self.tf_train[cluster],axis=1)
        index = cluster[np.argsort(dist)[:k]]
        return index

if __name__ == "__main__":
    import scipy.io as sio
    import pickle
    from time import time
    import cv2
    import matplotlib.pyplot as plt

    def vis_boundary(b):
        img = np.ones((256,256,3))
        img = cv2.line(img,tuple(b[0,:2]),tuple(b[1,:2]),(1.,1.,0.),thickness=2)
        for i in range(1,len(b)-1):
            img = cv2.line(img,tuple(b[i,:2]),tuple(b[i+1,:2]),(0.,0.,0.),thickness=2)
        img = cv2.line(img,tuple(b[0,:2]),tuple(b[-1,:2]),(0.,0.,0.),thickness=2)
        plt.imshow(img)
        plt.show()

    #data_train = sio.loadmat('data_train70.mat',squeeze_me=True,struct_as_record=False)['data']
    #data_test = #sio.loadmat('data_test15.mat',squeeze_me=True,struct_as_record=False)['data']
    t1 = time()
    train_data = pickle.load(open('data_train_converted.pkl','rb'))['data']
    t2 = time()
    print('load train',t2-t1)

    t1 = time()
    test_data = pickle.load(open('data_test_converted.pkl','rb'))
    test_data, testNameList, trainNameList = test_data['data'], list(test_data['testNameList']), list(
        test_data['trainNameList'])
    t2 = time()
    print('load test',t2-t1)

    t1 = time()
    tf_train = np.load('tf_train.npy')
    centroids = np.load('centroids_train.npy')
    clusters = np.load('clusters_train.npy')
    t2 = time()
    print('load tf/centroids/clusters',t2-t1)

    retriever = DataRetriever(tf_train,centroids,clusters)

    datum = np.random.choice(test_data,1)[0]
    vis_boundary(datum.boundary)

    t1 = time()
    index = retriever.retrieve_cluster(datum,k=10,multi_clusters=False)
    t2 = time()
    print('cluster',t2-t1)
    data_retrieval = train_data[index]
    vis_boundary(data_retrieval[0].boundary)

    t1 = time()
    index = retriever.retrieve_bf(datum,k=10)
    t2 = time()
    print('bf',t2-t1)
    data_retrieval = train_data[index]
    vis_boundary(data_retrieval[0].boundary)
