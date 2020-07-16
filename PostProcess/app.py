from g2p.model import Model
from g2p.box_utils import centers_to_extents
from g2p.retrieval import DataRetriever
from g2p.floorplan import FloorPlan
from g2p.align import align_fp_refine
from g2p.add_archs import add_door_window

import numpy as np
import pickle
import torch

class App():
    def __init__(self,model_path,device='cpu', data_path=None,tf_path=None,centroid_path=None,cluster_path=None):
        super().__init__()
        self.load_model(model_path,device=device)
        if data_path is not None: self.load_database(data_path,tf_path,centroid_path,cluster_path)
    
    def load_model(self,model_path,device='cpu'):
        model = Model()
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()
        self.model = model
        self.device = device

    def load_database(self,data_path,tf_path,centroid_path,cluster_path):
        assert tf_path is not None
        assert centroid_path is not None
        assert cluster_path is not None

        self.data = pickle.load(open(data_path,'rb'))['data']
        tf_train = np.load(tf_path)
        centroids = np.load(centroid_path)
        clusters = np.load(cluster_path)
        self.retriever = DataRetriever(tf_train,centroids,clusters)

    def retrieve(self,data_query,k=10,multi_clusters=False):
        index = self.retriever.retrieve_cluster(data_query,k=k,multi_clusters=multi_clusters)
        data = self.data[index]
        return data

    def transfer(self,data_boundary,data_graph):
        fp_boundary = FloorPlan(data_boundary)
        fp_graph = FloorPlan(data_graph)
        fp_transfer = fp_boundary.adapt_graph(fp_graph)
        fp_transfer.adjust_graph()
        fp_transfer.data.rType = fp_transfer.get_rooms(tensor=False)
        fp_transfer.data.rEdge = fp_transfer.get_triples(tensor=False)[:, [0, 2, 1]]
        return fp_transfer.data

    def forward(self,data,network_data=False):
        if network_data:
            data.box = np.concatenate([data.gtBoxNew,data.rType.reshape(-1,1)],axis=-1)
            data.edge = data.rEdge
        fp = FloorPlan(data)
        boundary, inside_box, rooms, attrs, triples = fp.get_test_data()
        boundary = boundary.unsqueeze(0).to(self.device)
        inside_box = inside_box.to(self.device)
        rooms = rooms.to(self.device)
        attrs = attrs.to(self.device)
        triples = triples.to(self.device)
        with torch.no_grad():
            model_out = self.model(
                rooms, 
                triples, 
                boundary,
                obj_to_img = None,
                attributes = attrs,
                boxes_gt= None, 
                generate = True,
                refine = True,
                relative = True,
                inside_box=inside_box
            )        
        boxes_pred,  gene_layout, boxes_refine= model_out
        boxes_pred = centers_to_extents(boxes_pred)*255
        boxes_pred = boxes_pred.squeeze().cpu().numpy().astype(int)

        boxes_refine = centers_to_extents(boxes_refine)*255
        boxes_refine = boxes_refine.squeeze().cpu().numpy().astype(int)

        #gene_layout = gene_layout*boundary[:,:1]
        gene_preds = torch.argmax(gene_layout.softmax(1).detach(),dim=1).squeeze()
        gene_preds[boundary[0,0]==0]=13
        gene_preds = gene_preds.cpu().numpy().astype(int)
       
        fp.data.predBox = boxes_pred
        fp.data.refineBox = boxes_refine
        fp.data.gene = gene_preds
        return fp.data

    def align(self,data):
        boxes_aligned, order, room_boundaries = align_fp_refine(
            data.boundary,
            data.refineBox,
            data.rType,
            data.rEdge,
            data.gene
        )

        data.newBox = boxes_aligned
        data.order = order
        data.rBoundary = room_boundaries

        return data

    def decorate(self,data):
        doors,windows = add_door_window(data)
        data.doors = doors
        data.windows = windows
        return data
    
    def generate(self,data_boundary):
        data = self.retrieve(data_boundary)[0]
        data = self.transfer(data_boundary,data)
        data = self.forward(data)
        data = self.align(data)
        data = self.decorate(data)
        return data