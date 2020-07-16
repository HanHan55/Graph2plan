from  model.floorplan import *
from  model.box_utils import *
from  model.model import Model
import os
from  model.utils import *

import Houseweb.views as vw

import numpy as np
import time
import math
import matlab.engine
  

global adjust,indxlist
adjust=False

def get_data(fp):
    batch = list(fp.get_test_data())
    batch[0] = batch[0].unsqueeze(0).cuda()
    batch[1] = batch[1].cuda()
    batch[2] = batch[2].cuda()
    batch[3] = batch[3].cuda()
    batch[4] = batch[4].cuda()
    return batch

def test(model,fp):
    with torch.no_grad():
        batch = get_data(fp)
        boundary,inside_box,rooms,attrs,triples = batch
        model_out = model(
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
        boxes_pred = boxes_pred.detach()
        boxes_pred = centers_to_extents(boxes_pred)
        boxes_refine = boxes_refine.detach()
        boxes_refine = centers_to_extents(boxes_refine)
        gene_layout = gene_layout*boundary[:,:1]
        gene_preds = torch.argmax(gene_layout.softmax(1).detach(),dim=1)
        return boxes_pred.squeeze().cpu().numpy(),gene_preds.squeeze().cpu().double().numpy(),boxes_refine.squeeze().cpu().numpy()

def load_model():
    
    model = Model()
    model.cuda(0)
    model.load_state_dict(
        torch.load('./model/model.pth', map_location={'cuda:0': 'cuda:0'}))
    model.eval()
    return model

def get_userinfo(userRoomID,adptRoomID):
    start = time.clock()
    global model
    test_index = vw.testNameList.index(userRoomID.split(".")[0])
    test_data = vw.test_data[test_index]

    # boundary
    Boundary = test_data.boundary
    boundary=[[float(x),float(y),float(z),float(k)] for x,y,z,k in list(Boundary)]
    
    test_fp =FloorPlan(test_data)

    train_index = vw.trainNameList.index(adptRoomID.split(".")[0])
    train_data = vw.train_data[train_index]
    train_fp =FloorPlan(train_data,train=True)
    fp_end = test_fp.adapt_graph(train_fp)
    fp_end.adjust_graph()
    return fp_end


def get_userinfo_adjust(userRoomID,adptRoomID,NewGraph):
    global adjust,indxlist
    test_index = vw.testNameList.index(userRoomID.split(".")[0])
    test_data = vw.test_data[test_index]
    # boundary
    Boundary = test_data.boundary
    boundary=[[float(x),float(y),float(z),float(k)] for x,y,z,k in list(Boundary)]
    
    test_fp =FloorPlan(test_data)

    train_index = vw.trainNameList.index(adptRoomID.split(".")[0])
    train_data = vw.train_data[train_index]
    train_fp =FloorPlan(train_data,train=True)
    fp_end = test_fp.adapt_graph(train_fp)
    fp_end.adjust_graph()

    
    newNode = NewGraph[0]
    newEdge = NewGraph[1]
    oldNode = NewGraph[2]
    
    temp = []
    for newindx, newrmname, newx, newy,scalesize in newNode:
        for type, oldrmname, oldx, oldy, oldindx in oldNode:
            if (int(newindx) == oldindx):
                tmp=int(newindx), (newx - oldx), ( newy- oldy),float(scalesize)
                temp.append(tmp)
    newbox=[]
    print(adjust)
    if adjust==True:
        oldbox = []
        for i in range(len(vw.boxes_pred)):
            indxtmp=[vw.boxes_pred[i][0],vw.boxes_pred[i][1],vw.boxes_pred[i][2],vw.boxes_pred[i][3],vw.boxes_pred[i][0]]
            oldbox.append(indxtmp)
    if adjust==False:
        indxlist=[]
        oldbox=fp_end.data.box.tolist()
        for i in range(len(oldbox)):
            indxlist.append([oldbox[i][4]])
        indxlist=np.array(indxlist)
        adjust=True
    oldbox=fp_end.data.box.tolist()

    # print("oldbox",oldbox)
    # print(oldbox,"oldbox")
    X=0
    Y=0
    for i in range(len(oldbox)):
        X= X+(oldbox[i][2]-oldbox[i][0])
        Y= Y+(oldbox[i][3]-oldbox[i][1])
    x_ave=(X/len(oldbox))/2
    y_ave=(Y/len(oldbox))/2

    index_mapping = {}
    #  The room that already exists
    #  Move: Just by the distance
    for newindx, tempx, tempy,scalesize in temp:
        index_mapping[newindx] = len(newbox)
        tmpbox=[]
        scalesize = int(scalesize)
        if scalesize<1:
            scale = math.sqrt(scalesize)
            scalex = (oldbox[newindx][2] - oldbox[newindx][0]) * (1 - scale) / 2
            scaley = (oldbox[newindx][3] - oldbox[newindx][1]) * (1 - scale) / 2
            tmpbox = [(oldbox[newindx][0] + tempx) + scalex, (oldbox[newindx][1] + tempy)+scaley,
                      (oldbox[newindx][2] + tempx) - scalex, (oldbox[newindx][3] + tempy) - scaley, oldbox[newindx][4]]
        if scalesize == 1:
            tmpbox = [(oldbox[newindx][0] + tempx) , (oldbox[newindx][1] + tempy) ,(oldbox[newindx][2] + tempx), (oldbox[newindx][3] + tempy), oldbox[newindx][4]]

        if scalesize>1:
            scale=math.sqrt(scalesize)
            scalex = (oldbox[newindx][2] - oldbox[newindx][0]) * ( scale-1) / 2
            scaley = (oldbox[newindx][3] - oldbox[newindx][1]) * (scale-1) / 2
            tmpbox = [(oldbox[newindx][0] + tempx) - scalex, (oldbox[newindx][1] + tempy) - scaley,
                      (oldbox[newindx][2] + tempx) + scalex, (oldbox[newindx][3] + tempy) + scaley, oldbox[newindx][4]]

           
        newbox.append(tmpbox)

    #  The room just added
    #  Move: The room node with the average size of the existing room
    for newindx, newrmname, newx, newy,scalesize in newNode:
        if int(newindx)>(len(oldbox)-1):
            scalesize=int(scalesize)
            index_mapping[int(newindx)] = (len(newbox))
            tmpbox=[]
            if scalesize < 1:
                scale = math.sqrt(scalesize)
                scalex = x_ave * (1 - scale) / 2
                scaley = y_ave* (1 - scale) / 2
                tmpbox = [(newx-x_ave) +scalex,(newy-y_ave) +scaley,(newx+x_ave)-scalex,(newy+y_ave)-scaley,vocab['object_name_to_idx'][newrmname]]

            if scalesize == 1:
                tmpbox = [(newx - x_ave), (newy - y_ave), (newx + x_ave), (newy + y_ave),vocab['object_name_to_idx'][newrmname]]
            if scalesize > 1:
                scale = math.sqrt(scalesize)
                scalex = x_ave * (scale - 1) / 2
                scaley = y_ave * (scale - 1) / 2
                tmpbox = [(newx-x_ave) - scalex, (newy-y_ave)  - scaley,(newx+x_ave) + scalex, (newy+y_ave) + scaley,vocab['object_name_to_idx'][newrmname]]
            print(scalesize)
            newbox.append(tmpbox)

    fp_end.data.box=np.array(newbox)
    adjust_Edge=[]
    for u, v in newEdge:
        tmp=[index_mapping[int(u)],index_mapping[int(v)], 0]
        adjust_Edge.append(tmp)
    fp_end.data.edge=np.array(adjust_Edge)
    rNode = fp_end.get_rooms(tensor=False)

    rEdge = fp_end.get_triples(tensor=False)[:, [0, 2, 1]]
    Edge = [[float(u), float(v), float(type2)] for u, v, type2 in rEdge]

    s=time.clock()
    boxes_pred, gene_layout, boxes_refeine = test(vw.model, fp_end)

    e=time.clock()
    print(' model test time: %s Seconds' % (e - s))

    boxes_pred = boxes_pred * 255
    
    fp_end.data.gene = gene_layout
    rBox = boxes_pred[:]
    Box = [[float(x), float(y), float(z), float(k)] for x, y, z, k in rBox]

    boundary_mat = matlab.double(boundary)
    rNode_mat = matlab.double(rNode.tolist())
    print("rNode.tolist()",rNode.tolist())
    Edge_mat = matlab.double(Edge)
    
    Box_mat=matlab.double(Box)
    
    fp_end.data.boundary =np.array(boundary)
    fp_end.data.rType =np.array(rNode).astype(int)
    fp_end.data.refineBox=np.array(Box)
    fp_end.data.rEdge=np.array(Edge)
    gene_mat=matlab.double(np.array(fp_end.data.gene).tolist())
    startcom= time.clock()
    box_refine =  vw.engview.align_fp(boundary_mat, Box_mat,  rNode_mat,Edge_mat,matlab.double(fp_end.data.gene.astype(float).copy().tolist()) ,18,False, nargout=3)
    endcom = time.clock()
    print(' matlab.compute time: %s Seconds' % (endcom - startcom))
    box_out=box_refine[0]
    box_order=box_refine[1]

    rBoundary=box_refine[2]
    fp_end.data.newBox = np.array(box_out)
    fp_end.data.order = np.array(box_order)
    fp_end.data.rBoundary = [np.array(rb) for rb in rBoundary]
    return fp_end,box_out,box_order, gene_layout, boxes_refeine


def get_userinfo_net(userRoomID,adptRoomID):
    global model
    test_index = vw.testNameList.index(userRoomID.split(".")[0])
    test_data = vw.test_data[test_index]

    # boundary
    Boundary = test_data.boundary
    boundary = [[float(x), float(y), float(z), float(k)] for x, y, z, k in list(Boundary)]
    test_fp = FloorPlan(test_data)

    train_index = vw.trainNameList.index(adptRoomID.split(".")[0])
    train_data = vw.train_data[train_index]
    train_fp = FloorPlan(train_data, train=True)
    fp_end = test_fp.adapt_graph(train_fp)
    fp_end.adjust_graph()
    boxes_pred, gene_layout, boxes_refeine = test(model, fp_end)
    boxes_pred=boxes_pred*255
    for i in range(len(boxes_pred)):
        for j in range(len(boxes_pred[i])):
            boxes_pred[i][j]=float(boxes_pred[i][j])
    return fp_end,boxes_pred, gene_layout, boxes_refeine

if __name__ == "__main__":
    pass
