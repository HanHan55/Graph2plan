import numpy as np
import matlab
import matlab.engine as engine
import os
eng = engine.start_matlab()
eng.addpath(os.path.join(os.path.dirname(__file__),'matlab'),nargout=0)

GT_ThRESHOLD = 6
PRED_ThRESHOLD = 12
REFINE_ThRESHOLD = 18

def align_fp(boundary, boxes, types, edges, image, threshold, dtype=int):
    boundary = np.array(boundary,dtype=int).tolist()
    boxes    = np.array(boxes,dtype=int).tolist()
    types    = np.array(types,dtype=int).tolist()
    edges    = np.array(edges,dtype=int).tolist()
    image    = np.array(image,dtype=int).tolist()

    boxes_aligned, order, room_boundaries = eng.align_fp(
        matlab.double(boundary),
        matlab.double(boxes),
        matlab.double(types),
        matlab.double(edges),
        matlab.double(image),
        threshold,False,nargout=3
    )

    boxes_aligned   = np.array(boxes_aligned,dtype=dtype)
    order           = np.array(order,dtype=dtype).reshape(-1)-1
    room_boundaries = np.array([np.array(rb,dtype=float) for rb in room_boundaries]) # poly with hole has value 'nan'

    return boxes_aligned, order, room_boundaries

def align_fp_gt(boundary, boxes, types, edges, dtype=int):
    return align_fp(boundary, boxes, types, edges, [], GT_ThRESHOLD, dtype)

def align_fp_pred(boundary, boxes, types, edges, dtype=int):
    return align_fp(boundary, boxes, types, edges, [], PRED_ThRESHOLD, dtype)

def align_fp_refine(boundary, boxes, types, edges, image, dtype=int):
    return align_fp(boundary, boxes, types, edges, image, REFINE_ThRESHOLD, dtype)