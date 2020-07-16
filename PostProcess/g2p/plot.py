import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely import geometry
from .utils import room_label

def get_color_map():
    color = np.array([
        [244,242,229], # living room
        [253,244,171], # bedroom
        [234,216,214], # kitchen
        [205,233,252], # bathroom
        [208,216,135], # balcony
        [249,222,189], # Storage
        [ 79, 79, 79], # exterior wall
        [255,225, 25], # FrontDoor
        [128,128,128], # interior wall
        [255,255,255]
    ],dtype=np.int64)
    cIdx  = np.array([1,2,3,4,1,2,2,2,2,5,1,6,1,10,7,8,9,10])-1
    return color[cIdx]
cmap = get_color_map()/255.0

def get_figure(size=512):
    if np.array(size).size==1:
        w,h = size,size
    else:
        w,h = size[0],size[1]
    fig = plt.figure()
    dpi = fig.get_dpi()
    fig.set_size_inches(w/dpi,h/dpi)
    fig.set_frameon(False)
    return fig    

def get_axes(size=512,fig=None,rect=[0,0,1,1]):
    if fig is None: fig = get_figure(size)

    ax = fig.add_axes(rect)
    #ax.set_frame_on(False)
    ax.set_aspect('equal')
    ax.set_xlim([0,255])
    ax.set_ylim([0,255])
    ax.invert_yaxis()
    ax.set_axis_off()
    
    return ax

def plot_category(category,show_boundary=True,ax=None):
    if ax is None: ax = get_axes()
    img = np.ones((category.shape[0],category.shape[1],4))
    img[...,:3] = cmap[category]
    img[category==13,3] = 0
    if not show_boundary: 
        img[np.isin(category,[14,15])]=[1.,1.,1.,0.]
    ax.imshow(img)
    return ax

def plot_boundary(boundary, wall_thickness=6,ax=None):
    if ax is None: ax = get_axes()
    
    is_new = boundary[:,-1]==1
    poly_boundary = geometry.Polygon(boundary[~is_new,:2])
    x,y = poly_boundary.exterior.xy
    ax.fill(x,y,fc='none',ec=cmap[14],lw=wall_thickness,joinstyle='round')

    door = boundary[:2,:2]
    idx = np.argmin(np.sum(door,axis=-1), axis=0)
    if idx==1: door = door[[1,0]]

    ori = boundary[0,2]
    if ori%2==0:
        door = door+np.array([
            [wall_thickness/4,0],[-wall_thickness/4,0]
        ])
    else:
        door = door+np.array([
            [0,wall_thickness/4],
            [0,-wall_thickness/4]
        ])
    ax.plot(door[:,0],door[:,1],color=cmap[15],lw=wall_thickness+1)

    return ax

def plot_graph(boundary, boxes, types, edges, wall_thickness=6,with_boundary=True,ax=None):
    if ax is None: ax = get_axes()

    if with_boundary: plot_boundary(boundary,wall_thickness,ax)

    boxes = boxes.astype(float)
    r_node = np.zeros((len(boxes),3))
    for k in range(len(boxes)):
        r_node[k,:2] = (boxes[k,:2]+boxes[k,2:])/2
        r_node[k,2] = (boxes[k,2]-boxes[k,0])*(boxes[k,3]-boxes[k,1])
    
    for i in range(len(edges)):
        idx = edges[i,:2]
        ax.plot(r_node[idx,0],r_node[idx,1],'-',color=[0.7,0.7,0.7],lw=wall_thickness/2)

    is_new = boundary[:,-1]==1
    poly_boundary = geometry.Polygon(boundary[~is_new,:2])
    for i in range(len(r_node)):
        s = round(10*r_node[i,2])/poly_boundary.area*3+wall_thickness*3
        ax.plot(r_node[i,0],r_node[i,1],'o',mec=cmap[16],mfc=cmap[types[i]],ms=s)

    return ax

def plot_fp(boundary, boxes, types, doors=[], windows=[], wall_thickness=6, fontsize=0, keep_box=False, alpha=1.0, ax=None):
    if ax is None: ax = get_axes()

    is_new = boundary[:,-1]==1
    poly_boundary = geometry.Polygon(boundary[~is_new,:2])
    poly = dict()
    
    for k in range(len(boxes)):
        poly_room = geometry.box(*boxes[k])
        poly[k] = poly_boundary.intersection(poly_room)
        if poly[k].area==0: 
            print(f'ploting empty box {k}!') 
            continue

        if keep_box:
            poly[k] = geometry.box(*poly[k].bounds)
            x,y = poly[k].exterior.xy
            ax.fill(x,y,fc=cmap[types[k]],ec=cmap[16],alpha=alpha,lw=wall_thickness,joinstyle='round')
        else:
            if poly[k].geom_type!='Polygon':
                for p in poly[k]:
                    if p.geom_type!='Polygon': continue
                    x,y = p.exterior.xy
                    ax.fill(x,y,fc=cmap[types[k]],ec=cmap[16],alpha=alpha,lw=wall_thickness,joinstyle='round')
            else:
                x,y = poly[k].exterior.xy
                ax.fill(x,y,fc=cmap[types[k]],ec=cmap[16],alpha=alpha,lw=wall_thickness,joinstyle='round')

    plot_boundary(boundary,wall_thickness,ax)

    if len(doors)>0:
        plot_door(doors, wall_thickness/3, ax)
    if len(windows)>0:
        plot_window(windows, wall_thickness/3, ax)

    if fontsize!=0:         
        for k in range(len(boxes)):
            if poly[k].area==0: continue
            cx, cy = poly[k].centroid.x,poly[k].centroid.y
            ax.text(cx, cy, room_label[types[k]][1], fontsize=fontsize,horizontalalignment='center',verticalalignment='center')
    
    return ax

def plot_window(windows, thickness=2, ax=None):
    if ax is None: ax = get_axes()
    for k in range(len(windows)):
        window = windows[k]
        seg = np.zeros((2,2))
        seg[0] = window[1:3]
        seg[1] = window[1:3]+ window[3:5]
    
        box = np.concatenate([seg.min(0),seg.max(0)],axis=-1)

        if window[3] < window[4]:
            box = box + np.array([-1,0,1,0]) * thickness
            if window[4] > 0:
                box[1] = box[1] + thickness
                seg[0,1] = seg[0,1] + thickness
            else:
                box[3] = box[3] - thickness
                seg[0,1] = seg[0,1] - thickness
        else:
            box = box + np.array([0,-1,0,1]) * thickness
            if window[3] > 0:
                box[0] = box[0] + thickness
                seg[0,0] = seg[0,0] + thickness
            else:
                box[2] = box[2] - thickness
                seg[0,0] = seg[0,0] - thickness
        
        ax.fill(box[[0,0,2,2,0]], box[[1,3,3,1,1]], 'w')
        
        ax.plot(box[[0,0,2,2,0]], box[[1,3,3,1,1]], color=[0.4,0.4,0.4], lw=1)
        ax.plot(seg[:,0], seg[:,1], color=[0.4,0.4,0.4], lw=1)
    
    return ax

def plot_door(doors, thickness, ax=None):
    if ax is None: ax = get_axes()
    for k in range(len(doors)):
        door = doors[k]
        seg = np.zeros((2,2))
        seg[0] = door[1:3]
        seg[1] = door[1:3]+ door[3:5]
    
        box = np.concatenate([seg.min(0),seg.max(0)],axis=-1)

        if door[3] < door[4]:
            box = box + np.array([-1,0,1,0]) * thickness
            if door[4] > 0:
                box[1] = box[1] + thickness
            else:
                box[3] = box[3] - thickness
        else:
            box = box + np.array([0,-1,0,1]) * thickness
            if door[3] > 0:
                box[0] = box[0] + thickness
            else:
                box[2] = box[2] - thickness
        
        ax.fill(box[[0,0,2,2,0]], box[[1,3,3,1,1]], 'w', ec=cmap[16])

    return ax
        