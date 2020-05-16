import numpy as np

def get_color_map():
    color = np.array([
        [244,242,229], # living room
        [253,244,171], # bedroom
        [234,216,214], # kitchen
        [205,233,252], # bathroom
        [208,216,135], # balcony
        [185,231,168], # balcony
        [249,222,189], # Storage
        [ 79, 79, 79], # exterior wall
        [255,225, 25], # FrontDoor
        [128,128,128], # interior wall
        [255,255,255]
    ],dtype=np.int64)
    cIdx  = np.array([1,2,3,4,1,2,2,2,2,5,1,6,1,10,7,8,9,10])-1
    return color[cIdx]

def rot90_2D(pts,k=1,cnt=np.array([127.5,127.5])):
    ang = k*np.pi/2
    R = np.array([[np.cos(ang),np.sin(ang)],[-np.sin(ang),np.cos(ang)]])
    return np.dot(pts-cnt,R)+cnt
    
def fliplr_2D(pts,size=255):
    return np.stack([pts[:,0],size-pts[:,1]],1)

def align_image(image,rot_old,rot_new=0):
    k = np.ceil((rot_old-rot_new+2*np.pi)%(2*np.pi)/(np.pi/4))//2
    return np.rot90(image,k)
    
def align_box(box,rot_old,rot_new=0):
    box = box-np.array([0,0,1,1])
    k = np.ceil((rot_old-rot_new+2*np.pi)%(2*np.pi)/(np.pi/4))//2
    box = rot90_2D(box.reshape(-1,2),k).reshape(-1,4)
    return np.concatenate([np.minimum(box[:,:2],box[:,2:]),np.maximum(box[:,:2],box[:,2:])+1],-1).round().astype(int)

def fliplr_box(box,size=255):
    box = box-np.array([0,0,1,1])
    box=fliplr_2D(box.reshape(-1,2),size=size).reshape(-1,4)
    return np.concatenate([np.minimum(box[:,:2],box[:,2:]),np.maximum(box[:,:2],box[:,2:]+1)],-1).round().astype(int)

def align_points(pts, rot_old, rot_new=0):
    k = np.ceil((rot_old - rot_new + 2 * np.pi) % (2 * np.pi) / (np.pi / 4)) // 2
    pts = rot90_2D(pts, k)
    return pts.round().astype(int)

def point_box_relation(u,vbox):
    uy,ux = u
    vy0, vx0, vy1, vx1 = vbox

    if (ux<vx0 and uy<=vy0) or (ux==vx0 and uy==vy0):
        relation = 'left-above'
    elif (vx0<=ux<vx1 and uy<=vy0):
        relation = 'above'
    elif (vx1<=ux and uy<vy0) or (ux==vx1 and uy==vy0):
        relation = 'right-above'
    elif (vx1<=ux and vy0<=uy<vy1):
        relation = 'right-of'
    elif (vx1<ux and vy1<=uy) or (ux==vx1 and uy==vy1):
        relation = 'right-below'
    elif (vx0<ux<=vx1 and vy1<=uy):
        relation = 'below'
    elif (ux<=vx0 and vy1<uy) or (ux==vx0 and uy==vy1):
        relation = 'left-below'
    elif(ux<=vx0 and vy0<uy<=vy1):
        relation = "left-of"
    elif(vx0<ux<vx1 and vy0<uy<vy1):
        relation = "inside"

    return relation

def get_vocab():
    room_label = [(0, 'LivingRoom', 1, "PublicArea"),
              (1, 'MasterRoom', 0, "Bedroom"),
              (2, 'Kitchen', 1, "FunctionArea"),
              (3, 'Bathroom', 0, "FunctionArea"),
              (4, 'DiningRoom', 1, "FunctionArea"),
              (5, 'ChildRoom', 0, "Bedroom"),
              (6, 'StudyRoom', 0, "Bedroom"),
              (7, 'SecondRoom', 0, "Bedroom"),
              (8, 'GuestRoom', 0, "Bedroom"),
              (9, 'Balcony', 1, "PublicArea"),
              (10, 'Entrance', 1, "PublicArea"),
              (11, 'Storage', 0, "PublicArea"),
              (12, 'Wall-in', 0, "PublicArea"),
              (13, 'External', 0, "External"),
              (14, 'ExteriorWall', 0, "ExteriorWall")
    ]
    
    predicates = [
        'left-above',
        'left-below',
        'left-of',
        'above',
        'inside',
        'surrounding',
        'below',
        'right-of',
        'right-above',
        'right-below'
    ]

    door_pos = [
        'nan',
        'bottom',
        'bottom-right','right-bottom',
        'right',
        'right-top','top-right',
        'top',
        'top-left','left-top',
        'left',
        'left-bottom','bottom-left'
    ]

    vocab = {
        'object_name_to_idx':{},
        'object_to_idx':{},
        'object_idx_to_name':[],
        'pred_idx_to_name':[],
        'pred_name_to_idx':{},
        'door_idx_to_name':[],
        'door_name_to_idx':{}
    }
    
    vocab['object_name_to_idx'] = { label:index for index,label,_,_ in room_label[:] }
    vocab['object_to_idx'] = {str(index):index for index,lable,_,_ in room_label}
    vocab['object_idx_to_name'] = [label for index,label,_,_ in room_label]
    vocab['pred_idx_to_name'] = [p for i,p in enumerate(predicates)]
    vocab['pred_name_to_idx'] = {p:i for i,p in enumerate(predicates)}
    vocab['door_idx_to_name'] = [p for i,p in enumerate(door_pos)]
    vocab['door_name_to_idx'] = {p:i for i,p in enumerate(door_pos)}

    return vocab

def int_tuple(s):
    return tuple(int(i) for i in s.split(','))


def float_tuple(s):
    return tuple(float(i) for i in s.split(','))


def str_tuple(s):
    return tuple(s.split(','))


def bool_flag(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)