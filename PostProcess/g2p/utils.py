import numpy as np

# index,name,type(private/public),floorTexture
room_label = [(0, 'LivingRoom', 1, "PublicArea",[220, 213, 205]),
              (1, 'MasterRoom', 0, "Bedroom",[138, 113, 91]),
              (2, 'Kitchen', 1, "FunctionArea",[244, 245, 247]),
              (3, 'Bathroom', 0, "FunctionArea",[224, 225, 227]),
              (4, 'DiningRoom', 1, "FunctionArea",[200, 193, 185]),
              (5, 'ChildRoom', 0, "Bedroom",[198, 173, 151]),
              (6, 'StudyRoom', 0, "Bedroom",[178, 153, 131]),
              (7, 'SecondRoom', 0, "Bedroom",[158, 133, 111]),
              (8, 'GuestRoom', 0, "Bedroom",[189, 172, 146]),
              (9, 'Balcony', 1, "PublicArea",[244, 237, 224]),
              (10, 'Entrance', 1, "PublicArea",[238, 235, 230]),
              (11, 'Storage', 0, "PublicArea",[226, 220, 206]),
              (12, 'Wall-in', 0, "PublicArea",[226, 220, 206]),
              (13, 'External', 0, "External",[255, 255, 255]),
              (14, 'ExteriorWall', 0, "ExteriorWall",[0, 0, 0]),
              (15, 'FrontDoor', 0, "FrontDoor",[255,255,0]),
              (16, 'InteriorWall', 0, "InteriorWall",[128,128,128]),
              (17, 'InteriorDoor', 0, "InteriorDoor",[255,255,255])]

# color palette for nyu40 labels
def create_color_palette():
    return [
       (0, 0, 0),
       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair
       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf
       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),  
       (247, 182, 210),		# desk
       (66, 188, 102), 
       (219, 219, 141),		# curtain
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14), 		# refrigerator
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),		# shower curtain
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209), 
       (227, 119, 194),		# bathtub
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)
    ]
color_palette = create_color_palette()[1:]

semantics_cmap = {
    'living room': '#e6194b',#[230,25,75]
    'kitchen': '#3cb44b',#[60,180,75]
    'bedroom': '#ffe119',#[255,225,25]
    'bathroom': '#0082c8',#[0,130,200]
    'balcony': '#f58230',#[245,130,48]
    'corridor': '#911eb4',#[145,30,180]
    'dining room': '#46f0f0',#[70,240,240]
    'study': '#f032e6',#[240,50,230]
    'studio': '#d2f53c',#[210,245,60]
    'store room': '#fabebe',#[250,190,190]
    'garden': '#008080',#[0,128,128]
    'laundry room': '#e6beff',#[230,190,255]
    'office': '#aa6e28',#[170,110,40]
    'basement': '#fffac8',#[255,250,200]
    'garage': '#800000',#[128,0,0]
    'undefined': '#aaffc3',#[170,255,195]
    'door': '#808000',#[128,128,0]
    'window': '#ffd7b4',#[255,215,180]
    'outwall': '#000000',#[0,0,0]
}

colormap_255 = [
    [230,  25,  75],#LivingRoom
    [ 60, 180,  75],#MasterRoom
    [170, 255, 195],#Kitchen
    [  0, 130, 200],#Bathroom
    [245, 130,  48],#DiningRoom
    [145,  30, 180],#ChildRoom
    [ 70, 240, 240],#StudyRoom
    [240,  50, 230],#SecondRoom
    [210, 245,  60],#GuestRoom
    [250, 190, 190],#Balcony
    [  0, 128, 128],#Entrance
    [230, 190, 255],#Storage
    [170, 110,  40],#Wall-in
    [255, 255, 255],#External
    [128,   0,   0],#ExteriorWall
    [255, 225,  25],#FrontDoor
    [128, 128, 128],#InteriorWall
    [255, 255, 255],#InteriorDoor
    #[255, 215, 180],
    [  0,   0, 128],
    [128, 128,   0],
    [255, 255, 255],
    [  0,   0,   0]
]

cmaps = {
    'nyu40': color_palette,
    'semantics': semantics_cmap,
    '255': colormap_255
}

category = [category for category in room_label if category[1] not in set(['External',
                                                                           'ExteriorWall', 'FrontDoor', 'InteriorWall', 'InteriorDoor'])]

num_category = len(category)

pixel2length = 18/256

def label2name(label=0):
    if label < 0 or label > 17:
        raise Exception("Invalid label!", label)
    else:
        return room_label[label][1]


def label2index(label=0):
    if label < 0 or label > 17:
        raise Exception("Invalid label!", label)
    else:
        return label


def index2label(index=0):
    if index < 0 or index > 17:
        raise Exception("Invalid index!", index)
    else:
        return index


def compute_centroid(mask):
    sum_h = 0
    sum_w = 0
    count = 0
    shape_array = mask.shape
    for h in range(shape_array[0]):
        for w in range(shape_array[1]):
            if mask[h, w] != 0:
                sum_h += h
                sum_w += w
                count += 1
    return (sum_h//count, sum_w//count)


def log(file, msg='', is_print=True):
    if is_print:
        print(msg)
    file.write(msg + '\n')
    file.flush()


def collide2d(bbox1, bbox2, th=0):
    return not(
        (bbox1[0]-th > bbox2[2]) or
        (bbox1[2]+th < bbox2[0]) or
        (bbox1[1]-th > bbox2[3]) or
        (bbox1[3]+th < bbox2[1])
    )
#
# def rot90_2D(pts,k=1,cnt=np.array([127.5,127.5])):
#     ang = k*np.pi/2
#     R = np.array([[np.cos(ang),np.sin(ang)],[-np.sin(ang),np.cos(ang)]])
#     return np.dot(pts-cnt,R)+cnt
# def fliplr_2D(pts,size=255):
#     return np.stack([pts[:,0],size-pts[:,1]],1)
#
# def align_image(image,rot_old,rot_new=0):
#     k = np.ceil((rot_old-rot_new+2*np.pi)%(2*np.pi)/(np.pi/4))//2
#     return np.rot90(image,k)
#
# def align_box(box,rot_old,rot_new=0):
#     k = np.ceil((rot_old-rot_new+2*np.pi)%(2*np.pi)/(np.pi/4))//2
#     box = rot90_2D(box.reshape(-1,2),k).reshape(-1,4)
#     return np.concatenate([np.minimum(box[:,:2],box[:,2:]),np.maximum(box[:,:2],box[:,2:])],-1)#.round().astype(int)
#
# def fliplr_box(box,size=255):
#     box=fliplr_2D(box.reshape(-1,2),size=size).reshape(-1,4)
#     return np.concatenate([np.minimum(box[:,:2],box[:,2:]),np.maximum(box[:,:2],box[:,2:])],-1)#.round().astype(int)


def rot90_2D(pts, k=1, cnt=np.array([127.5, 127.5])):
    ang = k * np.pi / 2
    R = np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])
    return np.dot(pts - cnt, R) + cnt


def fliplr_2D(pts, size=255):
    return np.stack([pts[:, 0], size - pts[:, 1]], 1)


def align_image(image, rot_old, rot_new=0):
    k = np.ceil((rot_old - rot_new + 2 * np.pi) % (2 * np.pi) / (np.pi / 4)) // 2
    return np.rot90(image, k)


def align_box(box, rot_old, rot_new=0):
    k = np.ceil((rot_old - rot_new + 2 * np.pi) % (2 * np.pi) / (np.pi / 4)) // 2
    box = rot90_2D(box.reshape(-1, 2), k).reshape(-1, 4)
    return np.concatenate([np.minimum(box[:, :2], box[:, 2:]), np.maximum(box[:, :2], box[:, 2:]) + 1],
                          -1).round().astype(int)


def align_points(points, rot_old, rot_new=0):
    k = np.ceil((rot_old - rot_new + 2 * np.pi) % (2 * np.pi) / (np.pi / 4)) // 2
    points = rot90_2D(points, k)
    return points.round().astype(int)

def graph2labels(graph):
    edges = graph.edges
    return sorted([
        tuple(sorted((room_label[graph.nodes[u]['category']][1],
        room_label[graph.nodes[v]['category']][1])))
        for u,v in edges
    ])

def graph2labels_withtype(graph):
    edges = graph.edges(data=True)
    return sorted([
        ('acc' if d['type'] else 'adj',
        *sorted(
            (room_label[graph.nodes[u]['category']][1],
            room_label[graph.nodes[v]['category']][1]))
        ) 
        for u,v,d in edges
    ])

def graph2functions(graph):
    edges = graph.edges
    return sorted([
        tuple(sorted((graph.nodes[u]['function'],
        graph.nodes[v]['function'])))
        for u,v in edges
    ])

def graph2functions_withtype(graph):
    edges = graph.edges(data=True)
    return sorted([
        ('acc' if d['type'] else 'adj',
        *sorted(
            (graph.nodes[u]['function'],
            graph.nodes[v]['function']))
        )
        for u,v,d in edges
    ])

def counter2labels(counter):
    return sorted({
        room_label[key][1]:value 
        for key,value in counter.items()
    }.items())

def counter2functions(counter):
    counter_new = {
        room_label[key][1]:value 
        for key,value in counter.items()
    }
    counter_new['Bedroom']=0
    for key in counter:
        if room_label[key][3]=='Bedroom':
            counter_new['Bedroom']+=counter_new.pop(room_label[key][1])
    return sorted(counter_new.items())

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
              (14, 'Internal', 0, "Internal")]
    
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

vocab = get_vocab()
