import numpy as np
import copy

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

class DirectedLine():
    def __init__(self,p1,p2):
        '''search direction : 0(horizontal) / 1(vertical)'''
        if np.abs(p1[0]-p2[0])<1e-6:
            self.dir = 1
            self.level = p1[0]
            self.minLevel = min(p1[1],p2[1])
            self.maxLevel = max(p1[1],p2[1])
        else:
            self.dir = 0
            self.level = p1[1]
            self.minLevel = min(p1[0],p2[0])
            self.maxLevel = max(p1[0],p2[0])
    
    def __repr__(self):
        if self.dir==0: return f'({self.level},[{self.minLevel},{self.maxLevel}])'
        else: return f'([{self.minLevel},{self.maxLevel}],{self.level})'
    
    @property
    def length(self):return self.maxLevel-self.minLevel
    
    def is_contact(self,line): 
        minl = min(self.minLevel,line.minLevel)
        maxl = max(self.maxLevel,line.maxLevel)
        length = maxl-minl
        return (
            self.dir==line.dir and 
            #self.level!=line.level and 
            abs(self.level-line.level)<6 and
            length < self.length+line.length
        )
    
    @staticmethod
    def lines_from_boundary(boundary):
        if len(boundary)==0:return []
        pts = boundary.tolist()+[boundary[0].tolist()]
        lines = [ DirectedLine(pts[i],pts[i+1]) for i in range(len(pts)-1)]
        return lines

class DirectedWall():
    def __init__(self):
        '''orientation : 0(right) / 1(down) / 2(left) / 3(up)'''
        self.dir = 0
        self.rect = np.array([0,0,0,0])
    
    @property
    def width(self):return self.rect[2]
    
    @property
    def height(self):return self.rect[3]
    
    @property
    def center(self):return self.rect[:2]+self.rect[2:]/2
    
    def setX(self,x):self.rect[0]=x
    def setY(self,y):self.rect[1]=y
    def setWidth(self,w):self.rect[2]=w
    def setHeight(self,h):self.rect[2]=h
    def setLeft(self,x):
        self.rect[2]=(self.rect[0]-x)+self.rect[2]
        self.rect[0]=x
    def setTop(self,y):
        self.rect[3]=(self.rect[1]-y)+self.rect[3]
        self.rect[1]=y
    
    def to_line(self):
        if self.dir in [0,2]:
            return DirectedLine([self.rect[0],self.rect[1]],[self.rect[0],self.rect[1]+self.rect[3]])
        else:
            return DirectedLine([self.rect[0],self.rect[1]],[self.rect[0]+self.rect[2],self.rect[1]])
    
    def __repr__(self):
        pos = ['right','down','left','up','None'][self.dir]
        return f'({pos},{self.rect})'

class Entry():
    def __init__(self):
        '''door type : 0(door) / 1(open wall)'''
        self.type = -1
        self.entry = None
        
    def __repr__(self):
        if self.type==0: return f'(door,{self.entry})'
        else: return f'(open wall,{self.entry})'   
        
class Room():
    def __init__(self):
        self.box = None
        self.category = None
        self.boundary = None
        
        self.map = None
        self.entry = None
        self.windows = []
    
    @property
    def label(self):return room_label[self.category][1]
    
    @property
    def type(self): return room_label[self.category][3]
    
    @property
    def center(self): return self.box.reshape(-1,2).mean(0)

    @staticmethod
    def rooms_from_data(data):
        rooms = []
        for i in range(len(data.rType)):
            room = Room()
            room.box = data.newBox[i]
            room.category = data.rType[i]
            room.boundary = data.rBoundary[i]
            room.lines = DirectedLine.lines_from_boundary(room.boundary)
            rooms.append(room)
        return rooms

    @staticmethod
    def from_node_box(node,box):
        x0,y0,x1,y1 = box
        room = Room()
        room.box = box
        room.category = node[-1]
        room.boundary = np.array([
                [x0,y0],
                [x0,y1],
                [x1,y1],
                [x1,y0]
            ])
        room.lines = DirectedLine.lines_from_boundary(room.boundary)
        # room.boundary = [
        #     DirectedLine((x0,y0),(x1,y0)), # top
        #     DirectedLine((x1,y0),(x1,y1)), # right
        #     DirectedLine((x0,y1),(x1,y1)), # bottom
        #     DirectedLine((x0,y0),(x0,y1)), # left
        #     ]
        return room
        
    @staticmethod
    def from_boundary(boundary):
        # pts = boundary[:,:2].tolist()
        # pts = pts+[pts[0]]
        room = Room()
        
        room.category = 0
        room.box = np.array([np.min(boundary[:,0]),np.min(boundary[:,1]),np.max(boundary[:,0]),np.max(boundary[:,1])])
        room.boundary = boundary[:,:2]
        room.lines = DirectedLine.lines_from_boundary(room.boundary)
        # room.boundary = [
        #     DirectedLine(pts[i],pts[i+1])
        #     for i in range(len(pts)-1)
        #     ]
        return room
        
    def __repr__(self):
        return f'({self.label},{self.type},{self.box},{self.entry},{self.windows})'
        
def find_contact_walls(room1,room2,reverse=False):
    contactWalls = []
    lines1 = copy.deepcopy(room1.lines) #DirectedLine.from_boundary(room1.boundary)#room1.lines
    center1 = room1.center
    lines2 = copy.deepcopy(room2.lines) #DirectedLine.from_boundary(room2.boundary)# room2.lines
    center2 = room2.center
    temp = []
    
    for i in range(len(lines1)):
        line1 = lines1[i]
        
        for j in range(len(lines2)):
            line2 = lines2[j]
            
            if line1.is_contact(line2):
                contactWall = DirectedWall()
                if line1.dir==0:
                    minh = line1.level if not reverse else line2.level #min(line1.level,line2.level)
                    maxh = line1.level if not reverse else line2.level #max(line1.level,line2.level)
                    minw = max(line1.minLevel,line2.minLevel)
                    maxw = min(line1.maxLevel,line2.maxLevel)
                    # @todo:boudanry not work!
                    if center1[1] > line1.level: contactWall.dir=1
                    else: contactWall.dir=3
                    contactWall.rect = np.array([minw,minh,maxw-minw,maxh-minh])
                else:
                    minw = line1.level if not reverse else line2.level#min(line1.level,line2.level)
                    maxw = line1.level if not reverse else line2.level#max(line1.level,line2.level)
                    minh = max(line1.minLevel,line2.minLevel)
                    maxh = min(line1.maxLevel,line2.maxLevel)
                    if center1[0] > line1.level: contactWall.dir=0
                    else: contactWall.dir=2
                    contactWall.rect = np.array([minw,minh,maxw-minw,maxh-minh])
                contactWalls.append(contactWall)
    return contactWalls

def find_longest_wall(contactWalls,dtype=1):
    contactLength = 0
    openWall = None
    for i in range(len(contactWalls)):
        maxLength = max(contactWalls[i].width,contactWalls[i].height)
        if maxLength>contactLength:
            contactLength = maxLength
            openWall = contactWalls[i]
    if contactLength!=0:
        # @todo: adjust door
        openWall = adjust_door(openWall,dtype)
        entry = Entry()
        entry.type = dtype
        entry.entry = openWall
        assert entry.entry.dir!=-1, "find longest wall with dir -1!"
        return entry
    return None

def find_closest_wall(candidateDoors,frontDoorCenter,dtype=1,boundary_lines=[]):
    dis = 1e8
    door = None
    for i in range(len(candidateDoors)):
        maxLength = max(candidateDoors[i].width,candidateDoors[i].height)
        if maxLength<12:continue

        valid = True
        line = candidateDoors[i].to_line()
        for b_line in boundary_lines:
            if line.is_contact(b_line):
                valid = False
                break
        if not valid: continue

        center = candidateDoors[i].center
        candidateDis = np.sum(np.power((center-frontDoorCenter),2))
        if dis>candidateDis:
            dis = candidateDis
            door = candidateDoors[i]
    if door is not None:
        door = adjust_door(door,dtype)
        entry = Entry()
        entry.type = dtype
        entry.entry = door
        assert entry.entry.dir!=-1, "find closest wall with dir -1!"
        return entry
    return None

def adjust_door(door,dtype=1):
    if door.dir in [1,3]:
        if dtype==1: 
            door.rect[0] = door.rect[0]+door.rect[2]/8
            door.rect[2] = door.rect[2]*3/4
        else:
            # door.rect[0] = door.center[0]-6
            door.rect[2] = min(2*6,door.rect[2])
    else:
        if dtype==1:
            door.rect[1] = door.rect[1]+door.rect[3]/8
            door.rect[3] = door.rect[3]*3/4
        else:
            # door.rect[1] = door.center[1]-6
            door.rect[3] = min(2*6,door.rect[3])
    return door

def add_interior_door(rooms,living_idx,house):
    frontDoorCenter = house.boundary[:2].mean(0)
    for i in range(len(rooms)):
        if i==living_idx:continue
        # 1. Balcony: find the longest door
        # 2. Public Area: find the longest door
        # 3. Others:
        #    3.1 Contact with living room: find the cloest door with the front door
        #    3.2 Others: find the longest wall
        if rooms[i].label == 'Balcony':
            contactWalls = []
            for j in range(len(rooms)):
                if i!=j:
                    contactWalls.extend(find_contact_walls(rooms[i],rooms[j]))
            
            rooms[i].entry = find_longest_wall(contactWalls,dtype=1)
        else:
            contactWalls = find_contact_walls(rooms[i],rooms[living_idx])
            if len(contactWalls)>0:
                if rooms[i].type == 'PublicArea':
                    rooms[i].entry = find_longest_wall(contactWalls,dtype=1)
                else:
                    candidateDoors = [ 
                        wall for wall in contactWalls 
                        if (wall.width>wall.height and wall.width>2*6) or 
                        (wall.height>wall.width and wall.height>2*6)
                    ]
                    if len(candidateDoors)==0:
                        rooms[i].entry = find_longest_wall(contactWalls,dtype=0)
                    else:
                        rooms[i].entry = find_closest_wall(contactWalls,frontDoorCenter,dtype=0,boundary_lines=house.lines)
            else:
                contactWalls = []
                for j in range(len(rooms)):
                    if i!=j:
                        contactWalls.extend(find_contact_walls(rooms[i],rooms[j]))
                if len(contactWalls)>0:
                    rooms[i].entry = find_closest_wall(contactWalls,frontDoorCenter,dtype=1,boundary_lines=house.lines)
            
    return rooms

def find_windows(contactWalls,wtypes=['mid'],keep_longest=False):
    windows = []
    contactLength = 1e8
    for i in range(len(contactWalls)):
        contactWall = contactWalls[i]
        maxLength = max(contactWall.width,contactWall.height)
        if ('large' in wtypes and maxLength>3*12):
            contactWalls[i] = adjust_window(contactWalls[i],'large')
            windows.append(contactWall)
        elif 'mid' in wtypes and maxLength>3*9:
            contactWalls[i] = adjust_window(contactWalls[i],'mid')
            windows.append(contactWall)
        elif 'small' in wtypes and maxLength>2*5:
            contactWalls[i] = adjust_window(contactWalls[i],'small')
            windows.append(contactWall)
        elif 'balcony' in wtypes and maxLength>2*5:
            contactWalls[i] = adjust_window(contactWalls[i],'balcony')
            windows.append(contactWall)
    return windows

def find_window_by_length(contactWalls,wtypes=['mid'],ltype='max'):
    window = None
    contactLength = 0 if ltype=='max' else 1e8
    ufunc = np.greater if ltype=='max' else np.less
    for i in range(len(contactWalls)):
        contactWall = contactWalls[i]
        maxLength = max(contactWall.width,contactWall.height)
        if ufunc(maxLength,contactLength):
            if 'large' in wtypes and maxLength>3*12:
                contactWalls[i] = adjust_window(contactWalls[i],'large')
                window = contactWalls[i]
                contactLength = maxLength
            elif 'mid' in wtypes and maxLength>3*9:
                contactWalls[i] = adjust_window(contactWalls[i],'mid')
                window = contactWalls[i]
                contactLength = maxLength
            elif 'small' in wtypes and maxLength>2*5:
                contactWalls[i] = adjust_window(contactWalls[i],'small')
                window = contactWalls[i]
                contactLength = maxLength
    return [window] if window is not None else []

def adjust_window(window,wtype='mid'):
    hl = {'small':5,'mid':9,'large':12}
    if window.dir in [1,3]:
        if wtype=='balcony':
            window.rect[0] = window.rect[0]+window.rect[2]/10
            window.rect[2] = window.rect[2]*4/5
        else:
            length = hl[wtype]
            window.rect[0] = window.center[0]-length
            window.rect[2] = 2*length
    else:
        if wtype=='balcony':
            window.rect[1] = window.rect[1]+window.rect[3]/10
            window.rect[3] = window.rect[3]*4/5
        else:
            length = hl[wtype]
            window.rect[1] = window.center[1]-length
            window.rect[3] = 2*length
    return window

def add_window(rooms,house):
    for i in range(len(rooms)):
        # 1. Balcony: small(half=5)
        # 2. Living Room: mid(half=9),large(half=12)
        # 3. Bathroom: shortest wall, small
        # 4. Others: longest wall, mid
        contactWalls = find_contact_walls(rooms[i],house,reverse=True)
        if rooms[i].label == 'Balcony':
            rooms[i].windows.extend(find_windows(contactWalls,['balcony']))
        elif rooms[i].label == 'LivingRoom':
            rooms[i].windows.extend(find_windows(contactWalls,['mid','large']))
        elif rooms[i].label == 'Bathroom':
            rooms[i].windows.extend(find_window_by_length(contactWalls,['small'],'min'))
        else:
            rooms[i].windows.extend(find_window_by_length(contactWalls,['mid'],'max'))
    return rooms

def rooms_to_numpy(rooms):
    doors = []
    windows = []
    for i in range(len(rooms)):
        if rooms[i].entry is not None:
            door = rooms[i].entry.entry
            doors.append([i,door.rect[0],door.rect[1],door.rect[2],door.rect[3],door.dir])
        if len(rooms[i].windows) > 0:
            ws = [[i,w.rect[0],w.rect[1],w.rect[2],w.rect[3],w.dir] for w in rooms[i].windows]
            windows.extend(ws)
    return np.array(doors),np.array(windows)

def add_door_window(data):
    boundary = data.boundary
    living_idx = np.where(data.rType==0)[0][0]
    rooms = Room.rooms_from_data(data)
    house = Room.from_boundary(boundary[:,:2])
    house.lines = house.lines[1:]    

    rooms = add_interior_door(rooms,living_idx,house)
    rooms = add_window(rooms,house)
    
    return rooms_to_numpy(rooms)

def add_dw_fp(data):
    doors,windows = add_door_window(data)
    data.doors = doors
    data.windows = windows
    return data
