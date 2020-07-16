import torch
import scipy.io as sio
import numpy as np
import cv2
import copy
from .utils import *


class FloorPlan():

    def __init__(self, data, train=False, rot=None):
        self.data = copy.deepcopy(data)
        self._get_rot()
        if rot is not None:
            if train:
                boxes = self.data.box[:, :4][:, [1, 0, 3, 2]]
                boxes = align_box(boxes, self.rot, rot)[:, [1, 0, 3, 2]]
                self.data.box[:, :4] = boxes
            points = self.data.boundary[:, :2][:, [1, 0]]
            points = align_points(points, self.rot, rot)[:, [1, 0]]
            self.data.boundary[:, :2] = points
            self._get_rot()

    def _get_rot(self):
        door_line = self.data.boundary[:2, :2]  # [:,[1,0]]
        c = door_line.mean(0) - np.array([127.5,127.5])
        theta = np.arctan2(c[1], c[0]) + np.pi  # [-pi,pi]
        self.rot = theta

    def get_input_boundary(self, tensor=True):
        external = self.data.boundary[:, :2]
        door = self.data.boundary[:2, :2]

        boundary = np.zeros((128, 128), dtype=float)
        inside = np.zeros((128, 128), dtype=float)
        front = np.zeros((128, 128), dtype=float)

        pts = np.concatenate([external, external[:1]]) // 2
        pts_door = door // 2

        cv2.fillPoly(inside, pts.reshape(1, -1, 2), 1.0)
        cv2.polylines(boundary, pts.reshape(1, -1, 2), True, 1.0, 3)
        cv2.polylines(boundary, pts_door.reshape(1, -1, 2), True, 0.5, 3)
        cv2.polylines(front, pts_door.reshape(1, -1, 2), True, 1.0, 3)

        input_image = np.stack([inside, boundary, front], -1)
        if tensor: input_image = torch.tensor(input_image).permute((2, 0, 1)).float()
        return input_image

    def get_inside_box(self, tensor=True):
        external = self.data.boundary[:, :2]

        X, Y = np.linspace(0, 1, 256), np.linspace(0, 1, 256)
        x0, x1 = np.min(external[:, 0]), np.max(external[:, 0])
        y0, y1 = np.min(external[:, 1]), np.max(external[:, 1])
        box = np.array([[X[x0], Y[y0], X[x1], Y[y1]]])
        if tensor: box = torch.tensor(box).float()
        return box

    def get_rooms(self, tensor=True):
        rooms = self.data.box[:, -1]
        if tensor: rooms = torch.tensor(rooms).long()
        return rooms

    def get_attributes(self, gsize=5, alevel=10, relative=True, tensor=True):
        boxes = self.data.box[:, :4][:, [1, 0, 3, 2]]
        external = self.data.boundary

        h, w = 256, 256
        if relative:
            external = np.asarray(external)
            x0, x1 = np.min(external[:, 0]), np.max(external[:, 0])
            y0, y1 = np.min(external[:, 1]), np.max(external[:, 1])
            h, w = y1 - y0, x1 - x0
            boxes = boxes - np.array([y0, x0, y0, x0], dtype=float)
        boxes /= np.array([h, w, h, w])
        boxes[:, 2:] -= boxes[:, :2]  # y1,x1->h,w
        boxes[:, :2] += boxes[:, 2:] / 2  # y0,x0->yc,xc
        
        l = len(boxes)
        gbins = np.linspace(0,1,gsize+1) # [1,gsize]
        gbins[0],gbins[-1]=-np.inf,np.inf
        abins = np.linspace(0,1,alevel+1) # [1,gsize]
        abins[0],abins[-1]=-np.inf,np.inf

        attributes = np.zeros((l,gsize*gsize+alevel))
        # pos: xc*gsize+yc*gsize*gsize
        attributes[range(l),(np.digitize(boxes[:,0],gbins)-1)*gsize+np.digitize(boxes[:,1],gbins)-1]=1
        # area:(w*h)
        attributes[range(l),gsize*gsize+np.digitize(boxes[:,2:].prod(1),abins)-1]=1
        if tensor: attributes = torch.tensor(attributes).float()
        return attributes

    def get_triples(self, random=False, tensor=True):
        boxes = self.data.box[:, :4][:, [1, 0, 3, 2]]

        triples = []
        # add edge relation
        for u, v, _ in self.data.edge:
            uy0, ux0, uy1, ux1 = boxes[u]
            vy0, vx0, vy1, vx1 = boxes[v]
            uc = (uy0 + uy1) / 2, (ux0 + ux1) / 2
            vc = (vy0 + vy1) / 2, (vx0 + vx1) / 2

            # surrounding/inside -> X four quadrants
            if ux0 < vx0 and ux1 > vx1 and uy0 < vy0 and uy1 > vy1:
                relation = 'surrounding'
            elif ux0 >= vx0 and ux1 <= vx1 and uy0 >= vy0 and uy1 <= vy1:
                relation = 'inside'
            else:
                relation = point_box_relation(uc, boxes[v])

            triples.append([u, vocab['pred_name_to_idx'][relation], v])

        triples = np.array(triples, dtype=int)
        if tensor: triples = torch.tensor(triples).long()
        return triples

    def vis_box(self):
        h, w = 128, 128
        image = np.full((h, w, 4), 0, dtype=np.uint8)

        boxes = self.data.box[:, :4] // 2
        objs = self.data.box[:, -1]

        for i, obj in enumerate(objs):
            if obj == 14: continue
            color = colormap_255[obj]
            box = boxes[i]
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (color[0], color[1], color[2], 255), 3)

        return image

    def get_test_data(self, tensor=True):
        boundary = self.get_input_boundary(tensor=tensor)
        inside_box = self.get_inside_box(tensor=tensor)
        rooms = self.get_rooms(tensor=tensor)
        attrs = self.get_attributes(tensor=tensor)
        triples = self.get_triples(random=False, tensor=tensor)
        return boundary, inside_box, rooms, attrs, triples

    def adapt_graph(self, fp_graph):
        fp = FloorPlan(fp_graph.data, train=True, rot=self.rot)
        g_external = fp.data.boundary[:, :2]
        gx0, gx1 = np.min(g_external[:, 0]), np.max(g_external[:, 0])
        gy0, gy1 = np.min(g_external[:, 1]), np.max(g_external[:, 1])
        gw, gh = gx1 - gx0, gy1 - gy0

        fp.data.boundary = self.data.boundary
        b_external = self.data.boundary[:, :2]
        bx0, bx1 = np.min(b_external[:, 0]), np.max(b_external[:, 0])
        by0, by1 = np.min(b_external[:, 1]), np.max(b_external[:, 1])
        bh, bw = by1 - by0, bx1 - bx0
        box_adapter = lambda box: (((box - np.array([gx0, gy0, gx0, gy0])) * np.array([bw, bh, bw, bh])) / np.array(
            [gw, gh, gw, gh]) + np.array([bx0, by0, bx0, by0])).astype(int)

        fp.data.box[:, :4] = np.apply_along_axis(box_adapter, 1, fp.data.box[:, :4])
        return fp

    def adjust_graph(self):
        external = self.data.boundary[:, :2]
        bx0, bx1 = np.min(external[:, 0]), np.max(external[:, 0])
        by0, by1 = np.min(external[:, 1]), np.max(external[:, 1])
        hw_b = np.array([by1 - by0, bx1 - bx0])
        step = hw_b / 10

        pts = np.concatenate([external, external[:1]])
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.fillPoly(mask, pts.reshape(1, -1, 2), 255)
        # plt.imshow(mask)
        # plt.show()
        mask = cv2.resize(mask[by0:by1 + 1, bx0:bx1 + 1], (10, 10))
        # plt.imshow(mask)
        # plt.show()
        mask[mask > 0] = 255

        outside_rooms = []
        for i in range(len(self.data.box)):
            box = self.data.box[i][:4][[1, 0, 3, 2]]
            center = (box[:2] + box[2:]) / 2
            center55 = ((center - np.array([by0, bx0])) * 10 / hw_b).astype(int)

            if not mask[center55[0], center55[1]]:
                outside_rooms.append([i, center55])

        candicate_coords55 = {}
        for i, coords55 in outside_rooms:
            row, col = coords55
            # left/right/up/down
            candicate_coords55[i] = np.array([
                next((col-c for c in range(col,-1,-1) if mask[row,c]==255),255),
                next((c-col for c in range(col+1,5) if mask[row,c]==255),255),
                next((row-r for r in range(row,-1,-1) if mask[r,col]==255),255),
                next((r-row for r in range(row+1,5) if mask[r,col]==255),255)])

        signs = np.array([
            [0, -1, 0, -1],
            [0, 1, 0, 1],
            [-1, 0, -1, 0],
            [1, 0, 1, 0]
        ])

        for i, coords55 in outside_rooms:
            deltas = candicate_coords55[i]
            idx = np.argmin(deltas)
            self.data.box[i, :4] += (signs[idx] * deltas[idx] * np.tile(step, 2)).astype(int)[[1, 0, 3, 2]]


if __name__ == "__main__":
    pass
