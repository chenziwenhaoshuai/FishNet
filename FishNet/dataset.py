import torchvision
from torch.utils.data import Dataset
import torch
import numpy as np
import math
from utils import make_image_data, poly2xy
from PIL import Image,ImageDraw

image_path = './dataset/images/'
cache_label = './dataset/output/cache_label.txt'

# ANCHORS_GROUP = {
#     13: [[430, 143], [420, 150], [400, 130]],
#     26: [[175, 222], [112, 235], [175, 140]],
#     52: [[81, 118], [53, 142], [44, 28]]
# }
'''kmeans anchor
thr=0.25: 0.9992 best possible recall, 8.80 anchors past thr
autoanchor: n=9, img_size=416, metric_all=0.607/0.885-mean/best, past_thr=0.617-mean: 129,56,  175,82,  231,99,  261,127,  148,245,  340,119,  333,151,  363,173,  353,216
'''
ANCHORS_GROUP = {
    13: [[333,151], [363,173], [353,216]],
    26: [[261,127], [148,245], [340,119]],
    52: [[129,56], [175,82], [231,99]]
}

ANCHORS_GROUP_AREA = {
    13: [x * y for x, y in ANCHORS_GROUP[13]],
    26: [x * y for x, y in ANCHORS_GROUP[26]],
    52: [x * y for x, y in ANCHORS_GROUP[52]],
}


CLASS_NUM = 1
input_image_width,input_image_height = 416,416

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


def one_hot(cls_num, v):
    b = np.zeros(cls_num)
    b[v] = 1.
    return b


class Mydataset(Dataset):
    def __init__(self):
        with open(cache_label) as f:
            self.all_label = f.readlines()

    def __len__(self):
        return len(self.all_label)

    def __getitem__(self,index):
        object_labels = {}
        line = self.all_label[index]
        line = line.replace('\n','')
        line = line.split(' ')
        image_name = line[0]
        object_label = line[1:6] # cls xc yc w h
        image, w, h = make_image_data(image_path+image_name) # 将图像填充黑色成正方形
        # image.show()

        image = image.resize((input_image_width,input_image_height))
        resize_ratio = input_image_height/max(w,h)
        draw = ImageDraw.Draw(image)
        image = transforms(image) # should convert to tensor when run the program
        # for label in object_label: # 后期为多目标标签使用
        #     pass
        """box"""
        box = []
        box.append(float(object_label[0]))
        box.append(float(object_label[1]) * w * resize_ratio) # xc
        box.append(float(object_label[2]) * h * resize_ratio +(input_image_height-h*resize_ratio)*0.5) # yc
        box.append(float(object_label[3]) * w*resize_ratio) # w
        box.append(float(object_label[4]) * h*resize_ratio) # h
        """keypoint"""
        points = []
        points.extend(float(i) for i in line[6:18])
        point = []
        for i in range(0,len(points),2):
            px = points[i] * w * resize_ratio
            py = points[i+1] * h * resize_ratio +(input_image_height-h*resize_ratio)*0.5
            point.append(px)
            point.append(py)
        """polygon"""
        polygons = []
        polygons.extend(float(i) for i in line[18:])
        polygon = []
        for i in range(0,len(polygons),3):
            dist = polygons[i] * resize_ratio
            angle_offsst = polygons[i+1]
            conf = polygons[i+2]
            polygon.append(dist)
            polygon.append(angle_offsst)
            polygon.append(conf)

        for feature_size, anchors in ANCHORS_GROUP.items():
            object_labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + CLASS_NUM+len(point)+len(polygon)))

            # for box in boxes: # 后期为多目标标签使用
            cls, cx, cy, w, h = box
            '''
            # --------------------take one box for object detection-----------
            draw.rectangle((cx-w*0.5,cy-h*0.5,cx+w*0.5,cy+h*0.5),outline='red',width=1)
            # ---------------------------------end----------------------------
            # --------------------take six points for keypoint detection-----------
            draw.ellipse((point[0] - 3, point[1] - 3,
                          point[0] + 3, point[1] + 3), fill=(255, 0, 0))
            draw.ellipse((point[2] - 3, point[3] - 3,
                          point[2] + 3, point[3] + 3), fill=(255, 255, 0))
            draw.ellipse((point[4] - 3, point[5] - 3,
                          point[4] + 3, point[5] + 3), fill=(255, 0, 255))
            draw.ellipse((point[6] - 3, point[7] - 3,
                          point[6] + 3, point[7] + 3), fill=(0, 255, 0))
            draw.ellipse((point[8] - 3, point[9] - 3,
                          point[8] + 3, point[9] + 3), fill=(0, 0, 255))
            draw.ellipse((point[10] - 3, point[11] - 3,
                          point[10] + 3, point[11] + 3), fill=(255, 255, 255))
            # ------------------------------end-------------------------------------
            # --------------------show polygon in instance segmentation-----------
            polygon_xy_points = poly2xy(cx,cy,polygon,step=15)
            draw.polygon(polygon_xy_points, fill=(255, 0, 0, 127), outline=(255, 255, 255, 255))
            for i in range(0,len(polygon_xy_points),2):#逐点显示向量
                x = polygon_xy_points[i]
                y = polygon_xy_points[i+1]
                draw.ellipse((x-5,y-5,x+5,y+5),fill=(255,0,0))
                draw.line([cx,cy,x,y],fill=(114,0,0))
                image.show()
            # --------------------end---------------------------------------------
            image.show()
            '''
            cx_offset, cx_index = math.modf(cx * feature_size / input_image_width)
            cy_offset, cy_index = math.modf(cy * feature_size / input_image_height)
            diagonal = math.sqrt((w**2)+(h**2)) # the diagonal of one bbox
            # point = point/
            for i in range(0,len(point),2): # 将关键点坐标xy偏移除以对角线长度来进行数据归一化
                point[i] = (cx-point[i])/diagonal
                point[i+1] = (cy-point[i+1])/diagonal
            # point = torch.tensor(point) # 测试是否使用sigmoid处理点的偏移数据会更好
            # point = torch.sigmoid(point).tolist()
            for i in range(0,len(polygon),3):
                polygon[i] = polygon[i]/diagonal



            for i, anchor in enumerate(anchors):

                anchor_area = ANCHORS_GROUP_AREA[feature_size][i]
                p_w, p_h = w / (anchor[0]), h / (anchor[1])
                p_area = w * h
                iou = min(p_area, anchor_area) / max(p_area, anchor_area)
                # index+=1
                # print(feature_size, cx_index, cy_index, i)
                # print(box)
                if object_labels[feature_size][int(cy_index), int(cx_index), i][0]<iou:
                    object_labels[feature_size][int(cy_index), int(cx_index), i] = \
                        np.array([iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h), *one_hot(CLASS_NUM, int(cls)), *(i for i in point) , *(i for i in polygon)])

        return image , object_labels#[13], object_label[26], object_labels[52]


if __name__ == '__main__':
    Mydataset = Mydataset().__getitem__(0)
    print(Mydataset[0].shape)
    print(Mydataset[1][13].shape)
    print(Mydataset[1][26].shape)
    print(Mydataset[1][52].shape)