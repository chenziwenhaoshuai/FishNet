# from make_label import load_seg_data
import os
import json
import cv2
from pycocotools import mask as maskUtils

image_path = '../dataset_test/images/'
seg_label_path = '../dataset_test/seg/'
seg_json_save_path = '../dataset_test/annotations/seg_label.json'
point_json_save_path = '../dataset_test/annotations/point_label.json'
point_label_path = '../dataset_test/keypoint/'

image_list = os.listdir(image_path)
seg_label_list = os.listdir(image_path)
point_label_list = os.listdir(point_label_path)


def load_seg_data(path, image_name, width, height):
    with open(path+image_name+'.json','r') as file:
        load_dict = json.load(file)
        points = []
        for point in load_dict['shapes'][0]['points']: # 目前仅支持单目标检测,仅提取第一个值的多边形point
            # nolm_x = '%.5f' % (point[0]/width)
            # nolm_y = '%.5f' % (point[1]/height)
            points.append(point[0])
            points.append(point[1])
    # points = points+[0]*(MAX_VERTICES-len(points))
    return list(map(int,points))


def load_keypoint_data(path,image_name, width, height):
    with open(path+image_name+'.txt','r') as file:
        lines = file.readlines()
        keypoint = []
        for line in lines[1:]:
            line = line.replace('\n', '')
            line = line.split(' ')
            keypoint.append(float(line[0])*width)
            keypoint.append(float(line[1])*height)
            keypoint.append(2)
    return list(map(int,keypoint))


seg_dataset = {'categories': [], 'annotations': [], 'images': []}
seg_dataset['categories'].append({'id': 0, 'name': 'fish', 'supercategory': 'mark'})

point_dataset = {'categories': [], 'annotations': [], 'images': []}
point_dataset['categories'].append({'id': 0,
                                  'name': 'fish',
                                  'supercategory': 'mark',
                                  'keypoints':["head","headUp","headDown","tail","tailUp","tailDown"],
                                  'skeleton':[[1,2],[1,3],[1,4],[4,5],[4,6]]})


for i, image_name in enumerate(image_list):
    image = cv2.imread(image_path + image_name)
    width, height = image.shape[1], image.shape[0]
    seg_dataset['images'].append({'file_name': image_name,
                              'id': i,
                              'width': width,
                              'height': height})
    point_dataset['images'].append({'file_name': image_name,
                                  'id': i,
                                  'width': width,
                                  'height': height})

ann_id_cnt = 0
for k,label_name in enumerate(seg_label_list):
    image = cv2.imread(image_path+label_name.split('.')[0]+'.jpg')
    width,height = image.shape[1],image.shape[0]
    seg_point = load_seg_data(seg_label_path,label_name.split('.')[0],width,height)
    keypoint = load_keypoint_data(point_label_path,label_name.split('.')[0], width, height)
    seg_x = seg_point[0::2]
    seg_y = seg_point[1::2]
    x1 = min(seg_x)
    y1 = min(seg_y)
    w = max(seg_x)-x1
    h = max(seg_y)-y1
    rles = maskUtils.frPyObjects([seg_point], height, width)
    rle = maskUtils.merge(rles)
    area = float(maskUtils.area(rle))
    seg_dataset['annotations'].append({
        'area': area,
        'bbox': [x1, y1, w, h],
        'category_id': 0,
        'id': ann_id_cnt,
        'image_id': k,
        'iscrowd': 0,
        # mask, 矩形是从左上角点按顺时针的四个顶点
        'segmentation': [seg_point]
    })
    point_dataset['annotations'].append({
        'area': area,
        'bbox': [x1, y1, w, h],
        'category_id': 0,
        'id': ann_id_cnt,
        'image_id': k,
        'iscrowd': 0,
        # mask, 矩形是从左上角点按顺时针的四个顶点
        'segmentation': [seg_point],
        'num_keypoints':6,
        'keypoints':keypoint
    })
    ann_id_cnt+=1

with open(seg_json_save_path,'w') as f:
    json.dump(seg_dataset, f)
    print('seg json label success')
with open(point_json_save_path, 'w') as f:
    json.dump(point_dataset, f)
    print('point json label success')

