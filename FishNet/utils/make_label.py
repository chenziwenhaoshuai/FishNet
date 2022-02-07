"""记录一下毕业论文的第一行代码，开动于2021年10月28日17点18分"""
"""本脚本作用为合并目标检测、关键点、分割任务的标签"""
"""格式为<cls，bbox[xc,yc,w,h]，keypoint[x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6],polygon[dist,angle1_offset,conf,dist,angle2_offset,conf,......]>"""
"""cls:1  bbox:4   keypoint:12  polygon:72"""


import cv2
import os
import json
import numpy as np

dir_path = "../dataset/"
save_path = "../dataset/output/"
image_dir = "../dataset/images/"
detect_label_dir = "../dataset/detect/"
keypoint_label_dir = "../dataset/keypoint/"
seg_label_dir = "../dataset/seg/"
MAX_VERTICES = 1000 # that allows the labels to have 1000 vertices per polygon at max. They are reduced for training
ANGLE_STEP = 15 # that means Poly-YOLO will detect 360/15=24 vertices per polygon at max
NUM_ANGLES3 = int(360 // ANGLE_STEP * 3)
NUM_ANGLES = int(360 // ANGLE_STEP)


def convert_polygon(width,height,object_box,seg_point):
    box_data = np.zeros(NUM_ANGLES3)
    box_data = list(box_data)
    wh = [width,height]
    a0 = [0]
    box = xcyc2xy(width,height,object_box)
    box.extend(a0)
    seg_point = xcyc2xy(width,height,seg_point)#[:48]
    box.extend(seg_point)

    # for b in range(0, len(box)): #最长支持1000个点的边缘，xy共2000个，加上xyxy cls所以长度为2005
    # boxes_xy = [a*b for a,b in zip(box[0:2],wh[0:2])] ##目标中心点xy
    boxes_xy = box[0:2]
    for i in range(5, MAX_VERTICES * 2, 2): # 每个两个点取一个点，取出来所有的顶点数，i为每一个点xy的索引
        if box[i] == 0 and box[i + 1] == 0: # 如果取到0就退出循环
            break
        # box[b, i:i+2] = box[b, i:i+2] - [crop_coords[2], crop_coords[0]] + [new_img_coords[2], new_img_coords[0]]
        # 坐标映射到裁切的图像上
        # if flip:
        #     box[b, i] = (w - 1) - box[b, i]
        dist_x = boxes_xy[0] - box[i]
        dist_y = boxes_xy[1] - box[i + 1]
        dist = np.sqrt(np.power(dist_x, 2) + np.power(dist_y, 2))
        # 计算中心点到当前顶点的距离
        if (dist < 1):
            dist = 1
        # 如果距离小于1则设定为1
        angle = np.degrees(np.arctan2(dist_y, dist_x))
        # 计算以中心点为圆心，y轴与圆心到到当前顶点射线的度数，单位为度
        if (angle < 0):
            angle += 360
        # 度数小于0度则加360度

        iangle = int(angle) // ANGLE_STEP
        # 计算应该属于第几个边的顶点

        if iangle>=NUM_ANGLES:  # 如果属于的边大于最大的射线数了，就设置为最后一个边
            iangle = NUM_ANGLES-1

        if dist > box_data[iangle * 3]: # check for vertex existence. only the most distant is taken
            box_data[iangle * 3] = dist  # 每隔3个为一组，第一个为距离，第二个为角度偏移值的百分比，第三个为1,
            box_data[iangle * 3 + 1] = (angle - (iangle * int(ANGLE_STEP))) / ANGLE_STEP # 与标准的偏移
            box_data[iangle * 3 + 2] = 1 # 置信度
    return box_data         # 得到数据三个一组，第一个为未归一化的顶点到中心点的距离，第二个为每个射线到最近分割角度的偏移量，第三个为标签用的置信度


def xcyc2xy(w,h,point):
    new_point = []
    for i in range(0,len(point),2):
        new_point.append(w * point[i])
        new_point.append(h * point[i+1])
    return new_point


def load_image_data(path):
    image = cv2.imread(path)
    image_width = image.shape[1]
    image_height = image.shape[0]

    return image_width,image_height


def load_detect_data(label_dir,file_name):
    with open(label_dir+file_name+'.txt','r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.replace('\n','')
            line = line.split(' ')
            cls = line[0]
            object_box = line[1:]
            object_box = list(map(float,object_box))
    return cls,object_box


def load_keypoint_data(path,image_name):
    with open(path+image_name+'.txt','r') as file:
        lines = file.readlines()
        keypoint = []
        for line in lines[1:]:
            line = line.replace('\n', '')
            line = line.split(' ')
            keypoint.append(line[0])
            keypoint.append(line[1])
    return list(map(float,keypoint))


def load_seg_data(path, image_name, width, height):
    with open(path+image_name+'.json','r') as file:
        load_dict = json.load(file)
        points = []
        for point in load_dict['shapes'][0]['points']: # 目前仅支持单目标检测,仅提取第一个值的多边形point
            nolm_x = '%.5f' % (point[0]/width)
            nolm_y = '%.5f' % (point[1]/height)
            points.append(nolm_x)
            points.append(nolm_y)
    points = points+[0]*(MAX_VERTICES-len(points))
    return list(map(float,points))


if __name__ == '__main__':

    image_list = os.listdir(image_dir)
    for image_name in image_list:
        output_list = []
        image_name = image_name.split('.')[0]
        width, height = load_image_data(image_dir+image_name+'.jpg')
        cls,object_box = load_detect_data(detect_label_dir,image_name) # xc,yc,w,h
        keypoint = load_keypoint_data(keypoint_label_dir,image_name) # x1c,y1c,....
        seg_point = load_seg_data(seg_label_dir,image_name,width,height) # x1c,y1c,....
        poly_point = convert_polygon(width,height,object_box,seg_point) # 极坐标:射线截距，每隔15度的角度offset
        output_list.append(float(cls))
        output_list.extend(object_box)
        output_list.extend(keypoint)
        output_list.extend(poly_point)

        blank = " "
        output_str = blank.join(list(map(str,output_list)))

        with open(save_path+image_name+'.txt','a') as output_file:
            output_file.write(output_str)

        cache_str = image_name+'.jpg '+output_str
        # make label
        with open(save_path+'cache_label.txt','a') as make_label_txt:
            make_label_txt.write(cache_str+'\n')
            """image_name:1  cls:1  bbox:4   keypoint:12  polygon:72"""
