import json
from pycocotools import mask as maskUtils
import cv2

file = "../json_output/polygon.csv"
jsonpath = "../dataset_test/annotations/seg_label.json" # 标签
image_path = '../dataset_test/images/'
# label_path = "./labels/"
# labelsss = []
output = {}
name = set()
file_name = []
with open(jsonpath,'r') as ojson:
    load_dict = json.load(ojson)
    images = load_dict['images']
with open(file, 'r') as csv:
    csv = csv.read().split('\n')
    for line in csv:
        if len(line)>3:
            line = line.split(',')
            cls = line[0]
            file_name.append(line[1])
            conf = line[2]
            polygon = line[3]
            # x = int(line[3])
            # y = int(line[4])
            # w = int(line[5])-int(line[3])
            # h = int(line[6])-int(line[4])
    # print(file_name)
    name = list(set(file_name))
    # images = []
    # for index,imgname in enumerate(name):
    #     bianhao = {"file_name":imgname,"id":index}
    #     # bianhao = json.dumps(bianhao)
    #     images.append(bianhao)
    #     print(bianhao)
    annotations = []
    for line in csv:
        if len(line)>3:
            line = line.split(',')

            file_name = line[1]
            for lines in images:
                if file_name == lines["file_name"]:
                    id = lines["id"]
                    conf = float(line[2])
                    cls = int(line[0])
                    polygon = []
                    for k in line[3:]:
                        if len(k)>0:
                            polygon.append(k)

                    image = cv2.imread(image_path + line[1])
                    width, height = image.shape[1], image.shape[0]
                    seg_x = polygon[0::2]
                    seg_y = polygon[1::2]
                    x1 = int(min(seg_x))
                    y1 = int(min(seg_y))
                    w = int(max(seg_x)) - x1
                    h = int(max(seg_y)) - y1
                    rles = maskUtils.frPyObjects([polygon], height, width)
                    rle = maskUtils.merge(rles)
                    size = rle['size']
                    counts = rle['counts'].decode()
                    segmentation = {"size":size,"counts":counts}
                    kuang = {"image_id":id,"category_id":cls,"segmentation":segmentation,"score":conf}
                    # # kuang = json.dumps(kuang)
                    # kuang = kuang.replace('\'','\"')
                    # with open('../dataset_test/annotations/x6-res.json', 'a') as json:
                    #     json.write(str(kuang))
                    annotations.append(kuang)
                    print(kuang)
    # output = {"images":images,"annotations":annotations}
    output = annotations
    output = json.dumps(output)
    output = output.replace('\'','\"')
    with open('../json_output/seg.json', 'a') as json:
        json.write(str(output))
