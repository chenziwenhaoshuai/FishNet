"""2021年12月14日，记录第一次全部功能测试完成，效果挺好，下一步完成后续工作，加油！"""


import torch
import PIL
import numpy as np
from PIL import Image, ImageDraw
# from model_resnet import mymodel
from model_darknet import mymodel
from utils import nms, poly2xy
import os
from utils import make_image_data
import torchvision
from dataset import *

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


class Detector(torch.nn.Module):
    def __init__(self):
        super(Detector, self).__init__()

        weights = './weights/checkpoint.pt'
        # weights = './experment_log/12三任务300epo3e-4kmanchor0.1segloss    lastvision/checkpoint.pt'
        self.net = mymodel()
        self.net.load_state_dict(torch.load(weights))
        self.net.eval()

    def forward(self, input, thresh, anchors, case):
        output_13, output_26, output_52 = self.net(input)
        # output_13 = self.net(input) # for demo test
        #
        idxs_13, vecs_13 = self._filter(output_13,thresh)
        boxes_13 = self._parse(idxs_13, vecs_13, 32, anchors[13], case)

        idxs_26, vecs_26 = self._filter(output_26, thresh)
        boxes_26 = self._parse(idxs_26, vecs_26, 16, anchors[26], case)

        idxs_52, vecs_52 = self._filter(output_52, thresh)
        boxes_52 = self._parse(idxs_52, vecs_52, 8, anchors[52], case)
        boxes = torch.cat([boxes_13, boxes_26, boxes_52], dim=0)

        # test = list(boxes[:,6:18])
        # boxes = nms(boxes, 0.1, mode='inter')
        # test1 = list(boxes[:,6:18])
        print(1)
        return boxes

    def _filter(self, output, thresh):
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
        # mask: N H W 3 90

        mask = torch.sigmoid(output[..., 0]) > thresh

        idxs = torch.nonzero(mask) # 返回[N H W Anchor]上不为0的索引，例如[0,11,3,1]代表第11列3行的第1个anchor为有效目标
        vecs = output[mask] # 返回原始格式的tensor
        return idxs, vecs

    def _parse(self, idxs, vecs, t, anchors, case):
        anchors = torch.Tensor(anchors)

        n = idxs[:, 0] # 所属于的图片
        a = idxs[:, 3] # 所属于的anchor

        cy = (idxs[:, 1].float() + vecs[:, 2]) * t # / case
        cx = (idxs[:, 2].float() + vecs[:, 1]) * t # / case

        w = anchors[a, 0] * torch.exp(vecs[:, 3]) # /case
        h = anchors[a, 1] * torch.exp(vecs[:, 4]) # /case

        p = vecs[:, 0]
        # for multi class target
        # cls_p = vecs[:, 5:5 + num_cls]
        # cls_p = torch.softmax(cls_p, dim = 1)
        # cls_index = torch.argmax(cls_p, dim=1)

        diagonal = torch.sqrt(w**2+h**2)

        for i in range(6,18,2):
            # print(vecs[:, i])
            # print(vecs[:, i+1]*370)
            vecs[:, i] = vecs[:, i] * diagonal / case
            vecs[:, i+1] = vecs[:, i+1] * diagonal / case
        point = vecs[:, 6:18]
        for i in range(18,90,3):
            vecs[:, i] = vecs[:, i] * diagonal / case
            vecs[:, i+1] = torch.sigmoid(vecs[:, i+1])
            vecs[:, i + 2] = torch.sigmoid(vecs[:, i + 2])
        seg = vecs[:,18:90]
        # cx = cx.squeeze()
        # print(cx.shape)
        # print(point.shape)
        obj = torch.stack((n.float(), torch.sigmoid(p), cx, cy, w, h), dim=1)
        # print(s.shape)
        obj_point = torch.cat((obj, point), 1)
        obj_point_seg = torch.cat((obj_point, seg), 1)
        # print(s.shape)

        return obj_point_seg


if __name__ == '__main__':
    detector = Detector()
    image_path = './detect_image/'
    # image_path = './dataset_test/images/'
    # image_path = 'G:/imagenet里面的fish/n01440764/' 
    json_output_path = './json_output/'
    image_list = os.listdir(image_path)
    thresh = 0.01
    poly_thresh = 0.1
    for filename in image_list:
        img = Image.open(image_path+filename)
        _img, _w, _h = make_image_data(image_path+filename) # _h:375 _w:500
        w, h = _img.size[0], _img.size[1] # w:500 h:500
        case = input_image_height / max(w, h) # resize ratio
        _img = _img.resize((416,416))
        _img_data = transforms(_img)
        _img_data = torch.unsqueeze(_img_data, dim=0)
        # resize_ratio = input_image_height / max(w, h)

        results = detector(_img_data, thresh, ANCHORS_GROUP, case)
        draw = ImageDraw.Draw(img)
        for rst in results:
            rst = rst.detach().numpy()

            if len(rst) == 0:
                continue
            else:
                # rst[n p cx cy w h]
                cx = rst[2] / case
                cy = (rst[3] - (input_image_height-_h*case)*0.5) / case
                obj_w = rst[4] / case
                obj_h = rst[5] / case

                x1 = cx - 0.5 * obj_w
                y1 = cy - 0.5 * obj_h
                x2 = cx + 0.5 * obj_w
                y2 = cy + 0.5 * obj_h

                draw.rectangle((x1, y1, x2, y2), width=1, outline='red')
                draw.text((x1, y1 - 10), str(rst[1]), (255, 0, 0))

                with open(json_output_path+'detect.csv','a') as f:
                    f.write('0'+','+str(filename)+','+str(rst[1])+','+str(int(x1))+','+str(int(y1))+','+str(int(x2))+','+str(int(y2))+'\n')
                    # json_box = [x1, y1, x2, y2]

                point = []
                for i in range(6,18,2):
                    point.append((cx-rst[i] ) )
                    point.append((cy-rst[i+1]) )

                draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=(255, 0, 0)) # 头
                draw.line((point[0],point[1],point[2],point[3]),fill=(230, 25, 75),width=4) # 1,2
                draw.ellipse((point[2] - 3, point[3] - 3, point[2] + 3, point[3] + 3), fill=(255, 255, 0)) # 头上
                draw.line((point[0], point[1], point[4], point[5]), fill=(60, 180, 75), width=4) # 1,3
                draw.ellipse((point[4] - 3, point[5] - 3, point[4] + 3, point[5] + 3), fill=(255, 0, 255)) # 头下
                draw.line((point[0], point[1], point[6], point[7]), fill=(255, 225, 25), width=4) # 1,4
                draw.ellipse((point[6] - 3, point[7] - 3, point[6] + 3, point[7] + 3), fill=(0, 255, 0)) # 尾
                draw.line((point[6], point[7], point[8], point[9]), fill=(0, 130, 200), width=4) # 4,5
                draw.ellipse((point[8] - 3, point[9] - 3, point[8] + 3, point[9] + 3), fill=(0, 0, 255)) # 尾上
                draw.line((point[6], point[7], point[10], point[11]), fill=(245, 130, 48), width=4) # 4,6
                draw.ellipse((point[10] - 3, point[11] - 3, point[10] + 3, point[11] + 3), fill=(255, 255, 255)) # 尾下


                with open(json_output_path+'point.csv','a') as f:
                    f.write('0'+','+str(filename)+','+str(rst[1])+',')
                    for i in range(0,len(point),2):
                        f.write(str(int(point[i]))+','+str(int(point[i+1]))+',2,')
                    f.write('\n')


                poly = []
                for i in range(18,90,3):
                    if rst[i+2] > poly_thresh:
                        poly.append(rst[i])
                        poly.append(rst[i+1])
                        poly.append(rst[i+2])
                    else:
                        poly.append(0)
                        poly.append(0)
                        poly.append(0)

                polygon_xy_points = poly2xy(cx, cy, poly, step=15)
                # for i in range(0, len(polygon_xy_points), 2):  # 逐点显示向量
                #     x = polygon_xy_points[i]
                #     y = polygon_xy_points[i + 1]
                #     draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 0, 0))
                #     draw.line([cx, cy, x, y], fill=(114, 0, 0))
                #     img.show()
                draw.polygon(polygon_xy_points, outline=(70, 240, 240))

                with open(json_output_path+'polygon.csv','a') as f:
                    f.write('0'+','+str(filename)+','+str(rst[1])+',')#+str(i for i in polygon_xy_points)+'\n')
                    for i in range(0,len(polygon_xy_points),2):
                        f.write(str(int(polygon_xy_points[i]))+','+str(int(polygon_xy_points[i+1]))+',')
                    f.write('\n')

                print(1)


        img.show()


