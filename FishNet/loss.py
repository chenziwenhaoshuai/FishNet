from torch import nn
import torch
from model_darknet import mymodel


def loss_func(output,target,alpha,nc):
    output = output.permute(0, 2, 3, 1)  # N,270,13,13==>N,13,13,270
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)  # N,13,13,3,90
    # iou x y w h cls point polygon
    mask_obj = target[..., 0] > 0  # N,13,13,3
    mask_noobj = target[..., 0] == 0

    loss_obj_p_func = nn.BCELoss()
    loss_obj_box_func = nn.MSELoss()
    loss_obj_cls_func = nn.CrossEntropyLoss()
    loss_obj_p = loss_obj_p_func(torch.sigmoid(output[...,0]),target[...,0].float())
    loss_obj_box = loss_obj_box_func(output[mask_obj][...,1:5],target[mask_obj][...,1:5])
    if nc > 1:
        loss_obj_cls = loss_obj_cls_func(output[mask_obj][...,5],torch.argmax(target[mask_obj][...,5],dim=1,keepdim=True).squeeze(dim=1))
        loss_obj = alpha * loss_obj_p + (1 - alpha) * 0.5 * loss_obj_box + (1 - alpha) * 0.5 * loss_obj_cls
    else:
        loss_obj = alpha * loss_obj_p + (1 - alpha) * loss_obj_box

    loss_point_func = nn.MSELoss()
    loss_point = loss_point_func(output[mask_obj][...,6:18],target[mask_obj][...,6:18])

    loss_seg_dist_func = nn.MSELoss()
    loss_seg_offset_func = nn.BCELoss()
    loss_sge_conf_func = nn.BCELoss()
    loss_seg_dist = loss_seg_dist_func(output[mask_obj][...,18::3], target[mask_obj][...,18::3])
    loss_seg_conf = loss_sge_conf_func(torch.sigmoid(output[mask_obj][...,19::3]), target[mask_obj][...,19::3])
    loss_seg_offset = loss_seg_offset_func(torch.sigmoid(output[mask_obj][...,20::3]), target[mask_obj][...,20::3])
    loss_seg = loss_seg_dist + loss_seg_offset + loss_seg_conf

    loss = loss_point + loss_obj + 0.1*loss_seg




    return loss.float()

if __name__ == '__main__':
    net = mymodel()
    image = torch.randn(5,3,416,416)
    output = net(image)
    target = torch.randn(5,13,13,3,90)
    loss = loss_func(output,target,0.6,1)
    print(loss)