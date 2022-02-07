from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask
"""评价过程，1.使用yolo2coco.py生成标签json 2.使用detect.py生成检测csv文件 3.使用csv2json.py将检测的csv转换为json 4.使用cocoeval.py
对标签json与检测的json进行评价"""

is_coco = 0
detect_anno_json = '../dataset_test/annotations/detect_label.json'  # annotations json
detect_pred_json = '../json_output/detect.json'
point_anno_json = '../dataset_test/annotations/point_label.json'  # annotations json
point_pred_json = '../json_output/point.json'
seg_anno_json = '../dataset_test/annotations/seg_label.json'  # annotations json
seg_pred_json = '../json_output/seg.json'

print("-------------bbox eval----------")
detect_anno = COCO(detect_anno_json)  # init annotations api
detect_pred = detect_anno.loadRes(detect_pred_json)  # init predictions api
# pred = COCO(pred_json)
eval = COCOeval(detect_anno, detect_pred, 'bbox')
# if is_coco:
#     eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
eval.evaluate()
eval.accumulate()
eval.summarize()
map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
print("-------------end----------")

print("-------------seg eval----------")
seg_anno = COCO(seg_anno_json)  # init annotations api
seg_pred = seg_anno.loadRes(seg_pred_json)  # init predictions api
# pred = COCO(pred_json)
eval = COCOeval(seg_anno, seg_pred, 'segm')
# if is_coco:
#     eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
eval.evaluate()
eval.accumulate()
eval.summarize()
map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
print("-------------end----------")

print("-------------keypoint eval----------")
keypoint_anno = COCO(point_anno_json)  # init annotations api
keypoint_pred = detect_anno.loadRes(point_pred_json)  # init predictions api
# pred = COCO(pred_json)
eval = COCOeval(keypoint_anno, keypoint_pred, 'keypoints')
# if is_coco:
#     eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
eval.evaluate()
eval.accumulate()
eval.summarize()
map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
print("-------------end----------")