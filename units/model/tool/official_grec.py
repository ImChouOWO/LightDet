# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Evaluator and box functions copied unchanged from henghuiding/gRefCOCO.
# Upstream commit: 41a8f008006bb1fb0f1df2547f9477bb97e36593
import copy
from types import SimpleNamespace
import numpy as np
import torch
from torchvision.ops.boxes import box_area

# This entry point runs one process; no distributed synchronization is needed.
dist=SimpleNamespace(is_main_process=lambda:True,all_gather=lambda value:[value])

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

class RefExpEvaluator(object):
    def __init__(self, refexp_gt, iou_types, k=(1, 5, 10), thresh_iou=0.5, thresh_score=0.7, thresh_F1=1.0):
        assert isinstance(k, (list, tuple))
        refexp_gt = copy.deepcopy(refexp_gt)
        self.refexp_gt = refexp_gt
        self.iou_types = iou_types
        self.img_ids = self.refexp_gt.imgs.keys()
        self.predictions = {}
        self.thresh_iou = thresh_iou
        self.thresh_score = thresh_score
        self.thresh_F1 = thresh_F1
        print(self.thresh_score)
    def accumulate(self):
        pass

    def update(self, predictions):
        self.predictions.update(predictions)

    def synchronize_between_processes(self):
        all_predictions = dist.all_gather(self.predictions)
        merged_predictions = {}
        for p in all_predictions:
            merged_predictions.update(p)
        self.predictions = merged_predictions

    def summarize(self):
        if dist.is_main_process():
            correct_image = 0
            num_image = 0
            nt = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
            for image_id in self.img_ids:
                TP = 0
                ann_ids = self.refexp_gt.getAnnIds(imgIds=image_id)
                # assert len(ann_ids) == 1
                img_info = self.refexp_gt.loadImgs(image_id)[0]

                target = self.refexp_gt.loadAnns(ann_ids)
                prediction = self.predictions[image_id]
                assert prediction is not None
                sorted_scores_boxes = sorted(
                    zip(prediction["scores"].tolist(), prediction["boxes"].tolist()), reverse=True
                )
                sorted_scores, sorted_boxes = zip(*sorted_scores_boxes)
                sorted_boxes = torch.cat([torch.as_tensor(x).view(1, 4) for x in sorted_boxes])
                converted_bbox_all = []
                no_target_flag = False
                for one_target in target:
                    if one_target['category_id'] == -1:
                        no_target_flag = True
                    target_bbox = one_target["bbox"]
                    converted_bbox = [
                    target_bbox[0],
                    target_bbox[1],
                    target_bbox[2] + target_bbox[0],
                    target_bbox[3] + target_bbox[1],
                ]
                    converted_bbox_all.append(torch.tensor(converted_bbox))
                gt_bbox_all = torch.stack(converted_bbox_all, dim=0)

                sorted_scores_array = np.array(sorted_scores)
                idx = sorted_scores_array >= self.thresh_score
                filtered_boxes = sorted_boxes[idx]
                # filtered_boxes = sorted_boxes[0:1]
                giou = generalized_box_iou(filtered_boxes, gt_bbox_all.view(-1, 4))
                num_prediction = filtered_boxes.shape[0]
                num_gt = gt_bbox_all.shape[0]
                if no_target_flag:
                    if num_prediction >= 1:
                        nt["FN"] += 1
                    else:
                        nt["TP"] += 1
                    if num_prediction >= 1:
                        F_1 = 0.
                    else:
                        F_1 = 1.0
                else:
                    if num_prediction >= 1:
                        nt["TN"] += 1
                    else:
                        nt["FP"] += 1
                    for i in range(min(num_prediction, num_gt)):
                        top_value, top_index = torch.topk(giou.flatten(0, 1), 1)
                        if top_value < self.thresh_iou:
                            break
                        else:
                            top_index_x = top_index // num_gt
                            top_index_y = top_index % num_gt
                            TP += 1
                            giou[top_index_x[0], :] = 0.0
                            giou[:, top_index_y[0]] = 0.0
                    FP = num_prediction - TP
                    FN = num_gt - TP
                    F_1 = 2 * TP / (2 * TP + FP + FN)

                if F_1 >= self.thresh_F1:
                    correct_image += 1
                num_image += 1

            score = correct_image / num_image
            results = {}
            results['F1_score'] = score
            results['T_acc'] = nt['TN'] / (nt['TN'] + nt['FP'])
            results['N_acc'] = nt['TP'] / (nt['TP'] + nt['FN'])
            print(results, 'results')
            return results
        return None
