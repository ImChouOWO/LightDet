"""Training-only GT ROI semantic warmup; shared heads learn from clean crops while predicted ROI supervision stays active."""
import math

import torch
import torch.nn.functional as F

from units.model.cards.relation_loss import phrase_predictions, ranking_preferences, select_ranking_negatives
from units.model.cards.phrase_matching import phrase_matrices


def gt_roi_scale(epoch, config):
    weight = float(config.get('weight', 0.0))
    hold, end = int(config.get('hold_epochs', 5)), int(config.get('end_epoch', 15))
    if weight < 0 or hold < 0 or end <= hold:
        raise ValueError('GT ROI requires weight >= 0 and end_epoch > hold_epochs >= 0')
    if epoch <= hold:
        return weight
    if epoch >= end:
        return 0.0
    return weight * .5 * (1 + math.cos(math.pi * (epoch-hold)/(end-hold)))


def gt_roi_loss(model, outputs, targets, criterion, epoch, config):
    scale = gt_roi_scale(epoch, config)
    zero = outputs['bbox'].sum() * 0
    names = ('alignment', 'order', 'contrast')
    metrics = {f'loss_gt_roi_{name}': zero.detach() for name in names}
    metrics.update(gt_roi_scale=scale, gt_roi_objects=0., loss_gt_roi_contrib=zero.detach())
    if scale == 0:
        return zero, metrics
    terms = []
    counts = [len(target['boxes']) if target['phrase_char_spans'] else 0 for target in targets]
    if not max(counts, default=0):
        return zero, metrics
    padded_boxes = outputs['bbox'].new_zeros(len(targets), max(counts), 4)
    for b, target in enumerate(targets):
        if not counts[b]:
            continue
        boxes = target['boxes'].to(outputs['bbox']).detach().reshape(-1, 4)
        padded_boxes[b, :counts[b]] = boxes
    query = model._condition_score_queries(padded_boxes)
    batch_score, _ = model.score_transformer._forward_group(
        outputs['fpn_features'], query,
        padded_boxes, outputs['input_image_size'], model.score_transformer.MAIN_GROUP)
    for b, target in enumerate(targets):
        boxes = target['boxes'].to(outputs['bbox']).detach().reshape(-1, 4)
        if not len(boxes) or not target['phrase_char_spans']:
            continue
        text = outputs['text_token'][b:b+1]
        score = batch_score[b:b+1, :counts[b]]
        mask = outputs['alignment_text_mask'][b:b+1]
        offsets = outputs['token_offsets'][b:b+1]
        logits = model.head.predict_phrase_matching(score, text, mask)
        phrase = phrase_predictions(logits, offsets, mask, [target])[0].float()
        labels = target['phrase_relations'].to(phrase.device).T
        pos, neg = labels == 1, labels == 0
        probability = phrase.sigmoid()
        element = F.binary_cross_entropy_with_logits(phrase, pos.float(), reduction='none')
        focal = element * (pos.float()-probability).abs().pow(criterion.text_alignment_focal_gamma)
        active = (pos | neg).any(1)
        alignment = ((focal * pos).sum(1)/pos.sum(1).clamp_min(1)
                     + criterion.text_alignment_negative_weight * (focal * neg).sum(1)/neg.sum(1).clamp_min(1))
        alignment = alignment[active].mean() if active.any() else phrase.sum()*0
        with torch.autocast(device_type=phrase.device.type, enabled=False):
            similarity = phrase_matrices(logits, offsets, mask,
                         [target['phrase_char_spans']])[1][0] / criterion.ranking_contrast_temperature
            orders, contrasts = [], []
            log_probability = F.logsigmoid(phrase)
            phrase_preference, object_preference = ranking_preferences(target, labels)
            for scores, sim, rel, preferred in (
                (log_probability, similarity, labels, phrase_preference),
                (log_probability.T, similarity.T, labels.T, object_preference)):
                hard, _ = select_ranking_negatives(scores, rel == 0, preferred,
                    criterion.ranking_negative_top_k, criterion.ranking_hard_negative_quota)
                for row, values, relation, candidates in zip(scores, sim, rel, hard):
                    positive, negative = relation == 1, relation == 0
                    if positive.any() and negative.any():
                        candidates = candidates[torch.isfinite(candidates)]
                        orders.append(F.relu(criterion.text_alignment_ranking_margin-row[positive,None]+candidates[None]).mean())
                        contrasts.append(F.softplus(values[negative].logsumexp(0)-values[positive]).mean())
            order = torch.stack(orders).mean() if orders else similarity.sum()*0
            contrast = torch.stack(contrasts).mean() if contrasts else similarity.sum()*0
        terms.append(torch.stack((alignment, order, contrast)))
        metrics['gt_roi_objects'] += len(boxes)
    if not terms:
        return zero, metrics
    alignment, order, contrast = torch.stack(terms).mean(0)
    loss = scale * (criterion.text_alignment_loss_weight * alignment
                    + criterion.text_alignment_ranking_weight * order
                    + criterion.ranking_contrast_weight * contrast)
    metrics.update({f'loss_gt_roi_{name}': value.detach() for name,value in zip(names,(alignment,order,contrast))})
    metrics['loss_gt_roi_contrib'] = loss.detach()
    return loss, metrics
