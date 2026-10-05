"""Phrase scores shared by assignment and bidirectional semantic supervision."""

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou


def phrase_predictions(token_logits, offsets, token_mask, targets):
    # Autocast matmul can round probabilities to 1 before logit and poison gradients.
    with torch.autocast(device_type=token_logits.device.type, enabled=False):
        return _phrase_predictions_fp32(token_logits.float(), offsets, token_mask, targets)


def _phrase_predictions_fp32(token_logits, offsets, token_mask, targets):
    if offsets is None:
        raise ValueError("Relation supervision requires tokenizer offsets")
    predictions = []
    for b, target in enumerate(targets):
        spans = target["phrase_char_spans"]
        rows = []
        for group in spans:
            mask = torch.zeros_like(token_mask[b], dtype=torch.bool)
            for start, end in group:
                hits = ((offsets[b, :, 1] > start) & (offsets[b, :, 0] < end)
                        & token_mask[b].bool())
                covered = offsets[b][hits]
                if not covered.numel() or covered[:, 0].min() > start or covered[:, 1].max() < end:
                    raise ValueError("Caption truncated or phrase offsets invalid; increase text_max_length")
                mask |= hits
            rows.append(mask.float() / mask.sum().clamp_min(1))
        maps = torch.stack(rows)
        # Match inference: mean of token probabilities, then convert to logits.
        probability = token_logits[b].float().sigmoid() @ maps.T
        predictions.append(torch.logit(probability.clamp(1e-6, 1 - 1e-6)))
    return predictions


def relation_cost(logits, target, negative_weight=0.25):
    with torch.autocast(device_type=logits.device.type, enabled=False):
        return _relation_cost_fp32(logits.float(), target, negative_weight)


def _relation_cost_fp32(logits, target, negative_weight):
    relations = target["phrase_relations"].to(logits.device)
    positive, negative = (relations == 1).float(), (relations == 0).float()
    pos = F.softplus(-logits) @ positive / positive.sum(0).clamp_min(1)
    neg = F.softplus(logits) @ negative / negative.sum(0).clamp_min(1)
    return pos + negative_weight * neg


def relation_loss(predictions, targets, assignments, gamma, margin, negative_weight,
                  *, pred_boxes=None, quality_logits=None, unmatched_weight=0.0):
    zero = sum(p.sum() * 0 for p in predictions)
    losses, ranks = [], []
    positives, negatives = [], []
    ignored, rank_pairs, extra_count, background_count = 0, 0, 0, 0
    for batch_index, (logits, target, (queries, objects)) in enumerate(zip(predictions, targets, assignments)):
        relations = target["phrase_relations"].to(logits.device).T
        supervised_queries = queries
        values, labels = logits[queries], relations[objects]
        _, inverse, counts = objects.unique(return_inverse=True, return_counts=True)
        weights = 1.0 / counts[inverse].float()
        normalizer = max(1, counts.numel())
        if pred_boxes is not None and unmatched_weight > 0:
            # Geometry chooses the object; annotation relations alone choose semantic labels.
            with torch.no_grad():
                boxes = pred_boxes[batch_index].detach().float()
                gt = target["boxes"].to(boxes).reshape(-1, 4)
                overlap = box_iou(boxes, gt)
                available = torch.ones(len(logits), device=logits.device, dtype=torch.bool)
                available[queries] = False
                confident = overlap >= 0.5
                extra = (available & (confident.sum(1) == 1)).nonzero().flatten()
                extra_objects = overlap[extra].argmax(1) if extra.numel() else objects[:0]
                _, extra_inverse, extra_counts = extra_objects.unique(return_inverse=True, return_counts=True)
                extra_weights = unmatched_weight / extra_counts[extra_inverse].float()
                background = queries[:0]
                # Attribute completeness does not imply that every object was annotated.
                if target.get("objects_complete", False) is True:
                    low_overlap = (overlap < 0.1).all(1)
                    candidates = (available & low_overlap).nonzero().flatten()
                    k = min(10, candidates.numel())
                    if k:
                        hardness = logits[candidates].detach().sigmoid().amax(1)
                        background = candidates[hardness.topk(k).indices]
            supervised_queries = torch.cat((queries, extra, background))
            extra_count += extra.numel()
            background_count += background.numel()
            values = torch.cat((values, logits[extra], logits[background]))
            labels = torch.cat((labels, relations[extra_objects],
                                relations.new_zeros((background.numel(), logits.shape[1]))))
            weights = torch.cat((weights, extra_weights,
                                 logits.new_full((background.numel(),), unmatched_weight / max(1, background.numel()))))
        if not values.numel():
            continue
        pos, neg = labels == 1, labels == 0
        probability = values.sigmoid()
        bce = F.binary_cross_entropy_with_logits(values, pos.float(), reduction="none")
        focal = bce * (pos.float() - probability).abs().pow(gamma)
        row_loss = (focal * pos).sum(1) / pos.sum(1).clamp_min(1)
        row_loss += negative_weight * (focal * neg).sum(1) / neg.sum(1).clamp_min(1)
        # Extra candidate counts cannot increase an object's total supervision weight.
        losses.append((row_loss * weights).sum() / normalizer)
        if pos.any():
            positives.append(probability[pos].mean())
        if neg.any():
            negatives.append(probability[neg].mean())
        ignored += int((labels < 0).sum())
        # Object ranking uses exactly the probability fusion used at inference.
        final_probability = probability
        if quality_logits is not None:
            quality = quality_logits[batch_index].float().reshape(-1)[supervised_queries].sigmoid()
            final_probability = (probability * quality[:, None]).clamp_min(1e-12).sqrt()
        directions = []
        for scores, positive, negative, transposed in (
            (probability, pos, neg, False), (final_probability.T, pos.T, neg.T, True)
        ):
            # All positives compete only with confirmed negatives; ignore is never mined.
            hardest = scores.masked_fill(~negative, -torch.inf).max(1).values
            valid = positive & negative.any(1, keepdim=True)
            terms = F.relu(margin - scores + hardest[:, None])
            if valid.any():
                pair_weights = weights[None, :] if transposed else weights[:, None]
                pair_weights = pair_weights.expand_as(scores)
                directions.append((terms[valid] * pair_weights[valid]).sum()
                                  / pair_weights[valid].sum())
                rank_pairs += int(valid.sum())
        if directions:
            ranks.append(torch.stack(directions).mean())
    mean = lambda values: torch.stack(values).mean() if values else zero
    pos_mean, neg_mean = mean(positives), mean(negatives)
    metrics = dict(positive_score_mean=pos_mean, negative_score_mean=neg_mean,
                   positive_negative_margin=pos_mean-neg_mean,
                   positive_count=zero.detach().new_tensor(sum(int((t["phrase_relations"] == 1).sum()) for t in targets)),
                   negative_count=zero.detach().new_tensor(sum(int((t["phrase_relations"] == 0).sum()) for t in targets)),
                   ignored_count=zero.detach().new_tensor(ignored), rank_pair_count=zero.detach().new_tensor(rank_pairs),
                   extra_query_count=zero.detach().new_tensor(extra_count),
                   background_query_count=zero.detach().new_tensor(background_count))
    return mean(losses), mean(ranks), metrics
