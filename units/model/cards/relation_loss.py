"""Phrase scores shared by assignment and bidirectional semantic supervision."""

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou
from units.model.cards.background import BackgroundPolicy, background_candidates, mine_background
from units.model.cards.phrase_matching import phrase_matrices


def ranking_preferences(target, labels, objects=None):
    """Prefer shared-color negatives; relation labels remain the only source of positive/negative truth."""
    object_preference = torch.zeros_like(labels.T, dtype=torch.bool)
    phrase_preference = torch.zeros_like(labels, dtype=torch.bool)
    if "ranking_object_preferences" in target:
        preference = target["ranking_object_preferences"].to(labels.device).bool()
        object_preference = preference if objects is None else preference[:, objects]
        if object_preference.shape != labels.T.shape:
            raise ValueError("Object ranking preference/label shape mismatch")
    if "ranking_phrase_preferences" in target:
        preference = target["ranking_phrase_preferences"].to(labels.device).float()
        if preference.shape != (labels.shape[1], labels.shape[1]):
            raise ValueError("Phrase ranking preference/label shape mismatch")
        phrase_preference = (labels == 1).float() @ preference > 0
    return phrase_preference, object_preference


def select_ranking_negatives(scores, negative, preferred, top_k=5, preferred_quota=3):
    """Reserve semantic hard-negative slots, then fill the budget with the highest-scoring remaining negatives."""
    if top_k < 1 or not 0 <= preferred_quota <= top_k:
        raise ValueError("Ranking requires top_k >= 1 and 0 <= preferred_quota <= top_k")
    k = min(top_k, scores.shape[1])
    preferred = preferred & negative
    selected = torch.zeros_like(negative)
    quota = min(preferred_quota, k)
    if quota:
        values, indices = scores.detach().masked_fill(~preferred, -torch.inf).topk(quota, dim=1)
        selected.scatter_(1, indices, torch.isfinite(values))
    values, indices = scores.detach().masked_fill(~(negative & ~selected), -torch.inf).topk(k, dim=1)
    remaining = k - selected.sum(1)
    valid = torch.isfinite(values) & (torch.arange(k, device=scores.device)[None] < remaining[:, None])
    selected.scatter_(1, indices, selected.gather(1, indices) | valid)
    values, indices = scores.masked_fill(~selected, -torch.inf).topk(k, dim=1)
    return values, indices


def ranking_contrast_loss(visual, text, offsets, token_mask, targets, pred_boxes, temperature, matching=None):
    """Pool duplicate ROIs per GT; compare every positive only against confirmed negatives in both directions."""
    if temperature <= 0:
        raise ValueError("Ranking contrast temperature must be positive")
    with torch.autocast(device_type=visual.device.type, enabled=False):
        zero = (visual.float().sum() + text.float().sum()) * 0
        similarities = (phrase_matrices(matching, offsets, token_mask,
                         [t['phrase_char_spans'] for t in targets])[1] if matching is not None else None)
        losses = []
        for b, target in enumerate(targets):
            gt = target["boxes"].to(pred_boxes).reshape(-1, 4)
            if not len(gt) or not target["phrase_char_spans"]:
                continue
            with torch.no_grad():
                overlap = box_iou(pred_boxes[b].detach().float(), gt.float())
                keep = (overlap >= .5).sum(1) == 1
                ids = overlap[keep].argmax(1)
                objects = ids.unique()
            if not len(objects):
                continue
            if similarities is not None:
                similarity = torch.stack([similarities[b][keep][ids == obj].mean(0) for obj in objects]) / temperature
                labels = target["phrase_relations"].to(visual.device).T[objects]
                losses.append(_contrast_directions(similarity, labels, zero))
                continue
            object_features = torch.stack([visual[b, keep][ids == obj].float().mean(0) for obj in objects])
            phrase_features = []
            for spans in target["phrase_char_spans"]:
                mask = torch.zeros_like(token_mask[b], dtype=torch.bool)
                for start, end in spans:
                    hits = (offsets[b, :, 1] > start) & (offsets[b, :, 0] < end) & token_mask[b].bool()
                    covered = offsets[b][hits]
                    if not covered.numel() or covered[:, 0].min() > start or covered[:, 1].max() < end:
                        raise ValueError("Invalid or truncated phrase in ranking contrast supervision")
                    mask |= hits
                if not mask.any():
                    raise ValueError("Ranking contrast requires nonempty phrase spans")
                phrase_features.append(text[b, mask].float().mean(0))
            similarity = F.normalize(object_features, dim=-1) @ F.normalize(torch.stack(phrase_features), dim=-1).T / temperature
            labels = target["phrase_relations"].to(visual.device).T[objects]
            directions = []
            for scores, relation in ((similarity, labels), (similarity.T, labels.T)):
                pos, neg = relation == 1, relation == 0
                active = pos.any(1) & neg.any(1)
                if active.any():
                    scores, pos, neg = scores[active], pos[active], neg[active]
                    negative_partition = scores.masked_fill(~neg, -torch.inf).logsumexp(1)
                    terms = F.softplus(negative_partition[:, None] - scores)
                    directions.append(((terms * pos).sum(1) / pos.sum(1)).mean())
            if directions:
                losses.append(torch.stack(directions).mean())
        return torch.stack(losses).mean() if losses else zero


def _contrast_directions(similarity, labels, zero):
    losses = []
    for scores, relation in ((similarity, labels), (similarity.T, labels.T)):
        pos, neg = relation == 1, relation == 0
        active = pos.any(1) & neg.any(1)
        if active.any():
            scores, pos, neg = scores[active], pos[active], neg[active]
            partition = scores.masked_fill(~neg, -torch.inf).logsumexp(1)
            terms = F.softplus(partition[:,None] - scores)
            losses.append(((terms * pos).sum(1) / pos.sum(1)).mean())
    return torch.stack(losses).mean() if losses else zero


def phrase_predictions(matching, offsets, token_mask, targets):
    return phrase_matrices(matching, offsets, token_mask,
                           [t['phrase_char_spans'] for t in targets])[0]


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
                  *, pred_boxes, quality_logits=None, background_policy=None, background_scale=1.0,
                  ranking_top_k=5, ranking_preferred_quota=3):
    policy = background_policy or BackgroundPolicy()
    zero = sum(p.sum() * 0 for p in predictions)
    losses, ranks = [], []
    positives, negatives = [], []
    ignored, rank_pairs, extra_count, background_count = 0, 0, 0, 0
    background_pairs, background_rank_pairs, eligible_count = 0, 0, 0
    background_losses, background_ranks = [], []
    ranking_counts = {name: zero.detach().clone() for name in (
        "ranking_negative_candidates", "ranking_hard_negative_candidates",
        "ranking_selected_negatives", "ranking_selected_hard_negatives")}
    for batch_index, (logits, target, (queries, objects)) in enumerate(zip(predictions, targets, assignments)):
        relations = target["phrase_relations"].to(logits.device).T
        if pred_boxes is None:
            raise ValueError("Semantic supervision requires candidate boxes")
        with torch.no_grad():
            boxes = pred_boxes[batch_index].detach().float()
            gt = target["boxes"].to(boxes).reshape(-1, 4)
            overlap = box_iou(boxes, gt)
            confident = overlap >= 0.5
            supervised_queries = (confident.sum(1) == 1).nonzero().flatten()
            objects = (overlap[supervised_queries].argmax(1) if gt.numel()
                       else torch.empty(0, device=boxes.device, dtype=torch.long))
            _, inverse, counts = objects.unique(return_inverse=True, return_counts=True)
            weights = 1.0 / counts[inverse].float()
            normalizer = max(1, counts.numel())
        values, labels = logits[supervised_queries], relations[objects]
        extra_count += int((~torch.isin(supervised_queries, queries.to(logits.device))).sum())
        all_probability = logits.float().sigmoid()
        all_log_probability = F.logsigmoid(logits.float())
        all_final = all_probability
        all_log_final = all_log_probability
        if quality_logits is not None:
            quality_logit = quality_logits[batch_index].detach().float().reshape(-1)
            quality = quality_logit.sigmoid()
            all_final = (all_probability * quality[:, None]).clamp_min(1e-12).sqrt()
            all_log_final = .5 * (all_log_probability + F.logsigmoid(quality_logit)[:, None])
        candidates = background_candidates(boxes, gt, target.get("objects_complete", False), policy)
        eligible_count += int(candidates.sum())
        background = torch.zeros_like(logits, dtype=torch.bool)
        image_loss = logits.sum() * 0
        if background_scale > 0:
            background = mine_background(candidates, all_final, (relations == 1).sum(0), policy)
            if background.any():
                element = F.softplus(logits.float()) * all_probability.pow(gamma)
                active = background.any(0)
                background_loss = ((element * background).sum(0) / background.sum(0).clamp_min(1))[active].mean()
                image_loss = background_scale * policy.alignment_weight * background_loss
                background_losses.append(background_loss)
                background_count += int(background.any(1).sum())
                background_pairs += int(background.sum())
        if not values.numel():
            losses.append(image_loss)
            continue
        pos, neg = labels == 1, labels == 0
        probability = values.sigmoid()
        bce = F.binary_cross_entropy_with_logits(values, pos.float(), reduction="none")
        focal = bce * (pos.float() - probability).abs().pow(gamma)
        row_loss = (focal * pos).sum(1) / pos.sum(1).clamp_min(1)
        row_loss += negative_weight * (focal * neg).sum(1) / neg.sum(1).clamp_min(1)
        # Extra candidate counts cannot increase an object's total supervision weight.
        losses.append(image_loss + (row_loss * weights).sum() / normalizer)
        if pos.any():
            positives.append(probability[pos].mean())
        if neg.any():
            negatives.append(probability[neg].mean())
        ignored += int((labels < 0).sum())
        # Log fusion preserves inference order; the margin now specifies a score ratio.
        log_probability = all_log_probability[supervised_queries]
        log_final = all_log_final[supervised_queries]
        phrase_preference, object_preference = ranking_preferences(target, labels, objects)
        directions = []
        for scores, positive, negative, preferred, transposed in (
            (log_probability, pos, neg, phrase_preference, False),
            (log_final.T, pos.T, neg.T, object_preference, True)
        ):
            direction_loss = zero
            has_pairs = False
            if transposed:
                grouped_negative = torch.zeros_like(negative)
                for object_id in objects.unique():
                    group = objects == object_id
                    candidates = scores.masked_fill(~(negative & group[None, :]), -torch.inf)
                    best, index = candidates.max(1)
                    rows = torch.where(torch.isfinite(best))[0]
                    grouped_negative[rows, index[rows]] = True
                negative = grouped_negative
            # All positives compete only with confirmed negatives; ignore is never mined.
            top, indices = select_ranking_negatives(scores, negative, preferred,
                                                    ranking_top_k, ranking_preferred_quota)
            finite = torch.isfinite(top)
            active = positive.any(1, keepdim=True)
            ranking_counts["ranking_negative_candidates"] += (negative & active).sum()
            ranking_counts["ranking_hard_negative_candidates"] += (negative & preferred & active).sum()
            ranking_counts["ranking_selected_negatives"] += (finite & active).sum()
            ranking_counts["ranking_selected_hard_negatives"] += (finite & preferred.gather(1, indices) & active).sum()
            valid = positive & negative.any(1, keepdim=True)
            pair_loss = F.relu(margin - scores[:, :, None] + top[:, None, :])
            terms = pair_loss.masked_fill(~finite[:, None, :], 0).sum(2) / finite.sum(1).clamp_min(1)[:, None]
            if valid.any():
                pair_weights = weights[None, :] if transposed else weights[:, None]
                pair_weights = pair_weights.expand_as(scores)
                direction_loss = (terms[valid] * pair_weights[valid]).sum() / pair_weights[valid].sum()
                has_pairs = True
                rank_pairs += int(valid.sum())
            if transposed and background.any():
                bg_scores = all_log_final.T.masked_fill(~background.T, -torch.inf)
                top = bg_scores.topk(min(policy.top_k, bg_scores.shape[1]), dim=1).values
                finite = torch.isfinite(top)
                valid = positive & finite.any(1, keepdim=True)
                if valid.any():
                    terms = F.relu(margin - scores[:, :, None] + top[:, None, :])
                    terms = terms.masked_fill(~finite[:, None, :], 0).sum(2) / finite.sum(1).clamp_min(1)[:, None]
                    pair_weights = weights[None, :].expand_as(scores)
                    bg_rank = (terms[valid] * pair_weights[valid]).sum() / pair_weights[valid].sum()
                    direction_loss = direction_loss + background_scale * policy.ranking_weight * bg_rank
                    background_ranks.append(bg_rank)
                    background_rank_pairs += int((valid.sum(1) * finite.sum(1)).sum())
                    has_pairs = True
            if has_pairs:
                directions.append(direction_loss)
        if directions:
            ranks.append(torch.stack(directions).mean())
    mean = lambda values: torch.stack(values).mean() if values else zero
    pos_mean, neg_mean = mean(positives), mean(negatives)
    metrics = dict(**ranking_counts, positive_score_mean=pos_mean, negative_score_mean=neg_mean,
                   positive_negative_margin=pos_mean-neg_mean,
                   positive_count=zero.detach().new_tensor(sum(int((t["phrase_relations"] == 1).sum()) for t in targets)),
                   negative_count=zero.detach().new_tensor(sum(int((t["phrase_relations"] == 0).sum()) for t in targets)),
                   ignored_count=zero.detach().new_tensor(ignored), rank_pair_count=zero.detach().new_tensor(rank_pairs),
                   extra_query_count=zero.detach().new_tensor(extra_count),
                   background_query_count=zero.detach().new_tensor(background_count),
                   background_candidate_count=zero.detach().new_tensor(eligible_count),
                   background_pair_count=zero.detach().new_tensor(background_pairs),
                   background_rank_pair_count=zero.detach().new_tensor(background_rank_pairs),
                   background_bce=mean(background_losses), background_rank=mean(background_ranks))
    return mean(losses), mean(ranks), metrics
