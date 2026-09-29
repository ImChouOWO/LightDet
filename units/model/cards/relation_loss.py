"""Phrase scores shared by assignment and bidirectional semantic supervision."""

import torch
import torch.nn.functional as F


def phrase_predictions(token_logits, offsets, token_mask, targets):
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


def relation_cost(logits, target):
    relations = target["phrase_relations"].to(logits.device)
    positive, negative = (relations == 1).float(), (relations == 0).float()
    pos = F.softplus(-logits) @ positive / positive.sum(0).clamp_min(1)
    neg = F.softplus(logits) @ negative / negative.sum(0).clamp_min(1)
    return pos + neg


def relation_loss(predictions, targets, assignments, gamma, margin, negative_weight):
    zero = sum(p.sum() * 0 for p in predictions)
    losses, ranks = [], []
    positives, negatives = [], []
    ignored, rank_pairs = 0, 0
    for logits, target, (queries, objects) in zip(predictions, targets, assignments):
        if not queries.numel():
            continue
        values = logits[queries]
        labels = target["phrase_relations"].to(logits.device).T[objects]
        pos, neg = labels == 1, labels == 0
        probability = values.sigmoid()
        bce = F.binary_cross_entropy_with_logits(values, pos.float(), reduction="none")
        focal = bce * (pos.float() - probability).abs().pow(gamma)
        row_loss = (focal * pos).sum(1) / pos.sum(1).clamp_min(1)
        row_loss += negative_weight * (focal * neg).sum(1) / neg.sum(1).clamp_min(1)
        # Each GT has equal weight even when its Aux positive count differs.
        _, inverse, counts = objects.unique(return_inverse=True, return_counts=True)
        weights = 1.0 / counts[inverse].float()
        losses.append((row_loss * weights).sum() / counts.numel())
        if pos.any():
            positives.append(probability[pos].mean())
        if neg.any():
            negatives.append(probability[neg].mean())
        ignored += int((labels < 0).sum())
        directions = []
        for scores, positive, negative, transposed in (
            (values, pos, neg, False), (values.T, pos.T, neg.T, True)
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
                   positive_top1_score=pos_mean, negative_text_top1_score=neg_mean,
                   positive_negative_margin=pos_mean-neg_mean,
                   positive_count=zero.detach().new_tensor(sum(int((t["phrase_relations"] == 1).sum()) for t in targets)),
                   negative_count=zero.detach().new_tensor(sum(int((t["phrase_relations"] == 0).sum()) for t in targets)),
                   ignored_count=zero.detach().new_tensor(ignored), rank_pair_count=zero.detach().new_tensor(rank_pairs))
    return mean(losses), mean(ranks), metrics
