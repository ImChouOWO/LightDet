"""Phrase Matching Head: pool tokens inside each description, then combine normalized ROI/phrase interactions with cosine affinity for shared alignment and ranking scores."""
import torch
from torch import nn
import torch.nn.functional as F


class PhraseMatchingHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.token_attention = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.zeros_(self.token_attention.weight)
        self.pair_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, max(4, hidden_dim // 4)),
            nn.GELU(),
            nn.Linear(max(4, hidden_dim // 4), 1),
        )
        nn.init.normal_(self.pair_head[-1].weight, std=0.01)
        nn.init.zeros_(self.pair_head[-1].bias)
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, visual, text, logit_scale):
        return dict(visual=visual, text=text, token_weights=self.token_attention(text).squeeze(-1),
                    scale=logit_scale.exp().clamp(1., 100.), bias=self.bias,
                    pair_head=self.pair_head)


def matching_for_image(matching, index):
    return {key: value[index:index+1] if key in ('visual', 'text', 'token_weights') else value
            for key,value in matching.items() if not key.startswith('_')}


def phrase_matrices(matching, offsets, token_mask, phrase_spans):
    key = (tuple(tuple(tuple(tuple(span) for span in group) for group in groups) for groups in phrase_spans),
           id(offsets), id(token_mask), id(matching['visual']), id(matching['text']),
           id(matching['token_weights']), id(matching['pair_head']))
    cache = matching.setdefault('_matrix_cache', {})
    if key in cache:
        return cache[key]
    sizes = [len(groups) for groups in phrase_spans]
    count = max(sizes, default=0)
    mask_key = (key[0], id(token_mask))
    masks_cache = getattr(offsets, '_phrase_masks', None)
    if masks_cache is None:
        masks_cache = offsets._phrase_masks = {}
    if mask_key not in masks_cache:
        masks_cache[mask_key] = _phrase_masks(offsets, token_mask, phrase_spans).to(
            matching['visual'].device, non_blocking=True)
    masks = masks_cache[mask_key]
    with torch.autocast(device_type=matching['visual'].device.type, enabled=False):
        scores = matching['token_weights'].float()[:,None,:].expand(-1,count,-1)
        scores = scores.masked_fill(~masks, -torch.inf)
        real = masks.any(-1, keepdim=True)
        scores = torch.where(real, scores, torch.zeros_like(scores))
        weights = scores.softmax(-1) * masks
        phrase = F.normalize(torch.bmm(weights, matching['text'].float()), dim=-1)
        visual = F.normalize(matching['visual'].float(), dim=-1)
        similarity = torch.bmm(visual, phrase.transpose(1,2))
        interactions = torch.cat((visual[:, :, None] * phrase[:, None],
                                  (visual[:, :, None] - phrase[:, None]).abs()), dim=-1)
        similarity = similarity + 0.25 * matching['pair_head'](interactions).squeeze(-1).tanh()
        logits = similarity * matching['scale'].float() + matching['bias'].float()
    result = ([logits[b,:,:n] for b,n in enumerate(sizes)], [similarity[b,:,:n] for b,n in enumerate(sizes)])
    cache[key] = result
    return result


def _phrase_masks(offsets, token_mask, phrase_spans):
    # Span masks depend on tokenization, not on Main/Aux head weights.
    offsets_cpu, valid_cpu = offsets.detach().cpu(), token_mask.detach().cpu().bool()
    batch, length = valid_cpu.shape
    count = max((len(groups) for groups in phrase_spans), default=0)
    masks = torch.zeros(batch, count, length, dtype=torch.bool)
    for b, groups in enumerate(phrase_spans):
        for p, spans in enumerate(groups):
            for start,end in spans:
                hits = (offsets_cpu[b,:,1] > start) & (offsets_cpu[b,:,0] < end) & valid_cpu[b]
                covered = offsets_cpu[b][hits]
                if not covered.numel() or covered[:,0].min() > start or covered[:,1].max() < end:
                    raise ValueError('Invalid or truncated phrase in Phrase Matching Head')
                masks[b,p] |= hits
            if not masks[b,p].any():
                raise ValueError('Phrase Matching Head requires nonempty descriptions')
    return masks
