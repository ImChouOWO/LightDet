"""Re-evaluate exported raw predictions without running model inference."""
import argparse
import gzip
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from units.model.tool.grounding_metrics import GroundingMetrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    metric = GroundingMetrics()
    images = set()
    with gzip.open(args.predictions, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            images.add(row['filename'])
            objects = torch.tensor(row['object_boxes']).reshape(-1, 4)
            relations = torch.tensor(row['relations'])
            metric.update(
                pred_boxes=torch.tensor(row['boxes']).reshape(-1, 4),
                final_scores=torch.tensor(row['final_scores']),
                quality_scores=torch.tensor(row['quality_scores']),
                alignment_scores=torch.tensor(row['alignment_scores']),
                gt_boxes=objects[relations == 1], ignored_boxes=objects[relations < 0],
                selected_boxes=torch.tensor(row['selected_boxes']).reshape(-1, 4))
    source = args.predictions.parent / 'summary.json'
    provenance = json.loads(source.read_text()) if source.exists() else {}
    result = dict(predictions=str(args.predictions), images=len(images),
                  checkpoint=provenance.get('checkpoint'),
                  partial=provenance.get('partial'), metrics=metric.compute())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
