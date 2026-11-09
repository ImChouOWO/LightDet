"""Print the latest gRefCOCO epoch metrics."""
import argparse
import json
from pathlib import Path

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metrics',type=Path)
    args=parser.parse_args()
    lines=args.metrics.read_text().splitlines()
    if not lines:raise ValueError('No completed epochs')
    row=json.loads(lines[-1]);print('Epoch:',row['epoch'])
    for name in ('grec_precision_f1','rec_acc50','n_acc','t_acc'):
        value=row['validation'][name]
        print(name, 'N/A' if value is None else f'{value:.4f}')
