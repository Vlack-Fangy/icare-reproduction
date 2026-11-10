"""Upstream metric settings with the pandas Series.lower bug corrected.

One result per aspect; no invented overall 'iCARE score'. Metric failures are
recorded, never silently replaced by zero. Re-running retries missing metrics.
"""
import json
import argparse
import hashlib
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
# Scoring stays on CPU so TensorFlow cannot reserve another GPU.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_USE_LEGACY_KERAS'] = '1'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    args = ap.parse_args()
    import evaluate
    import nltk
    from torchmetrics.text.infolm import InfoLM
    for name in ['punkt_tab', 'wordnet', 'omw-1.4']:
        nltk.download(name, quiet=True, raise_on_error=True)
    metric_names = ['bleu', 'meteor', 'rouge', 'bertscore', 'bleurt']
    metrics = {}
    failures = {}
    for name in metric_names:
        try:
            metrics[name] = evaluate.load(str(ROOT / 'metrics' / name / (name + '.py')))
        except Exception as e:
            failures[name] = str(e)
    try:
        metrics['infolm'] = InfoLM('google/bert_uncased_L-2_H-128_A-2',
            information_measure='fisher_rao_distance', idf=False)
    except Exception as e:
        failures['infolm'] = str(e)
    dest = args.results_dir / 'scores.json'
    rows = json.loads(dest.read_text()) if dest.exists() else []
    for path in sorted(args.results_dir.glob('Gemma_*.xlsx')):
        df = pd.read_excel(path).fillna('Nil')
        features = list(df.columns[1:18])
        for feature in features:
            refs = df[feature].astype(str).str.lower().tolist()
            preds = df['By Model ' + feature].astype(str).str.lower().tolist()
            fingerprint = hashlib.sha256(json.dumps([refs, preds], ensure_ascii=False).encode()).hexdigest()
            row = next((r for r in rows if r['model'] == path.name and r['feature'] == feature), None)
            if row is None:
                row = {'model': path.name, 'feature': feature, 'sessions': len(df), 'errors': {}}
                rows.append(row)
            if row['sessions'] != len(df) or row.get('input_sha256') != fingerprint:
                row.clear()
                row.update(model=path.name, feature=feature, sessions=len(df), errors={}, input_sha256=fingerprint)
            for name in metric_names + ['infolm']:
                if name in row:
                    continue
                try:
                    if name in failures:
                        raise RuntimeError(failures[name])
                    metric = metrics[name]
                    if name == 'infolm':
                        value = metric(preds, refs).item()
                        metric.reset()
                    elif name == 'bertscore':
                        value = np.mean(metric.compute(predictions=preds, references=refs,
                            lang='en', model_type='roberta-large', device='cpu')['f1'])
                    elif name == 'bleurt':
                        value = np.mean(metric.compute(predictions=preds, references=refs)['scores'])
                    else:
                        result = metric.compute(predictions=preds, references=refs)
                        value = result[{'bleu':'bleu', 'meteor':'meteor', 'rouge':'rougeL'}[name]]
                    row[name] = float(value)
                    row['errors'].pop(name, None)
                except Exception as e:
                    row['errors'][name] = str(e)
                dest.write_text(json.dumps(rows, indent=2))
                print(path.name, feature, name, row.get(name, 'FAILED'), flush=True)
    pd.DataFrame([{k:v for k,v in row.items() if k != 'errors'} for row in rows]).to_excel(
        args.results_dir / 'scores.xlsx', index=False)
    if any(row['errors'] for row in rows):
        raise SystemExit('Some metrics failed; see results/scores.json. No substitute scores reported.')


if __name__ == '__main__':
    main()
