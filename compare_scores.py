"""Build an offline comparison to the user-supplied three-decimal target table."""
import html
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent


def normalize(s):
    return ' '.join(s.strip().lower().replace('by the therapist', 'by therapist').split())


def main():
    scorefile = ROOT / 'results/scores.json'
    results = json.loads(scorefile.read_text()) if scorefile.exists() else []
    rows = []
    for setting in ['zeroshot', 'oneshot']:
        reference_file = 'reference_scores_oneshot.csv' if setting == 'oneshot' else 'reference_scores.csv'
        refs = pd.read_csv(ROOT / reference_file)
        for ref in refs.to_dict('records'):
            row = next((r for r in results if setting in r['model'] and normalize(r['feature']) == normalize(ref['feature'])), {})
            for metric in ['bleu', 'meteor', 'rouge', 'bertscore', 'bleurt', 'infolm']:
                value = row.get(metric)
                rows.append({'setting': setting, 'aspect': ref['feature'], 'metric': metric,
                    'reference': ref[metric], 'ours': value,
                    'delta_ours_minus_reference': value-ref[metric] if value is not None else None,
                    'matches_displayed_3_decimals': round(value, 3) == ref[metric] if value is not None else None,
                    'evaluated_rows': row.get('sessions'),
                    'status': 'scored' if value is not None else row.get('errors', {}).get(metric, 'pending')})
    dest = ROOT / 'results'
    dest.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(dest / 'comparison.csv', index=False)
    df.to_excel(dest / 'comparison.xlsx', index=False)
    (dest / 'comparison.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<title>iCARE Gemma reproduction comparison</title><style>
body{font:16px/1.5 system-ui;background:#10151c;color:#eef;padding:2rem}
table{border-collapse:collapse}th,td{padding:.5rem;border:1px solid #567;text-align:left}
th{position:sticky;top:0;background:#234}a{color:#acf}</style>
<h1>Gemma 7B IT: iCARE reproduction</h1>
<p>References: two user-supplied tables, rounded to three decimals. We provisionally assign
the first table to zero-shot and the second table to one-shot; table captions have not been confirmed.
Zero-shot and one-shot are reported separately. Delta is ours minus reference; lower InfoLM is better.</p>
<p>Pending means no score yet. A three-decimal match is not proof of identical outputs.
This is a saved offline snapshot.</p>''' + df.to_html(index=False, na_rep='pending', escape=True) + '</html>')
    print('Comparison:', dest / 'comparison.html')


if __name__ == '__main__':
    main()
