"""Vendor the same Hugging Face metric implementations for Evaluate 0.4.2."""
from pathlib import Path
import urllib.request
import hashlib
import json
from huggingface_hub import HfApi

root = Path(__file__).resolve().parent / 'metrics'
files = ['bleu/bleu.py', 'bleu/nmt_bleu.py', 'bleu/tokenizer_13a.py', 'meteor/meteor.py', 'rouge/rouge.py',
         'bertscore/bertscore.py', 'bleurt/bleurt.py']
manifest = {}
for name in files:
    metric, filename = name.split('/')
    revision = HfApi().repo_info('evaluate-metric/' + metric, repo_type='space').sha
    url = f'https://huggingface.co/spaces/evaluate-metric/{metric}/resolve/{revision}/{filename}'
    if filename == 'nmt_bleu.py':
        url = 'https://raw.githubusercontent.com/tensorflow/nmt/master/nmt/scripts/bleu.py'
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data = urllib.request.urlopen(url, timeout=60).read()
    path.write_bytes(data)
    manifest[name] = {'url': url, 'sha256': hashlib.sha256(data).hexdigest()}
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print('Saved metric sources', flush=True)
