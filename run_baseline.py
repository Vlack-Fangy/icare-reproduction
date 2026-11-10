"""Checkpointed execution of the upstream Gemma test baseline.

Loads upstream definitions verbatim except the local model path/device. Keeps
upstream character-sliced output for comparability and saves token-sliced output
alongside it to expose decoding boundary differences.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--setting', choices=['zeroshot', 'oneshot'], required=True)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--anonymized-root', type=Path)
    ap.add_argument('--anonymize-example', action='store_true')
    ap.add_argument('--results-dir', type=Path)
    args = ap.parse_args()
    if args.anonymized_root:
        args.anonymized_root = args.anonymized_root.resolve()
    if args.results_dir:
        args.results_dir = args.results_dir.resolve()
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu is None or not gpu.isdigit():
        raise SystemExit('Expose exactly one free GPU with CUDA_VISIBLE_DEVICES=<index>.')
    os.chdir(ROOT / 'upstream')
    import pandas as pd
    import torch
    import transformers

    if not Path('train.xlsx').exists() or not Path('test.xlsx').exists():
        runpy.run_path('Train_Test_Builder.py', run_name='__main__')
    anonymous_dialogs, input_provenance = {}, {}
    if args.anonymized_root:
        from anonymized_inputs import load_batch
        ids = [('test', int(s)) for s in pd.read_excel('test.xlsx')['Summary Number'].unique()]
        if args.anonymize_example:
            ids += [('train', int(s)) for s in pd.read_excel('train.xlsx')['Summary Number'].unique()]
        anonymous_dialogs, input_provenance = load_batch(args.anonymized_root, ids, ROOT / 'upstream/Data')
    elif args.anonymize_example:
        raise ValueError('--anonymize-example requires --anonymized-root')
    code = Path('Baselines.py').read_text()
    code = code[:code.index('Summ = [')]
    code = code.replace("device = 'cuda:3'", "device = 'cuda:0'")
    code = code.replace('/home/models/gemma-7b-it', str(ROOT / 'models/gemma-7b-it'))
    sys.argv = ['Baselines.py', 'Gemma', args.setting, 'test']
    scope = {'__name__': 'upstream_baseline'}
    exec(compile(code, 'upstream/Baselines.py', 'exec'), scope)
    if anonymous_dialogs:
        original_dialog = scope['CSVtoStringDialog']
        def experiment_dialog(sid):
            return anonymous_dialogs[int(sid)] if int(sid) in anonymous_dialogs else original_dialog(sid)
        scope['CSVtoStringDialog'] = experiment_dialog
    output = args.results_dir or ROOT / ('results_anonymized' if args.anonymized_root else 'results')
    output.mkdir(exist_ok=True)
    manifest = {
        'upstream_commit': 'ce36a39eeb6da1bfb1f572730ad1f0cf0f68f02b',
        'model': 'google/gemma-7b-it', 'setting': args.setting, 'split': 'test',
        'gpu': int(gpu), 'torch': torch.__version__, 'transformers': transformers.__version__,
        'max_new_tokens': 100, 'do_sample': False, 'dtype': 'bfloat16',
        'output_extraction': 'upstream character slice; token slice also recorded',
        'input_condition': 'anonymized' if args.anonymized_root else 'original',
        'one_shot_example_condition': 'anonymized transcript with original expert answer' if args.anonymize_example else 'original',
        'input_provenance': input_provenance,
        'train_order': scope['df_train']['Summary Number'].tolist(),
        'test_order': scope['df_test']['Summary Number'].tolist(),
        'upstream_baseline_sha256': hashlib.sha256(Path('Baselines.py').read_bytes()).hexdigest(),
    }
    (output / f'{args.setting}-manifest.json').write_text(json.dumps(manifest, indent=2))
    checkpoint = output / f'{args.setting}.jsonl'
    saved = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            item = json.loads(line)
            saved[(item['session'], item['aspect'])] = item
    df = scope['df_test'].copy()
    if args.limit:
        df = df.head(args.limit).copy()
    features = scope['features']
    tokenizer, model = scope['tokenizer'], scope['model']
    for sid in df['Summary Number']:
        sid = int(sid)
        prompts = scope['PromptGen'](sid, args.setting)
        for feature, prompt in zip(features, prompts):
            key = (sid, feature)
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            if key in saved:
                if saved[key]['prompt_sha256'] != prompt_hash:
                    raise RuntimeError('Checkpoint prompt mismatch; do not mix experiments')
                continue
            start = time.monotonic()
            inputs = tokenizer(prompt, return_tensors='pt').to(scope['device'])
            with torch.no_grad():
                tokens = model.generate(**inputs, max_new_tokens=100, do_sample=False,
                                        pad_token_id=tokenizer.pad_token_id)
            legacy = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0][len(prompt):]
            token_text = tokenizer.decode(tokens[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            item = {'session': sid, 'aspect': feature, 'prediction': legacy,
                    'token_prediction': token_text, 'prompt_sha256': prompt_hash,
                    'input_tokens': inputs['input_ids'].shape[1],
                    'over_model_context': inputs['input_ids'].shape[1] + 100 > model.config.max_position_embeddings,
                    'seconds': time.monotonic() - start, 'gpu': int(gpu)}
            with checkpoint.open('a') as f:
                f.write(json.dumps(item) + '\n')
                f.flush()
            saved[key] = item
            print(f'{args.setting} session={sid} aspect={features.index(feature)+1}/17 '
                  f'saved={len(saved)} elapsed={item["seconds"]:.1f}s', flush=True)
        for feature in features:
            df.loc[df['Summary Number'] == sid, 'By Model ' + feature] = saved[(sid, feature)]['prediction']
        done = df[df['Summary Number'].map(lambda s: all((int(s), f) in saved for f in features))]
        dest = output / f'Gemma_doctors_prompts_{args.setting}.xlsx'
        temp = dest.with_name(dest.stem + '.tmp.xlsx')
        done.to_excel(temp, index=False)
        temp.replace(dest)
    print(f'COMPLETE {args.setting}: {len(df)} sessions', flush=True)


if __name__ == '__main__':
    main()
