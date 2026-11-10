"""Read verified anonymizer drafts without falling back to original text."""
import hashlib
import csv
import json
from pathlib import Path


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def load_dialog(directory):
    directory = Path(directory)
    doc = json.loads((directory / 'document.json').read_text())
    source = (directory / 'source.txt').read_text()
    candidate = (directory / 'output.txt').read_text()
    analysis = doc['analysis']
    if analysis.get('transformation', {}).get('status') != 'complete':
        raise ValueError(f'Incomplete anonymization: {directory}')
    if digest(source) != doc['source_sha256']:
        raise ValueError(f'Source hash mismatch: {directory}')
    revision = analysis['transformation']['draft_revision']
    draft = next(d for d in analysis['drafts'] if d['revision_id'] == revision)
    if draft['text_sha256'] != digest(candidate) or draft['status'] != 'complete':
        raise ValueError(f'Candidate does not match verified revision: {directory}')
    by_id = {e['edit_id']: e for e in doc['edits']}
    edits = sorted([by_id[e] for e in draft['applied_edit_ids']], key=lambda e:e['char_start'])
    rebuilt = source
    for e in reversed(edits):
        rebuilt = rebuilt[:e['char_start']] + e['replacement'] + rebuilt[e['char_end']:]
    if rebuilt != candidate:
        raise ValueError(f'Applied edits do not reconstruct saved candidate: {directory}')
    dialog, previous = '', ''
    assigned = set()
    omitted_speakers = {}
    for utterance in doc['utterances']:
        start, end = utterance['char_start'], utterance['char_end']
        text = source[start:end]
        local = [e for e in edits if start <= e['char_start'] and e['char_end'] <= end]
        for e in reversed(local):
            text = text[:e['char_start']-start] + e['replacement'] + text[e['char_end']-start:]
            assigned.add(e['edit_id'])
        speaker = utterance['speaker']
        if speaker not in ('SPEAKER_00','SPEAKER_01'):
            # Match upstream CSVtoStringDialog, which only emits these two
            # labels and leaves `old` unchanged when another speaker occurs.
            # The anonymizer still processes every turn; record this existing
            # evaluation limitation explicitly rather than silently hiding it.
            omitted_speakers[speaker] = omitted_speakers.get(speaker, 0) + 1
            continue
        if speaker != previous:
            dialog += '#Therapist#: ' if speaker == 'SPEAKER_00' else '#Patient#: '
        dialog += text + ' '
        previous = speaker
    if assigned != set(draft['applied_edit_ids']):
        raise ValueError(f'Edit crosses an utterance boundary: {directory}')
    return dialog, {'directory':str(directory.resolve()), 'source_sha256':digest(source),
                    'candidate_sha256':digest(candidate), 'revision':revision,
                    'applied_edit_count':len(edits), 'transformation_status':'complete',
                    'upstream_formatter_omitted_speaker_turns':omitted_speakers}


def load_batch(root, ids, original_root=None):
    root = Path(root)
    batch = json.loads((root / 'batch.json').read_text())
    cases = {c['id']:c for c in batch['cases']}
    dialogs, provenance = {}, {}
    for split, sid in ids:
        key = f'icare-{split}-{sid}'
        if key not in cases:
            raise ValueError(f'Missing anonymizer case: {key}')
        directory = root / cases[key]['directory']
        dialogs[int(sid)], provenance[key] = load_dialog(directory)
        if original_root is not None:
            original = Path(original_root) / split / str(sid) / f'{sid}.csv'
            with original.open(encoding='utf-8-sig', newline='') as stream:
                source_rows = [(r['speaker'].strip(), r['text'].strip())
                               for r in csv.DictReader(stream) if r.get('text', '').strip()]
            if digest('\n'.join(text for _, text in source_rows)) != provenance[key]['source_sha256']:
                raise ValueError(f'Anonymizer source differs from upstream session {key}')
            doc = json.loads((directory / 'document.json').read_text())
            if [u['speaker'] for u in doc['utterances']] != [speaker for speaker, _ in source_rows]:
                raise ValueError(f'Anonymizer speaker sequence differs for {key}')
    return dialogs, provenance
