"""Build and run the authorized 23-group E2 v2 pilot; old artifacts are read-only.

Build uses only the standard library. Run reuses E2's prompt, loader and generation
function without changing their settings. Analysis is performed separately.
"""

import argparse
import copy
import hashlib
import json
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'data/e2_samples_v2_pilot.json'
PROVENANCE = ROOT / 'results/e2_v2_pilot_provenance.json'
CONDITIONS = ['sufficient', 'insufficient', 'conflicting']
EXCLUDED = {'G18', 'G22', 'G23', 'G25', 'G26', 'G27', 'G30'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def build():
    if DATASET.exists() or PROVENANCE.exists():
        raise FileExistsError('Pilot inputs/provenance already exist; refusing to overwrite.')
    audit_path = ROOT / 'results/e21_conflict_adjudication.md'
    audit = audit_path.read_text(encoding='utf-8')
    rows = [[c.strip() for c in line.split('|')[1:-1]]
            for line in audit.splitlines() if re.match(r'^\| G\d{2} \|', line)]
    assert len(rows) == 30 and len({r[0] for r in rows}) == 30
    held = {r[0] for r in rows if r[2] == '暂不用'}
    assert held == EXCLUDED
    groups = sorted(r[0] for r in rows if r[0] not in held)
    assert len(groups) == 23
    v1 = read(ROOT / 'data/e2_samples_v1.json')['items']
    v2 = read(ROOT / 'data/e2_conflict_v2.json')['items']
    original = {(x['group_id'], x['condition']): x for x in v1}
    conflicts = {x['group_id']: x for x in v2}
    assert len(original) == len(v1) == 90
    assert len(conflicts) == len(v2) == 30
    items = []
    for gid in groups:
        for condition in CONDITIONS:
            item = copy.deepcopy(original[gid, condition])
            source = 'data/e2_samples_v1.json'
            if condition == 'conflicting':
                revised = conflicts[gid]
                for field in ['question', 'core_gold', 'core_other', 'main_relation']:
                    assert item[field] == revised[field], (gid, field)
                assert item['gold_answer'] == revised['gold']
                assert item['other_answer'] == revised['other']
                assert item['evidence'].split('；')[0] == revised['clause1']
                assert original[gid, 'sufficient']['evidence'].split('；')[0] == revised['clause1']
                item['evidence'] = revised['evidence']
                item['conflict_pattern'] = revised['pattern']
                source = 'data/e2_conflict_v2.json'
            item['source_file'] = source
            items.append(item)
        assert len({x['question'] for x in items[-3:]}) == 1
    assert len(items) == len({x['item_id'] for x in items}) == 69
    assert all(sum(x['condition'] == c for x in items) == 23 for c in CONDITIONS)
    sources = ['data/e2_samples_v1.json', 'data/e2_conflict_v2.json',
               'results/e21_conflict_adjudication.md', 'notes/e2_v2_behavior_retest_plan.md']
    source_hashes = {p: sha(ROOT / p) for p in sources}
    protected = set(sources)
    for folder in ['data', 'results']:
        for path in (ROOT / folder).iterdir():
            if path.is_file() and path.suffix in {'.json', '.csv', '.md'}:
                if not path.name.startswith('e2_v2_pilot') and 'e21' not in path.name:
                    protected.add(path.relative_to(ROOT).as_posix())
    provenance = {
        'created_utc': stamp(), 'authorization': 'User: 开始吧 (2026-11-19)',
        'scope': '23-group paired behavior pilot, no training or hidden-state extraction',
        'protected_sha256': {p: sha(ROOT / p) for p in sorted(protected)},
        'source_sha256': source_hashes,
    }
    dataset = {
        'experiment': 'e2_evidence_state_v2_pilot', 'created_utc': stamp(),
        'n_items': 69, 'n_groups': 23, 'included_groups': groups,
        'excluded_groups': sorted(EXCLUDED), 'condition_order': CONDITIONS,
        'source_sha256': source_hashes,
        'audit_source': 'Model-assisted checks and conversational review; no independent human agreement evidence.',
        'note': 'Development diagnostic set. gold/other are inherited institution identifiers, not a unique correct answer under conflict.',
        'items': items,
    }
    save(DATASET, dataset)
    provenance['dataset_sha256'] = sha(DATASET)
    save(PROVENANCE, provenance)
    print(f'Built {len(items)} items / {len(groups)} groups; all pairing/source assertions passed.')


def verify_protected():
    record = read(PROVENANCE)
    changed = [p for p, digest in record['protected_sha256'].items() if sha(ROOT / p) != digest]
    assert not changed, f'Protected files changed: {changed}'
    assert sha(DATASET) == record['dataset_sha256']
    return record


def run(smoke):
    provenance = verify_protected()
    dataset = read(DATASET)
    items = dataset['items'][:6] if smoke else dataset['items']
    name = 'e2_v2_pilot_smoke.json' if smoke else 'e2_v2_pilot_responses.json'
    output = ROOT / 'results' / name
    if output.exists():
        raise FileExistsError(f'{output} exists; refusing to overwrite/retry automatically.')
    sys.path.insert(0, str(ROOT / 'experiments'))
    import torch
    import transformers
    from e2_run_inference import PROMPT_TEMPLATE, build_user_content, generate_batch
    from hidden_states_sufficiency import DEFAULT_MODEL, check_model_files, load_model_and_tokenizer

    torch.manual_seed(42)
    check_model_files(DEFAULT_MODEL)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable.')
    print('Loading local Qwen2.5-3B-Instruct (bfloat16)...', flush=True)
    tokenizer, model = load_model_and_tokenizer(DEFAULT_MODEL, torch.bfloat16)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    record = {
        'experiment': dataset['experiment'], 'stage': 'smoke' if smoke else 'full',
        'timestamp_utc': stamp(), 'script': 'experiments/e2_v2_pilot.py',
        'script_sha256': sha(Path(__file__)),
        'reused_scripts_sha256': {p: sha(ROOT / p) for p in
            ['experiments/e2_run_inference.py', 'experiments/hidden_states_sufficiency.py']},
        'samples_file': str(DATASET), 'samples_sha256': provenance['dataset_sha256'],
        'model_path': str(DEFAULT_MODEL), 'model_config_sha256': sha(DEFAULT_MODEL / 'config.json'),
        'decoding': 'greedy (do_sample=False, repetition_penalty=1.0)',
        'dtype': str(next(model.parameters()).dtype), 'seed': 42, 'batch_size': 12,
        'max_new_tokens': 48, 'prompt_template': PROMPT_TEMPLATE,
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'python': platform.python_version(), 'executable': sys.executable,
        'gpu': torch.cuda.get_device_name(0), 'planned_n_items': len(items),
        'complete': False, 'generations': [],
    }
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    for start in range(0, len(items), 12):
        batch = items[start:start + 12]
        contents = [build_user_content(item) for item in batch]
        results = generate_batch(model, tokenizer, contents, 48)
        assert len(results) == len(batch)
        for item, content, result in zip(batch, contents, results):
            record['generations'].append({**item, 'user_content': content, **result})
            print(f"{item['item_id']}: {result['answer_raw']}", flush=True)
        record['n_items'] = len(record['generations'])
        save(output, record)
    record.update(complete=True, elapsed_seconds=round(time.perf_counter() - start_time, 3),
                  max_peak_gb=round(torch.cuda.max_memory_allocated() / 1024**3, 3))
    assert len(record['generations']) == len(items)
    assert not any(x['truncated'] for x in record['generations']), 'Truncated outputs; inspect before proceeding.'
    verify_protected()
    save(output, record)
    print(f"Saved {output.name}: {record['n_items']} complete responses, {record['elapsed_seconds']}s, {record['max_peak_gb']} GB.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['build', 'smoke', 'run', 'verify'])
    action = parser.parse_args().action
    if action == 'build':
        build()
    elif action == 'verify':
        verify_protected()
        print('Protected input/historical output hashes unchanged.')
    else:
        run(smoke=action == 'smoke')
