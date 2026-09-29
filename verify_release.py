"""Validate packaged data without invoking an LLM or reading credentials."""
from pathlib import Path
import collections
import hashlib
import json

ROOT = Path(__file__).resolve().parent

def verify():
    loaded = {}
    for entry in json.loads((ROOT / 'datasets/manifest.json').read_text(encoding='utf-8')):
        path = ROOT / 'datasets' / entry['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], path.name
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        assert len(rows) == entry['count'], path.name
        assert len({c['case_id'] for c in rows}) == len(rows), path.name
        review_counts = collections.Counter(c['review']['formal_acc_eligible'] for c in rows)
        assert review_counts[True] == {'master_311.jsonl': 311, 'hard_55.jsonl': 35, 'holdout_48.jsonl': 32}[path.name]
        print('  review eligible:', review_counts[True], 'not eligible:', review_counts[False])
        if path.name != 'master_311.jsonl':
            assert all(c['runtime'].get('execution_policy', {}).get('protocol') == 'round6-v1' for c in rows)
        loaded[path.name] = rows
        print(path.name, len(rows), dict(collections.Counter(c['split'] for c in rows)))
    master, hard, holdout = (loaded[n] for n in ('master_311.jsonl', 'hard_55.jsonl', 'holdout_48.jsonl'))
    assert not ({c['case_id'] for c in master} & {c['case_id'] for c in holdout})
    def user_text(c):
        return '\n'.join(t['content'].strip() for t in c['input']['conversation'] if t['role'] == 'user')
    assert not ({user_text(c) for c in master} & {user_text(c) for c in holdout})
    print('Checksums, counts, review eligibility, runtime policies and master/holdout ID/text separation: OK')
    return loaded

if __name__ == '__main__':
    verify()
