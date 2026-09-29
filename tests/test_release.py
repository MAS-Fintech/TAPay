"""Checks for release packaging and entry-point defaults (no provider calls)."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]

def test_release_checksums_and_visibility():
    spec = importlib.util.spec_from_file_location('verify_release', ROOT / 'verify_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = module.verify()
    from payment.cases import load_cases
    for name in rows:
        assert len(load_cases(str(ROOT / 'datasets' / name))) == len(rows[name])

def test_entry_point_full_and_ablation(monkeypatch):
    spec = importlib.util.spec_from_file_location('tapay_main', ROOT / 'main.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import run_experiment
    seen = []
    monkeypatch.setattr(run_experiment, 'main', lambda: seen.append(list(sys.argv)) or 0)
    for variant, expected in module.VARIANTS.items():
        monkeypatch.setattr(sys, 'argv', ['main.py', '--variant', variant, '--input', 'x', '--output', 'y'])
        assert module.main() == 0
        args = seen[-1]
        assert tuple(args[args.index(f) + 1] for f in ('--h1', '--h2', '--h3')) == expected
