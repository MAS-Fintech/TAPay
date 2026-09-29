"""TAPay release entry point. Defaults to all three control agents enabled."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model"))
VARIANTS = {
    "full": ("on", "on", "on"),
    "backbone": ("off", "off", "off"),
    "without-intent": ("off", "on", "on"),
    "without-alignment": ("on", "off", "on"),
    "without-boundary": ("on", "on", "off"),
}

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    args, remaining = parser.parse_known_args()
    if "--help" in remaining or "-h" in remaining:
        print("TAPay --variant: " + ", ".join(VARIANTS) + " (default: full)\n")
    def present(flag):
        return any(x == flag or x.startswith(flag + "=") for x in remaining)
    for flag, value in zip(("--h1", "--h2", "--h3"), VARIANTS[args.variant]):
        if not present(flag):
            remaining += [flag, value]
    if not present("--config"):
        remaining += ["--config", str(ROOT / "config/config.ini")]
    if not present("--config-llm"):
        config = ROOT / "config/llm.ini"
        remaining += ["--config-llm", str(config if config.exists() else ROOT / "config/llm.example.ini")]
    sys.argv = [sys.argv[0], *remaining]
    from run_experiment import main as run
    return run()

if __name__ == "__main__":
    raise SystemExit(main())
