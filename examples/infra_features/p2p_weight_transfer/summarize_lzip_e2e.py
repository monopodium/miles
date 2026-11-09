"""Summarize the end-to-end RL runs of the LZip-P2P experiment from miles driver logs.

miles prints one ``perf <rollout_id>: {...}`` line per rollout step from the trainer and one from the rollout
manager. This merges them per step, drops warm-up steps (the first sync also connects, builds the replicas and
registers memory), and prints the mean time of every phase plus the weight-update share of the step.

    python summarize_lzip_e2e.py --skip 2 baseline=logs/uccl_none.log lzip=logs/uccl_split.log [mooncake=...]
"""

import argparse
import ast
import re
import statistics
from collections import defaultdict

_PERF_LINE = re.compile(r"perf (\d+): (\{.*\})\s*$")
_KEYS = [
    "perf/update_weights_implementation_time",
    "perf/finalize_and_resume_engines_time",
    "perf/rollout_time",
    "perf/actor_train_time",
    "perf/train_time",
    "perf/train_wait_time",
    "perf/step_time",
]


def parse(path: str) -> dict[int, dict[str, float]]:
    steps: dict[int, dict[str, float]] = defaultdict(dict)
    with open(path, errors="replace") as f:
        for line in f:
            match = _PERF_LINE.search(line)
            if not match:
                continue
            try:
                values = ast.literal_eval(match.group(2))
            except (ValueError, SyntaxError):
                continue
            steps[int(match.group(1))].update(
                {key: float(value) for key, value in values.items() if isinstance(value, (int, float))}
            )
    return steps


def summarize(steps: dict[int, dict[str, float]], skip: int) -> dict[str, float]:
    kept = [steps[step] for step in sorted(steps)[skip:]]
    out = {"steps": len(kept)}
    for key in _KEYS:
        values = [step[key] for step in kept if key in step]
        if values:
            out[key] = statistics.mean(values)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=path/to/driver.log")
    ap.add_argument("--skip", type=int, default=2, help="warm-up steps to drop")
    args = ap.parse_args()

    results = {}
    for run in args.runs:
        label, path = run.split("=", 1)
        results[label] = summarize(parse(path), args.skip)

    keys = [key for key in _KEYS if any(key in result for result in results.values())]
    print(f"{'metric (s, mean)':<44}" + "".join(f"{label:>14}" for label in results))
    print(f"{'steps kept':<44}" + "".join(f"{result['steps']:>14}" for result in results.values()))
    for key in keys:
        print(f"{key:<44}" + "".join(f"{result.get(key, float('nan')):>14.3f}" for result in results.values()))
    labels = list(results)
    if len(labels) >= 2:
        base, new = results[labels[0]], results[labels[1]]
        for key in ("perf/update_weights_implementation_time", "perf/step_time"):
            if key in base and key in new and new[key] > 0:
                print(f"speedup {labels[1]} vs {labels[0]} on {key}: {base[key] / new[key]:.3f}x "
                      f"({(1 - new[key] / base[key]) * 100:.1f}% less time)")
        if "perf/step_time" in base and "perf/update_weights_implementation_time" in base:
            share = base["perf/update_weights_implementation_time"] / base["perf/step_time"]
            print(f"weight-update share of the {labels[0]} step: {share * 100:.1f}%")


if __name__ == "__main__":
    main()
