"""Build train / held-out task-id splits per LIBERO-Plus perturbation category.

LIBERO-Plus ships ``libero/libero/benchmark/task_classification.json`` mapping
tasks to perturbation categories (camera, robot init, language, light,
background, noise, layout) and difficulty. The exact JSON layout is not
documented, so this script accepts the common shapes and prints what it found;
check the printed summary in Week 1.

Output: ``<out>/<suite>_<category>.yaml`` with ``train_task_ids`` and
``heldout_task_ids`` usable as ``env.train.task_id_filter`` /
``env.eval.task_id_filter`` overrides.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path


def find_classification_file() -> Path:
    import liberoplus.liberoplus as lp

    root = Path(lp.__file__).resolve().parent
    for path in root.rglob("task_classification.json"):
        return path
    raise FileNotFoundError(f"task_classification.json not found under {root}")


def entries(data):
    """Yield (task_name, category, difficulty) from the supported layouts."""
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                cat = value.get("category") or value.get("type") or value.get("perturbation")
                diff = value.get("difficulty") or value.get("level")
                if cat is not None:
                    yield key, str(cat), diff
                else:  # {category: [task names]}
                    for name in value.get("tasks", []):
                        yield name, str(key), None
            elif isinstance(value, list):  # {category: [task names]}
                for name in value:
                    yield (name if isinstance(name, str) else name.get("name")), str(key), None
    elif isinstance(data, list):
        for item in data:
            name = item.get("name") or item.get("task") or item.get("task_name")
            cat = item.get("category") or item.get("type") or item.get("perturbation")
            yield name, str(cat), item.get("difficulty") or item.get("level")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser.add_argument("--heldout-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--classification", default=None)
    args = parser.parse_args()

    from liberoplus.liberoplus import benchmark

    path = Path(args.classification) if args.classification else find_classification_file()
    data = json.loads(path.read_text())
    name_to_cat = {}
    for name, cat, _diff in entries(data):
        if name:
            name_to_cat[str(name)] = cat
    print(f"[splits] {path}: {len(name_to_cat)} classified tasks, "
          f"categories={sorted(set(name_to_cat.values()))}")

    os.makedirs(args.out, exist_ok=True)
    bench = benchmark.get_benchmark_dict()
    rng = random.Random(args.seed)
    for suite_name in args.suites:
        suite = bench[suite_name]()
        by_cat = defaultdict(list)
        for task_id in range(suite.n_tasks):
            name = str(suite.get_task(task_id).name)
            by_cat[name_to_cat.get(name, "unclassified")].append(task_id)
        for cat, ids in sorted(by_cat.items()):
            ids = sorted(ids)
            rng.shuffle(ids)
            n_held = max(1, int(round(len(ids) * args.heldout_frac)))
            held, train = sorted(ids[:n_held]), sorted(ids[n_held:])
            out = Path(args.out) / f"{suite_name}_{cat}.yaml"
            out.write_text(
                f"# {suite_name} / {cat}: {len(train)} train, {len(held)} held-out\n"
                f"train_task_ids: {train}\nheldout_task_ids: {held}\n"
            )
            print(f"[splits] {out.name}: train={len(train)} heldout={len(held)}")


if __name__ == "__main__":
    main()
