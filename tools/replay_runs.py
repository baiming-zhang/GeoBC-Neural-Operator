"""Replay the completed DeepJEB experiments into a separate output directory.

Use ``python tools/replay_runs.py --dry-run`` to inspect all eight stages without
starting training. The commands preserve the saved seed, ordered case keys,
sampling, learning-rate override flags, and best/last checkpoint lineage.
CUDA reductions may vary numerically between executions or environments.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = (
    "runs/hashgrid_single_1000",
    "runs/hashgrid_single_refine",
    "runs/hashgrid_five_1000",
    "runs/hashgrid_five_refine",
    "runs/hashgrid_five_converge",
    "runs/hashgrid_five_low_lr",
    "runs/hashgrid_five_tail",
    "runs/hashgrid_five_tail_final",
)
REFERENCE_HASH_GRID = {
    "n_levels": 12, "n_features_per_level": 2, "log2_hashmap_size": 19,
    "base_resolution": 8, "per_level_scale": 1.6,
}


@dataclass(frozen=True)
class ReplayStage:
    name: str
    source: Path
    output: Path
    resume: Path | None
    command: list[str]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _source_directory(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        candidate = PROJECT_ROOT / path
        # Permit either "runs/stage" or a bare stage name in --runs.
        path = PROJECT_ROOT / "runs" / path if len(path.parts) == 1 and not candidate.exists() else candidate
    return path.resolve()


def _recorded_parts(value: str | Path) -> PurePosixPath:
    # Configs contain WSL absolute paths; this also permits Windows dry runs.
    return PurePosixPath(str(value).replace("\\", "/"))


def _overlap(first: Path, second: Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def _protect_outputs(output_root: Path, targets: list[Path], sources: list[Path]) -> None:
    protected = set(sources)
    for summary in (PROJECT_ROOT / "runs").glob("*/summary.json"):
        if _read_json(summary).get("status") == "completed":
            protected.add(summary.parent.resolve())
    for output in [output_root, *targets]:
        for source in protected:
            if _overlap(output, source):
                raise ValueError(f"Replay output overlaps completed source results: {output} and {source}")
        for parent in [output, *output.parents]:
            summary = parent / "summary.json"
            if summary.is_file() and _read_json(summary).get("status") == "completed":
                raise ValueError(f"Replay output is inside completed results: {output}")
    for target in targets:
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            raise FileExistsError(f"Replay stage output is already nonempty; choose a fresh --output-root: {target}")


def _model_arguments(config: dict[str, Any]) -> dict[str, Any]:
    arguments = dict(config["arguments"])
    model = config["model"]
    for key in ("encoder", "input_mode", "hidden_dim", "hidden_layers"):
        effective = model[key]
        recorded = arguments.get(key)
        if recorded is None or (key == "input_mode" and recorded == "auto"):
            arguments[key] = effective
        elif recorded != effective:
            raise ValueError(f"Saved CLI/model disagreement for {key}: {recorded!r} versus {effective!r}")
    if model.get("condition_dim", 25) != 25:
        raise ValueError("The training CLI only supports the prepared 25-channel feature contract")
    if model.get("hash_grid", REFERENCE_HASH_GRID) != REFERENCE_HASH_GRID:
        raise ValueError("The training CLI cannot replay nondefault hash-grid parameters")
    return arguments


def _cli_arguments(arguments: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for key, value in arguments.items():
        flag = "--" + key.replace("_", "-")
        if value is None or value is False:
            continue
        if value is True:
            result.append(flag)
        elif isinstance(value, (list, tuple)):
            if not value:
                raise ValueError(f"Saved argument {flag} has an empty list")
            result.extend([flag, *(str(item) for item in value)])
        else:
            result.extend([flag, str(value)])
    return result


def build_plan(run_dirs: list[str | Path], output_root: str | Path,
               python_executable: str | None = None) -> list[ReplayStage]:
    sources = [_source_directory(value) for value in run_dirs]
    if len(set(sources)) != len(sources):
        raise ValueError("--runs contains duplicate source directories")
    names = [path.name for path in sources]
    if len(set(names)) != len(names):
        raise ValueError("Replay stage directory names must be unique")
    root = Path(output_root).expanduser()
    root = (root if root.is_absolute() else PROJECT_ROOT / root).resolve()
    targets = [(root / name).resolve() for name in names]
    _protect_outputs(root, targets, sources)
    configs = [_read_json(path / "config.json") for path in sources]
    for source in sources:
        if _read_json(source / "summary.json").get("status") != "completed":
            raise ValueError(f"Source experiment is not completed: {source}")
    aliases: dict[str, int] = {}
    for index, (source, config) in enumerate(zip(sources, configs)):
        # The recorded basename remains valid after moving the repository or
        # accessing WSL-created configs from Windows; ambiguous names are rejected.
        recorded = config["arguments"].get("output")
        for alias in {source.name, _recorded_parts(recorded).name if recorded else source.name}:
            if alias in aliases and aliases[alias] != index:
                raise ValueError(f"Ambiguous saved output directory name: {alias}")
            aliases[alias] = index
    stages: list[ReplayStage] = []
    for index, (source, target, config) in enumerate(zip(sources, targets, configs)):
        arguments = _model_arguments(config)
        resume_value = config.get("resumed_from") or arguments.get("resume")
        resume: Path | None = None
        if resume_value:
            recorded_resume = _recorded_parts(resume_value)
            source_index = aliases.get(recorded_resume.parent.name)
            if source_index is None or source_index >= index:
                raise ValueError(f"Resume parent must occur earlier in --runs: {resume_value}")
            original_checkpoint = sources[source_index] / recorded_resume.name
            if not original_checkpoint.is_file():
                raise FileNotFoundError(f"Saved lineage checkpoint is missing: {original_checkpoint}")
            resume = targets[source_index] / recorded_resume.name
        arguments["output"] = str(target)
        arguments["resume"] = str(resume) if resume is not None else None
        command = [python_executable or sys.executable, "-m", "geobc.train", *_cli_arguments(arguments)]
        stages.append(ReplayStage(source.name, source, target, resume, command))
    return stages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS),
                        help="Completed source run directories in lineage order; default is all eight stages")
    parser.add_argument("--output-root", type=Path, default=Path("runs/replay"),
                        help="Fresh replay root; source results and nonempty stage outputs are protected")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without starting training or writing outputs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stages = build_plan(args.runs, args.output_root)
    for number, stage in enumerate(stages, 1):
        print(f"[{number}/{len(stages)}] {stage.name}", flush=True)
        print(f"  output: {stage.output}", flush=True)
        print(f"  resume: {stage.resume or '(fresh start)'}", flush=True)
        rendered = subprocess.list2cmdline(stage.command) if os.name == "nt" else shlex.join(stage.command)
        print(f"  {rendered}", flush=True)
        if not args.dry_run:
            subprocess.run(stage.command, cwd=PROJECT_ROOT, check=True)
    if args.dry_run:
        print("Dry run complete; no training started and no replay outputs written.", flush=True)


if __name__ == "__main__":
    main()
