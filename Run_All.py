#!/usr/bin/env python3
"""Run the self-contained similarity pipeline; all filesystem paths are configured here."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

BASE = Path(__file__).resolve().parent
# Resolve sibling stage modules even when Python is launched in isolated mode.
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
# fmt: off
# ===== Input paths: paste absolute paths or paths relative to this file =====
# The large trade archive is not bundled. Set its path here or use --archive.
trade_archive_path = ""
# Optional daily CSV directory; leave empty to read all trades from the archive.
trade_csv_directory_path = ""

optionmetrics_2016_2023_path = "assets/optionmetrics/vsurfd_30d_SPY_QQQ_IWM_2016_2023.csv"
optionmetrics_2020_2025_path = "assets/optionmetrics/vsurfd_30d_SPY_QQQ_IWM_2020_2025.csv"
rates_path = "assets/market_data/sofr.csv"
dividends_path = "assets/market_data/Dividends.xlsx"
table_shell_template_path = "assets/templates/main_shell_template.tex"
output_directory_path = "runs/baseline"
configuration_path = "config.py"
# ===== End of path configuration =====
# fmt: on

PATHS = {
    "output": output_directory_path,
    "config": configuration_path,
    "raw_dir": trade_csv_directory_path,
    "rates_file": rates_path,
    "dividends_file": dividends_path,
    "archive": trade_archive_path,
    "optionmetrics_early": optionmetrics_2016_2023_path,
    "optionmetrics_late": optionmetrics_2020_2025_path,
    "shell_template": table_shell_template_path,
}
STAGES = [
    "Build_Panel",
    "Build_Event_Panel",
    "Estimate_Lambda",
    "Measure_Similarity",
    "Structural_Model",
    "Reduced_Form",
    "Inference",
    "Make_Results",
]
DEPS = {
    "Build_Panel": [],
    "Build_Event_Panel": ["Build_Panel"],
    "Estimate_Lambda": ["Build_Event_Panel"],
    "Measure_Similarity": ["Build_Event_Panel"],
    "Structural_Model": ["Estimate_Lambda", "Measure_Similarity"],
    "Reduced_Form": ["Structural_Model"],
    "Inference": ["Reduced_Form"],
    "Make_Results": ["Inference"],
}


def resolve(value):
    if not str(value).strip():
        return None
    path = Path(value).expanduser()
    return (path if path.is_absolute() else BASE / path).resolve()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    temporary.replace(path)


@dataclass(frozen=True)
class Paths:
    output: Path
    config: Path
    raw_dir: Path | None
    rates_file: Path
    dividends_file: Path
    archive: Path | None
    optionmetrics_early: Path | None
    optionmetrics_late: Path | None
    shell_template: Path

    @property
    def data(self):
        return self.output / "data"

    @property
    def events(self):
        return self.data / "events"

    @property
    def results(self):
        return self.output / "results"

    @property
    def full(self):
        return self.data / "full"

    @property
    def shells(self):
        return self.output / "table_shells"

    @property
    def context(self):
        return self.output / "runtime_context.json"

    @property
    def manifest(self):
        return self.output / "manifest.json"

    @property
    def settings(self):
        return self.output / "settings.json"

    @property
    def log_file(self):
        return self.output / "run.log"

    @property
    def optionmetrics_files(self):
        return [
            p
            for p in [self.optionmetrics_early, self.optionmetrics_late]
            if p is not None
        ]

    @property
    def raw_data(self):
        return self.rates_file.parent

    def executable(self, name):
        location = shutil.which(name)
        if not location:
            raise FileNotFoundError(f"{name} is not on PATH")
        return location

    def shell_file(self, name):
        return self.shells / name

    def code_files(self):
        return sorted(BASE.glob("*.py")) + sorted(BASE.glob("*.R"))

    def input_files(self):
        return [
            self.config,
            self.shell_template,
            self.rates_file,
            self.dividends_file,
            *self.optionmetrics_files,
        ]


def runtime_paths():
    """Stages import this through config.py; importing a stage has no write side effects."""
    from types import SimpleNamespace

    context = os.environ.get("SIMDIV_PIPELINE_CONTEXT")
    if context:
        state = json.loads(Path(context).read_text())
        values = {
            key: (
                [Path(f) for f in value]
                if isinstance(value, list)
                else Path(value) if value is not None else None
            )
            for key, value in state["paths"].items()
        }
        values["raw_dir"] = values["raw_dir"] or BASE / "inputs" / "trades"
        values["raw_data"] = values["rates_file"].parent
        return SimpleNamespace(**values)
    paths = Paths(**{key: resolve(value) for key, value in PATHS.items()})
    values = dict(vars(paths))
    values.update(
        data=paths.data,
        events=paths.events,
        results=paths.results,
        full=paths.full,
        shells=paths.shells,
        optionmetrics_files=paths.optionmetrics_files,
        raw_dir=paths.raw_dir or BASE / "inputs" / "trades",
        raw_data=paths.raw_data,
    )
    return SimpleNamespace(**values)


def _execute(paths, log, command, label):
    log("Run " + label)
    proc = subprocess.Popen(
        command,
        cwd=paths.output,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=os.environ.copy(),
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
        with paths.log_file.open("a") as stream:
            stream.write(line)
    if proc.wait():
        raise RuntimeError(f"{label} exited {proc.returncode}; see {paths.log_file}")


def invoke(paths, log, stage, task, args=()):
    _execute(
        paths,
        log,
        [sys.executable, str(BASE / (stage + ".py")), "--task", task, *map(str, args)],
        stage + ":" + task,
    )


def invoke_r(paths, log, mode):
    _execute(
        paths,
        log,
        [
            paths.executable("Rscript"),
            str(BASE / "Reduced_Form.R"),
            str(paths.data),
            str(paths.results),
            mode,
        ],
        "Reduced_Form.R:" + mode,
    )


def reduced_form(paths, cfg, log):
    invoke(paths, log, "Measure_Similarity", "similarity_exposure")
    invoke(paths, log, "Measure_Similarity", "phantom")
    invoke_r(paths, log, "reduced_form")
    return (
        [paths.data / "reg_panel_rf.csv", paths.data / "did18_all.csv"]
        + sorted(paths.data.glob("*/did_panel.parquet"))
        + sorted(paths.results.glob("reduced_form_*"))
    )


def main(default_stage=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in PATHS.items():
        parser.add_argument("--" + key.replace("_", "-"), default=value)
    parser.add_argument(
        "--stage", choices=["all"] + STAGES, default=default_stage or "all"
    )
    parser.add_argument(
        "--force", action="store_true", help="Recompute the selected stages"
    )
    parser.add_argument(
        "--results-only",
        action="store_true",
        help="Refill shells from a completed numerical run",
    )
    parser.add_argument("--show-paths", action="store_true")
    args = parser.parse_args()
    paths = Paths(**{key: resolve(getattr(args, key)) for key in PATHS})
    if args.show_paths:
        print(
            json.dumps(
                {
                    key: str(value) if value is not None else None
                    for key, value in vars(paths).items()
                },
                indent=2,
            )
        )
        return
    protected = [BASE / "assets"]
    if paths.output == BASE or any(
        paths.output == p or paths.output.is_relative_to(p) for p in protected
    ):
        parser.error(
            "Output must be separate from code and inputs"
        )
    configuration = runpy.run_path(str(paths.config))
    cfg = dict(configuration["SETTINGS"])
    missing = [
        str(p)
        for p in paths.input_files()
        if p is None or not p.is_file()
    ]
    if not paths.optionmetrics_files:
        missing.append("optionmetrics_2016_2023_path / optionmetrics_2020_2025_path")
    if not args.results_only and (paths.archive is None or not paths.archive.is_file()):
        missing.append(
            "trade_archive_path (fill the quoted path in Run_All.py, or use --archive)"
        )
    if missing:
        parser.error("Missing inputs: " + ", ".join(missing))
    for directory in [
        paths.output,
        paths.data,
        paths.events,
        paths.results,
        paths.full,
        paths.shells,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    def log(message):
        line = f"{datetime.now(timezone.utc).isoformat()} {message}"
        print(line, flush=True)
        with paths.log_file.open("a") as stream:
            stream.write(line + "\n")

    runtime = {
        key: str(value) if value is not None else None
        for key, value in vars(paths).items()
    }
    runtime.update(
        data=str(paths.data),
        events=str(paths.events),
        results=str(paths.results),
        full=str(paths.full),
        shells=str(paths.shells),
        optionmetrics_files=[str(f) for f in paths.optionmetrics_files],
    )
    atomic_json(paths.context, {"paths": runtime, "config": str(paths.config)})
    os.environ["SIMDIV_PIPELINE_CONTEXT"] = str(paths.context)
    for name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]:
        os.environ[name] = "1"
    os.environ["MPLCONFIGDIR"] = str(paths.output / "cache" / "matplotlib")
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "code": {str(f): sha256(f) for f in paths.code_files()},
                "inputs": {
                    str(f): sha256(f) for f in paths.input_files()
                },
                "paths": runtime,
                "settings": cfg,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    previous = json.loads(paths.manifest.read_text()) if paths.manifest.exists() else {}
    if args.results_only:
        for stage in STAGES[:-1]:
            if previous.get("stages", {}).get(stage, {}).get("status") != "complete":
                parser.error("Complete numerical run required")
        from Make_Results import run

        products = run(paths, cfg, log)
        atomic_json(
            paths.shell_file("export_manifest.json"),
            {"outputs": {str(f): sha256(f) for f in products}},
        )
        return
    if previous.get("fingerprint") not in [None, fingerprint] and not args.force:
        parser.error("Run settings/code changed; choose a new output or --force")
    manifest = (
        previous
        if previous.get("fingerprint") == fingerprint and not args.force
        else {"fingerprint": fingerprint, "stages": {}}
    )
    atomic_json(
        paths.settings,
        {
            "settings": cfg,
            "events": configuration["EVENT_SPEC"],
            "paths": runtime,
            "note": "Self-contained implementations; original numerical specification retained.",
        },
    )
    needed = set()

    def add(stage):
        for dependency in DEPS[stage]:
            add(dependency)
        needed.add(stage)

    if args.stage == "all":
        needed = set(STAGES)
    else:
        add(args.stage)
    try:
        reran = set()
        for stage in STAGES:
            if stage not in needed:
                continue
            state = manifest["stages"].get(stage, {})
            intact = (
                state.get("status") == "complete"
                and bool(state.get("outputs"))
                and all(
                    Path(f).is_file() and sha256(Path(f)) == h
                    for f, h in state.get("outputs", {}).items()
                )
            )
            if intact and not any(dependency in reran for dependency in DEPS[stage]):
                log(stage + ": reuse verified outputs")
                continue
            log(stage + ": start")
            manifest["stages"][stage] = {"status": "running"}
            atomic_json(paths.manifest, manifest)
            products = (
                reduced_form(paths, cfg, log)
                if stage == "Reduced_Form"
                else importlib.import_module(stage).run(paths, cfg, log)
            )
            manifest["stages"][stage] = {
                "status": "complete",
                "outputs": {str(f): sha256(f) for f in products},
            }
            atomic_json(paths.manifest, manifest)
            reran.add(stage)
            log(stage + ": complete")
    except Exception as error:
        manifest["stages"][stage] = {"status": "failed", "error": str(error)}
        atomic_json(paths.manifest, manifest)
        raise


if __name__ == "__main__":
    main()
