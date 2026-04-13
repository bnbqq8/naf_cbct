from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import yaml


def safe_get(dct: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = dct
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def try_parse_from_name(exp_name: str) -> tuple[Any, Any, Any]:
    weight = None
    interval = None
    warmup = None

    m = re.search(r"sds([0-9]*\.?[0-9]+)", exp_name)
    if m:
        try:
            weight = float(m.group(1))
        except ValueError:
            pass

    m = re.search(r"interval(\d+)", exp_name)
    if m:
        interval = int(m.group(1))

    m = re.search(r"startFrom(\d+)", exp_name)
    if m:
        warmup = int(m.group(1))

    return weight, interval, warmup


def find_config_file(exp_dir: Path) -> Path | None:
    yaml_files = sorted(exp_dir.glob("*.yaml"))
    if yaml_files:
        return yaml_files[0]
    return None


def summarize_logs(logs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if not logs_dir.exists():
        raise FileNotFoundError(f"logs dir not found: {logs_dir}")

    for exp_dir in sorted(p for p in logs_dir.iterdir() if p.is_dir()):
        exp_name = exp_dir.name
        cfg_file = find_config_file(exp_dir)

        cfg: dict[str, Any] = {}
        if cfg_file is not None:
            try:
                cfg = yaml.safe_load(cfg_file.read_text()) or {}
            except Exception:
                cfg = {}

        w_name, i_name, wu_name = try_parse_from_name(exp_name)

        sds_weight = safe_get(cfg, ["train", "sds", "sds_weight"], w_name)
        interval = safe_get(cfg, ["train", "sds", "interval"], i_name)
        warmup = safe_get(cfg, ["train", "sds", "warmup_epochs"], wu_name)
        epochs = safe_get(cfg, ["train", "epoch"], None)
        i_eval = safe_get(cfg, ["log", "i_eval"], None)

        patient_dirs = [p for p in exp_dir.iterdir() if p.is_dir()]
        patient_count = len(patient_dirs)

        eval_count = 0
        has_pred_npy = False
        has_gt_npy = False
        for pdir in patient_dirs:
            eval_dir = pdir / "eval"
            if eval_dir.exists():
                eval_count += 1
                has_pred_npy = has_pred_npy or (eval_dir / "image_pred.npy").exists()
                has_gt_npy = has_gt_npy or (eval_dir / "image_gt.npy").exists()

        rows.append(
            {
                "exp_name": exp_name,
                "sds_weight": sds_weight,
                "interval": interval,
                "warmup_epochs": warmup,
                "epochs": epochs,
                "i_eval": i_eval,
                "patients": patient_count,
                "eval_dirs": eval_count,
                "has_pred_npy": has_pred_npy,
                "has_gt_npy": has_gt_npy,
            }
        )

    rows.sort(
        key=lambda x: (
            float("inf") if x["sds_weight"] is None else float(x["sds_weight"]),
            float("inf") if x["interval"] is None else int(x["interval"]),
            float("inf") if x["warmup_epochs"] is None else int(x["warmup_epochs"]),
            x["exp_name"],
        )
    )
    return rows


def to_markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        "exp_name",
        "sds_weight",
        "interval",
        "warmup_epochs",
        "epochs",
        "i_eval",
        "patients",
        "eval_dirs",
        "has_pred_npy",
        "has_gt_npy",
    ]

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        vals = [str(row.get(h, "")) for h in headers]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "exp_name",
        "sds_weight",
        "interval",
        "warmup_epochs",
        "epochs",
        "i_eval",
        "patients",
        "eval_dirs",
        "has_pred_npy",
        "has_gt_npy",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize experiments in logs by sds_weight / interval / warmup_epochs."
    )
    parser.add_argument("--logs-dir", type=Path, default=Path("./logs"))
    parser.add_argument(
        "--out-csv", type=Path, default=Path("./logs/sds_experiment_summary.csv")
    )
    parser.add_argument(
        "--out-md", type=Path, default=Path("./logs/sds_experiment_summary.md")
    )
    args = parser.parse_args()

    rows = summarize_logs(args.logs_dir)
    md_table = to_markdown(rows)

    write_csv(rows, args.out_csv)
    args.out_md.write_text(md_table + "\n", encoding="utf-8")

    print(f"found experiments: {len(rows)}")
    print(f"csv saved: {args.out_csv}")
    print(f"md saved : {args.out_md}")
    print("\n" + md_table)


if __name__ == "__main__":
    main()
