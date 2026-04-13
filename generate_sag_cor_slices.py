from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_3d(arr: np.ndarray, path: Path) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape={arr.shape} at {path}")
    return arr


def get_sag_slice(vol: np.ndarray, index: int | None = None) -> np.ndarray:
    h, w, d = vol.shape
    idx = w // 2 if index is None else int(index)
    idx = max(0, min(w - 1, idx))
    return vol[:, idx, :]


def get_cor_slice(vol: np.ndarray, index: int | None = None) -> np.ndarray:
    h, w, d = vol.shape
    idx = h // 2 if index is None else int(index)
    idx = max(0, min(h - 1, idx))
    return vol[idx, :, :]


def save_plane_png(
    pred_plane: np.ndarray,
    out_path: Path,
    title: str,
    gt_plane: np.ndarray | None = None,
    dpi: int = 140,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if gt_plane is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi)
        ax.set_title(f"{title} | Pred")
        ax.imshow(pred_plane, cmap="gray")
        ax.axis("off")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=dpi)
        axes[0].set_title(f"{title} | Pred")
        axes[0].imshow(pred_plane, cmap="gray")
        axes[0].axis("off")
        axes[1].set_title(f"{title} | GT")
        axes[1].imshow(gt_plane, cmap="gray")
        axes[1].axis("off")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def find_candidate_dirs(logs_dir: Path, pred_name: str, gt_name: str) -> list[Path]:
    pred_files = list(logs_dir.rglob(pred_name))
    gt_files = list(logs_dir.rglob(gt_name))
    candidate_dirs = {p.parent for p in pred_files}
    candidate_dirs.update({g.parent for g in gt_files})
    return sorted(candidate_dirs)


def process_one_dir(
    eval_dir: Path,
    pred_name: str,
    gt_name: str,
    sag_name: str,
    cor_name: str,
    sag_index: int | None,
    cor_index: int | None,
) -> tuple[str, str]:
    pred_path = eval_dir / pred_name
    gt_path = eval_dir / gt_name
    sag_png = eval_dir / sag_name
    cor_png = eval_dir / cor_name

    if sag_png.exists() and cor_png.exists():
        return "skipped", f"exists: {eval_dir}"

    if not pred_path.exists():
        return "missing", f"missing pred: {pred_path}"

    try:
        pred = ensure_3d(np.load(pred_path), pred_path)
        gt = None
        if gt_path.exists():
            gt = ensure_3d(np.load(gt_path), gt_path)

        if not sag_png.exists():
            pred_sag = get_sag_slice(pred, sag_index)
            gt_sag = get_sag_slice(gt, sag_index) if gt is not None else None
            save_plane_png(pred_sag, sag_png, title="Sagittal", gt_plane=gt_sag)

        if not cor_png.exists():
            pred_cor = get_cor_slice(pred, cor_index)
            gt_cor = get_cor_slice(gt, cor_index) if gt is not None else None
            save_plane_png(pred_cor, cor_png, title="Coronal", gt_plane=gt_cor)

        return "processed", f"done: {eval_dir}"
    except Exception as exc:
        return "error", f"{eval_dir} | {exc}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-generate sag.png and cor.png for logs experiments."
    )
    parser.add_argument("--logs-dir", type=Path, default=Path("./logs"))
    parser.add_argument("--pred-name", type=str, default="image_pred.npy")
    parser.add_argument("--gt-name", type=str, default="image_gt.npy")
    parser.add_argument("--sag-name", type=str, default="sag.png")
    parser.add_argument("--cor-name", type=str, default="cor.png")
    parser.add_argument("--sag-index", type=int, default=None)
    parser.add_argument("--cor-index", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    logs_dir = args.logs_dir
    if not logs_dir.exists():
        raise FileNotFoundError(f"logs dir not found: {logs_dir}")

    candidate_dirs = find_candidate_dirs(logs_dir, args.pred_name, args.gt_name)
    print(f"logs_dir: {logs_dir.resolve()}")
    print(f"candidate dirs: {len(candidate_dirs)}")

    stats = {"processed": 0, "skipped": 0, "missing": 0, "error": 0}
    errors: list[str] = []

    for eval_dir in candidate_dirs:
        status, msg = process_one_dir(
            eval_dir=eval_dir,
            pred_name=args.pred_name,
            gt_name=args.gt_name,
            sag_name=args.sag_name,
            cor_name=args.cor_name,
            sag_index=args.sag_index,
            cor_index=args.cor_index,
        )
        stats[status] += 1
        if args.verbose or status in {"error", "missing"}:
            print(f"[{status}] {msg}")
        if status == "error":
            errors.append(msg)

    print("\n=== Summary ===")
    for key, value in stats.items():
        print(f"{key:>9}: {value}")

    if errors:
        print("\n=== Errors (first 20) ===")
        for line in errors[:20]:
            print(" -", line)


if __name__ == "__main__":
    main()
