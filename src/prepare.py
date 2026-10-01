"""Prepare VOLTRACER participant data and private answers from its organizer input."""

from __future__ import annotations

import base64
import csv
import io
import json
import shutil
import sys
import zipfile
import zlib
from pathlib import Path

import numpy as np


TRAIN_DAYS = 7
TEST_DAYS = 6
POLICY_KEYS = ("first", "threshold", "second_low", "second_high")


class Input:
    def __init__(self, location: Path):
        self.location = location
        self.archive = zipfile.ZipFile(location) if location.is_file() else None
        self.members = set(self.archive.namelist()) if self.archive else None

    def read(self, name: str) -> bytes:
        if self.archive:
            return self.archive.read(name)
        return (self.location / name).read_bytes()

    def close(self) -> None:
        if self.archive:
            self.archive.close()


def _is_input(path: Path) -> bool:
    if path.is_dir():
        return (path / "manifest.csv").is_file() and (path / "SOURCE_MANIFEST.json").is_file()
    if not path.is_file() or path.stat().st_size < 200_000_000:
        return False
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            return {"manifest.csv", "SOURCE_MANIFEST.json", "PRIVATE_ID_SALT.txt"}.issubset(names) and any(n.startswith("cases/") and n.endswith(".npz") for n in names) and any(n.startswith("truth/") and n.endswith(".npz") for n in names)
    except (OSError, zipfile.BadZipFile):
        return False


def find_input(root: Path) -> Input:
    if _is_input(root):
        return Input(root)
    if not root.is_dir():
        raise FileNotFoundError("VOLTRACER organizer input not found; upload VOLTRACER_Eris_PRIVATE_INPUT_v1.zip, not the participant ZIP")
    todo = [root]
    seen = set()
    inspected = 0
    while todo and inspected < 5000:
        path = todo.pop(0)
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        inspected += 1
        if _is_input(path):
            return Input(path)
        if path.is_dir() and path.name not in {"cases", "truth", "arrays", "private", "public"}:
            try:
                todo.extend(sorted(path.iterdir()))
            except OSError:
                pass
    raise FileNotFoundError("VOLTRACER organizer input not found; upload VOLTRACER_Eris_PRIVATE_INPUT_v1.zip, not the participant ZIP")


def _clean(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in ("arrays", "train_truth"):
        target = out / name
        if target.exists():
            assert target.resolve().is_relative_to(out.resolve())
            shutil.rmtree(target)
    for name in ("train.csv", "test.csv", "train_labels.csv", "sample_submission.csv", "answers.csv", "metric.json"):
        target = out / name
        if target.is_file():
            target.unlink()


def _policy(first: int, threshold: int, low: int, high: int) -> dict:
    return {"first": int(first), "threshold": int(threshold), "second_low": int(low), "second_high": int(high)}


def _bounds(loss: np.ndarray, response: np.ndarray, anchors: tuple[int, ...]) -> tuple[dict, float, float]:
    candidates = [i for i in range(14) if i not in anchors]
    best = float("inf")
    worst = float("-inf")
    oracle = None
    for first in candidates:
        second = [i for i in candidates if i != first]
        for threshold in range(7):
            low_mask = response[:, first] < threshold
            choice = np.asarray(loss[:, first, second], dtype=np.float64)
            selected = np.where(low_mask[None, None, :], choice.T[:, None, :], choice.T[None, :, :])
            mean = selected.mean(axis=2)
            tail = np.sort(selected, axis=2)[:, :, -2:].mean(axis=2)
            risk = 0.75 * mean + 0.25 * tail
            lo_min, hi_min = np.unravel_index(np.argmin(risk), risk.shape)
            lo_max, hi_max = np.unravel_index(np.argmax(risk), risk.shape)
            minimum = float(risk[lo_min, hi_min])
            maximum = float(risk[lo_max, hi_max])
            if minimum < best:
                best = minimum
                oracle = _policy(first, threshold, second[lo_min], second[hi_min])
            if maximum > worst:
                worst = maximum
    assert oracle is not None and np.isfinite(best) and np.isfinite(worst) and worst >= best
    return oracle, best, worst


def _truth_payload(loss: np.ndarray, response: np.ndarray, best: float, worst: float) -> dict:
    loss_bytes = np.asarray(loss, dtype="<f4", order="C").tobytes()
    response_bytes = np.asarray(response, dtype=np.uint8, order="C").tobytes()
    return {
        "loss_z": base64.b64encode(zlib.compress(loss_bytes, 9)).decode("ascii"),
        "response_z": base64.b64encode(zlib.compress(response_bytes, 9)).decode("ascii"),
        "best": best,
        "worst": worst,
    }


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare(raw_root: str | Path, public_dir: str | Path, private_dir: str | Path) -> None:
    source = find_input(Path(raw_root))
    public = Path(public_dir)
    private = Path(private_dir)
    assert public.resolve() != private.resolve()
    _clean(public)
    _clean(private)
    (public / "arrays").mkdir(exist_ok=True)
    (public / "train_truth").mkdir(exist_ok=True)
    meta = json.loads(source.read("SOURCE_MANIFEST.json"))
    assert meta["license"] == "CC BY 4.0" and len(meta["files"]) == 13
    assert len(source.read("PRIVATE_ID_SALT.txt").strip()) == 32
    rows = list(csv.DictReader(io.StringIO(source.read("manifest.csv").decode("utf-8"))))
    assert rows and len({r["case_id"] for r in rows}) == len(rows)
    assert len({r["source_day"] for r in rows if r["split"] == "train"}) == TRAIN_DAYS
    assert len({r["source_day"] for r in rows if r["split"] == "test"}) == TEST_DAYS
    assert not ({r["source_day"] for r in rows if r["split"] == "train"} & {r["source_day"] for r in rows if r["split"] == "test"})
    train, test, labels, sample, answers = [], [], [], [], []
    for row in sorted(rows, key=lambda r: r["case_id"]):
        case_id = row["case_id"]
        assert case_id.startswith("vt_") and len(case_id) == 23
        anchors = tuple(json.loads(row["anchors_json"]))
        assert len(anchors) == len(set(anchors)) == 3 and all(isinstance(i, int) and 0 <= i < 14 for i in anchors)
        with np.load(io.BytesIO(source.read(row["case_file"])), allow_pickle=False) as z:
            traces = z["traces"]
        with np.load(io.BytesIO(source.read(row["truth_file"])), allow_pickle=False) as z:
            loss, response = z["loss"], z["response"]
        assert traces.shape == (8, 14, 14900) and traces.dtype == np.int16
        assert loss.shape == (8, 14, 14) and np.isfinite(loss).all() and (loss >= 0).all()
        assert response.shape == (8, 14) and response.dtype == np.uint8 and (response <= 5).all()
        oracle, best, worst = _bounds(loss, response, anchors)
        pub_file = "arrays/" + case_id + ".npz"
        common = {"case_id": case_id, "file": pub_file, "anchors_json": row["anchors_json"]}
        if row["split"] == "train":
            (public / pub_file).write_bytes(source.read(row["case_file"]))
            truth_file = "train_truth/" + case_id + ".npz"
            (public / truth_file).write_bytes(source.read(row["truth_file"]))
            train.append(common)
            labels.append({"case_id": case_id, "policy_json": json.dumps(oracle, separators=(",", ":")), "truth_file": truth_file})
        elif row["split"] == "test":
            np.savez_compressed(public / pub_file, traces=traces[:, list(anchors), :])
            test.append(common)
            answer = {**oracle, "anchors": list(anchors), "truth": _truth_payload(loss, response, best, worst)}
            answers.append({"case_id": case_id, "policy_json": json.dumps(answer, separators=(",", ":"), allow_nan=False)})
            candidates = [i for i in range(14) if i not in anchors]
            first = candidates[len(candidates) // 3]
            second = candidates[(2 * len(candidates)) // 3]
            if second == first:
                second = candidates[-1]
            baseline = _policy(first, int(case_id[-1], 16) % 7, second, second)
            sample.append({"case_id": case_id, "policy_json": json.dumps(baseline, separators=(",", ":"))})
        else:
            raise ValueError("unknown split")
    columns = ("case_id", "file", "anchors_json")
    _write_csv(public / "train.csv", columns, train)
    _write_csv(public / "test.csv", columns, test)
    _write_csv(public / "train_labels.csv", ("case_id", "policy_json", "truth_file"), labels)
    _write_csv(public / "sample_submission.csv", ("case_id", "policy_json"), sample)
    _write_csv(private / "answers.csv", ("case_id", "policy_json"), answers)
    (private / "metric.json").write_text(json.dumps({"name": "tail_robust_normalized_reconstruction_regret", "higher_is_better": True, "range": [0, 1], "batch_loss": "0.75*mean(event_loss)+0.25*mean(largest_two_event_losses)", "case_score": "clip((worst-policy)/(worst-best),0,1); tied policies score 1", "aggregation": "arithmetic mean over evaluated answer IDs"}, indent=2) + "\n", encoding="utf-8")
    source.close()
    print(json.dumps({"train": len(train), "test": len(test), "source_days": TRAIN_DAYS + TEST_DAYS}))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: prepare.py RAW_ROOT PUBLIC_DIR PRIVATE_DIR")
    prepare(*sys.argv[1:])
