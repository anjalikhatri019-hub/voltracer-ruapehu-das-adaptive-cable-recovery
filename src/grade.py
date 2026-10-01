"""Standalone VOLTRACER Eris grader. Return one scalar in [0, 1]."""

from __future__ import annotations

import base64
import csv
import json
import math
import struct
import zlib
from pathlib import Path


COLS = ("case_id", "policy_json")
KEYS = ("first", "threshold", "second_low", "second_high")
TRUTH_KEYS = ("loss_z", "response_z", "best", "worst")


def _rows(obj):
    if isinstance(obj, (str, Path)):
        with open(obj, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            columns = tuple(reader.fieldnames or ())
            rows = list(reader)
    elif hasattr(obj, "to_dict") and hasattr(obj, "columns"):
        columns = tuple(str(c) for c in obj.columns)
        rows = obj.to_dict(orient="records")
    else:
        raise ValueError("expected CSV path or DataFrame")
    if len(columns) != len(set(columns)) or tuple(c for c in columns if c != "visibility") != COLS or len(columns) not in (2, 3):
        raise ValueError("wrong columns")
    result = {}
    for row in rows:
        key = row["case_id"]
        value = row["policy_json"]
        if not isinstance(key, str) or not key or key in result:
            raise ValueError("invalid or duplicate ID")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("missing policy")
        result[key] = value
    return result


def _answer_marker(rows):
    if not rows:
        return False
    try:
        value = json.loads(next(iter(rows.values())))
        return isinstance(value, dict) and isinstance(value.get("truth"), dict) and set(value["truth"]) == set(TRUTH_KEYS)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False


def _policy(obj, anchors):
    if not isinstance(obj, dict) or not set(KEYS).issubset(obj):
        raise ValueError("missing policy fields")
    for name in KEYS:
        if type(obj[name]) is not int:
            raise ValueError("policy values must be integers")
    first, threshold, low, high = (obj[name] for name in KEYS)
    available = set(range(14)) - set(anchors)
    if first not in available or low not in available or high not in available:
        raise ValueError("probe is unavailable")
    if low == first or high == first or not 0 <= threshold <= 6:
        raise ValueError("invalid branch or threshold")
    return first, threshold, low, high


def _truth(obj):
    truth = obj.get("truth")
    if not isinstance(truth, dict) or set(truth) != set(TRUTH_KEYS):
        raise ValueError("bad private truth")
    loss_bytes = zlib.decompress(base64.b64decode(truth["loss_z"], validate=True))
    response_bytes = zlib.decompress(base64.b64decode(truth["response_z"], validate=True))
    if len(loss_bytes) != 8 * 14 * 14 * 4 or len(response_bytes) != 8 * 14:
        raise ValueError("wrong private truth size")
    loss = struct.unpack("<" + "f" * (8 * 14 * 14), loss_bytes)
    response = response_bytes
    best, worst = float(truth["best"]), float(truth["worst"])
    if not (math.isfinite(best) and math.isfinite(worst) and 0 <= best <= worst):
        raise ValueError("invalid private normalization")
    return loss, response, best, worst


def _case_score(submitted, answer):
    try:
        answer_obj = json.loads(answer)
        policy = json.loads(submitted)
        anchors = answer_obj.get("anchors")
        # Anchor membership is encoded in the private answer to keep the grader
        # independent of public files while preserving the two-column contract.
        if not isinstance(anchors, list) or len(anchors) != 3:
            raise ValueError("missing private anchors")
        if "truth" in policy and policy != answer_obj:
            raise ValueError("participant supplied private fields")
        if set(policy) not in (set(KEYS), set(KEYS) | {"truth", "anchors"}):
            raise ValueError("unexpected policy fields")
        first, threshold, low, high = _policy(policy, anchors)
        loss, response, best, worst = _truth(answer_obj)
        selected = []
        for event in range(8):
            branch = low if response[event * 14 + first] < threshold else high
            value = loss[event * 196 + first * 14 + branch]
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid private loss")
            selected.append(float(value))
        actual = 0.75 * (math.fsum(selected) / 8) + 0.25 * (math.fsum(sorted(selected)[-2:]) / 2)
        if worst - best <= 1e-12:
            return 1.0
        return max(0.0, min(1.0, (worst - actual) / (worst - best)))
    except (ValueError, TypeError, KeyError, OverflowError, zlib.error, struct.error, base64.binascii.Error):
        return 0.0


def grade(answers, submission):
    try:
        a = _rows(answers)
        s = _rows(submission)
        if not _answer_marker(a) and _answer_marker(s):
            a, s = s, a
        if not a or not _answer_marker(a) or not set(a).issubset(s):
            return 0.0
        return float(math.fsum(_case_score(s[key], a[key]) for key in sorted(a)) / len(a))
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, csv.Error):
        return 0.0


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        raise SystemExit("usage: grade.py ANSWERS_CSV SUBMISSION_CSV")
    print(grade(sys.argv[1], sys.argv[2]))
