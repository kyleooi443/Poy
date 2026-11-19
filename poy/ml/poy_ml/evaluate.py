"""Metrics for a held-out split: per-class precision/recall/F1 and a confusion matrix."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def evaluate(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> dict[str, Any]:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(labels), zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f)),
        "n_test": len(y_true),
        "per_class": {
            lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, lab in enumerate(labels)
        },
        "labels": list(labels),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist(),
    }


def format_report(m: dict[str, Any]) -> str:
    labels = m["labels"]
    lines = [f"accuracy {m['accuracy']:.3f}   macro-F1 {m['macro_f1']:.3f}   n_test {m['n_test']}", ""]
    lines.append(f"{'class':<10}{'precision':>10}{'recall':>8}{'f1':>7}{'support':>9}")
    for lab in labels:
        c = m["per_class"][lab]
        lines.append(f"{lab:<10}{c['precision']:>10.3f}{c['recall']:>8.3f}{c['f1']:>7.3f}{c['support']:>9}")
    lines += ["", "confusion matrix (rows = true, cols = predicted)"]
    lines.append(" " * 10 + "".join(f"{lab:>10}" for lab in labels))
    for lab, row in zip(labels, m["confusion_matrix"], strict=True):
        lines.append(f"{lab:<10}" + "".join(f"{v:>10}" for v in row))
    return "\n".join(lines)
