import json
import os
import threading
from datetime import datetime, timezone

STATS_FILE = "stats.json"
_lock = threading.Lock()


def _load_stats() -> dict:
    if not os.path.exists(STATS_FILE):
        return {"trainings": [], "predictions": []}
    try:
        with open(STATS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"trainings": [], "predictions": []}


def _save_stats(data: dict):
    tmp_path = STATS_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, STATS_FILE)


def record_training(model_name: str, metrics: dict):
    with _lock:
        stats = _load_stats()
        stats["trainings"].append({
            "model_name": model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
        })
        _save_stats(stats)


def record_prediction(model_name: str, total: int, fraud_count: int):
    with _lock:
        stats = _load_stats()
        stats["predictions"].append({
            "model_name": model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total": total,
            "fraud_count": fraud_count,
        })
        _save_stats(stats)


def get_stats() -> dict:
    stats = _load_stats()

    total_trainings = len(stats["trainings"])
    total_predictions = len(stats["predictions"])
    total_fraud_detected = sum(p.get("fraud_count", 0) for p in stats["predictions"])
    total_checked = sum(p.get("total", 0) for p in stats["predictions"])

    accuracies = []
    for t in stats["trainings"]:
        m = t.get("metrics", {})
        if "accuracy" in m:
            accuracies.append(m["accuracy"])
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0

    # Recent activity: last 20 events, merged and sorted by timestamp
    recent = []
    for t in stats["trainings"]:
        recent.append({
            "type": "training",
            "model_name": t["model_name"],
            "timestamp": t["timestamp"],
            "details": f"accuracy={t.get('metrics', {}).get('accuracy', 'N/A')}",
        })
    for p in stats["predictions"]:
        recent.append({
            "type": "prediction",
            "model_name": p["model_name"],
            "timestamp": p["timestamp"],
            "details": f"total={p['total']}, fraud={p['fraud_count']}",
        })
    recent.sort(key=lambda x: x["timestamp"], reverse=True)
    recent = recent[:20]

    return {
        "total_trainings": total_trainings,
        "total_predictions": total_predictions,
        "total_fraud_detected": total_fraud_detected,
        "total_checked": total_checked,
        "avg_accuracy": round(avg_accuracy, 4),
        "recent_activity": recent,
    }
