"""Bounded, filename-free history for one Streamlit session."""

import json
from datetime import datetime, timezone


def add_prediction(session, label: str, confidence: float, limit: int = 100):
    history = list(session.get("prediction_history", []))
    history.append(
        {
            "prediction": label,
            "confidence": round(confidence, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    session["prediction_history"] = history[-limit:]


def export_history(session) -> str:
    # Export JSON to avoid spreadsheet formulas; no user filenames are
    # stored.
    return json.dumps(session.get("prediction_history", []), indent=2)


def clear_history(session):
    session["prediction_history"] = []
