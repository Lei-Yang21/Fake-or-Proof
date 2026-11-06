import importlib
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[3]


def test_import_does_not_load_models_or_contact_cloud():
    with patch("fake_or_proof.model.InferenceService.load") as load:
        importlib.reload(importlib.import_module("app"))
        load.assert_not_called()


def test_missing_model_shows_generic_message_and_disables_prediction(
    monkeypatch,
):
    monkeypatch.setenv("FOP_MODEL_URI", "bundled")
    monkeypatch.setenv("FOP_MODEL_SHA256", "0" * 64)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert "temporarily unavailable" in app.error[0].value
    assert "checksum" not in app.error[0].value
    assert app.sidebar.button[0].disabled


def test_two_ui_sessions_have_independent_history():
    first = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    second = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not first.exception and not second.exception
    assert not first.error and not second.error
    first.session_state["prediction_history"] = [
        {"prediction": "Real Image", "confidence": 0.9}
    ]
    first.run()
    assert len(first.dataframe) == 1 and len(second.dataframe) == 0
    clear = next(
        button
        for button in first.button
        if button.label == "Clear your session history"
    )
    clear.click().run()
    assert not first.session_state["prediction_history"]
    assert len(second.dataframe) == 0
