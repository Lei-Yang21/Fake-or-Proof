import json

from fake_or_proof.history import add_prediction, clear_history, export_history


def test_two_sessions_do_not_share_read_export_or_delete():
    first, second = {}, {}
    add_prediction(first, "Real Image", 0.9)
    assert json.loads(export_history(second)) == []
    add_prediction(second, "AI Generated Image", 0.8)
    clear_history(first)
    assert json.loads(export_history(first)) == []
    assert (
        json.loads(export_history(second))[0]["prediction"]
        == "AI Generated Image"
    )


def test_history_is_bounded_and_does_not_contain_filenames():
    session = {}
    for index in range(110):
        add_prediction(session, "Real Image", index / 110)
    history = json.loads(export_history(session))
    assert len(history) == 100
    assert set(history[0]) == {"prediction", "confidence", "timestamp"}
