"""
End-to-end smoke tests: run the real Streamlit app with fake data and click around.

Slower than the other tests (~30-60 s). Skip them with:  pytest -m "not slow"
"""
import os

import pandas as pd
import pytest

import f1_predict as f1
from conftest import ROOT, make_history

pytestmark = pytest.mark.slow

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


@pytest.fixture
def app(monkeypatch, no_network):
    hist = make_history(seasons=(2025, 2026), rounds=6)
    monkeypatch.setattr(f1, "fetch_season", lambda s, refresh=False: hist[hist["season"] == s].copy()
                        if s in (2025, 2026) else hist.iloc[0:0].copy())
    sched = pd.DataFrame([dict(season=2026, round=r, race_name=f"Race {r} Grand Prix",
                               circuit=f"c{r}", country="Qatar", locality="Town",
                               date=f"2026-11-{r + 10}") for r in range(1, 9)])
    monkeypatch.setattr(f1, "fetch_schedule", lambda s: sched)
    quali = {f"d{i}": i + 1 for i in range(10)}
    monkeypatch.setattr(f1, "fetch_next_race", lambda: (
        dict(season=2026, round=7, race_name="Race 7 Grand Prix", circuit="c7", date="2026-11-17"),
        quali))
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.session_state["train_seasons"] = [2025, 2026]
    return at.run()


def test_prediction_page_loads(app):
    assert not app.exception, app.exception
    header = " ".join(c.value for c in app.caption)
    assert "qualifying order" in header and "penalties not applied" in header


def test_adding_a_penalty_changes_the_grid(app):
    app.selectbox(key="pen_driver").set_value("d0")
    app.selectbox(key="pen_kind").set_value("back")
    app = app.run()
    app = next(b for b in app.button if b.label == "Add penalty").click().run()
    assert not app.exception, app.exception
    assert app.session_state["penalties"][(2026, 7)] == [{"driver": "d0", "kind": "back", "places": 0}]
    header = " ".join(c.value for c in app.caption)
    assert "qualifying + 1 penalty" in header


@pytest.mark.parametrize("button", ["season dashboard", "Ask LightsToFlag", "R6 "])
def test_other_pages_load(app, button):
    app = next(b for b in app.sidebar.button if button in b.label).click().run()
    assert not app.exception, app.exception


def test_sound_button_plays_audio(app):
    app = next(b for b in app.button if "Lights out" in b.label).click().run()
    assert not app.exception, app.exception
    assert len(app.get("audio")) == 1
