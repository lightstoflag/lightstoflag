"""Charts, sounds and the chat context build without errors and contain what they should."""
import io
import wave

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import charts
import chat
import engine_audio
import f1_predict as f1


@pytest.fixture
def season(features):
    s = features[features["season"] == 2025].copy()
    s["Driver"], s["Team"] = s["driver_name"], s["team_name"]
    return s


@pytest.mark.parametrize("dark", [False, True])
def test_season_charts(season, dark):
    standings = season.groupby("Driver")["points"].sum().sort_values(ascending=False)
    assert isinstance(charts.championship_race(season, standings.index[:4].tolist(), dark), go.Figure)
    assert isinstance(charts.results_heatmap(season, standings.index.tolist(), dark), go.Figure)
    assert isinstance(charts.team_points(season, dark), go.Figure)


def test_heatmap_labels_dnfs_as_dnf(season):
    season = season.copy()
    season.loc[season.index[0], ["dnf", "finish"]] = [1, f1.DNF_POSITION]
    fig = charts.results_heatmap(season, season["Driver"].unique().tolist())
    labels = [t for trace in fig.data if trace.type == "scatter" for t in trace.text]
    assert "DNF" in labels
    assert str(f1.DNF_POSITION) not in labels


def test_race_charts(season):
    race = season[season["round"] == 6].copy()
    race["Pos"] = race["finish"]
    assert isinstance(charts.position_shift(race, "grid", "finish", "Grid", "Finish"), go.Figure)
    race = race.assign(score=race["finish"], win=10.0, podium=30.0)
    assert isinstance(charts.chances(race), go.Figure)
    h = race.assign(race=race["driver"])
    assert isinstance(charts.driver_form(h), go.Figure)


def test_model_charts(features):
    imp = pd.Series(1 / len(f1.FEATURES), index=f1.FEATURES)
    fig = charts.feature_importance(imp)
    assert len(fig.data[0].y) == len(f1.FEATURES)
    stats = f1.backtest(features, n_races=2)
    assert isinstance(charts.backtest_chart(stats["per_race"]), go.Figure)


def test_every_model_input_has_a_readable_chart_label():
    imp = pd.Series(1 / len(f1.FEATURES), index=f1.FEATURES)
    shown = set(charts.feature_importance(imp).data[0].y)
    raw_names = shown & set(f1.FEATURES)
    assert not raw_names, f"Add labels in charts.feature_importance for: {raw_names}"


@pytest.mark.parametrize("make", [engine_audio.race_start, engine_audio.fly_by])
def test_sounds_are_valid_wav_files(make):
    data = make()
    with wave.open(io.BytesIO(data)) as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == engine_audio.SR
        seconds = w.getnframes() / w.getframerate()
    assert 2 < seconds < 20


def test_chat_context_includes_prediction_and_standings(season):
    info = dict(season=2025, round=7, race_name="Race 7 Grand Prix", date="2025-12-01")
    pred = pd.DataFrame({"Pos": [1, 2], "Driver": ["Alice Ace", "Bob Bold"],
                         "Team": ["Team Zero"] * 2, "grid": [1, 2],
                         "win": [60.0, 20.0], "podium": [90.0, 70.0], "top10": [99.0, 98.0]})
    ctx = chat.build_context(info, pred, season,
                             {"n_races": 8, "winners_hit": 5, "podium_overlap": .7, "mae": 2.1},
                             "qualifying")
    assert "Race 7 Grand Prix" in ctx and "Alice Ace" in ctx and "win 60%" in ctx
    assert "Drivers' standings" in ctx and "Constructors" in ctx
    assert "Winner correct 5/8" in ctx


def test_chat_context_without_prediction(season):
    ctx = chat.build_context({"season": 2025}, None, season)
    assert "Prediction" not in ctx and "Drivers' standings" in ctx
