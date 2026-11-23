"""Model training, prediction and walk-forward back-testing."""
import numpy as np
import pytest

import f1_predict as f1


def test_recent_seasons_weigh_more(features):
    w = f1.sample_weights(features)
    newest = features["season"] == features["season"].max()
    assert (w[newest] == 1).all()
    assert (w[~newest] == 0.5).all()


def test_model_trains_and_predicts_every_driver(features):
    model = f1.fit_model(features)
    preds = model.predict(features[f1.FEATURES])
    assert len(preds) == len(features)
    assert np.isfinite(preds).all()


def test_model_learns_that_better_drivers_finish_higher(features):
    model = f1.fit_model(features)
    last = features[features["round"] == features["round"].max()]
    last = last[last["season"] == last["season"].max()].copy()
    last["pred"] = model.predict(last[f1.FEATURES])
    # d0 is the fastest driver in the fake data, d9 the slowest
    assert last.set_index("driver").loc["d0", "pred"] < last.set_index("driver").loc["d9", "pred"]


def test_predict_returns_full_order_and_importances(hist, features, next_info):
    rows = f1.next_race_rows(hist, next_info, None)
    result, imp = f1.predict(features, rows)
    assert result["predicted_pos"].tolist() == list(range(1, len(rows) + 1))
    assert set(imp.index) == set(f1.FEATURES)
    assert imp.sum() == pytest.approx(1.0)
    assert (imp >= 0).all()


def test_backtest_reports_what_the_app_reads(features):
    stats = f1.backtest(features, n_races=3)
    for key in ["n_races", "winners_hit", "podium_overlap", "mae", "per_race"]:
        assert key in stats
    assert stats["n_races"] == 3
    assert 0 <= stats["winners_hit"] <= 3
    assert 0 <= stats["podium_overlap"] <= 1
    assert len(stats["per_race"]) == 3


def test_backtest_only_trains_on_earlier_races(features, monkeypatch):
    """Every model in the walk-forward test must be trained strictly on the past."""
    seen = []
    real_fit = f1.fit_model

    def spy(df):
        seen.append(df[["season", "round"]].drop_duplicates().values.tolist())
        return real_fit(df)

    monkeypatch.setattr(f1, "fit_model", spy)
    f1.backtest(features, n_races=3)

    tested = features[["season", "round"]].drop_duplicates().sort_values(["season", "round"]).tail(3)
    for trained_on, (s, r) in zip(seen, tested.values.tolist()):
        assert all((ts, tr) < (s, r) for ts, tr in trained_on), f"Race {s} R{r} saw its own or future data"
