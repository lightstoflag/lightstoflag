"""Feature engineering: correct values and, above all, no peeking at the future."""
import numpy as np
import pandas as pd
import pytest

import f1_predict as f1
from conftest import make_history


def test_all_model_inputs_exist_and_have_no_gaps(features):
    missing = [c for c in f1.FEATURES if c not in features.columns]
    assert not missing, f"FEATURES lists columns that build_features() doesn't create: {missing}"
    assert not features[f1.FEATURES].isna().any().any()


def test_features_do_not_depend_on_future_races(hist):
    """The key leakage test: rewriting the last race must not change any earlier row."""
    before = f1.build_features(hist)

    last = (hist["season"] == hist["season"].max()) & (hist["round"] == hist["round"].max())
    tampered = hist.copy()
    tampered.loc[last, "finish"] = tampered.loc[last, "finish"][::-1].values
    tampered.loc[last, "points"] = 99.0
    tampered.loc[last, "dnf"] = 1 - tampered.loc[last, "dnf"]
    after = f1.build_features(tampered)

    key = ["season", "round", "driver"]
    b = before[~before.set_index(key).index.isin(after[last.values].set_index(key).index)]
    a = after.set_index(key).loc[b.set_index(key).index].reset_index()
    pd.testing.assert_frame_equal(
        b[f1.FEATURES].reset_index(drop=True), a[f1.FEATURES].reset_index(drop=True),
        check_dtype=False)


def test_race_result_itself_never_feeds_its_own_features(hist):
    """Changing a race's own finishing order must not change that race's inputs."""
    r = (hist["season"] == 2025) & (hist["round"] == 3)
    tampered = hist.copy()
    tampered.loc[r, "finish"] = tampered.loc[r, "finish"][::-1].values
    a = f1.build_features(hist)
    b = f1.build_features(tampered)
    cols = [c for c in f1.FEATURES if c != "grid"]
    sel = lambda df: df[(df["season"] == 2025) & (df["round"] == 3)].sort_values("driver")[cols]
    pd.testing.assert_frame_equal(sel(a).reset_index(drop=True), sel(b).reset_index(drop=True))


def test_first_ever_race_uses_rookie_defaults(features):
    first = features[(features["season"] == 2024) & (features["round"] == 1)]
    assert (first["drv_avg_finish_5"] == 12).all()
    assert (first["drv_avg_points_5"] == 0).all()
    assert (first["drv_season_points"] == 0).all()
    assert (first["races_in_season"] == 0).all()


def test_rolling_average_uses_previous_races_only():
    h = make_history(seasons=(2025,), rounds=4, dnf_rate=0)
    feats = f1.build_features(h)
    d = feats[feats["driver"] == "d3"].sort_values("round")
    finishes = d["finish"].tolist()
    # Round 3's 5-race average = mean of rounds 1 and 2
    assert d["drv_avg_finish_5"].iloc[2] == pytest.approx(np.mean(finishes[:2]))


def test_season_points_accumulate_before_each_race():
    h = make_history(seasons=(2025,), rounds=3, dnf_rate=0)
    feats = f1.build_features(h)
    d = feats[feats["driver"] == "d0"].sort_values("round")
    pts = d["points"].tolist()
    assert d["drv_season_points"].tolist() == [0, pts[0], pts[0] + pts[1]]


def test_team_form_resets_each_season():
    feats = f1.build_features(make_history(seasons=(2024, 2025), rounds=3))
    r1_2025 = feats[(feats["season"] == 2025) & (feats["round"] == 1)]
    # No 2025 team history yet -> falls back to the driver's own recent average
    assert (r1_2025["team_avg_finish_5"] == r1_2025["drv_avg_finish_5"]).all()


def test_grid_vs_teammate():
    rows = pd.DataFrame({
        "season": 2025, "round": 1,
        "driver": ["a", "b", "c"], "team": ["x", "x", "y"], "grid": [3, 7, 5],
        "team_avg_grid_5": [4.0, 4.0, 9.0],
    })
    out = f1.add_grid_features(rows).set_index("driver")["grid_vs_teammate"]
    assert out["a"] == -4      # beat teammate by 4 places
    assert out["b"] == 4
    assert out["c"] == 0       # no teammate in this race


def test_training_rows_exclude_dnfs(features):
    tr = f1.training_rows(features)
    assert (tr["dnf"] == 0).all()
    assert len(tr) == (features["dnf"] == 0).sum()


def test_next_race_rows_with_qualifying_grid(hist, next_info):
    quali = {f"d{i}": 10 - i for i in range(10)}   # reverse order
    rows = f1.next_race_rows(hist, next_info, quali)
    assert (rows["grid_source"] == "qualifying").all()
    assert rows.set_index("driver")["grid"].to_dict() == quali
    assert not rows[f1.FEATURES].isna().any().any()


def test_next_race_rows_estimates_grid_before_qualifying(hist, next_info):
    rows = f1.next_race_rows(hist, next_info, None)
    assert (rows["grid_source"] == "estimated").all()
    assert sorted(rows["grid"]) == list(range(1, len(rows) + 1))


def test_driver_missing_from_qualifying_starts_last(hist, next_info):
    quali = {f"d{i}": i + 1 for i in range(9)}     # d9 didn't set a time
    rows = f1.next_race_rows(hist, next_info, quali)
    assert rows.set_index("driver").loc["d9", "grid"] == 20
