#!/usr/bin/env python3
"""
F1 race predictor.

Pulls historical results + qualifying from the Jolpica (Ergast-compatible) API,
builds per-driver form features, trains a gradient-boosted model, back-tests it,
and predicts the finishing order of the next race.

Usage:
    pip install pandas scikit-learn requests
    python f1_predict.py                 # predict next race (uses quali if published)
    python f1_predict.py --seasons 2022 2023 2024 2025 2026
    python f1_predict.py --refresh       # ignore cache and re-download

Run it after Saturday qualifying for the best accuracy - grid position is the
single strongest signal. Before qualifying, it estimates grid from recent form.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

API = "https://api.jolpi.ca/ergast/f1"
CACHE_DIR = "f1_cache"
DNF_POSITION = 21  # finishing value assigned to non-classified finishers

FEATURES = [
    "grid",
    "drv_avg_finish_5",
    "drv_avg_grid_5",
    "drv_avg_points_5",
    "drv_dnf_rate_10",
    "team_avg_finish_5",
    "team_avg_grid_5",
    "drv_track_avg_finish",
    "drv_season_points",
    "races_in_season",
    # added after tuning on 2022-2026 results
    "drv_avg_gain_5",
    "drv_last_finish",
    "drv_avg_finish_season",
    "grid_vs_teammate",
    "team_dnf_rate_10",
]


# --------------------------------------------------------------------------- #
# Data fetching
# --------------------------------------------------------------------------- #
def _get(url):
    for attempt in range(5):
        r = requests.get(url, timeout=30)
        if r.status_code == 429:  # rate limited (4 req/s, 500/hr)
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        time.sleep(0.3)
        return r.json()["MRData"]
    raise RuntimeError(f"Rate limited too many times: {url}")


def _paged(endpoint, table_key, list_key):
    out, offset = [], 0
    while True:
        data = _get(f"{API}/{endpoint}.json?limit=100&offset={offset}")
        races = data[table_key]["Races"]
        out.extend(races)
        offset += 100
        if offset >= int(data["total"]):
            break
    # Pages split a race's rows across responses; merge them by round
    merged = {}
    for race in out:
        key = (race["season"], race["round"])
        if key in merged:
            merged[key][list_key].extend(race[list_key])
        else:
            merged[key] = race
    return list(merged.values())


def fetch_season(season, refresh=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{season}.csv")
    current = season >= pd.Timestamp.now().year
    if os.path.exists(path) and not refresh and not current:
        cached = pd.read_csv(path)
        if {"driver_name", "status", "nationality"}.issubset(cached.columns):  # older cache format -> refetch
            return cached

    print(f"Downloading {season}...")
    results = _paged(f"{season}/results", "RaceTable", "Results")
    rows = []
    for race in results:
        for r in race["Results"]:
            classified = r["positionText"].isdigit()
            rows.append({
                "season": int(race["season"]),
                "round": int(race["round"]),
                "race_name": race["raceName"],
                "circuit": race["Circuit"]["circuitId"],
                "country": race["Circuit"]["Location"]["country"],
                "date": race["date"],
                "driver": r["Driver"]["driverId"],
                "driver_name": f"{r['Driver']['givenName']} {r['Driver']['familyName']}",
                "nationality": r["Driver"].get("nationality", ""),
                "driver_code": r["Driver"].get("code", r["Driver"]["familyName"][:3].upper()),
                "team": r["Constructor"]["constructorId"],
                "team_name": r["Constructor"]["name"],
                "status": r["status"],
                "grid": int(r["grid"]) or 20,  # 0 = pit lane start
                "finish": int(r["position"]) if classified else DNF_POSITION,
                "points": float(r["points"]),
                "dnf": 0 if classified else 1,
            })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def fetch_schedule(season):
    """Full calendar for a season (completed and upcoming rounds)."""
    races = _get(f"{API}/{season}.json?limit=100")["RaceTable"]["Races"]
    return pd.DataFrame([{
        "season": int(r["season"]), "round": int(r["round"]), "race_name": r["raceName"],
        "circuit": r["Circuit"]["circuitId"], "country": r["Circuit"]["Location"]["country"],
        "locality": r["Circuit"]["Location"]["locality"], "date": r["date"],
    } for r in races])


def fetch_next_race():
    data = _get(f"{API}/current/next.json")
    race = data["RaceTable"]["Races"][0]
    season, rnd = int(race["season"]), int(race["round"])
    info = {
        "season": season,
        "round": rnd,
        "race_name": race["raceName"],
        "circuit": race["Circuit"]["circuitId"],
        "date": race["date"],
    }
    quali = _get(f"{API}/{season}/{rnd}/qualifying.json")["RaceTable"]["Races"]
    grid = None
    if quali:
        grid = {q["Driver"]["driverId"]: int(q["position"]) for q in quali[0]["QualifyingResults"]}
    return info, grid


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def build_features(df):
    """Every feature only uses races strictly before the current one (no leakage)."""
    df = df.sort_values(["season", "round"]).reset_index(drop=True)
    df["race_idx"] = df.groupby(["season", "round"]).ngroup()

    def roll(group_cols, col, window):
        return (df.groupby(group_cols)[col]
                  .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean()))

    df["drv_avg_finish_5"] = roll("driver", "finish", 5)
    df["drv_avg_grid_5"] = roll("driver", "grid", 5)
    df["drv_avg_points_5"] = roll("driver", "points", 5)
    df["drv_dnf_rate_10"] = roll("driver", "dnf", 10)
    df["drv_track_avg_finish"] = (df.groupby(["driver", "circuit"])["finish"]
                                    .transform(lambda s: s.shift(1).expanding().mean()))

    # Team form is computed within a season (cars change a lot between seasons,
    # especially across the 2026 regulation reset). Average both cars per race first.
    team_race = (df.groupby(["season", "team", "race_idx"])[["finish", "grid"]]
                   .mean().reset_index().sort_values("race_idx"))
    for col in ["finish", "grid"]:
        team_race[f"team_avg_{col}_5"] = (team_race.groupby(["season", "team"])[col]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean()))
    df = df.merge(team_race[["season", "team", "race_idx", "team_avg_finish_5", "team_avg_grid_5"]],
                  on=["season", "team", "race_idx"], how="left")

    df["drv_season_points"] = (df.groupby(["season", "driver"])["points"]
                                 .transform(lambda s: s.shift(1).cumsum()).fillna(0))
    df["races_in_season"] = df.groupby(["season", "driver"]).cumcount()

    # Sensible defaults for rookies / first race of a season
    df["drv_avg_finish_5"] = df["drv_avg_finish_5"].fillna(12)
    df["drv_avg_grid_5"] = df["drv_avg_grid_5"].fillna(12)
    df["drv_avg_points_5"] = df["drv_avg_points_5"].fillna(0)
    df["drv_dnf_rate_10"] = df["drv_dnf_rate_10"].fillna(0.1)
    df["team_avg_finish_5"] = df["team_avg_finish_5"].fillna(df["drv_avg_finish_5"])
    df["team_avg_grid_5"] = df["team_avg_grid_5"].fillna(df["drv_avg_grid_5"])
    df["drv_track_avg_finish"] = df["drv_track_avg_finish"].fillna(df["drv_avg_finish_5"])

    # Race craft: average places gained from grid when finishing (last 5)
    gain = (df["grid"] - df["finish"]).where(df["dnf"] == 0)
    df["drv_avg_gain_5"] = (gain.groupby(df["driver"])
                              .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())).fillna(0)
    df["drv_last_finish"] = df.groupby("driver")["finish"].shift(1).fillna(12)
    df["drv_avg_finish_season"] = (df.groupby(["season", "driver"])["finish"]
                                     .transform(lambda s: s.shift(1).expanding().mean())
                                     ).fillna(df["drv_avg_finish_5"])
    team_dnf = (df.groupby(["season", "team", "race_idx"])["dnf"].mean().reset_index()
                  .sort_values("race_idx"))
    team_dnf["team_dnf_rate_10"] = team_dnf.groupby(["season", "team"])["dnf"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    df = df.merge(team_dnf[["season", "team", "race_idx", "team_dnf_rate_10"]],
                  on=["season", "team", "race_idx"], how="left")
    df["team_dnf_rate_10"] = df["team_dnf_rate_10"].fillna(0.1)
    return add_grid_features(df)


def add_grid_features(df):
    """Features that depend on this race's grid. Re-run after the grid changes (what-if edits)."""
    df = df.copy()
    key = ["season", "round", "team"]
    total = df.groupby(key)["grid"].transform("sum")
    count = df.groupby(key)["grid"].transform("count")
    mate = (total - df["grid"]) / (count - 1).clip(lower=1)
    df["grid_vs_teammate"] = np.where(count > 1, df["grid"] - mate, 0.0)
    df["out_of_position"] =  df["grid"] - df["team_avg_grid_5"]
    return df


def training_rows(df):
    """DNFs are mostly bad luck (crashes, failures) and only add noise to the finishing-order target."""
    return df[df["dnf"] == 0]


def next_race_rows(hist, info, grid):
    """Build feature rows for the upcoming race by appending a placeholder race."""
    season = info["season"]
    last = hist[hist["season"] == season]
    if last.empty:
        last = hist[hist["season"] == hist["season"].max()]
    entrants = last[last["round"] == last["round"].max()][
        [c for c in ["driver", "driver_code", "driver_name", "nationality", "team", "team_name"] if c in last.columns]]

    placeholder = entrants.assign(
        season=season, round=info["round"], race_name=info["race_name"],
        circuit=info["circuit"], date=info["date"],
        grid=np.nan, finish=np.nan, points=0.0, dnf=0,
    )
    combined = build_features(pd.concat([hist, placeholder], ignore_index=True))
    rows = combined[(combined["season"] == season) & (combined["round"] == info["round"])].copy()

    if grid:
        rows["grid"] = rows["driver"].map(grid)
        rows["grid_source"] = "qualifying"
    else:
        # No quali yet: rank drivers by blended recent grid of driver and team
        est = 0.6 * rows["drv_avg_grid_5"] + 0.4 * rows["team_avg_grid_5"]
        rows["grid"] = est.rank(method="first")
        rows["grid_source"] = "estimated"
    rows["grid"] = rows["grid"].fillna(20)
    return add_grid_features(rows)


# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
def make_model():
    # Chosen by walk-forward tuning on 2025-26 races (see README "Model tuning")
    return HistGradientBoostingRegressor(max_iter=200, max_depth=3, learning_rate=0.03,
                                         min_samples_leaf=40, random_state=42)


def fit_model(df):
    tr = training_rows(df)
    return make_model().fit(tr[FEATURES], tr["finish"], sample_weight=sample_weights(tr))


def importances(model, df, n=600):
    """Permutation importance: how much accuracy drops when each input is shuffled."""
    tr = training_rows(df)
    tr = tr.tail(n)
    r = permutation_importance(model, tr[FEATURES], tr["finish"], n_repeats=3, random_state=0)
    imp = pd.Series(np.clip(r.importances_mean, 0, None), index=FEATURES)
    return imp / imp.sum() if imp.sum() > 0 else imp


def sample_weights(df):
    # Recent seasons matter more, especially after the 2026 regulation change
    newest = df["season"].max()
    return np.power(0.5, newest - df["season"])


def backtest(df, n_races=8):
    """Walk-forward test: for each of the last n races, train on everything before it."""
    races = df[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    tests = races.tail(n_races)
    winners_hit, podium_overlap, maes, per_race = 0, [], [], []
    for _, (s, r) in tests.iterrows():
        before = (df["season"] < s) | ((df["season"] == s) & (df["round"] < r))
        train, test = df[before], df[(df["season"] == s) & (df["round"] == r)].copy()
        model = fit_model(train)
        test["pred"] = model.predict(test[FEATURES])
        test["pred_rank"] = test["pred"].rank(method="first")
        winners_hit += int(test.loc[test["pred_rank"] == 1, "finish"].iloc[0] == 1)
        pred_podium = set(test.nsmallest(3, "pred")["driver"])
        real_podium = set(test[test["finish"] <= 3]["driver"])
        podium_overlap.append(len(pred_podium & real_podium) / 3)
        maes.append((test["pred_rank"] - test["finish"].clip(upper=20)).abs().mean())
        per_race.append({"season": s, "round": r, "race": f"{s} R{r}",
                         "mae": maes[-1], "podium": podium_overlap[-1],
                         "win_hit": bool(test.loc[test["pred_rank"] == 1, "finish"].iloc[0] == 1)})
    stats = {"per_race": pd.DataFrame(per_race), "n_races": n_races, "winners_hit": winners_hit,
             "podium_overlap": float(np.mean(podium_overlap)), "mae": float(np.mean(maes))}
    print(f"\nBack-test on last {n_races} races:")
    print(f"  Winner correct:      {winners_hit}/{n_races}")
    print(f"  Podium overlap:      {stats['podium_overlap']:.0%}")
    print(f"  Mean position error: {stats['mae']:.2f} places")
    return stats


def predict(df, rows):
    model = fit_model(df)
    rows = rows.copy()
    rows["pred_score"] = model.predict(rows[FEATURES])
    rows = rows.sort_values("pred_score").reset_index(drop=True)
    rows["predicted_pos"] = np.arange(1, len(rows) + 1)

    return rows, importances(model, df).sort_values(ascending=False)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    this_year = pd.Timestamp.now().year
    p.add_argument("--seasons", nargs="+", type=int, default=list(range(this_year - 4, this_year + 1)))
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--backtest", type=int, default=8, help="races to back-test (0 to skip)")
    args = p.parse_args()

    hist = pd.concat([fetch_season(s, args.refresh) for s in args.seasons], ignore_index=True)
    df = build_features(hist)
    print(f"Loaded {df[['season','round']].drop_duplicates().shape[0]} races, {len(df)} driver results.")

    if args.backtest:
        backtest(df, args.backtest)

    info, grid = fetch_next_race()
    rows = next_race_rows(hist, info, grid)
    result, importances = predict(df, rows)

    print(f"\n=== {info['season']} {info['race_name']} ({info['date']}) ===")
    print(f"Grid source: {result['grid_source'].iloc[0]}\n")
    print(f"{'Pos':>3}  {'Driver':<6} {'Team':<16} {'Grid':>4}")
    for _, r in result.iterrows():
        print(f"{r['predicted_pos']:>3}  {r['driver_code']:<6} {r['team']:<16} {int(r['grid']):>4}")

    print("\nWhat the model leans on:")
    for f, v in importances.head(5).items():
        print(f"  {f:<22} {v:.0%}")


if __name__ == "__main__":
    main()
