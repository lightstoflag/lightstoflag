#!/usr/bin/env python3
"""
Walk-forward evaluation harness for the LightsToFlag challenges (see CHALLENGES.md).

    python evaluate.py              # score on the TUNE races (use this while you work)
    python evaluate.py --final      # score on the held-out TEST races (run once, at the end)

Every race is predicted using only races before it. Edit `my_scores()` (Challenge 2)
and `NOISE_MULT` / `my_probabilities()` (Challenge 1); leave the rest alone so the
results stay comparable.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import f1_predict as f1

SEASONS = range(2022, 2027)
N_TUNE, N_TEST = 20, 10          # last 30 races: first 20 to tune on, last 10 locked away


# --------------------------------------------------------------------------- #
# ✏️  CHALLENGE 2: change how a race is scored (lower score = better finish)
# --------------------------------------------------------------------------- #
def my_scores(train, race):
    return f1.fit_model(train).predict(race[f1.FEATURES])


# --------------------------------------------------------------------------- #
# ✏️  CHALLENGE 1: turn scores into win / podium probabilities
# --------------------------------------------------------------------------- #
NOISE_MULT = 0.75   # the app's current guess: residual std x 0.75


def my_probabilities(train, race, scores, n_sims=5000, seed=7):
    """Return (p_win, p_podium) arrays, one value per driver in `race`."""
    model = f1.fit_model(train)
    tr = f1.training_rows(train)
    noise = (tr["finish"] - model.predict(tr[f1.FEATURES])).std() * NOISE_MULT
    rng = np.random.default_rng(seed)
    sims = scores[None, :] + rng.normal(0, noise, size=(n_sims, len(scores)))
    ranks = sims.argsort(axis=1).argsort(axis=1) + 1
    return (ranks == 1).mean(axis=0), (ranks <= 3).mean(axis=0)


# --------------------------------------------------------------------------- #
# Harness (don't edit below, so scores stay comparable)
# --------------------------------------------------------------------------- #
def load():
    hist = pd.concat([pd.read_csv(f"f1_cache/{s}.csv") for s in SEASONS], ignore_index=True)
    df = f1.build_features(hist)
    races = df[["season", "round"]].drop_duplicates().sort_values(["season", "round"]).values.tolist()
    return df, races


def evaluate(df, races, probs=True):
    rows, cal = [], []
    for s, r in races:
        before = (df["season"] < s) | ((df["season"] == s) & (df["round"] < r))
        train, race = df[before], df[(df["season"] == s) & (df["round"] == r)].copy()
        race["score"] = np.asarray(my_scores(train, race), dtype=float)
        race["prank"] = race["score"].rank(method="first")
        fin = race[race["dnf"] == 0]
        rows.append({
            "race": f"{s} R{r}",
            "win": int(race.loc[race["prank"] == 1, "finish"].iloc[0] == 1),
            "podium": len(set(race.nsmallest(3, "score")["driver"]) &
                          set(race[race["finish"] <= 3]["driver"])) / 3,
            "rho": spearmanr(fin["prank"], fin["finish"]).correlation,
            "mae": (fin["score"].rank() - fin["finish"].rank()).abs().mean(),
        })
        if probs:
            pw, pp = my_probabilities(train, race, race["score"].values)
            cal.append(pd.DataFrame({"p_win": pw, "p_pod": pp,
                                     "won": (race["finish"] == 1).astype(int).values,
                                     "podium": (race["finish"] <= 3).astype(int).values}))
    return pd.DataFrame(rows), (pd.concat(cal, ignore_index=True) if cal else None)


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def log_loss(p, y, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def report(per_race, cal, label):
    print(f"\n=== {label}: {len(per_race)} races ===")
    print(f"Ordering  | winner {per_race['win'].sum()}/{len(per_race)}  "
          f"podium {per_race['podium'].mean():.0%}  "
          f"rank-corr {per_race['rho'].mean():.3f}  "
          f"pos-error {per_race['mae'].mean():.2f}")
    if cal is not None:
        print(f"Win prob  | Brier {brier(cal.p_win, cal.won):.4f}  log-loss {log_loss(cal.p_win, cal.won):.4f}")
        print(f"Pod prob  | Brier {brier(cal.p_pod, cal.podium):.4f}  log-loss {log_loss(cal.p_pod, cal.podium):.4f}")
        # Reliability table: when we said X%, how often did it happen?
        bins = [0, .05, .15, .30, .50, 1.0]
        t = cal.assign(bucket=pd.cut(cal.p_pod, bins, include_lowest=True))
        rel = t.groupby("bucket", observed=True).agg(n=("podium", "size"),
                                                     predicted=("p_pod", "mean"),
                                                     actual=("podium", "mean"))
        print("Podium reliability (predicted vs actual):")
        print(rel.to_string(float_format=lambda v: f"{v:.0%}" if v <= 1 else f"{v:.0f}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="score the locked TEST races (do this once)")
    ap.add_argument("--no-probs", action="store_true", help="skip Challenge 1 metrics (faster)")
    a = ap.parse_args()
    df, races = load()
    tune, test = races[-(N_TUNE + N_TEST):-N_TEST], races[-N_TEST:]
    chosen, label = (test, "TEST (held-out)") if a.final else (tune, "TUNE")
    per_race, cal = evaluate(df, chosen, probs=not a.no_probs)
    report(per_race, cal, label)
