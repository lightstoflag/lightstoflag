"""
Shared fixtures for the LightsToFlag test suite.

Every test runs offline: the F1 API is replaced with small, made-up data, so the
suite is fast and gives the same result every time.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import f1_predict as f1  # noqa: E402

# 10 drivers in 5 teams. Lower "skill" = faster, so d0 is the best driver.
DRIVERS = [
    # driver id, code, name, nationality, team id, team name
    ("d0", "AAA", "Alice Ace", "British", "t0", "Team Zero"),
    ("d1", "BBB", "Bob Bold", "Dutch", "t0", "Team Zero"),
    ("d2", "CCC", "Cara Cruz", "Spanish", "t1", "Team One"),
    ("d3", "DDD", "Dev Dash", "Indian", "t1", "Team One"),
    ("d4", "EEE", "Eli Edge", "French", "t2", "Team Two"),
    ("d5", "FFF", "Fay Fast", "Italian", "t2", "Team Two"),
    ("d6", "GGG", "Gus Grit", "German", "t3", "Team Three"),
    ("d7", "HHH", "Hana Hale", "Japanese", "t3", "Team Three"),
    ("d8", "III", "Ivo Ion", "Brazilian", "t4", "Team Four"),
    ("d9", "JJJ", "Jo Jet", "Australian", "t4", "Team Four"),
]
CIRCUITS = ["c_a", "c_b", "c_c", "c_d", "c_e", "c_f"]
POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]


def make_history(seasons=(2024, 2025), rounds=6, seed=0, dnf_rate=0.08):
    """Synthetic results in the same columns fetch_season() produces."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for rnd in range(1, rounds + 1):
            skill = {d[0]: i + rng.normal(0, 1.5) for i, d in enumerate(DRIVERS)}
            grid_order = sorted(DRIVERS, key=lambda d: skill[d[0]])
            race_order = sorted(grid_order, key=lambda d: grid_order.index(d) + rng.normal(0, 1.5))
            for drv, code, name, nat, team, team_name in DRIVERS:
                dnf = rng.random() < dnf_rate
                pos = race_order.index(next(d for d in DRIVERS if d[0] == drv)) + 1
                rows.append(dict(
                    season=season, round=rnd, race_name=f"Race {rnd} Grand Prix",
                    circuit=CIRCUITS[(rnd - 1) % len(CIRCUITS)], country="Testland",
                    date=f"{season}-0{min(rnd, 9)}-01",
                    driver=drv, driver_name=name, nationality=nat, driver_code=code,
                    team=team, team_name=team_name,
                    status="Accident" if dnf else "Finished",
                    grid=[d[0] for d in grid_order].index(drv) + 1,
                    finish=f1.DNF_POSITION if dnf else pos,
                    points=0.0 if dnf else float(POINTS[pos - 1]),
                    dnf=int(dnf),
                ))
    return pd.DataFrame(rows)


@pytest.fixture
def hist():
    return make_history()


@pytest.fixture
def features(hist):
    return f1.build_features(hist)


@pytest.fixture
def next_info():
    return dict(season=2025, round=7, race_name="Race 7 Grand Prix", circuit="c_a", date="2025-12-01")


# --------------------------------------------------------------------------- #
# Fake Jolpica API responses
# --------------------------------------------------------------------------- #
def api_result(position_text, grid, points, status="Finished", driver_id="d0", code="AAA"):
    return {
        "positionText": position_text,
        "position": position_text if position_text.isdigit() else "20",
        "grid": str(grid), "points": str(points), "status": status,
        "Driver": {"driverId": driver_id, "code": code, "givenName": "Alice",
                   "familyName": "Ace", "nationality": "British"},
        "Constructor": {"constructorId": "t0", "name": "Team Zero"},
    }


def api_race(season, rnd, results):
    return {
        "season": str(season), "round": str(rnd), "raceName": f"Race {rnd} Grand Prix",
        "date": f"{season}-03-0{rnd}",
        "Circuit": {"circuitId": f"c{rnd}", "Location": {"country": "Testland", "locality": "Testville"}},
        "Results": results,
    }


@pytest.fixture
def no_network(monkeypatch):
    """Fail loudly if any test accidentally tries to reach the real API."""
    def blocked(*a, **k):
        raise AssertionError("Tests must not call the real F1 API")
    monkeypatch.setattr(f1.requests, "get", blocked)
