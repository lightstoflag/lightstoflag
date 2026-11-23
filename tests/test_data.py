"""Data download and parsing, with the API replaced by fake responses."""
import pandas as pd
import pytest

import f1_predict as f1
from conftest import api_race, api_result


@pytest.fixture
def fake_api(monkeypatch, tmp_path, no_network):
    """Route _get() to a dict of canned responses and cache to a temp folder."""
    responses = {}

    def fake_get(url):
        for key, value in responses.items():
            if key in url:
                return value
        raise AssertionError(f"Unexpected URL in test: {url}")

    monkeypatch.setattr(f1, "_get", fake_get)
    monkeypatch.setattr(f1, "CACHE_DIR", str(tmp_path))
    return responses


def mrdata(races, total=None):
    return {"total": str(total if total is not None else len(races)), "RaceTable": {"Races": races}}


def test_fetch_season_parses_results(fake_api):
    fake_api["2024/results"] = mrdata([api_race(2024, 1, [
        api_result("1", 2, 25),
        api_result("R", 0, 0, status="Engine", driver_id="d1", code="BBB"),
    ])])
    df = f1.fetch_season(2024, refresh=True)

    winner = df[df["driver"] == "d0"].iloc[0]
    assert (winner["finish"], winner["grid"], winner["points"], winner["dnf"]) == (1, 2, 25.0, 0)
    assert winner["driver_name"] == "Alice Ace" and winner["nationality"] == "British"

    retired = df[df["driver"] == "d1"].iloc[0]
    assert retired["dnf"] == 1
    assert retired["finish"] == f1.DNF_POSITION
    assert retired["grid"] == 20            # grid 0 = pit-lane start
    assert retired["status"] == "Engine"


def test_fetch_season_writes_cache(fake_api, tmp_path):
    fake_api["2023/results"] = mrdata([api_race(2023, 1, [api_result("1", 1, 25)])])
    f1.fetch_season(2023, refresh=True)
    cached = pd.read_csv(tmp_path / "2023.csv")
    assert len(cached) == 1


def test_old_seasons_are_read_from_cache(fake_api, tmp_path):
    fake_api["2022/results"] = mrdata([api_race(2022, 1, [api_result("1", 1, 25)])])
    f1.fetch_season(2022, refresh=True)
    fake_api.clear()                         # any further API call would now fail
    assert len(f1.fetch_season(2022)) == 1


def test_old_cache_without_new_columns_is_refetched(fake_api, tmp_path):
    pd.DataFrame({"season": [2022], "driver": ["x"]}).to_csv(tmp_path / "2022.csv", index=False)
    fake_api["2022/results"] = mrdata([api_race(2022, 1, [api_result("1", 1, 25)])])
    df = f1.fetch_season(2022)
    assert "driver_name" in df.columns


def test_paged_merges_a_race_split_across_pages(monkeypatch):
    page1 = mrdata([api_race(2024, 1, [api_result("1", 1, 25)])], total=150)
    page2 = mrdata([api_race(2024, 1, [api_result("2", 2, 18, driver_id="d1")])], total=150)
    pages = iter([page1, page2])
    monkeypatch.setattr(f1, "_get", lambda url: next(pages))
    races = f1._paged("2024/results", "RaceTable", "Results")
    assert len(races) == 1
    assert len(races[0]["Results"]) == 2


def test_fetch_next_race_with_and_without_qualifying(fake_api):
    race = api_race(2026, 22, [])
    fake_api["current/next"] = mrdata([race])
    fake_api["2026/22/qualifying"] = mrdata([{
        "QualifyingResults": [{"Driver": {"driverId": "d0"}, "position": "1"},
                              {"Driver": {"driverId": "d1"}, "position": "2"}]}])
    info, grid = f1.fetch_next_race()
    assert (info["season"], info["round"]) == (2026, 22)
    assert grid == {"d0": 1, "d1": 2}

    fake_api["2026/22/qualifying"] = mrdata([])
    _, grid = f1.fetch_next_race()
    assert grid is None


def test_get_retries_after_rate_limit(monkeypatch):
    calls = []

    class Resp:
        def __init__(self, code):
            self.status_code = code

        def raise_for_status(self):
            pass

        def json(self):
            return {"MRData": {"ok": True}}

    def fake_requests_get(url, timeout):
        calls.append(url)
        return Resp(429 if len(calls) == 1 else 200)

    monkeypatch.setattr(f1.requests, "get", fake_requests_get)
    monkeypatch.setattr(f1.time, "sleep", lambda s: None)
    assert f1._get("https://example.test") == {"ok": True}
    assert len(calls) == 2
