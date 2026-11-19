"""
"Ask LightsToFlag" chat: answers questions using the app's own predictions and results.

Two backends:
  * Claude (Anthropic API) - needs ANTHROPIC_API_KEY in your environment or .streamlit/secrets.toml
  * Ollama (free, runs locally) - needs `ollama serve` running and a model pulled, e.g. `ollama pull llama3.1`
"""
import json
import os

import pandas as pd
import requests

CLAUDE_MODELS = {
    "Claude Haiku 5.5 (fast, cheapest)": "claude-haiku-5-5",
    "Claude Sonnet 5.5 (smarter)": "claude-sonnet-5-5",
}
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

SYSTEM_PROMPT = """You are the race analyst inside LightsToFlag, an F1 race-prediction app.
Answer the user's questions using ONLY the data below: the model's prediction for the next race,
this season's results and standings, and the model's track record. Be concise and specific: name
drivers, positions and percentages. If the data doesn't contain the answer (e.g. weather, news,
tyre strategy, anything after the latest result), say so plainly instead of guessing.
Remember the predictions are probabilities from a model, not certainties.

{context}"""


# --------------------------------------------------------------------------- #
# Context: a compact text snapshot of what the app knows
# --------------------------------------------------------------------------- #
def build_context(info, prediction, season_hist, backtest=None, grid_source="estimated"):
    parts = []

    if prediction is not None and len(prediction):
        p = prediction.sort_values("Pos")
        lines = [f"P{int(r.Pos)} {r.Driver} ({r.Team}) grid P{int(r.grid)} | "
                 f"win {r.win:.0f}% podium {r.podium:.0f}% top10 {r.top10:.0f}%"
                 for r in p.itertuples()]
        parts.append(f"## Prediction: {info['season']} {info['race_name']} on {info['date']} "
                     f"(grid: {grid_source})\n" + "\n".join(lines))

    if season_hist is not None and len(season_hist):
        sh = season_hist
        name = sh["driver_name"] if "driver_name" in sh else sh["driver_code"]
        team = sh["team_name"] if "team_name" in sh else sh["team"]
        sh = sh.assign(dname=name, tname=team)
        st = (sh.groupby(["dname", "tname"]).agg(pts=("points", "sum"),
                                           wins=("finish", lambda s: int((s == 1).sum())),
                                           podiums=("finish", lambda s: int((s <= 3).sum())),
                                           dnfs=("dnf", "sum"))
                .reset_index().sort_values("pts", ascending=False))
        parts.append("## Drivers' standings\n" + "\n".join(
            f"{i}. {r.dname} ({r.tname}) {r.pts:.0f} pts, {r.wins} wins, {r.podiums} podiums, {int(r.dnfs)} DNFs"
            for i, r in enumerate(st.itertuples(), 1)))
        teams = sh.groupby("tname")["points"].sum().sort_values(ascending=False)
        parts.append("## Constructors\n" + "\n".join(f"{t}: {v:.0f} pts" for t, v in teams.items()))

        races = []
        for (rnd, rname), g in sh.groupby(["round", "race_name"]):
            g = g.sort_values(["finish", "grid"])
            top = ", ".join(f"P{int(r.finish)} {r.dname} (from P{int(r.grid)})"
                            for r in g[g["dnf"] == 0].head(10).itertuples())
            dnf = ", ".join(g[g["dnf"] == 1]["dname"])
            races.append(f"R{rnd} {rname}: {top}" + (f" | DNF: {dnf}" if dnf else ""))
        parts.append("## Race results this season (top 10 + DNFs)\n" + "\n".join(races))

    if backtest:
        parts.append(f"## Model track record (last {backtest['n_races']} races, walk-forward)\n"
                     f"Winner correct {backtest['winners_hit']}/{backtest['n_races']}, "
                     f"podium overlap {backtest['podium_overlap']:.0%}, "
                     f"avg position error {backtest['mae']:.1f} places")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Backends (both stream text chunks)
# --------------------------------------------------------------------------- #
def claude_available(api_key):
    return bool(api_key)


def stream_claude(messages, context, model, api_key):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    with client.messages.stream(
        model=model, max_tokens=1024,
        system=SYSTEM_PROMPT.format(context=context),
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
    ) as stream:
        for text in stream.text_stream:
            yield text


def ollama_models():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=1.5)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def stream_ollama(messages, context, model):
    payload = {
        "model": model, "stream": True,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
                    + [{"role": m["role"], "content": m["content"]} for m in messages],
    }
    with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                chunk = json.loads(line)
                if "message" in chunk:
                    yield chunk["message"].get("content", "")


SUGGESTIONS = [
    "Who's most likely to win, and why?",
    "Who could surprise from further back?",
    "How has the championship leader done in the last 5 races?",
    "Which team has the most DNFs this season?",
]
