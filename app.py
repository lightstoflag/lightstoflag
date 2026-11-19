"""
Interactive UI for the F1 race predictor.

    pip install streamlit pandas scikit-learn requests
    streamlit run app.py

Keep this file next to f1_predict.py and engine_audio.py.
"""
import html
import os

import numpy as np
import pandas as pd
import streamlit as st

import charts
import engine_audio
import f1_predict as f1

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
st.set_page_config(page_title="LightsToFlag · F1 race predictions",
                   page_icon=os.path.join(ASSETS, "logo_icon.png"), layout="wide")

FLAGS = {
    "Australia": "🇦🇺", "China": "🇨🇳", "Japan": "🇯🇵", "Bahrain": "🇧🇭", "Saudi Arabia": "🇸🇦",
    "USA": "🇺🇸", "United States": "🇺🇸", "Italy": "🇮🇹", "Monaco": "🇲🇨", "Spain": "🇪🇸",
    "Canada": "🇨🇦", "Austria": "🇦🇹", "UK": "🇬🇧", "United Kingdom": "🇬🇧", "Belgium": "🇧🇪",
    "Hungary": "🇭🇺", "Netherlands": "🇳🇱", "Azerbaijan": "🇦🇿", "Singapore": "🇸🇬",
    "Mexico": "🇲🇽", "Brazil": "🇧🇷", "Qatar": "🇶🇦", "UAE": "🇦🇪", "United Arab Emirates": "🇦🇪",
    "France": "🇫🇷", "Portugal": "🇵🇹", "Germany": "🇩🇪", "Turkey": "🇹🇷", "Russia": "🇷🇺",
}
NATIONALITY_FLAGS = {
    "British": "🇬🇧", "Dutch": "🇳🇱", "Monegasque": "🇲🇨", "Spanish": "🇪🇸", "Mexican": "🇲🇽",
    "Australian": "🇦🇺", "Canadian": "🇨🇦", "French": "🇫🇷", "German": "🇩🇪", "Finnish": "🇫🇮",
    "Thai": "🇹🇭", "Japanese": "🇯🇵", "Chinese": "🇨🇳", "Danish": "🇩🇰", "American": "🇺🇸",
    "Italian": "🇮🇹", "New Zealander": "🇳🇿", "Brazilian": "🇧🇷", "Argentine": "🇦🇷",
    "Argentinian": "🇦🇷", "Russian": "🇷🇺", "Polish": "🇵🇱", "Swiss": "🇨🇭", "Belgian": "🇧🇪",
    "Swedish": "🇸🇪", "Austrian": "🇦🇹", "Indonesian": "🇮🇩", "Venezuelan": "🇻🇪",
    "Colombian": "🇨🇴", "Indian": "🇮🇳", "Estonian": "🇪🇪", "Irish": "🇮🇪", "Portuguese": "🇵🇹",
    "Hungarian": "🇭🇺", "Czech": "🇨🇿", "Malaysian": "🇲🇾", "South African": "🇿🇦",
}
MEDALS = ["🥇", "🥈", "🥉"]


def is_dark():
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


DARK = is_dark()
st.logo(os.path.join(ASSETS, "logo_badge.svg"),
        icon_image=os.path.join(ASSETS, "logo_icon.svg"), size="large")
PLOT = dict(width="stretch", config={"displayModeBar": False})

st.markdown("""
<style>
.f1-card { padding: 0.9rem 1rem; border-radius: 12px; border: 1px solid rgba(128,128,128,0.25);
           background: rgba(128,128,128,0.06); height: 100%; }
.f1-card .lbl { font-size: 0.85rem; opacity: 0.75; margin-bottom: 0.2rem; }
.f1-card .val { font-size: clamp(1.15rem, 1.8vw, 1.6rem); font-weight: 600; line-height: 1.25;
                overflow-wrap: anywhere; }
.f1-card .sub { font-size: 0.85rem; opacity: 0.75; margin-top: 0.35rem; }
[data-testid="stSidebarHeader"] img, [data-testid="stLogo"] { height: 3.4rem !important; max-width: 100%; }
section[data-testid="stSidebar"] .stButton button { justify-content: flex-start; text-align: left; }
section[data-testid="stSidebar"] .stButton button p { text-align: left; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


def card(col, label, value, sub=""):
    """Metric-style card whose value wraps instead of being cut off with '...'."""
    e = html.escape
    col.markdown(f'<div class="f1-card"><div class="lbl">{e(str(label))}</div>'
                 f'<div class="val">{e(str(value))}</div>'
                 + (f'<div class="sub">{e(str(sub))}</div>' if sub else "") + "</div>",
                 unsafe_allow_html=True)


def short(name):
    return name.replace(" Grand Prix", "").replace("Grand Prix", "").strip() or name


def name_col(frame):
    """Driver name with a nationality flag in front, e.g. '🇮🇹 Kimi Antonelli'."""
    names = frame["driver_name"] if "driver_name" in frame else frame["driver_code"]
    if "nationality" not in frame:
        return names
    flags = frame["nationality"].map(NATIONALITY_FLAGS).fillna("🏳️")
    return flags + " " + names


def flag(country):
    return FLAGS.get(country, "🏁")


# --------------------------------------------------------------------------- #
# Cached data + model
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=3600, show_spinner="Downloading race history...")
def load_history(seasons, refresh_token):
    return pd.concat([f1.fetch_season(s, refresh=refresh_token > 0) for s in seasons],
                     ignore_index=True)


@st.cache_data(ttl=3600, show_spinner="Loading calendar...")
def load_schedule(season, refresh_token):
    return f1.fetch_schedule(season)


@st.cache_data(ttl=900, show_spinner="Checking the next race and qualifying...")
def load_next_race(refresh_token):
    return f1.fetch_next_race()


@st.cache_resource(show_spinner="Training model...")
def train(df_key, _df):
    model = f1.make_model().fit(_df[f1.FEATURES], _df["finish"],
                                sample_weight=f1.sample_weights(_df))
    resid = _df["finish"] - model.predict(_df[f1.FEATURES])
    return model, float(resid.std() * 1.15)


@st.cache_data(show_spinner="Asking the model what it would have said...")
def hindsight(df_key, _df, season, rnd):
    """Predict a finished race using only data from before it."""
    before = (_df["season"] < season) | ((_df["season"] == season) & (_df["round"] < rnd))
    train_df = _df[before]
    if train_df[["season", "round"]].drop_duplicates().shape[0] < 5:
        return None
    model = f1.make_model().fit(train_df[f1.FEATURES], train_df["finish"],
                                sample_weight=f1.sample_weights(train_df))
    race = _df[(_df["season"] == season) & (_df["round"] == rnd)].copy()
    race["model_pos"] = model.predict(race[f1.FEATURES]).argsort().argsort() + 1
    return race.set_index("driver")["model_pos"]


@st.cache_data(show_spinner="Back-testing...")
def run_backtest(df_key, _df, n):
    return f1.backtest(_df, n)


@st.cache_data
def sound(name):
    """Use your own recording from ./sounds/<name>.mp3|.wav if present, else synthesise."""
    for ext, fmt in [("mp3", "audio/mpeg"), ("wav", "audio/wav")]:
        path = os.path.join("sounds", f"{name}.{ext}")
        if os.path.exists(path):
            with open(path, "rb") as fh:
                return fh.read(), fmt
    return getattr(engine_audio, name)(), "audio/wav"


def simulate(scores, noise, n_sims=5000, seed=7):
    rng = np.random.default_rng(seed)
    sims = scores[None, :] + rng.normal(0, noise, size=(n_sims, len(scores)))
    ranks = sims.argsort(axis=1).argsort(axis=1) + 1
    return (ranks == 1).mean(axis=0), (ranks <= 3).mean(axis=0), (ranks <= 10).mean(axis=0)


# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #
this_year = pd.Timestamp.now().year
ss = st.session_state
ss.setdefault("refresh", 0)
ss.setdefault("train_seasons", list(range(this_year - 4, this_year + 1)))

try:
    info, quali_grid = load_next_race(ss.refresh)
except Exception as e:
    st.error(f"Couldn't reach the F1 data API: {e}")
    st.stop()

# --------------------------------------------------------------------------- #
# Left nav: season + race list
# --------------------------------------------------------------------------- #
with st.sidebar:
    season = st.selectbox("Season", list(range(this_year, 2017, -1)), index=0)

try:
    sched = load_schedule(season, ss.refresh)
    train_seasons = sorted(set(ss.train_seasons) | {season})
    hist = load_history(tuple(train_seasons), ss.refresh)
except Exception as e:
    st.error(f"Couldn't load data: {e}")
    st.stop()

season_hist = hist[hist["season"] == season]
done_rounds = set(season_hist["round"].unique())
_w = season_hist[season_hist["finish"] == 1]
winners = dict(zip(_w["round"], name_col(_w)))
next_key = (info["season"], info["round"])

if "race" not in ss or ss.race[0] != season:
    ss.race = next_key if season == info["season"] else (season, int(sched["round"].max()))

with st.sidebar:
    dash_key = (season, "dash")
    if st.button(f"📊 {season} season dashboard", key="nav_dash", width="stretch",
                 type="primary" if ss.race == dash_key else "secondary"):
        ss.race = dash_key
        st.rerun()
    upcoming = sched[~sched["round"].isin(done_rounds)].sort_values("round")
    completed = sched[sched["round"].isin(done_rounds)].sort_values("round", ascending=False)

    if len(upcoming):
        st.caption("UPCOMING")
        for _, r in upcoming.iterrows():
            key = (season, int(r["round"]))
            is_next = key == next_key
            tag = "🔮 Prediction" if is_next else pd.Timestamp(r["date"]).strftime("%b %d")
            if st.button(f"R{r['round']} {flag(r['country'])} {short(r['race_name'])} · {tag}",
                         key=f"nav{key}", width="stretch", disabled=not is_next,
                         type="primary" if ss.race == key else "secondary"):
                ss.race = key
                st.rerun()

    st.caption("COMPLETED")
    for _, r in completed.iterrows():
        key = (season, int(r["round"]))
        w = winners.get(r["round"], "—")
        if isinstance(w, str) and " " in w:  # keep flag + surname
            w = f"{w.split()[0]} {w.split()[-1]}"
        if st.button(f"R{r['round']} {flag(r['country'])} {short(r['race_name'])} · 🏆 {w}",
                     key=f"nav{key}", width="stretch",
                     type="primary" if ss.race == key else "secondary"):
            ss.race = key
            st.rerun()

    with st.expander("⚙️ Settings"):
        ss.train_seasons = st.multiselect("Training seasons", list(range(2018, this_year + 1)),
                                          default=ss.train_seasons) or ss.train_seasons
        n_backtest = st.slider("Races to back-test", 0, 20, 8)
        sound_on = st.toggle("🔊 Engine sounds", value=True)
        if st.button("Re-download data"):
            ss.refresh += 1
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        st.caption("Data: Jolpica F1 API. Best predictions after qualifying.")

df = f1.build_features(hist)
df_key = (tuple(train_seasons), len(df), ss.refresh)
sel_season, sel_round = ss.race
race_meta = sched[sched["round"] == sel_round].iloc[0] if sel_round != "dash" else None


def header(meta, subtitle):
    st.title(f"{flag(meta['country'])} {meta['season']} {meta['race_name']}")
    st.caption(f"Round {meta['round']} · {meta.get('locality', '')} · "
               f"{pd.Timestamp(meta['date']).strftime('%a %d %b %Y')} · {subtitle}")


def driver_form(pool, default_circuit):
    names = pool[["driver", "label"]].drop_duplicates()
    pick = st.selectbox("Driver", names["label"].tolist())
    drv_id = names.loc[names["label"] == pick, "driver"].iloc[0]
    h = df[df["driver"] == drv_id].sort_values(["season", "round"]).tail(20).copy()
    h["race"] = h["season"].astype(str) + " R" + h["round"].astype(str).str.zfill(2)
    h["finish"] = h["finish"].where(h["dnf"] == 0)
    st.plotly_chart(charts.driver_form(h, DARK), **PLOT)
    st.caption("Higher on the chart = better position. ✕ marks a DNF.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Avg finish (last 5)", f"{h['finish'].tail(5).mean():.1f}")
    m2.metric("DNFs (last 20)", int(h["dnf"].sum()))
    track = df[(df["driver"] == drv_id) & (df["circuit"] == default_circuit)]
    best = track["finish"].min() if len(track) else 99
    m3.metric("Best at this track", f"P{int(best)}" if best < 21 else "—")


def team_col(frame):
    return frame["team_name"] if "team_name" in frame else frame["team"]


# --------------------------------------------------------------------------- #
# Season dashboard
# --------------------------------------------------------------------------- #
if sel_round == "dash":
    st.title(f"📊 {season} season dashboard")
    if season_hist.empty:
        st.info("No races completed yet this season.")
        st.stop()
    sd = season_hist.copy()
    sd["Driver"], sd["Team"] = name_col(sd), team_col(sd)
    standings = sd.groupby("Driver")["points"].sum().sort_values(ascending=False)
    wins = sd[sd["finish"] == 1]["Driver"].value_counts()
    podiums = sd[sd["finish"] <= 3]["Driver"].value_counts()

    k1, k2, k3, k4 = st.columns(4)
    card(k1, "Championship leader", standings.index[0], f"{standings.iloc[0]:.0f} pts")
    card(k2, "Gap to P2", f"{standings.iloc[0] - standings.iloc[1]:.0f} pts", standings.index[1])
    card(k3, "Most wins", wins.index[0], f"{wins.iloc[0]} wins")
    card(k4, "Races", f"{len(done_rounds)} / {len(sched)}", f"{len(sched) - len(done_rounds)} to go")

    st.subheader("Championship battle")
    picks = st.multiselect("Drivers (up to 4)", standings.index.tolist(),
                           default=standings.index[:4].tolist(), max_selections=4)
    if picks:
        st.plotly_chart(charts.championship_race(sd, picks, DARK), **PLOT)

    st.subheader("Every result this season")
    st.caption("Darker = better finish. Rows sorted by championship position.")
    st.plotly_chart(charts.results_heatmap(sd, standings.index.tolist(), DARK), **PLOT)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Constructors")
        st.plotly_chart(charts.team_points(sd, DARK), **PLOT)
    with c2:
        st.subheader("Standings")
        table = pd.DataFrame({"Points": standings, "Wins": wins, "Podiums": podiums}).fillna(0)
        table = table.astype(int).reset_index(names="Driver")
        table.insert(0, "Pos", range(1, len(table) + 1))
        st.dataframe(table, hide_index=True, width="stretch", height=30 * len(table) + 40)

# --------------------------------------------------------------------------- #
# Completed race view
# --------------------------------------------------------------------------- #
elif sel_round in done_rounds:
    race = season_hist[season_hist["round"] == sel_round].copy()
    race["Driver"], race["Team"] = name_col(race), team_col(race)
    race["label"] = race["Driver"]
    race = race.sort_values(["finish", "grid"])
    header(race_meta, "Final result")

    podium = race.head(3)
    cols = st.columns(3)
    for col, (_, r), medal in zip(cols, podium.iterrows(), MEDALS):
        card(col, f"{medal} P{int(r['finish'])}", r["Driver"],
             f"{r['Team']} · from P{int(r['grid'])}")

    tab_res, tab_drv = st.tabs(["Results", "Driver form"])
    with tab_res:
        hs = hindsight(df_key, df, sel_season, sel_round)
        race["Pos"] = np.where(race["dnf"] == 1, "DNF", race["finish"].astype(int).astype(str))
        race["Gained"] = np.where(race["dnf"] == 1, np.nan, race["grid"] - race["finish"])
        show = ["Pos", "Driver", "Team", "grid", "Gained", "points", "status"]
        if hs is not None:
            race["Model said"] = race["driver"].map(hs).astype("Int64")
            show.insert(4, "Model said")
            top3_hit = len(set(race.nsmallest(3, "finish")["driver"]) &
                           set(hs.nsmallest(3).index))
            win_hit = hs.idxmin() == race.iloc[0]["driver"]
            st.info(f"Model before this race: winner {'✅ correct' if win_hit else '❌ ' + race.set_index('driver').loc[hs.idxmin(), 'Driver']}"
                    f" · {top3_hit}/3 podium finishers")
        chart_col, table_col = st.columns([2, 3])
        with chart_col:
            options = ["Grid → Finish"] + (["Model → Finish"] if hs is not None else [])
            mode = st.segmented_control("Compare", options, default=options[0])
            src_col, src_label = (("Model said", "Model") if mode == "Model → Finish"
                                  else ("grid", "Grid"))
            finished = race[race["dnf"] == 0]
            st.plotly_chart(charts.position_shift(finished, src_col, "finish", src_label,
                                                  "Finish", DARK), **PLOT)
            st.caption("Blue = gained places, red = lost. DNFs not shown.")
        with table_col:
            st.dataframe(race[show], hide_index=True, width="stretch", height=740,
                         column_config={"grid": "Grid", "points": "Pts", "status": "Status",
                                        "Gained": st.column_config.NumberColumn("± Places", format="%+d")})
    with tab_drv:
        driver_form(race, race_meta["circuit"])

# --------------------------------------------------------------------------- #
# Next race: prediction view
# --------------------------------------------------------------------------- #
elif (sel_season, sel_round) == next_key:
    model, noise = train(df_key, df)
    base_rows = f1.next_race_rows(hist, info, quali_grid)
    base_rows["Driver"], base_rows["Team"] = name_col(base_rows), team_col(base_rows)
    base_rows["label"] = base_rows["Driver"]
    src = base_rows["grid_source"].iloc[0]
    header(race_meta, "Grid: **official qualifying**" if src == "qualifying"
           else "Grid: **estimated** (qualifying not published yet)")

    tab_pred, tab_model, tab_drv = st.tabs(["Prediction", "Model accuracy", "Driver form"])
    with tab_pred:
        left, right = st.columns([2, 3])
        with left:
            st.subheader("Starting grid")
            st.caption("Edit a grid slot to try a what-if (e.g. a grid penalty).")
            grid_df = (base_rows[["driver", "Driver", "Team", "grid"]]
                       .sort_values("grid").reset_index(drop=True))
            grid_df["grid"] = grid_df["grid"].astype(int)
            edited = st.data_editor(
                grid_df,
                column_config={
                    "driver": None,
                    "Driver": st.column_config.TextColumn(disabled=True),
                    "Team": st.column_config.TextColumn(disabled=True),
                    "grid": st.column_config.NumberColumn("Grid", min_value=1, max_value=22, step=1),
                },
                hide_index=True, width="stretch", key="grid_editor", height=740,
            )
            if st.button("Reset grid"):
                ss.pop("grid_editor", None)
                st.rerun()

        rows = base_rows.drop(columns="grid").merge(edited[["driver", "grid"]], on="driver")
        scores = model.predict(rows[f1.FEATURES])
        p_win, p_pod, p_pts = simulate(scores, noise)
        out = rows.assign(score=scores, win=p_win * 100, podium=p_pod * 100, top10=p_pts * 100)
        out = out.sort_values("score").reset_index(drop=True)
        out["Pos"] = np.arange(1, len(out) + 1)
        out["Δ vs grid"] = (out["grid"] - out["Pos"]).astype(int)

        with right:
            st.subheader("Predicted result")
            if sound_on:
                b1, b2, _ = st.columns([1, 1, 2])
                if b1.button("🚦 Lights out!", width="stretch"):
                    ss.play = "race_start"
                if b2.button("💨 Fly-by", width="stretch"):
                    ss.play = "fly_by"
                if ss.get("play"):
                    data, fmt = sound(ss.pop("play"))
                    st.audio(data, format=fmt, autoplay=True)
            c = st.columns(3)
            for col, (_, r), medal in zip(c, out.head(3).iterrows(), MEDALS):
                card(col, f"{medal} P{r['Pos']}", r["Driver"],
                     f"{r['Team']} · {r['win']:.0f}% win chance")
            pct = lambda label: st.column_config.ProgressColumn(label, format="%.0f%%",
                                                                min_value=0, max_value=100)
            st.dataframe(
                out[["Pos", "Driver", "Team", "grid", "Δ vs grid", "win", "podium", "top10"]],
                column_config={"grid": "Grid", "win": pct("Win"), "podium": pct("Podium"),
                               "top10": pct("Top 10")},
                hide_index=True, width="stretch", height=740,
            )
        st.caption("Win / podium / top-10 chances come from 5,000 simulated races "
                   "using the model's typical error.")
        g1, g2 = st.columns(2)
        with g1:
            st.subheader("Who can win?")
            st.plotly_chart(charts.chances(out, DARK), **PLOT)
        with g2:
            st.subheader("Grid → predicted finish")
            st.plotly_chart(charts.position_shift(out, "grid", "Pos", "Grid", "Predicted", DARK), **PLOT)
            st.caption("Blue = expected to gain places, red = expected to lose.")

    with tab_model:
        if n_backtest:
            bt = run_backtest(df_key + (n_backtest,), df, n_backtest)
            a, b, c3 = st.columns(3)
            a.metric("Winner correct", f"{bt['winners_hit']}/{bt['n_races']}")
            b.metric("Podium overlap", f"{bt['podium_overlap']:.0%}")
            c3.metric("Avg position error", f"{bt['mae']:.1f} places")
            st.caption("Walk-forward: each past race is predicted using only data from before it.")
            st.plotly_chart(charts.backtest_chart(bt["per_race"], DARK), **PLOT)
        st.subheader("What the model leans on")
        st.plotly_chart(charts.feature_importance(model, f1.FEATURES, DARK), **PLOT)
    with tab_drv:
        driver_form(out, info["circuit"])

# --------------------------------------------------------------------------- #
# Later rounds
# --------------------------------------------------------------------------- #
else:
    header(race_meta, "Upcoming")
    st.info("Predictions open once the previous round has finished.")
