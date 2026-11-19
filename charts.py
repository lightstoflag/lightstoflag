"""
Plotly charts for the F1 dashboard.

Colours follow a colour-blind-validated palette: categorical slots in fixed order,
a single-hue blue ramp for magnitude, blue/red with grey midpoint for gain/loss.
Pass dark=True to get the steps tuned for a dark background.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

SERIES = {
    False: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    True: ["#3987e5", "#d95926", "#199e70", "#c98500"],
}
SEQ_BLUE = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
GAIN, LOSS = "#2a78d6", "#e34948"
NEUTRAL = {False: "#c3c2b7", True: "#5d5c57"}
INK = {False: ("#0b0b0b", "#52514e", "#e9e8e3"), True: ("#f0efec", "#c3c2b7", "#33332f")}


def _layout(fig, dark, height, **kw):
    text, muted, grid = INK[dark]
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=text, size=13), hoverlabel=dict(font_size=13),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(color=muted)),
        **kw,
    )
    fig.update_xaxes(gridcolor=grid, zeroline=False, tickfont=dict(color=muted),
                     linecolor=grid, showspikes=False)
    fig.update_yaxes(gridcolor=grid, zeroline=False, tickfont=dict(color=muted), linecolor=grid)
    return fig


# --------------------------------------------------------------------------- #
# Season dashboard
# --------------------------------------------------------------------------- #
def championship_race(season_df, drivers, dark=False):
    """Cumulative points per round for up to 4 drivers, direct-labelled at the line ends."""
    colors = SERIES[dark]
    pts = (season_df.pivot_table(index="round", columns="Driver", values="points", aggfunc="sum")
           .fillna(0).cumsum())
    fig = go.Figure()
    drivers = drivers[:4]
    for i, d in enumerate(drivers):
        fig.add_trace(go.Scatter(
            x=pts.index, y=pts[d], name=d, mode="lines", line=dict(color=colors[i], width=2),
            hovertemplate=f"<b>{d}</b><br>After R%{{x}}: %{{y:.0f}} pts<extra></extra>",
        ))
    # End labels, nudged apart so close rivals don't overlap
    ends = sorted(((pts[d].iloc[-1], d) for d in drivers), reverse=True)
    min_gap = max(pts[drivers].values.max(), 1) * 0.055
    placed = []
    for val, d in ends:
        y = val if not placed else min(val, placed[-1] - min_gap)
        placed.append(y)
        fig.add_annotation(x=pts.index[-1], y=y, text=f"{d.split()[-1]} {val:.0f}",
                           xanchor="left", xshift=8, showarrow=False,
                           font=dict(color=INK[dark][0], size=12))
    fig.update_layout(margin=dict(r=110))
    fig.update_xaxes(title="Round", dtick=2)
    fig.update_yaxes(title="Points")
    _layout(fig, dark, 420, hovermode="x unified")
    fig.update_layout(margin=dict(l=10, r=110, t=30, b=10))
    return fig


def results_heatmap(season_df, order, dark=False):
    """Driver x round grid of finishing positions; darker = better, grey = DNF."""
    pos = season_df.pivot_table(index="Driver", columns="round", values="finish", aggfunc="first")
    dnf = season_df.pivot_table(index="Driver", columns="round", values="dnf", aggfunc="first")
    pos, dnf = pos.reindex(order), dnf.reindex(order)
    finished = pos.where(dnf == 0)
    did_not = pd.DataFrame(np.where(dnf == 1, 1, np.nan), index=pos.index, columns=pos.columns)
    cols = [f"R{c}" for c in pos.columns]
    scale = [[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=finished.values, x=cols, y=pos.index, colorscale=scale, zmin=1, zmax=20,
        xgap=2, ygap=2,
        colorbar=dict(title="Pos", tickvals=[1, 5, 10, 15, 20], len=0.6, thickness=10),
        hovertemplate="<b>%{y}</b><br>%{x}: P%{z}<extra></extra>",
    ))
    fig.add_trace(go.Heatmap(
        z=did_not.values, x=cols, y=pos.index, colorscale=[[0, NEUTRAL[dark]], [1, NEUTRAL[dark]]],
        showscale=False, xgap=2, ygap=2, hovertemplate="<b>%{y}</b><br>%{x}: DNF<extra></extra>",
    ))
    # Cell labels: light ink on dark cells, dark ink on light cells
    xs, ys, txt, ink = [], [], [], []
    for drv in pos.index:
        for c, col in zip(pos.columns, cols):
            v, gone = pos.at[drv, c], dnf.at[drv, c]
            if pd.isna(v):
                continue
            xs.append(col); ys.append(drv)
            if gone == 1:
                txt.append("DNF"); ink.append(INK[dark][1])
            else:
                txt.append(str(int(v))); ink.append("#ffffff" if v <= 9 else "#0b0b0b")
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="text", text=txt, textfont=dict(color=ink, size=10),
                             hoverinfo="skip", showlegend=False))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(side="top", showgrid=False)
    return _layout(fig, dark, max(420, 26 * len(order) + 80))


def team_points(season_df, dark=False):
    """Constructor totals, single series, sorted."""
    t = season_df.groupby("Team")["points"].sum().sort_values()
    fig = go.Figure(go.Bar(
        x=t.values, y=t.index, orientation="h", marker=dict(color=SERIES[dark][0], cornerradius=4),
        text=[f"{v:.0f}" for v in t.values], textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>%{x:.0f} pts<extra></extra>",
    ))
    fig.update_xaxes(title="Points", showgrid=True)
    fig.update_yaxes(showgrid=False)
    return _layout(fig, dark, 30 * len(t) + 80, bargap=0.35)


# --------------------------------------------------------------------------- #
# Race views
# --------------------------------------------------------------------------- #
def position_shift(frame, start_col, end_col, start_label, end_label, dark=False):
    """Slope chart: each driver's line from one ordering to another.
    Blue = gained places, red = lost, grey = unchanged. Names on both ends."""
    f = frame.dropna(subset=[start_col, end_col]).copy()
    fig = go.Figure()
    for _, r in f.iterrows():
        delta = r[start_col] - r[end_col]
        color = GAIN if delta > 0 else LOSS if delta < 0 else NEUTRAL[dark]
        width = 2.5 if abs(delta) >= 3 else 1.5
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[r[start_col], r[end_col]], mode="lines+markers",
            line=dict(color=color, width=width), marker=dict(size=8, color=color),
            showlegend=False,
            hovertemplate=(f"<b>{r['Driver']}</b><br>{start_label}: P{int(r[start_col])}"
                           f"<br>{end_label}: P{int(r[end_col])}<br>"
                           f"{'+' if delta > 0 else ''}{int(delta)} places<extra></extra>"),
        ))
        short = r["Driver"].split()[-1]
        fig.add_annotation(x=0, y=r[start_col], text=f"{short}  P{int(r[start_col])}",
                           xanchor="right", xshift=-10, showarrow=False, font=dict(size=11))
        fig.add_annotation(x=1, y=r[end_col], text=f"P{int(r[end_col])}  {short}",
                           xanchor="left", xshift=10, showarrow=False, font=dict(size=11))
    fig.update_xaxes(tickvals=[0, 1], ticktext=[start_label, end_label], range=[-0.6, 1.6],
                     showgrid=False, side="top")
    fig.update_yaxes(autorange="reversed", showticklabels=False, showgrid=False)
    return _layout(fig, dark, 26 * len(f) + 90)


def chances(out, dark=False, top=10):
    """Grouped bars: win vs podium chance for the top contenders (same unit, one axis)."""
    t = out.sort_values(["win", "podium"], ascending=False).head(top).iloc[::-1]
    names = [d for d in t["Driver"]]
    c = SERIES[dark]
    fig = go.Figure()
    for col, label, color in [("podium", "Podium", c[1]), ("win", "Win", c[0])]:
        fig.add_trace(go.Bar(
            y=names, x=t[col], name=label, orientation="h",
            marker=dict(color=color, cornerradius=4, line=dict(width=0)),
            text=[f"{v:.0f}%" for v in t[col]], textposition="outside", cliponaxis=False,
            hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:.0f}}%<extra></extra>",
        ))
    fig.update_xaxes(title="Chance (%)", range=[0, 105], ticksuffix="%")
    fig.update_yaxes(showgrid=False)
    _layout(fig, dark, 46 * len(t) + 90, barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_layout(legend_traceorder="reversed")
    return fig


def driver_form(h, dark=False):
    """Grid vs finish over recent races: two series on one shared position axis."""
    c = SERIES[dark]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["race"], y=h["grid"], name="Grid", mode="lines+markers",
                             line=dict(color=c[1], width=2, dash="dot"), marker=dict(size=8),
                             hovertemplate="%{x}<br>Grid P%{y}<extra></extra>"))
    fig.add_trace(go.Scatter(x=h["race"], y=h["finish"], name="Finish", mode="lines+markers",
                             line=dict(color=c[0], width=2), marker=dict(size=8),
                             connectgaps=False,
                             hovertemplate="%{x}<br>Finished P%{y}<extra></extra>"))
    dnfs = h[h["dnf"] == 1]
    if len(dnfs):
        fig.add_trace(go.Scatter(x=dnfs["race"], y=[21] * len(dnfs), name="DNF", mode="markers",
                                 marker=dict(symbol="x", size=10, color=LOSS),
                                 hovertemplate="%{x}<br>DNF<extra></extra>"))
    fig.update_yaxes(title="Position", range=[21.5, 0.5], tickvals=[1, 5, 10, 15, 20],
                     ticktext=["P1", "P5", "P10", "P15", "P20"])
    fig.update_xaxes(tickangle=-45)
    return _layout(fig, dark, 380, hovermode="x unified")


def feature_importance(imp, dark=False):
    imp = imp.sort_values()
    labels = {
        "grid": "Grid position", "drv_avg_finish_5": "Driver avg finish (5)",
        "drv_avg_grid_5": "Driver avg grid (5)", "drv_avg_points_5": "Driver avg points (5)",
        "drv_dnf_rate_10": "Driver DNF rate (10)", "team_avg_finish_5": "Team avg finish (5)",
        "team_avg_grid_5": "Team avg grid (5)", "drv_track_avg_finish": "Driver at this track",
        "drv_season_points": "Season points", "races_in_season": "Races this season",
        "drv_avg_gain_5": "Places gained (5)", "drv_last_finish": "Last race finish",
        "drv_avg_finish_season": "Season avg finish", "grid_vs_teammate": "Grid vs teammate",
        "team_dnf_rate_10": "Team DNF rate (10)",
    }
    fig = go.Figure(go.Bar(
        x=imp.values * 100, y=[labels.get(i, i) for i in imp.index], orientation="h",
        marker=dict(color=SERIES[dark][0], cornerradius=4),
        text=[f"{v*100:.0f}%" for v in imp.values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_xaxes(ticksuffix="%", title="Share of model's decisions")
    fig.update_yaxes(showgrid=False)
    return _layout(fig, dark, 30 * len(imp) + 80, bargap=0.35)


def backtest_chart(per_race, dark=False):
    """Per-race back-test: average position error, with winner hits marked."""
    c = SERIES[dark]
    fig = go.Figure(go.Bar(
        x=per_race["race"], y=per_race["mae"], marker=dict(color=c[0], cornerradius=4),
        text=["✓ winner" if w else "" for w in per_race["win_hit"]], textposition="outside",
        cliponaxis=False,
        customdata=np.stack([per_race["podium"] * 3], axis=-1),
        hovertemplate="%{x}<br>Avg error %{y:.1f} places<br>Podium %{customdata[0]:.0f}/3<extra></extra>",
    ))
    fig.update_yaxes(title="Avg position error (places)")
    return _layout(fig, dark, 340, bargap=0.4)
