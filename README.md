<p align="center"><img src="assets/logo_light.svg" alt="LightsToFlag" height="96"></p>

# LightsToFlag: full F1 race predictions

A machine-learning app that predicts the finishing order of the next Formula 1 race. It also lets you browse past races with their winners and see how the model would have called each one.

---

## What's in the folder

| File | What it does |
|---|---|
| `app.py` | The web app (Streamlit) |
| `f1_predict.py` | Data download, feature building, model training and back-testing. It also runs on its own from the command line. |
| `charts.py` | Interactive Plotly charts (colour-blind-safe palette) |
| `engine_audio.py` | Generates the engine sounds in code (no recordings needed) |
| `assets/` | Logo files (sidebar logo for light and dark mode, app icon, favicon) |
| `f1_cache/` | Created on first run. Saved race data, so later runs are fast. |
| `sounds/` | *Optional.* Put your own engine recordings here (see [Custom sounds](#custom-sounds)). |

---

## Setup (one time)

You need **Python 3.9 or newer**. Check with `python3 --version` (on Windows, `python --version`).

```bash
cd path/to/f1-predictor
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the app

```bash
source venv/bin/activate          # if not already active
streamlit run app.py
```

Your browser opens at **http://localhost:8501**. To stop the app, press `Ctrl+C` in the terminal.

The first launch downloads about 5 seasons of results from the free [Jolpica F1 API](https://github.com/jolpica/jolpica-f1), which takes 1–2 minutes. After that, the app starts in seconds.

---

## Using the app

### Left side: race list

- **Season** picks the year to browse, anywhere from 2018 to now.
- **Upcoming**
  - The next race is marked **🔮 Prediction**. Click it to open the prediction view.
  - Later races show their date. They're greyed out until the race before them has finished.
- **Completed** lists finished races, newest first, with the winner (e.g. `R21 🇧🇷 São Paulo · 🏆 Antonelli`). Click one to see its results.
- **⚙️ Settings** sits at the bottom of the list:
  - **Training seasons**: which years the model learns from (default: last 5).
  - **Races to back-test**: how many recent races are used to score the model's accuracy.
  - **🔊 Engine sounds**: shows or hides the sound buttons.
  - **Re-download data**: gets fresh results. Use it after qualifying or after a race.

### 📊 Season dashboard

Click **📊 Season dashboard** at the top of the race list.

- **Headline numbers:** championship leader, gap to second place, most wins, races done and races left.
- **Championship battle:** each driver's points total race by race. Pick up to 4 drivers to compare. Hover to see the totals after any race.
- **Every result this season:** a grid of drivers × races. Darker squares are better finishes and grey squares are DNFs. You can spot hot streaks and bad weekends at a glance.
- **Constructors:** total points per team.
- **Standings:** points, wins and podiums for every driver.

### Next race: prediction view

**Prediction tab**
- **Starting grid** (left): the official qualifying order if it's out, otherwise an estimate from recent form. The page header tells you which one you're looking at.
  - ✏️ **What-if mode:** edit any grid number to try a scenario, such as a 5-place penalty or a driver starting from the pit lane (set them to 20). The prediction updates right away. **Reset grid** puts the original back.
- **Predicted result** (right):
  - The predicted top three, each with their chance of winning.
  - A full predicted order, with places gained or lost compared with the grid.
  - **Win / Podium / Top 10** chances for every driver, from 5,000 simulated races.
- **Who can win?** compares win and podium chances for the top 10.
- **Grid → predicted finish** draws a line from each driver's grid slot to their predicted finish: blue for places gained, red for places lost.
- **🚦 Lights out!** plays a race start: five countdown beeps, then the launch and gear shifts.
- **💨 Fly-by** plays a car passing at 320 km/h.

**Model accuracy tab**
- A per-race chart of the back-test, with ✓ on races where the model picked the winner.
- Back-test scores: how often the model picked the winner, how many podium finishers it got right, and its average position error over recent races.
- Which inputs the model relies on most.

**Driver form tab**
- Pick a driver to see a chart of their grid and finish positions over their last 20 races (✕ marks a DNF), their recent average finish, DNF count, and best result at this track.

### Completed race view

- **Podium:** the top three with team and starting position.
- **Results table:**
  - finishing position (DNF if they didn't finish), grid, places gained or lost, points and status
  - **Model said:** where the model would have placed each driver, using only data from before that race
- **Position chart:** toggle between *Grid → Finish* (who gained or lost places) and *Model → Finish* (where the model was right or wrong).
- **Summary line:** whether the model picked the winner and how many podium finishers it got right.
- **Driver form tab:** same as above.

---

## Tips for better predictions

1. **Run it after qualifying.** Starting grid position is by far the strongest signal (about 70%+ of what the model uses). Before qualifying, the grid is only a guess.
2. Click **Re-download data** in Settings to pick up new qualifying or race results. The app only checks the API about every 15 minutes on its own.
3. **Keep training seasons recent.** The 2026 rule changes reshuffled the field, so older seasons say less about today's cars. The model already counts recent seasons more.
4. **Expect about 50% on picking the winner.** Crashes, safety cars, weather and strategy make F1 noisy. Use the win and podium percentages instead of treating the order as certain.

---

## Command-line use (no UI)

```bash
python3 f1_predict.py                          # back-test + next-race prediction
python3 f1_predict.py --refresh                # ignore the saved data and re-download
python3 f1_predict.py --seasons 2024 2025 2026 # choose training seasons
python3 f1_predict.py --backtest 0             # skip the back-test
```

---

## How the model works

- **Data:** race results and qualifying for each season, from the Jolpica (Ergast-compatible) API.
- **Inputs for each driver:**
  - grid position
  - average finish, grid and points over their last 5 races
  - DNF rate over their last 10 races
  - team's average finish and grid over its last 5 races (current season only)
  - the driver's past average finish at this track
  - season points so far
- **No peeking ahead:** every input only uses races that happened before the one being predicted.
- **Model:** scikit-learn gradient boosting that predicts a finishing position, trained with more weight on recent seasons.
- **Chances:** each driver's predicted score gets random noise the size of the model's typical error. Running that 5,000 times gives the win, podium and top-10 percentages.

---

## Custom sounds

To use real recordings instead of the generated sounds, create a `sounds/` folder next to `app.py` and add either or both of:

```
sounds/race_start.mp3   (or .wav)
sounds/fly_by.mp3       (or .wav)
```

The app uses them automatically.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Activate the venv (`source venv/bin/activate`) and run the `pip install` line again. |
| "Couldn't reach the F1 data API" | Check your internet connection. The API allows about 4 requests a second and 500 an hour, so wait a minute and click **Re-download data**. |
| `429 Too Many Requests` | Same as above: the rate limit was hit. Wait, then retry. |
| Grid says *estimated* after qualifying | Results can take a few minutes to appear. Click **Re-download data**. |
| No sound | Click anywhere on the page first (browsers block autoplay until you've interacted), and check that 🔊 is on in Settings. |
| Something looks stale or broken | Delete the `f1_cache/` folder and restart. |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
