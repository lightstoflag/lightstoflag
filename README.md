<p align="center"><img src="assets/logo_light.svg" alt="LightsToFlag" height="96"></p>

# LightsToFlag: full F1 race predictions

A machine-learning app that predicts the finishing order of the next Formula 1 race. It also lets you browse past races with their winners and see how the model would have called each one.

---

## What's in the folder

| File | What it does |
|---|---|
| `app.py` | The web app (Streamlit) |
| `f1_predict.py` | Data download, feature building, model training and back-testing. It also runs on its own from the command line. |
| `chat.py` | The "Ask LightsToFlag" chat: builds the data context and talks to Claude or Ollama |
| `charts.py` | Interactive Plotly charts (colour-blind-safe palette) |
| `engine_audio.py` | Generates the engine sounds in code (no recordings needed) |
| `tests/` | Automated test suite (see [Running the tests](#running-the-tests)) |
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

### 💬 Ask LightsToFlag (chat)

Click **💬 Ask LightsToFlag** in the race list and ask questions in plain English, for example:
- *Who's most likely to win, and why?*
- *Who could surprise from further back?*
- *How has Verstappen done in the last 5 races?*
- *Which team has the most DNFs?*

**What it knows:**
- the next-race prediction, with win, podium and top-10 chances
- this season's results, driver and constructor standings, and DNFs
- the model's track record

It's told to answer only from that data, so it will say when it doesn't know something, like weather or news.

**Choose a model** under *Chat settings*:

| Option | Cost | Setup |
|---|---|---|
| Claude Haiku 5.5 (default) | Fractions of a cent per question | Needs an Anthropic API key |
| Claude Sonnet 5.5 | A little more, smarter answers | Needs an Anthropic API key |
| Ollama (any local model) | Free, runs on your computer | `ollama pull llama3.1`, with Ollama running |

To avoid pasting your API key each time, create `.streamlit/secrets.toml` next to `app.py`:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```
`.streamlit/secrets.toml` is already in `.gitignore`, so your key never gets pushed to GitHub.

### Next race: prediction view

**Prediction tab**
- **🚩 Grid penalties** (top left): the free data only has the **qualifying order**, not the real starting grid. Penalties such as engine changes or impeding aren't included until after the race. Check race-day news and add each one here:
  - **Drop places**, e.g. "Piastri −10"
  - **Back of grid**
  - **Pit-lane start**

  The app rebuilds the grid (everyone else moves up to fill the gaps) and re-predicts. The page header shows whether penalties have been applied. Penalties last for your browser session and apply to the chat page too.
- **Starting grid** (left): the qualifying order with your penalties applied, or an estimate from recent form before qualifying.
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
- Back-test scores (tuned model: about 70% winners, 73% podium finishers over the last 30 races): how often the model picked the winner, how many podium finishers it got right, and its average position error over recent races.
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

1. **Run it after qualifying, then add penalties on race day.** Starting grid position is by far the strongest signal (about 70%+ of what the model uses). In Las Vegas 2026, Piastri qualified 7th but started 17th after a penalty, and the prediction missed it because the penalty wasn't entered.
   - **Limitation:** when a front-runner gets a big penalty, the model can be too optimistic about how far they'll climb. That situation is rare in the training data. Treat those predictions with extra caution.
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

- **Data:** race results and starting grids for each season, from the Jolpica (Ergast-compatible) API.
- **Inputs for each driver:**
  - starting grid position, and grid position compared with their teammate
  - recent form: average finish, grid and points over their last 5 races, last race result, season-average finish, and average places gained from the grid
  - reliability: the driver's DNF rate (last 10 races) and the team's DNF rate
  - team form: the team's average finish and grid over its last 5 races (current season only)
  - the driver's past results at this track, and season points so far
- **No peeking ahead:** every input only uses races that happened before the one being predicted.
- **Model:** scikit-learn `HistGradientBoostingRegressor` (200 trees, depth 3, learning rate 0.03, at least 40 results per leaf). It predicts a finishing position and is trained with more weight on recent seasons.
- **DNFs are left out of training.** Crashes and failures are mostly bad luck. Leaving them in taught the model to expect random retirements and made the finishing order worse.
- **Chances:** each driver's predicted score gets random noise the size of the model's typical error. Running that 5,000 times gives the win, podium and top-10 percentages.

## Model tuning

The model was tuned on real 2022–2026 results using **walk-forward testing**: each race is predicted using only data from before it.
- **Selection:** 20 races (2025 R15 – 2026 R10) were used to choose settings.
- **Final check:** 10 more races (2026 R11–R20) were held back until the end.

Results on the 10 held-back races:

| Model | Winner correct | Podium overlap | Rank correlation | Avg position error |
|---|---|---|---|---|
| Grid order only (no ML) | 7/10 | 60% | 0.73 | 2.7 |
| Original model | 6/10 | 50% | 0.79 | 2.5 |
| **Tuned model** | **6/10** | **83%** | **0.84** | **2.1** |

- **Rank correlation** measures how well the predicted order matches the real order, from 1.0 for perfect down to 0 for random.
- **Avg position error** is measured among drivers who finished.

**What changed:**
- **Algorithm:** gradient boosting was switched to HistGradientBoosting, with shallower, more regularised trees.
- **Training data:** DNF rows were dropped from training.
- **Inputs:** five new ones were added: places gained, last finish, season average, grid vs teammate, and team DNF rate.

**Honest note:** picking the *winner* is still about as good as "the pole-sitter wins". In 2025–26, pole won about 70% of races, and no results-only model beats that by much. The tuned model's gains are in the rest of the order. Adding practice pace and qualifying lap-time gaps (e.g. via [FastF1](https://github.com/theOehrly/Fast-F1)) is the next step for improving winner picks.

## Running the tests

The `tests/` folder has an automated test suite. It runs **offline**: the F1 API is replaced with small, made-up data, so results are the same every time.

```bash
pip install -r requirements-dev.txt
pytest                    # everything, about 15 seconds
pytest -m "not slow"      # skip the end-to-end app tests
pytest tests/test_penalties.py -v   # one file, with each test's name
```

| File | What it checks |
|---|---|
| `test_penalties.py` | Grid penalties: place drops, back of grid, pit lane, several at once, the Vegas/Piastri case |
| `test_features.py` | **No data leakage:** changing a race's result never changes the inputs for that race or earlier ones. Also rookie defaults, rolling averages, season points, teammate comparison and next-race grids |
| `test_data.py` | Parsing API results (DNFs, pit-lane starts), caching, pages split across responses, qualifying, rate-limit retries |
| `test_model.py` | Training, predictions, importances, and that every walk-forward back-test trains only on earlier races |
| `test_ui_helpers.py` | Every chart builds (light and dark), DNFs are labelled, every input has a readable chart label, sounds are valid WAV files, and the chat context contains the prediction |
| `test_app.py` | End-to-end: runs the real Streamlit app, opens every page, adds a penalty, plays a sound |

Run the tests before every push. If one fails, it tells you exactly what broke.

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
