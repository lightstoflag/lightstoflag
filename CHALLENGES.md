# LightsToFlag: two model-tuning challenges

Both challenges use the same scoring harness, `evaluate.py`. It predicts each race using only the races before it.

| Command | Scores | When to use it |
|---|---|---|
| `python evaluate.py` | 20 **TUNE** races (2025 R15 → 2026 R10) | While you work. Run it as often as you like. |
| `python evaluate.py --final` | 10 **TEST** races (2026 R11 → R20), locked away | **Once**, when you're done. |

> 🔒 **Rule:** don't make decisions by looking at `--final` scores. If you tune against the test races, the result stops telling you anything about the future. This is the most important habit in model tuning.

**Starting point.** The current model, scored on TUNE:
```
Ordering  | winner 15/20  podium 68%  rank-corr 0.791  pos-error 2.26
Win prob  | Brier 0.0278  log-loss 0.0925
Pod prob  | Brier 0.0681  log-loss 0.2118
```

---

## Challenge 1: Make the win and podium percentages honest (calibration)

**The problem.** The app turns predicted scores into chances by simulating 5,000 races. Each simulation adds random noise, and the noise size is the model's typical error × `1.15`. That `1.15` was a guess. If the percentages are well calibrated, things the app calls "30% likely" should happen about 30% of the time. Look at the reliability table `evaluate.py` prints:

```
predicted  actual
   22%      14%     ← the app is too confident here
   63%      76%     ← and not confident enough here
```

**Your task:** change `NOISE_MULT` and/or `my_probabilities()` in `evaluate.py` so that both of these are lower on TUNE:
- the win-probability **Brier score**
- the win-probability **log-loss**

Keep the podium Brier score no worse than **0.0681**.

**Targets (TUNE):**
- 🥉 Bronze: win Brier < **0.0260**
- 🥈 Silver: win Brier < **0.0240**, and win log-loss < **0.085**
- 🥇 Gold: Silver, plus podium Brier < **0.0670**, plus a reliability table where every bucket's predicted value is within about 8 points of the actual one

**Things to explore** (your call how):
- Is one fixed multiplier the right idea? Front-runners and backmarkers may need different amounts of noise.
- Is a normal distribution the right shape for the noise?
- Could you choose the noise size from data instead of guessing it? Hint: what single number would make the log-loss smallest on *past* races?

**Learn:** Brier score, log-loss, reliability diagrams, why "accurate order" ≠ "honest probabilities".

---

## Challenge 2: Train the model to rank, not to guess positions

**The problem.** The model is a *regressor*: it predicts a finishing position (like 4.3) for each driver separately, and we sort the numbers. But what we care about is the **order** within each race. *Learning-to-rank* models (like LightGBM's `LGBMRanker` with the `lambdarank` objective) are trained on the order directly, one race at a time.

A naive first attempt is **worse** than the current model:
```
naive LGBMRanker  | winner 11/20  podium 67%  rank-corr 0.755  pos-error 2.52
```
So you'll have to think, not just swap the class.

**Your task:** rewrite `my_scores()` in `evaluate.py` to use a ranking model that beats the current model on TUNE. Remember that a **lower** score means a better finish.

**Targets (TUNE, run with `--no-probs` for speed):**
- 🥉 Bronze: beat the naive ranker on rank-corr **and** pos-error
- 🥈 Silver: rank-corr ≥ **0.791** and pos-error ≤ **2.26** (match the current model)
- 🥇 Gold: beat the current model on rank-corr **and** podium (> 68%), without dropping below 15/20 winners

**Things to explore:**
- **Relevance labels.** The ranker needs a "how good was this finish" number per driver. Is `21 − finish` a good choice? What about labels where only the top 10 (the points places) count, or where P1 counts far more than P2?
- **Recency.** The current model weights recent seasons more heavily. How would you do that in a ranker, where weights apply to whole races?
- **DNFs.** Keep them, drop them, or give them the lowest label?
- **Hyperparameters.** For example `n_estimators`, `learning_rate`, `num_leaves`, `min_child_samples`, and `label_gain` (how much each relevance level counts). Tune them on TUNE only.
- Setup: `pip install lightgbm`. Rows must be grouped by race, sorted by race, with the `group=` sizes passed in the same order.

**Learn:** pointwise vs. listwise objectives, NDCG, query groups, label design.

---

## How to submit for review

1. Put your work in `evaluate.py`. You can add helper files if you like.
2. Write down:
   - your TUNE results
   - your **one** `--final` result
   - a few lines on what you tried, including what *didn't* work
3. Push to a branch, e.g. `git checkout -b challenge-1`, then commit and push. Or paste the code into the chat.

I'll review it for:
- **correctness**, especially data leakage, where future information sneaks into training
- **whether the gain is real**
- **whether it should go into the app**
