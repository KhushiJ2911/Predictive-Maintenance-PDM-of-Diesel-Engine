# Predictive Maintenance of a Diesel Engine — RUL Prediction with LSTMs

Run-to-failure data for engine faults is genuinely hard to get your hands on — nobody runs a real diesel engine to destruction just to generate a training set. This project trains a dedicated LSTM per fault to predict Remaining Useful Life (RUL) from synthetic run-to-failure sensor trajectories, covering five different fault mechanisms.

This code is the companion implementation for the paper *"Fault-Aware RUL Prediction for Diesel Engines Using Physics-Motivated Degradation Models."* Five of the six fault scenarios analyzed in the paper are implemented here (bearing fatigue, unbalance, cooling overheat, EGR malfunction, multi-fault vibration); the sixth, fuel injector fault, isn't included in this repo.

## Repo layout

```
Bearing_Fatigue_LSTM_RUL_Updated.ipynb
Fault3_LSTM_RUL_updated.ipynb
Fault5_RUL_EGR_Updated.ipynb
Unbalance_LSTM_RUL_Updated.ipynb
Vibration_RUL_updated.ipynb
```

Each notebook is self-contained: data loading, sequence windowing, model training, and evaluation for one fault. They were developed in Colab (GPU runtime), so each has an "Open in Colab" badge at the top if you'd rather run them there than locally.

## Model

All five faults use roughly the same LSTM setup so results are comparable across faults:

- Input: sliding windows of 30 timesteps
- `LSTM(64, return_sequences=True) → Dropout(0.3) → LSTM(32) → Dropout(0.3) → Dense(1)`
- Adam optimizer (`lr=5e-4`), MAE loss, early stopping on validation loss (patience 5) with best-weight restoration
- Target is RUL normalized by each run's initial life, so errors are comparable across faults with very different lifespans before being converted back to hours for interpretation

## The leakage problem, and fixing it

First pass at this used `Damage` and `DamageRate` — the simulator's own internal damage-accumulation variables — as model inputs. Unsurprisingly, that gave excellent-looking metrics (R² > 0.99 across the board), because the model was essentially being handed the answer.

Every notebook has a second, "leakage-free" section that redoes the evaluation using only what a real sensor could actually measure — engine RPM and a fault-specific severity metric (vibration amplitude, EGR flow deviation, etc.) — with `Damage`, `DamageRate`, and any other future-derived variable removed from the inputs. Errors go up substantially once the leakage is removed, which is expected and is the whole point: the leakage-free numbers below are the realistic ones.

## Results (leakage-free, held-out runs)

Numbers below match the paper's Table I (reliability) and Table II (absolute error) exactly. `Fault3_LSTM_RUL_updated.ipynb` implements what the paper calls the **cooling overheat** fault — the filename is a leftover from an early placeholder and was never renamed.

| Fault | Notebook | Late-life MAE (hrs) | Late-life RMSE (hrs) | PW@0.05 | PW@0.10 | PW@0.20 |
|---|---|---|---|---|---|---|
| Bearing Fatigue | `Bearing_Fatigue_LSTM_RUL_Updated.ipynb` | 850 | 1240 | 38.5% | 66.8% | 92.8% |
| Unbalance | `Unbalance_LSTM_RUL_Updated.ipynb` | 226 | 335 | 32.0% | 62.0% | 90.0% |
| Cooling Overheat | `Fault3_LSTM_RUL_updated.ipynb` | 371 | 641 | 74.0% | 97.0% | 99.6% |
| EGR Malfunction | `Fault5_RUL_EGR_Updated.ipynb` | 643 | 918 | 46.0% | 81.0% | 99.0% |
| Multi-Fault Vibration | `Vibration_RUL_updated.ipynb` | 1179 | 1775 | 43.3% | 71.0% | 92.1% |

PW@α = percentage of late-life predictions within α of the true normalized RUL. "Late-life" is the final portion of each run's trajectory, where accurate RUL estimation matters most for maintenance decisions.

A few things fall out of this pretty clearly:

- **Absolute error tracks degradation time scale, not model quality.** Bearing fatigue runs ~10,000 hours with weak early/mid-life signal, so even small timing errors in predicted degradation onset blow up into hundreds of hours of absolute error — despite the model doing a genuinely good job of learning the trend.
- **Fast-collapsing faults aren't automatically "easier."** Cooling overheat has the best reliability of any fault (97% within 10% of true RUL) because once thermal limits are crossed, RUL collapses fast and predictably. But that same steep slope means small timing misses still cost hundreds of hours in absolute terms. Unbalance degrades fast too, but its late-life slope varies more run-to-run, which hurts reliability at tight thresholds (only 62% at PW@0.10) even though its absolute error is low.
- **EGR and multi-fault vibration are the genuinely hard cases.** Soot buildup and nonlinear exhaust flow (EGR) and interacting fault mechanisms (multi-fault vibration) both produce noisy, non-smooth observability, so prediction variance stays elevated all the way to failure rather than tightening up.

## What I'd change next

- Only RPM + one fault-severity metric are used as inputs, by design, to keep the leakage-free comparison fair. Adding real sensor-derived features (frequency-domain vibration, exhaust temperature) would likely help late-life accuracy the most, where it's currently weakest.
- No monotonicity constraint is enforced, so predicted RUL occasionally ticks up slightly instead of only decreasing. A health-index formulation or a monotonic loss term would fix that.
- This is validated entirely on simulated degradation. The obvious next step is testing against real fleet or bench data, where noise and unmodeled effects will behave differently than in simulation.

## Running the notebooks

Open any notebook's Colab badge, or run locally with:

```
pip install tensorflow scikit-learn pandas numpy matplotlib
```

Each notebook expects its corresponding run-to-failure dataset (generated separately) to be available at the path referenced in the notebook.
