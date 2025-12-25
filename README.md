# Probabilistic Price Forecasting System (Time-Series)

**What the project does**

This project builds a machine-learning system that forecasts **future price movement as a probability distribution**, not a single predicted price.

Instead of saying "the price will be X tomorrow", the system answers:

> "Given current market conditions, here is the likely range of future returns and how confident we are."

---

### Core objective

* Predict **future log-returns** over a fixed horizon (for example 1 hour or 1 day)
* Output **multiple quantiles** (e.g. 10th, 50th, 90th percentile)
* Use those quantiles to represent **uncertainty and risk**

---

### Inputs

* Historical market data (OHLCV)
* Derived features such as:
  * Past returns
  * Rolling volatility
  * Price range
  * Volume-based features
  * Time/session indicators

---

### Model behavior

* Trains **quantile regression models** (LightGBM or equivalent)
* Produces:
  * Lower bound forecast (worst-case)
  * Median forecast (most likely)
  * Upper bound forecast (best-case)

These three together form a **confidence interval** for future price movement.

---

### Validation approach

* Uses **walk-forward validation** (time-aware training and testing)
* No random shuffling
* Evaluates:
  * How often actual returns fall inside the predicted range
  * Calibration of uncertainty
  * Stability of predictions over time

---

### Output / Deliverables

* Time-aligned forecasts with confidence bands
* Visualizations showing:
  * Price vs predicted ranges
  * Forecast uncertainty expanding or contracting over time
* Metrics that measure:
  * Prediction interval coverage
  * Quantile loss (pinball loss)

---

### Optional decision layer (lightweight)

* Demonstrates how forecasts can be used for decisions:
  * Only act when confidence interval is sufficiently wide
  * Direction taken from median forecast
* Evaluated using basic risk metrics (not a full trading system)

---

### Why this project matters

* Financial markets are **non-stationary and noisy**
* Point predictions fail under regime changes
* This system explicitly models **uncertainty**, which is critical for:
  * Real-world ML systems

---

### Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Training & Forecast**:
   ```bash
   # Train on last 50,000 candles and forecast 1 hour ahead
   python scripts/train_quantile_model.py --limit 50000 --horizon 4
   ```

3. **Check Output**:
   - Models saved to `models/`
   - Console output showing coverage metrics and example forecasts.

