# Credit Spread Monitor

Quantitative regime detection and stress testing for corporate bond markets.

A pipeline tool that ingests 20+ years of credit spread data, computes
professional fixed income analytics, detects market regimes using a
Hidden Markov Model, and generates automated stress testing reports.

> **Status:** Active development — core pipeline functional, report generation and documentation in progress.

> **Note:** This tool is calibrated on US corporate credit markets (ICE BofA indices).
> Adaptation to EUR indices would be a natural extension for European asset managers.

---

## Methodology

```
Data Ingestion (FRED API) → Feature Engineering → Regime Detection (HMM) →
Historical Stress Testing → Automated Excel Report
```

---

## Why This Project Matters

- **Automates the Monday morning workflow:** replaces manual spread monitoring
  with a reproducible pipeline that generates a multi-sheet Excel briefing.
- **Objective regime detection:** instead of arbitrary thresholds, a probabilistic
  HMM identifies risk-on, risk-off, and crisis states from market dynamics.
- **History-based stress testing:** measures real spread impact across multiple
  scenarios with statistical confidence intervals.


---

## Architecture

```
credit-spread-monitor/
    src/
        ingestion.py         # FRED API data fetching and caching
        features.py          # Feature engineering (z-scores, percentiles, ratios)
        regime.py            # HMM regime detection
        stress.py            # Historical stress testing
        report.py            # Automated Excel report generation
        pipeline.py          # End-to-end orchestrator
    notebooks/
        eda.ipynb            # Exploratory data analysis
    figures/                 # Generated visualizations
    results/                 # CSV/JSON outputs
    models/                  # Serialized HMM model
    .env.example             # FRED API key template
    requirements.txt
```

---

## Technical Highlights

### Regime Detection
- 3-state Hidden Markov Model (risk-on, risk-off, crisis) fitted on spread dynamics.
- Probabilistic classification replaces arbitrary threshold-based rules.
- Transition matrix captures the persistence and switching behavior of market states.

### Feature Engineering
- Rolling z-scores for mean-reversion signals.
- Percentile rankings for historical context.
- BBB/HY ratio as a quality compression indicator.
- Realized volatility for risk monitoring.

### Stress Testing
- Historical scenario simulation (GFC, COVID, Taper Tantrum, etc.).
- Confidence intervals via scenario-based analysis.
- Half-life comparison across different crisis types.

### Automated Report
- Excel workbook with 6 tabs: summary, spreads, regimes, stress tests, correlations, methodology.
- Fully reproducible — re-run the pipeline to update with latest data.

---

## Data

All data is sourced from the FRED API (Federal Reserve Economic Data).

| Series | Column | Description |
|--------|--------|-------------|
| BAMLC0A1CAAA | aaa_spread | ICE BofA AAA US Corporate OAS |
| BAMLC0A2CAA | aa_spread | ICE BofA AA US Corporate OAS |
| BAMLC0A4CBBB | bbb_spread | ICE BofA BBB US Corporate OAS |
| BAMLH0A0HYM2 | hy_spread | ICE BofA US High Yield OAS |
| DGS10 | treasury_10y | 10-Year Treasury Yield |
| DGS5 | treasury_5y | 5-Year Treasury |
| VIXCLS | vix | CBOE Volatility Index |

---

## How to Run

```bash
git clone https://github.com/Zekhayoub/credit-spread-monitor.git
cd credit-spread-monitor
python -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt

# Copy .env.example to .env and add your FRED API key
cp .env.example .env

# Run the full pipeline
python -m src.pipeline

# Or skip ingestion if data is already cached
python -m src.pipeline --skip-ingestion
```

---

## Limitations and Known Biases

- **US data only.** Calibrated on US markets; EUR adaptation needed for European focus.
- **OAS ≠ pure credit risk.** Includes callable option value driven by rate volatility.
- **Index composition drift.** Sector mix changed from telecoms (2000) to tech (2020).
- **Fallen angel bias.** Downgrades from BBB to HY artificially improve BBB index.
- **Gaussian HMM.** Underestimates early crisis probability due to fat-tailed distributions.
- **Stationary transitions.** Assumes constant regime dynamics across 25 years of structural change.
- **Pre/post-QE mixing.** Historical percentiles blend free-market and central bank-compressed periods.
- **VIX as credit proxy.** Equity vol, not direct credit stress (CDX/iTraxx unavailable on FRED).

---

## Tech Stack

Python · pandas · NumPy · fredapi · hmmlearn · scikit-learn · scipy ·
matplotlib · seaborn · openpyxl · joblib

---

## References

1. Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series"
2. Ang, A. & Timmermann, A. (2012). "Regime Changes and Financial Markets"
3. Longstaff, F.A. et al. (2005). "Corporate Yield Spreads: Default Risk or Liquidity?"
4. ICE BofA Index Methodology documentation
5. Bao, J. et al. (2011). "The Illiquidity of Corporate Bonds" — Journal of Finance