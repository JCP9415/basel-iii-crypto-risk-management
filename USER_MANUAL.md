# 📘 User Manual — Crypto Risk Manager Pro v4.6

## 🎯 Overview
**Crypto Risk Manager Pro - Operation Fortress v4.6** is a Basel III-inspired risk management and quantitative analysis platform for cryptocurrency portfolios. Powered by **Comparative Bayesian Quantile Regression Analysis (CBQRA)** and **GJR-GARCH volatility modeling**, it delivers institutional-grade rigor with an accessible interface designed for retail traders, educators, and hobbyist quants.

---

## 🛡️ Philosophy: Capital Preservation First

### Core Principles
1. **Risk Management > Return Chasing**: Better to preserve capital than chase moonshots
2. **Probabilistic Thinking**: Embrace uncertainty with Bayesian methods
3. **Tail Risk Awareness**: Model extreme events, not just averages
4. **Diversification Science**: Correlation-aware portfolio construction
5. **Dynamic Adaptation**: Time-varying volatility adjustments via GARCH

### Basel III Prudence+ Framework
Traditional portfolio theory assumes:
- Gaussian (normal) return distributions
- Constant volatility over time
- Stable correlations between assets
- 70/30 equity/bond optimal allocation

**Crypto breaks all these assumptions.** Our framework addresses this with:

| Traditional Approach | Basel III Prudence+ |
|---------------------|---------------------|
| Mean-variance optimization | Bayesian quantile regression (full distribution) |
| Historical volatility | GJR-GARCH conditional volatility |
| Static correlations | Rolling correlation monitoring |
| Equal treatment of assets | Speculative asset caps (15% max) |
| Buy-and-hold | Dynamic rebalancing with drift thresholds |
| Single scenario forecasts | Monte Carlo simulations (500-2000 paths) |

**Result**: Conservative allocations that survive black swans while capturing upside.

---

## 🏗️ Technical Architecture

### System Components

#### 1. Core Engines
```
MultiCryptoBQRAnalysis()
├── Data Ingestion: CSV parsing, validation, cleaning
├── MCMC Sampling: PyMC Bayesian inference (4 chains, 1000 samples each)
├── Quantile Regression: 5th, 50th, 95th percentile forecasts
├── Correlation Analysis: Static and 30-day rolling correlations
└── Visualization: 11+ professional charts (matplotlib, seaborn)

CryptoMonteCarlo()
├── Portfolio Simulation: Brownian motion with correlated assets
├── Scenario Generation: 500/1000/2000 paths over 90-730 days
├── Risk Metrics: VaR, CVaR, expected return, probability distributions
├── Stress Testing: 2008 crisis, COVID crash, bear market scenarios
└── Comparison: Beat SP500 probability, win rate analysis

GARCHEngine()
├── Model Fitting: GJR-GARCH(1,1) with ARCH effects
├── Leverage Detection: Asymmetric volatility (γ parameter)
├── Conditional Vol: Time-varying volatility forecasts
├── Crisis Detection: Extreme volatility regime alerts
└── Position Adjustment: Dynamic Kelly penalty based on vol spikes

AdvancedVisualizations()
├── Correlation Heatmaps: Static and rolling 30-day windows
├── Performance Dashboards: Multi-metric comparison grids
├── Return Distributions: Histogram overlays with KDE
├── Cumulative Returns: Time series with drawdown shading
├── Risk-Return Scatter: Efficient frontier visualization
└── Pairwise Comparisons: Side-by-side asset deep-dives
```

#### 2. Risk Management Layer
```
RiskMonitor()
├── Correlation Warnings: Alert if pairs exceed 90%
├── Speculative Caps: Enforce 15% max for meme coins
├── Flash Crash Detection: 3+ assets dropping >15%
├── Leverage Effect Alerts: High γ parameters (>0.08)
└── Portfolio Drift Tracking: Rebalance if drift >5%

PositionSizer()
├── Kelly Criterion: Optimal bet sizing per asset
├── Volatility Adjustment: Scale down high-vol positions
├── Correlation Penalty: Reduce overconcentrated pairs
├── Profile Scaling: Conservative (65%), Moderate (100%), Aggressive (135%)
└── GARCH Penalty: Reduce if conditional vol >> historical vol
```

#### 3. State Management
```
SessionManager()
├── Pre-Session Cleanup: Privacy-first artifact purging
├── State Validation: Corruption detection and repair
├── Freeze/Thaw: Persistence layer for expensive computations
├── Cache Management: Monte Carlo, Backtest, GARCH results
└── Session Isolation: Unique IDs, independent state trees
```

### Data Flow Architecture

```
┌─────────────────┐
│   CSV Upload    │  ← User uploads OR default dataset
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Validation │  ← Check Date/Price columns, min 100 rows
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CBQRA Engine   │  ← MCMC sampling (3-5 min)
└────────┬────────┘
         │
         ├──────────────────┐
         │                  │
         ▼                  ▼
┌─────────────────┐  ┌─────────────────┐
│  Risk Analysis  │  │ Visualizations  │
│  - Correlations │  │ - Heatmaps      │
│  - Volatility   │  │ - Returns       │
│  - Forecasts    │  │ - Drawdowns     │
└────────┬────────┘  └─────────────────┘
         │
         ▼
┌─────────────────┐
│ Position Sizing │  ← Kelly Criterion + adjustments
└────────┬────────┘
         │
         ├──────────┬──────────┬──────────┐
         │          │          │          │
         ▼          ▼          ▼          ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
  │  Monte   │ │ Backtest │ │  GARCH   │ │  Time    │
  │  Carlo   │ │  Engine  │ │  Engine  │ │ Machine  │
  └──────────┘ └──────────┘ └──────────┘ └──────────┘
       │             │            │            │
       └─────────────┴────────────┴────────────┘
                      │
                      ▼
              ┌─────────────┐
              │ Risk        │
              │ Dashboard   │
              │ (Allocations)│
              └─────────────┘
```

---

## 📊 User Interface Guide

### Tab 1: 🎯 Risk Dashboard

#### Purpose
Central command for portfolio configuration, allocation viewing, and risk monitoring.

#### Key Sections

##### 1.1 Portfolio Configuration
```
Inputs:
├── Portfolio Value ($): Capital to deploy (default: $10,000)
├── Risk Tolerance: Conservative / Moderate / Aggressive
└── Asset Selection: Default 5 cryptos OR custom uploads

Outputs:
├── Expected Volatility: Weighted average of asset volatilities
├── Sharpe Ratio: Risk-adjusted return estimate
└── Worst Case Drawdown: Maximum historical decline
```

**Risk Profile Details:**

| Profile      | Max Position | Kelly Scale | Stop Loss | Take Profit | Vol Tolerance |
|--------------|--------------|-------------|-----------|-------------|---------------|
| Conservative | 15%          | 65%         | 10%       | 25%         | 50%           |
| Moderate     | 25%          | 100%        | 15%       | 40%         | 75%           |
| Aggressive   | 35%          | 135%        | 20%       | 60%         | 100%          |

##### 1.2 Recommended Positions
```
For each asset:
├── Allocation %: Kelly-optimized position size
├── Dollar Value: Portfolio Value × Allocation
├── Stop Loss: Entry × (1 - Stop Loss %)
├── Take Profit: Entry × (1 + Take Profit %)
└── Warnings: Speculative flag if meme coin detected
```

**Example Output:**
```
XRP: 22.5% ($2,250)
  🔴 Stop Loss: $2,025
  🟢 Take Profit: $2,813

DOGE: 15.0% ($1,500) [⚠️ SPECULATIVE]
  🔴 Stop Loss: $1,350
  🟢 Take Profit: $1,875
  ⚠️ DOGE is high-risk. Use 25% trailing stop-loss.
```

##### 1.3 Active Risk Monitoring
```
Alerts:
├── High Correlation (>90%): "XRP & XLM correlation 0.94 - reduce combined by 20%"
├── Speculative Overweight: "DOGE allocation 18% exceeds 15% max"
├── Flash Crash Risk: "3 assets in portfolio dropped >15% today"
└── Leverage Effect: "BTC γ=0.12 - downside shocks amplified 1.12x"
```

Dismissible warnings with confirmation tracking (prevents alert fatigue).

##### 1.4 Monte Carlo Simulations (Optional)
```
Configuration:
├── Number of Simulations: 500 / 1000 / 2000
├── Time Horizon: 3 months / 6 months / 1 year / 2 years
├── Random Seed: Profile-specific for reproducibility
└── Stress Testing: Enable 2008/COVID/Bear/Mild scenarios

Outputs:
├── Expected Return: Mean across all paths
├── Best/Worst Case: 5th and 95th percentiles
├── VaR 95%: Value at Risk (worst loss at 95% confidence)
├── Probability Positive: Chance of any profit
├── Beat SP500 Chance: Outperformance probability
└── Stress Test Table: Performance under crisis scenarios
```

**Monte Carlo Visualization:**
- 2000 semi-transparent paths (gray)
- Median path (blue, thick)
- Mean path (green, dashed)
- 5th/95th percentile bands (red shading)

**Stress Test Results Example:**
| Scenario              | Expected Return | Median Final | Worst Case  |
|-----------------------|-----------------|--------------|-------------|
| 2008 Crisis (-50%)    | -35.2%          | $6,480       | $4,200      |
| 2020 COVID (-35%)     | -18.7%          | $8,130       | $6,800      |
| Bear Market (-20%)    | -8.3%           | $9,170       | $8,500      |
| Mild Correction (-10%)| +2.1%           | $10,210      | $9,700      |

##### 1.5 Smart Glossary Integration
```
Search: "sharpe ratio" → Fuzzy matches (70%+ threshold)
Results:
├── 🟢 Sharpe Ratio (100%): Risk-adjusted return measure (higher is better)
├── 🟡 Sortino Ratio (82%): Downside-only risk adjustment
└── 🟠 Information Ratio (75%): Active return per unit tracking error
```

Browse by category:
- 📈 Risk & Return Metrics
- 🔮 Simulation & Forecasting
- ⚖️ Portfolio Construction

---

### Tab 2: 🧠 CBQRA Engine

#### Purpose
Run Bayesian quantile regression analysis, generate forecasts, and produce professional visualizations.

#### Workflow

##### 2.1 Pre-Analysis Checks
```
System Validation:
├── Data Source Confirmed: Default OR X uploaded files
├── Risk Profile Set: Current = MODERATE
├── Module Availability: BQR ✅ | GARCH ✅ | Monte Carlo ✅
└── Previous Analysis: None (or "Completed with CONSERVATIVE profile")
```

##### 2.2 Running Analysis
```
Button: 🚀 Run CBQRA Analysis

Progress:
├── Phase 1: Initialization (0-30%) - Loading data, setting up models
├── Phase 2: MCMC Sampling (30-90%) - Monitor terminal for chain progress
│   ├── Asset 1: XRP_q0.05 → XRP_q0.5 → XRP_q0.95 (3 models × 4 chains)
│   ├── Asset 2: XLM_q0.05 → XLM_q0.5 → XLM_q0.95
│   └── ... (repeat for all assets)
└── Phase 3: Visualization (90-100%) - Chart generation, saving to disk
```

**Terminal Output Example:**
```
Sampling 4 chains for 1_000 tune and 1_000 draw iterations...
Progress | Draws | Divergences | Step size | Speed
████████ | 2000  | 0           | 0.559     | 79.54 draws/s
████████ | 2000  | 0           | 0.182     | 199.37 draws/s
```

**Expected Duration:**
- 2-core CPU: 15-25 minutes (5 assets)
- 4-core CPU: 5-10 minutes (5 assets)
- 8-core CPU: 2-5 minutes (5 assets)

##### 2.3 Analysis Outputs

###### Main Visualizations (11 total)
1. **Correlation Matrix Heatmap**: Static correlation coefficients
2. **Rolling Correlation Heatmap**: 30-day windowed correlations
3. **Volatility Comparison**: Bar chart of annualized volatilities
4. **Performance Dashboard**: Grid of Sharpe/Sortino/Max Drawdown
5. **Return Distributions**: Histogram overlays with KDE curves
6. **Cumulative Returns**: Time series with indexed growth
7. **Risk-Return Scatter**: Volatility (X) vs Return (Y) plot
8. **Drawdown Comparison**: Underwater equity curves
9. **Multi-Crypto Correlation**: Enhanced heatmap with dendrograms
10. **Multi-Asset Summary**: Table of key metrics
11. **Forecast Comparison**: Q0.05, Q0.5, Q0.95 side-by-side

###### Pairwise Comparisons (N×(N-1)/2 charts)
For 5 assets: 10 pairwise comparisons
- XRP vs XLM
- XRP vs XMR
- XRP vs TRX
- ... (all combinations)

Each pairwise chart shows:
- Normalized price overlay
- Return scatter plot
- Correlation coefficient
- Beta coefficient

##### 2.4 Downloading Results

###### Jumbo Pack (Recommended)
```
Button: 💾 Download Jumbo Pack

Contents:
├── Main Visualizations (11 PNG files)
├── Pairwise Comparisons (10 PNG files)
├── Performance Metrics (CSV)
├── Forecast Data (CSV)
└── Configuration Snapshot (TXT)

Filename: CRYPTO_FORTRESS_MODERATE_2025-01-15_1430.zip
Size: ~5-15 MB (highly compressed)
```

**Critical**: Download before ending session! Privacy-first design purges visualizations on next session start.

###### Individual Chart Downloads
Each visualization has a dedicated "📥 Download [Chart Name]" button.

---

### Tab 3: 📈 Backtesting

#### Purpose
Test portfolio strategies against historical data with configurable rebalancing frequencies.

#### Configuration

##### 3.1 Backtest Parameters
```
Inputs:
├── Backtest Period:
│   ├── Last 30 Days (for quick validation)
│   ├── Last 90 Days (for short-term trends)
│   ├── Last 180 Days (for seasonal effects)
│   ├── Last Year (for full market cycle)
│   └── All Available Data (for long-term analysis)
│
├── Rebalancing Frequency:
│   ├── Daily (most responsive, highest transaction costs)
│   ├── Weekly (balanced trade-off)
│   └── Monthly (low costs, drift tolerance)
│
└── Initial Capital: $1,000 - $1,000,000 (default: $10,000)
```

##### 3.2 Execution Flow
```
Button: Run Backtest

Process:
├── 1. Align Data: Find common date range across all assets
├── 2. Generate Rebalance Dates: Based on frequency selected
├── 3. Simulate Trades:
│   ├── Rebalance Event: Liquidate all holdings → recalculate targets
│   ├── Buy Shares: Allocate capital per position sizes
│   └── Mark-to-Market: Daily portfolio valuation
├── 4. Calculate Metrics: Returns, volatility, Sharpe, drawdown
└── 5. Generate Chart: Portfolio value over time
```

##### 3.3 Results Interpretation

###### Key Metrics
```
Total Return: (Final Value / Initial Capital - 1) × 100%
├── 15.7% over 180 days = good
└── -8.2% over 180 days = underperforming

Annualized Return: ((Final / Initial)^(365/days) - 1) × 100%
├── Normalizes for time period comparison
└── 32.4% annualized = excellent for moderate profile

Sharpe Ratio: (Ann. Return - Risk-Free Rate) / Ann. Volatility
├── > 1.0 = acceptable
├── > 2.0 = good
└── > 3.0 = excellent

Max Drawdown: Worst peak-to-trough decline
├── -15.3% = manageable
├── -35.8% = concerning
└── -60%+ = catastrophic (adjust allocations!)

Win Rate: Percentage of profitable days
├── 55%+ = positive trend
├── 45-55% = neutral
└── <45% = losing strategy
```

###### Portfolio Value Chart
- Blue line: Portfolio value over time
- Red dashed: Initial capital (break-even line)
- Green/Red shading: Profit/loss zones
- Annotations: Major drawdown events

##### 3.4 Allocation Table
Shows exact positions used in backtest:
```
| Asset | Allocation | Value      |
|-------|------------|------------|
| XRP   | 22.5%      | $2,250.00  |
| XLM   | 18.3%      | $1,830.00  |
| XMR   | 20.1%      | $2,010.00  |
| TRX   | 24.2%      | $2,420.00  |
| DOGE  | 15.0%      | $1,500.00  |
```

---

### Tab 4: 🌪️ GJR-GARCH

#### Purpose
Advanced volatility modeling with asymmetric leverage effects and conditional forecasting.

#### Concepts

##### 4.1 What is GJR-GARCH?
**Generalized Autoregressive Conditional Heteroskedasticity** with Glosten-Jagannathan-Runkle modification.

**Standard GARCH**: σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁

**GJR-GARCH**: σ²ₜ = ω + α·ε²ₜ₋₁ + **γ·Iₜ₋₁·ε²ₜ₋₁** + β·σ²ₜ₋₁

Where:
- **σ²ₜ**: Conditional variance (volatility squared) at time t
- **ε²ₜ₋₁**: Previous period's squared shock (return deviation)
- **Iₜ₋₁**: Indicator (1 if εₜ₋₁ < 0, else 0)
- **γ**: **Leverage effect parameter** (THE KEY INNOVATION!)

**Leverage Effect Interpretation:**
- **γ = 0**: Symmetric (bad news = good news for volatility)
- **γ > 0**: Asymmetric (bad news increases volatility MORE)
- **γ = 0.08**: -5% shock increases vol 1.08× more than +5% shock

##### 4.2 Why GJR-GARCH for Crypto?
```
Crypto Markets Exhibit:
├── Volatility Clustering: Big moves follow big moves
├── Leverage Effects: Crashes amplify volatility asymmetrically
├── Regime Shifts: Bull vs bear markets have different vol dynamics
└── Fat Tails: Extreme events more common than normal distribution predicts
```

**BQR vs GARCH Comparison:**

| Feature               | BQR                        | GJR-GARCH                  |
|-----------------------|----------------------------|----------------------------|
| Volatility Assumption | Constant over horizon      | Time-varying (dynamic)     |
| Tail Risk             | Quantile-based (5th/95th)  | Conditional variance       |
| Leverage Effects      | ❌ Not modeled             | ✅ γ parameter             |
| Forecast Type         | Return distribution        | Volatility distribution    |
| Best For              | Strategic allocation       | Tactical adjustments       |

**Combined Approach:**
1. BQR sets strategic weights (long-term)
2. GARCH applies tactical penalties (short-term vol spikes)
3. Result: Adaptive portfolio that respects both trend and risk

#### Workflow

##### 4.3 Fitting GARCH Models
```
Button: 🔄 Fit GARCH Models

Process (per asset):
├── 1. Extract Returns: prices.pct_change().dropna()
├── 2. Fit GJR-GARCH(1,1):
│   ├── Estimate ω, α, β, γ via Maximum Likelihood
│   ├── Check convergence (optimization warnings)
│   └── Validate parameters (α+β<1, γ>0)
├── 3. Calculate Conditional Vol: σₜ = √(σ²ₜ)
├── 4. Detect Leverage: Test γ significantly > 0
└── 5. Cache Model: Store fitted_models[crypto] = result

Duration:
├── 1-2 minutes per asset (single-core optimization)
└── 5-10 minutes total for 5 assets
```

**Terminal Output:**
```
Iteration:      1,   Func. Count:      7,   Neg. LLF: 1234.5678
Iteration:      2,   Func. Count:     14,   Neg. LLF: 1230.1234
...
Optimization terminated successfully    (Exit mode 0)
✅ XRP: GARCH fitted successfully
```

##### 4.4 GARCH Insights Dashboard
```
Table Columns:
├── Crypto: Asset ticker
├── Leverage Effect (γ): Asymmetry parameter
├── Status: ✅ Confirmed (γ>0.04) | ⚠️ Moderate (0.02<γ<0.04) | ✓ Mild (γ<0.02)
├── Conditional Vol: Current annualized volatility (%)
└── VaR 95%: Worst-case 1-day loss at 95% confidence
```

**Example Output:**
| Crypto | Leverage Effect | Status         | Conditional Vol | VaR 95% |
|--------|-----------------|----------------|-----------------|---------|
| BTC    | 0.1234          | ✅ Confirmed   | 67.8%           | -8.2%   |
| ETH    | 0.0987          | ✅ Confirmed   | 72.3%           | -9.1%   |
| XRP    | 0.0456          | ⚠️ Moderate    | 89.4%           | -11.3%  |
| DOGE   | 0.1523          | 🔥 STRONG      | 134.7%          | -17.8%  |

**Interpretation:**
- DOGE shows **strongest leverage** (γ=0.1523) → bad news hits 1.15× harder
- All assets confirm asymmetry → need tighter stop-losses on downside
- High conditional vol assets (DOGE: 134.7%) get position size penalties

##### 4.5 Quick GARCH Volatility Charts
```
Button: Generate Quick GARCH Volatility Charts

For each asset:
├── Historical Conditional Volatility (last 500 days)
├── Annualized % scale
├── Title: "{Crypto} — Same, same until it ISN'T"
└── Export: Download individual PNG
```

**Chart Features:**
- Red line: GARCH conditional volatility
- Shows clustering: quiet periods → explosive regimes
- Identifies current regime: Low/Normal/Elevated/EXTREME

##### 4.6 Volatility Forecast Comparison: BQR vs GARCH
```
Showdown Table:
├── Asset: Crypto ticker
├── Historical Vol: 30-day rolling standard deviation
├── BQR Risk Spread: (Q0.95 - Q0.05) × √365
├── GARCH Conditional Vol: σₜ × √365
├── GARCH - BQR (pp): Percentage point difference
├── Leverage γ: Asymmetry parameter
└── Winner: Which model closer to historical vol
```

**Example:**
| Asset | Hist Vol | BQR Vol | GARCH Vol | GARCH-BQR | γ      | Winner |
|-------|----------|---------|-----------|-----------|--------|--------|
| XRP   | 85.2%    | 78.3%   | 87.1%     | +8.8 pp   | 0.0456 | GARCH  |
| DOGE  | 128.7%   | 115.2%  | 131.4%    | +16.2 pp  | 0.1523 | GARCH  |

**Bar Chart Visualization:**
- Gray bars: Historical volatility (baseline)
- Blue bars: BQR risk spread
- Red bars: GARCH conditional vol
- "WIN" labels on most accurate model

**Key Finding:**
> **GARCH wins 4/5 assets — especially where leverage effect is strong (γ > 0.05)**

##### 4.7 GARCH-Adjusted Position Sizes
```
Side-by-Side Comparison:

Standard Kelly Allocation          GARCH-Penalized Allocation
(Volatility-adjusted only)         (Conditional vol penalty)
┌────────────────────┐             ┌────────────────────┐
│ XRP:   22.5%       │             │ XRP:   20.8%   ⬇️  │
│ XLM:   18.3%       │             │ XLM:   17.1%   ⬇️  │
│ XMR:   20.1%       │             │ XMR:   19.5%   ⬇️  │
│ TRX:   24.2%       │             │ TRX:   22.9%   ⬇️  │
│ DOGE:  15.0%       │             │ DOGE:  12.7%   ⬇️⬇️│
└────────────────────┘             └────────────────────┘
```

**Penalty Calculation:**
```python
if garch_conditional_vol > historical_vol * 1.2:
    penalty_factor = min(0.9, historical_vol / garch_conditional_vol)
    adjusted_allocation = standard_allocation * penalty_factor
```

**Intuition**: If GARCH detects current vol 20%+ above historical, reduce position proportionally.

##### 4.8 Current Volatility Regime
```
For each asset:
├── Current Vol: GARCH conditional vol (annualized %)
├── Regime Classification:
│   ├── 😴 Low Vol: <50%
│   ├── ⚠️ Normal: 50-80%
│   ├── 🔥 Elevated: 80-120%
│   └── 🌪️ EXTREME: >120%
└── Recommendation: Position size adjustment
```

**Example:**
```
XRP: 87.3% → 🔥 Elevated
Recommendation: Reduce position 10%, tighten stop-loss to 12%

DOGE: 134.2% → 🌪️ EXTREME
Recommendation: Reduce position 30%, consider 25% trailing stop
```

##### 4.9 The Elton John Leverage Blues 🎵
```
"Who's Got the Blues?" Leaderboard:
┌────────────────────────────────────┐
│ Asset  │ γ (Leverage) │ Impact     │
├────────┼──────────────┼────────────┤
│ DOGE   │ 0.1523       │ 1.15× worse│ 🎸🎸🎸
│ BTC    │ 0.1234       │ 1.12× worse│ 🎸🎸
│ ETH    │ 0.0987       │ 1.10× worse│ 🎸
│ XRP    │ 0.0456       │ 1.05× worse│ 🎵
└────────┴──────────────┴────────────┘

Status: 4 assets singing the blues!

"And I think it's gonna be a long, long time…
Till volatility comes down to earth again…"
— Elton γ John, 2025
```

##### 4.10 Volatility Forecasts (7-90 days)
```
Configuration:
└── Forecast Horizon: 7 / 14 / 30 / 60 / 90 days (slider)

For each asset:
├── Historical Conditional Vol (last 500 days, blue line)
├── GARCH Forecast (red dashed line extending forward)
├── Confidence Bands (20% above/below, red shading)
└── Download: Individual forecast PNG
```

**Use Cases:**
- **7-day**: Tactical rebalancing decisions
- **30-day**: Monthly portfolio reviews
- **90-day**: Quarterly strategic adjustments

##### 4.11 Crisis Detection System
```
Crisis Threshold: 100% annualized volatility

Alert Logic:
IF any asset shows conditional_vol > 100%:
    ├── 🚨 CRISIS MODE: Display red banner
    ├── List affected assets with current vol
    ├── Recommendation:
    │   ├── Reduce exposure by 50%
    │   ├── Implement tight stop-losses (5-10%)
    │   └── Keep 20-30% cash for dip-buying
    └── Export: Crisis report CSV

ELSE:
    └── ✅ No crisis-level volatility detected
```

---

## 🔧 Advanced Configuration

### Editing Core Parameters

#### File: `crypto_fortress.py`

##### MCMC Settings
```python
BASE_CONFIG = {
    'mcmc_samples': 1000,       # Increase for better convergence (slower)
    'mcmc_tune': 1000,          # Tuning iterations (warm-up phase)
    'mcmc_cores': 4,            # Match your CPU cores
    'mcmc_target_accept': 0.95  # Acceptance rate (0.8-0.99)
}
```

**Recommendations:**
- **mcmc_samples**: 500 (fast) | 1000 (default) | 2000 (robust)
- **mcmc_tune**: Keep equal to mcmc_samples for stability
- **mcmc_cores**: `os.cpu_count()` for maximum speed
- **mcmc_target_accept**: 0.95 (default), lower to 0
