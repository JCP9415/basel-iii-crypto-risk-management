[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://basel-iii-crypto-risk-management-kvqpfpeyh99fepbgzscck5.streamlit.app/)

# 🧠 Crypto Risk Manager Pro — Operation Fortress v4.6
**Basel III Crypto Risk Management: Industrial-Strength Quant for the Home**

![CBQRA in Action](assets/demo/analysis_complete.png)
![MCMC Progress](assets/mcmc_progress.png)
![Visualizations](assets/demo/visualizations.png)

---

## 🚀 What's New in v4.6: Operation Fortress

### 🛡️ Military-Grade Reliability
- **Comprehensive Error Boundaries**: Every function wrapped with defensive programming
- **State Validation**: Automatic corruption detection and repair
- **Graceful Degradation**: System continues operating even with partial failures
- **Pre-Session Cleanup**: Privacy-first design purges residual artifacts between sessions

### 💾 Time Machine Persistence Layer
- **Freeze & Thaw**: Save expensive CBQRA computations to disk
- **Instant Restoration**: Skip 5+ minute reruns — restore analysis in seconds
- **Selective State Management**: Preserves test results while respecting privacy
- **Jumbo Pack Downloads**: Export all visualizations before session end

### 🌪️ GJR-GARCH Volatility Engine
- **Asymmetric Leverage Effects**: Model how bad news impacts volatility more than good news
- **Volatility Clustering Detection**: Capture persistence of price shocks
- **Dynamic Risk Adjustment**: Real-time conditional volatility forecasts
- **GARCH vs BQR Showdown**: Compare forecast accuracy head-to-head

### 🎲 Enhanced Monte Carlo
- **Stress Testing**: Simulate 2008 financial crisis, COVID crash, and bear market scenarios
- **Risk-Adjusted Seeds**: Different random paths per risk profile
- **Intelligent Caching**: Configuration-aware result preservation
- **Profile Mismatch Guards**: Prevents running simulations with inconsistent settings

### 🔬 Advanced Backtesting
- **Rebalancing Strategies**: Daily, weekly, or monthly portfolio adjustments
- **Risk-Adjusted Metrics**: Sharpe ratio, max drawdown, win rate analysis
- **Historical Scenario Testing**: See how strategies performed in past regimes
- **Profile-Locked Analysis**: Ensures consistent risk parameters throughout tests

---

## 🚀 Mission Statement

This dashboard democratizes institutional-grade crypto quantitative analysis. Built for:

- **Retail traders** seeking professional-grade risk management
- **Hobbyist quants** learning advanced financial modeling
- **Educators** teaching modern portfolio theory with real data
- **Risk-averse investors** prioritizing capital preservation over hype

**Power users** can tweak MCMC parameters, quantile targets, and risk thresholds — but remember:
> _"Moving away from Prudent Person standards. You have been duly warned."_

---

## 🏦 Beyond 70/30 — Enter Basel III Prudence+

Standard portfolio theory defaults to **70/30 equity/bond splits**, assuming Gaussian returns and stable regimes. Crypto laughs at that.

**Crypto Risk Manager Pro** embraces:
- ✅ **Pessimistic Forward Drift** modeling with Bayesian uncertainty quantification
- ✅ **Speculative Asset Caps** — meme coins limited to 15% max allocation
- ✅ **Flash Crash Detection** — correlation-based early warning system
- ✅ **Kelly Criterion Scaling** — risk-adjusted position sizing per profile
- ✅ **Volatility-Adjusted Allocations** — GARCH penalties for elevated conditional vol
- ✅ **Correlation-Aware Rebalancing** — detect and mitigate concentration risk

This isn't just prudent — it's **Basel III Prudence+**.

---

## 📊 Why Bayesian Quantile Regression Beats Standard Methods

| Feature                     | Standard Regression | **Bayesian Quantile Regression (BQR)** |
|-----------------------------|---------------------|----------------------------------------|
| Assumes normality           | ✅ Yes              | ❌ No — models full distribution       |
| Sensitive to outliers       | ✅ Yes              | ❌ No — robust to extreme events       |
| Captures tail risk          | ❌ No               | ✅ Yes — 5th, 50th, 95th percentiles   |
| Forecasts full distribution | ❌ No               | ✅ Yes — probabilistic predictions     |
| Adapts to regime shifts     | ❌ No               | ✅ Yes — MCMC updates with new data    |
| Crypto-ready                | ❌ Not really       | ✅ Absolutely — built for volatility   |

**BQR + GJR-GARCH = The One-Two Punch for Crypto Risk**

---

## 🔬 Technical Architecture

### Core Engines
```python
MultiCryptoBQRAnalysis()    # Bayesian forecasting with MCMC sampling
CryptoMonteCarlo()          # 500-2000 simulation portfolio projections
AdvancedVisualizations()    # Professional-grade chart generation
GARCHEngine()               # GJR-GARCH volatility modeling
RiskMonitor()               # Real-time correlation/leverage alerts
```

### Risk Framework Flow
```
Portfolio Config → CBQRA Analysis → Risk Validation → Position Sizing
                                    ↓
                 GJR-GARCH Penalty → Kelly Scaling → Final Allocations
                                    ↓
            Monte Carlo Simulation → Stress Testing → Backtest Verification
```

### State Management
```
Session Start → Pre-Cleanup (Privacy) → CBQRA (5 min) → Freeze State
                                                         ↓
Next Session → Thaw State (Instant) → Monte Carlo/Backtest/GARCH (Restored)
```

---

## 🛠️ System Requirements

**Minimum (Tested on Lenovo 110S — Yes, Really!):**
- **CPU**: Dual-core Intel Atom or AMD equivalent
- **RAM**: 2GB (swap recommended for MCMC)
- **Storage**: 32GB eMMC or better
- **OS**: Windows 7+, Linux, macOS 10.12+
- **Browser**: Chrome 90+, Firefox 88+, Edge 90+
- **Python**: 3.8 - 3.11 (3.9 recommended)

**Recommended for Smoother Experience:**
- **CPU**: Quad-core Intel i5/AMD Ryzen 5 or better
- **RAM**: 8GB+
- **Storage**: SSD with 5GB free space
- **Cores**: 4+ for parallel MCMC chains

**Note**: MCMC sampling is CPU-intensive but respects your hardware. Expected runtime:
- 2-core: 3-5 minutes per asset
- 4-core: 1-2 minutes per asset
- 8-core: 30-60 seconds per asset

---

## 🎉 Feature Highlights

### 📈 CBQRA Engine
- Full Bayesian MCMC sampling with PyMC
- Multi-asset comparative analysis (1-10+ cryptos)
- Automatic professional chart generation
- Terminal progress tracking with divergence monitoring
- Pairwise asset correlation deep-dives

### 🎲 Monte Carlo Simulations
- 500/1000/2000 path configurations
- Time horizons: 3 months to 2 years
- Profile-specific random seeds for reproducibility
- Stress test scenarios (2008 crisis, COVID crash, bear markets)
- Expected return, VaR 95%, Conditional VaR

### 🔬 Backtesting Engine
- Historical performance simulation
- Rebalancing strategies (daily/weekly/monthly)
- Risk-adjusted returns analysis (Sharpe, Sortino)
- Drawdown scenarios and win rate analytics
- Profile-locked consistency checks

### 🌪️ GJR-GARCH Analysis
- Asymmetric leverage effect detection ("bad news hits harder")
- Conditional volatility forecasting (7-90 days)
- GARCH vs BQR accuracy showdown
- Crisis detection system (extreme volatility alerts)
- "Elton John Leverage Blues" — γ parameter visualization

### 🧠 Smart Glossary
- 100+ financial terms with plain-English definitions
- Fuzzy search with 70%+ match threshold
- Browse by category (risk metrics, forecasting, portfolio construction)
- Inline help tooltips throughout UI

### 🛡️ Risk Monitoring
- Real-time correlation heatmaps
- High correlation warnings (>90%)
- Speculative asset caps (15% max for meme coins)
- Flash crash detection (3+ assets dropping >15%)
- Dismissible alerts with confirmation tracking

### 💾 Time Machine
- **Freeze State**: Save analyzer, models, forecasts, allocations to `crypto_brain_freeze.pkl`
- **Thaw State**: Instant restore — skip CBQRA reruns entirely
- **Skeleton Detection**: Ignores empty initialization states
- **Jumbo Pack**: Download all PNG/CSV/HTML files as ZIP before session end

### 🔄 Data Management
- Default dataset: XRP, XLM, XMR, TRX, DOGE (5 assets)
- CSV upload support for custom assets
- Automatic duplicate detection by ticker
- File hash tracking for change detection
- Privacy-first: Uploads purged between sessions

### 🧹 System Controls
- **Nuclear Flush**: Complete state reset + temp cleanup
- **Selective Resets**: Clear analysis, warnings, Monte Carlo, uploads independently
- **Emergency Stop**: Interrupt runaway MCMC sampling
- **Session Isolation**: Each run gets unique session ID

---

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/JCP9415/basel-iii-crypto-risk-management.git
cd basel-iii-crypto-risk-management

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run crypto_fortress.py
```

### First Run Workflow
1. **Risk Dashboard Tab**: Set portfolio value ($10,000 default) and risk profile (Conservative/Moderate/Aggressive)
2. **CBQRA Tab**: Click "🚀 Run CBQRA Analysis" — monitor terminal for MCMC progress (3-5 min)
3. **Analysis Complete**: View 8+ main visualizations + pairwise comparisons
4. **Optional**: Download "Jumbo Pack" ZIP with all charts
5. **Time Machine**: Click "💾 Freeze State" to save analysis to disk
6. **Risk Dashboard Tab**: Enable Monte Carlo simulations, configure scenarios, run stress tests
7. **Backtesting Tab**: Test historical performance with rebalancing strategies
8. **GJR-GARCH Tab**: Fit volatility models, view forecasts, detect leverage effects

### Next Session Workflow
1. **Time Machine**: Click "📂 Thaw State" — instant restore (no CBQRA rerun needed!)
2. **Risk Dashboard/Backtesting/GARCH Tabs**: All cached results available immediately
3. **Change Risk Profile**: Requires fresh CBQRA run (allocations recalculated)

---

## 🧪 Advanced Mode (Use With Caution)

Edit `crypto_fortress.py` to customize:

### MCMC Configuration
```python
BASE_CONFIG = {
    'mcmc_samples': 1000,      # Default: 1000 (increase for better convergence)
    'mcmc_tune': 1000,         # Default: 1000 (tuning iterations)
    'mcmc_cores': 4,           # Default: 4 (match your CPU cores)
    'mcmc_target_accept': 0.95 # Default: 0.95 (acceptance rate)
}
```

### Risk Thresholds
```python
RISK_THRESHOLDS = {
    'max_correlation': 0.90,              # Correlation alert trigger
    'speculative_allocation_max': 0.15,   # Meme coin cap (15%)
    'daily_loss_limit': 0.40,             # Max drawdown tolerance
    'flash_crash_assets': 3,              # Flash crash: 3+ assets...
    'flash_crash_drop': 0.15,             # ...dropping >15%
    'rebalance_drift': 0.05               # Rebalance if drift >5%
}
```

### Kelly Criterion Scaling
```python
risk_profiles = {
    "conservative": {"kelly_scale": 0.65},  # 65% of full Kelly
    "moderate":     {"kelly_scale": 1.00},  # 100% of full Kelly
    "aggressive":   {"kelly_scale": 1.35}   # 135% of full Kelly (⚠️ risky!)
}
```

**⚠️ Warning**: Deviating from defaults moves away from Prudent Person standards. Proceed at your own risk.

---

## 📚 Documentation

- **[User Manual](USER_MANUAL.md)**: Comprehensive feature guide
- **[FAQ](FAQ.md)**: Common questions and troubleshooting
- **[Glossary](glossary.py)**: 100+ financial term definitions
- **[Technical Architecture](USER_MANUAL.md#technical-architecture)**: Under-the-hood details

---

## 🎸 Cultural DNA

This dashboard is powered by:
- 🐰 **Bugs Bunny Logic**: Fast, clever, irreverent — "Ain't I a stinker?"
- 🎩 **Chaplin Charm**: Silent but expressive — actions speak louder
- 🎸 **Kansas-Style Optimism**: "Carry on, wayward quant" — persistence wins
- 🎵 **Elton John Leverage Blues**: "And I think it's gonna be a long, long time..." — asymmetry matters

---

## 🤝 Contributing

Contributions welcome! Focus areas:
- Additional volatility models (EGARCH, TGARCH)
- Machine learning integrations (LSTM, transformer forecasts)
- Alternative risk metrics (Sortino, Omega, Calmar ratios)
- Multi-asset class support (stocks, commodities, FX)
- Enhanced stress testing scenarios

Open an issue or submit a PR. Let's make quant analysis accessible to everyone.

---

## 📜 License

MIT License — free to use, modify, and share. See [LICENSE](LICENSE) for details.

---

## 🚨 Disclaimer

**NOT FINANCIAL ADVICE**

- This software is for educational purposes only
- Past performance does not guarantee future results
- Cryptocurrency investing involves substantial risk of loss
- You could lose your entire investment
- Always consult qualified financial advisors before making investment decisions
- The authors assume no liability for financial losses resulting from use of this software

**Use at your own risk. DYOR (Do Your Own Research).**

---

## 🎵 Parting Shot

> *"And I think it's gonna be a long, long time,*
> *Until this dashboard ever fails on me,*
> *I'm not the dev they think I am at home,*
> *Oh no no no!*
> *I'm a ROCKET MAAAAAN!*
> *Rocket MAAAAN, building all the systems that can't be beat!"*

> *...Oh sweet freedom whispers in my ear...*

> ***In Lak'ech...🚀***

---

**Built with ❤️ for rigorous crypto risk management and portfolio optimization**

*Tested on hardware you'd find in a thrift store. Because good quant shouldn't require a data center.*
