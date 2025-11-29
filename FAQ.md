# ❓ FAQ — Crypto Risk Manager Pro v4.6

## 🤔 General Questions

### Is this safe to use?
**Yes, but with important caveats:**
- This is **educational software** for learning advanced portfolio risk management
- **NOT financial advice** — you make your own investment decisions
- We help you DYOR (Do Your Own Research) with institutional-grade tools
- The software itself is safe and contains no malicious code (open-source, MIT licensed)
- However, crypto investing involves substantial risk — you could lose your entire investment

---

### Do I need coding experience to use this?
**No!** The Streamlit interface is point-and-click friendly:
- Select risk profile with radio buttons
- Upload CSVs or use default dataset
- Click "Run CBQRA Analysis" and monitor terminal
- View visualizations, download charts, run simulations

**However**, understanding basic financial concepts helps:
- Sharpe ratio, volatility, correlation
- Monte Carlo simulation principles
- Risk-adjusted returns
- Use the built-in **Smart Glossary** (100+ terms) to learn as you go!

---

### What makes v4.6 "Operation Fortress" special?
**Military-grade reliability upgrades:**
- **Comprehensive error boundaries**: Every function has defensive programming
- **State validation**: Automatic corruption detection and repair
- **Graceful degradation**: System continues even with partial failures
- **Time Machine persistence**: Save/load analysis states to skip expensive reruns
- **Privacy-first design**: Pre-session cleanup purges residual artifacts
- **Enhanced logging**: Detailed debug logs in `crypto_dashboard_debug.log`

**Bottom line**: v4.6 is production-ready, not a fragile prototype.

---

## ⏱️ Performance & System Requirements

### Why does CBQRA analysis take so long?
**Because Bayesian statistics are computationally expensive:**
- **MCMC sampling**: Runs thousands of Markov Chain Monte Carlo iterations per asset
- **Multi-chain convergence**: 4 parallel chains ensure robust results
- **Quantile regression**: Fits 5th, 50th, and 95th percentiles simultaneously
- **Correlation matrices**: Computes rolling and static correlations across all pairs

**Expected duration:**
- 2-core CPU: 3-5 minutes per asset
- 4-core CPU: 1-2 minutes per asset
- 8-core CPU: 30-60 seconds per asset

**Pro tip**: Use **Time Machine** to freeze/thaw states — run CBQRA once, restore instantly forever!

---

### Can I speed up MCMC sampling?
**Yes, with caveats:**

1. **Increase CPU cores** (edit `BASE_CONFIG['mcmc_cores']` to match your hardware)
2. **Reduce sample size** (change `mcmc_samples` from 1000 to 500) — ⚠️ less convergence
3. **Reduce tuning iterations** (change `mcmc_tune` from 1000 to 500) — ⚠️ worse acceptance rates
4. **Use fewer assets** (analyze 3 cryptos instead of 5+)

**⚠️ Warning**: Lowering samples/tuning moves away from Prudent Person standards. Monitor divergences in terminal!

---

### It crashed on my old laptop! Help!
**Troubleshooting steps:**

1. **Check terminal error messages** — look for:
   - `MemoryError`: Insufficient RAM (close browser tabs, use fewer assets)
   - `ImportError`: Missing dependencies (run `pip install -r requirements.txt`)
   - `FileNotFoundError`: Missing CSV files (check file paths)

2. **Try fewer assets first**:
   - Start with 2-3 cryptos instead of 5+
   - Test with default dataset before custom uploads

3. **Reduce MCMC parameters**:
   ```python
   'mcmc_samples': 500,  # Down from 1000
   'mcmc_cores': 2,      # Match your CPU
   ```

4. **Check available RAM**:
   - Close memory-heavy apps (browsers, IDEs)
   - Consider adding swap space on Linux

5. **Review debug logs**:
   - Check `crypto_dashboard_debug.log` for detailed errors
   - Look for `CRITICAL` or `ERROR` level messages

**Still stuck?** Open a GitHub issue with:
- Your system specs (CPU, RAM, OS)
- Full error traceback from terminal
- Debug log excerpt

---

## 💰 Financial Questions

### Do I need a lot of money to use this?
**No!** Principles scale to any portfolio size:
- Test with $1,000 — Kelly scaling and risk management still apply
- Learn with $10,000 default settings
- Apply to $100,000+ real portfolio after paper trading

**Important**: Start small, validate strategies, then scale up. **Never invest money you can't afford to lose.**

---

### What's the expected ROI?
**We don't promise returns — that's not how prudent risk management works.**

Instead, this tool helps you:
- **Reduce catastrophic losses** via stop-losses and position sizing
- **Optimize risk-adjusted returns** using Kelly Criterion
- **Avoid emotional decisions** with data-driven allocations
- **Stress test strategies** before deploying real capital

**Realistic expectations:**
- Conservative profile: Target 10-20% annual return, 40-60% volatility
- Moderate profile: Target 20-40% annual return, 60-80% volatility
- Aggressive profile: Target 40-60% annual return, 80-120% volatility

**Remember**: Crypto is volatile. Past performance ≠ future results.

---

### Why are my allocations different from last run?
**Several factors cause allocation changes:**

1. **Risk profile switch**: Conservative/Moderate/Aggressive use different Kelly scales
2. **New data**: If you re-uploaded CSVs, price history affects volatility estimates
3. **GARCH engine toggle**: GJR-GARCH applies dynamic penalties vs BQR static estimates
4. **Correlation shifts**: If asset correlations changed, diversification benefits adjust
5. **Monte Carlo randomness**: Different simulation seeds generate varying forecasts

**To maintain consistency:**
- Use **Time Machine Freeze/Thaw** to preserve exact states
- Lock to one risk profile per analysis session
- Avoid re-running CBQRA unnecessarily (expensive computation!)

---

## 🛠️ Technical Questions

### What's the difference between BQR and GARCH?
**BQR (Bayesian Quantile Regression):**
- Forecasts return distributions (5th, 50th, 95th percentiles)
- Captures tail risk and asymmetry
- Assumes constant volatility within forecast horizon
- Best for: Long-term strategic allocation

**GJR-GARCH (Glosten-Jagannathan-Runkle GARCH):**
- Models time-varying conditional volatility
- Captures volatility clustering and leverage effects
- Assumes returns are conditionally normal
- Best for: Short-term tactical adjustments

**Why use both?**
- BQR sets strategic weights
- GARCH applies dynamic penalties for elevated volatility
- Combined approach balances long-term goals with short-term risk

---

### What is the "Time Machine" feature?
**A persistence layer for expensive computations:**

**Freeze State** (`💾 Freeze State` button):
- Saves analyzer, models, forecasts, allocations to `crypto_brain_freeze.pkl`
- Preserves GARCH fitted models, correlation matrices, Monte Carlo cache
- Allows closing the app without losing 5+ minutes of CBQRA work

**Thaw State** (`📂 Thaw State` button):
- Restores full analysis state in ~2 seconds
- Skips MCMC sampling entirely (instant dashboard refresh)
- Enables running tests (Monte Carlo, Backtest, GARCH) without CBQRA reruns

**What's NOT saved?**
- Visualizations (PNGs) — privacy-first design purges these between sessions
- Temporary upload files — cleared on Nuclear Flush
- Dismissed warnings — reset between sessions

**Best practice:**
1. Run CBQRA (5 min)
2. Download Jumbo Pack (get visualizations)
3. Freeze State (save computations)
4. Next session: Thaw State → instant tests!

---

### What does "Skeleton State" mean?
**A state with no meaningful data:**
- Only default initialization values (e.g., `allocations: None`)
- No completed CBQRA analysis
- No forecasts, models, or correlation matrices

**Why detect skeletons?**
- Prevents false "state restored" messages when nothing was actually saved
- Saves user time — won't load empty states
- Provides clear guidance: "Run CBQRA to create a full analysis"

**How to fix:**
- Run CBQRA analysis in the CBQRA tab
- Freeze State AFTER analysis completes
- Thaw will only work with full states

---

### Can I use my own CSV files?
**Yes! Upload custom crypto data:**

**Requirements:**
- CSV format with `Date` and `Price` columns
- Date format: `YYYY-MM-DD` or any pandas-parseable format
- At least 100 data points (preferably 365+ for rolling calculations)
- Ticker name in filename (e.g., `BTC_historical_data.csv` → ticker: BTC)

**How to upload:**
1. Sidebar → **📂 Data Source** → Select "Upload CSV Files"
2. Click "Select CSV files" → choose multiple files
3. Unique tickers auto-detected from filenames
4. Run CBQRA as normal

**Privacy note**: Uploaded files stored in temporary directory, purged on:
- Nuclear Flush
- Clear Uploads button
- Session end (after ~24 hours idle)

---

### What's the "Nuclear Flush" button?
**Complete system reset:**
- Deletes all session state (allocations, forecasts, models)
- Clears temporary uploaded files
- Resets warnings and caches
- Purges correlation matrices and GARCH insights
- Removes `crypto_brain_freeze.pkl` (frozen states)

**When to use:**
- Dashboard behaving unexpectedly (corrupted state)
- Starting completely fresh analysis
- Switching between different portfolios
- Testing after code changes

**⚠️ Warning**: Cannot be undone! Download Jumbo Pack first if you want to keep charts.

---

## 🔬 Methodology Questions

### What is the "Kelly Criterion"?
**Optimal bet sizing formula from information theory:**

**Formula**: `f* = (p×b - q) / b`
- `f*` = fraction of capital to bet
- `p` = probability of winning
- `q` = probability of losing (1 - p)
- `b` = odds (payout ratio)

**In crypto context:**
- `p` estimated from Bayesian forecasts (expected return)
- `b` derived from Sharpe ratio and volatility
- `f*` scaled by risk profile (Conservative: 65%, Moderate: 100%, Aggressive: 135%)

**Why use Kelly?**
- Maximizes long-term logarithmic growth
- Avoids over-betting (preserves capital in drawdowns)
- Adapts to changing win probabilities

**Caveats:**
- Assumes accurate probability estimates (hard in crypto!)
- Full Kelly can be volatile — we scale down for safety
- Works best with uncorrelated bets (hence correlation penalties)

---

### Why cap speculative assets at 15%?
**Risk management prudence:**
- Meme coins (DOGE, SHIB, PEPE) have extreme volatility (100-200% annualized)
- Lack fundamental value drivers (purely sentiment-based)
- Prone to flash crashes (80%+ drops in hours)
- High correlation with social media trends (unpredictable)

**15% allocation:**
- Allows participation in upside
- Limits catastrophic portfolio impact
- Aligns with speculative investment guidelines
- Enforced via `RISK_THRESHOLDS['speculative_allocation_max']`

**How to identify:**
```python
SPECULATIVE_PATTERNS = ['DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'MEME']
```
Add your own patterns by editing `crypto_fortress.py`.

---

### What do the risk profiles actually do?
**Three pre-configured parameter sets:**

| Parameter          | Conservative | Moderate | Aggressive |
|--------------------|--------------|----------|------------|
| Max Position Size  | 15%          | 25%      | 35%        |
| Max Portfolio Vol  | 50%          | 75%      | 100%       |
| Stop Loss          | 10%          | 15%      | 20%        |
| Take Profit        | 25%          | 40%      | 60%        |
| Kelly Scale        | 65%          | 100%     | 135%       |

**How they interact:**
- **Position sizing**: Kelly Criterion × Kelly Scale × Vol Adjustment
- **Rebalancing triggers**: Drift threshold varies by profile
- **Monte Carlo seeds**: Different random paths for reproducibility
- **GARCH penalties**: More aggressive profiles tolerate higher conditional vol

**Switching profiles:**
- Requires re-running CBQRA (allocations recalculated)
- Clears Monte Carlo and Backtest caches (no longer valid)
- Dashboard shows "Profile Mismatch" warning until resolved

---

## 🐛 Troubleshooting

### "CBQRA running" stuck forever?
**Possible causes:**

1. **MCMC divergence issues**:
   - Check terminal for `Divergence warnings`
   - Reduce `mcmc_target_accept` from 0.95 to 0.90
   - Increase `mcmc_tune` iterations

2. **Infinite loop in fitting**:
   - Click **🛑 EMERGENCY STOP** in sidebar
   - Check debug logs: `crypto_dashboard_debug.log`
   - Report issue with traceback

3. **System resource exhaustion**:
   - Close heavy applications
   - Reduce `mcmc_cores` to 2
   - Try fewer assets (3 instead of 5)

**Recovery:**
1. Emergency Stop → Nuclear Flush → Restart analysis
2. If persistent: restart Streamlit server (`Ctrl+C`, `streamlit run crypto_fortress.py`)

---

### "Profile Mismatch" error?
**What it means:**
- Risk Dashboard shows **Moderate** profile
- But CBQRA analysis was run with **Conservative** profile
- Monte Carlo/Backtest results no longer valid

**How to fix:**
1. **Option A**: Switch Dashboard profile back to match analysis
   - Allocations won't change (already calculated)
   - Tests become valid again

2. **Option B**: Re-run CBQRA with current Dashboard profile
   - New allocations calculated
   - Tests cleared (must re-run)
   - Preserves consistency

**Why locked?**
- Prevents nonsensical scenarios (aggressive allocations + conservative tests)
- Ensures reproducibility
- Maintains audit trail

---

### Visualizations not appearing?
**Common causes:**

1. **Pre-session cleanup purged them**:
   - Privacy feature: old charts deleted on new session start
   - Solution: Download Jumbo Pack before ending session

2. **Analysis not completed**:
   - Check "✅ Analysis completed successfully!" message
   - Verify `crypto_analysis_results/` folder exists
   - Re-run CBQRA if needed

3. **File permission issues**:
   - Check write permissions on `crypto_analysis_results/`
   - On Windows: run as Administrator
   - On Linux: `chmod 755 crypto_analysis_results/`

4. **Corrupted output directory**:
   - Nuclear Flush → re-run CBQRA
   - Manually delete `crypto_analysis_results/` folder

---

### "No saved analysis found" when trying to Thaw?
**Possible reasons:**

1. **Never ran Freeze State**:
   - Must click `💾 Freeze State` after CBQRA completes
   - File `crypto_brain_freeze.pkl` won't exist otherwise

2. **Skeleton state saved**:
   - Freeze was clicked before CBQRA finished
   - Thaw detects empty state and rejects it
   - Solution: Run CBQRA → Freeze again

3. **File deleted**:
   - Nuclear Flush removes `crypto_brain_freeze.pkl`
   - Manual deletion by user
   - Check if file exists in project directory

4. **Corrupted pickle file**:
   - Rare: incomplete write due to crash
   - Solution: Delete `crypto_brain_freeze.pkl` → re-run CBQRA → Freeze

---

## 📊 Interpretation Questions

### What's a "good" Sharpe ratio?
**General benchmarks:**
- < 0.5: Poor risk-adjusted returns
- 0.5 - 1.0: Acceptable
- 1.0 - 2.0: Good
- 2.0 - 3.0: Very good
- > 3.0: Excellent (rare in crypto!)

**Crypto context:**
- Sharpe ratios often lower due to high volatility
- Compare against HODL Bitcoin (typically 0.5-1.5)
- Focus on **relative improvement** not absolute values

---

### What does "leverage effect" mean?
**Asymmetric volatility response:**
- **Bad news** (negative returns) increases future volatility MORE than
- **Good news** (positive returns) of equal magnitude

**Example:**
- -5% drop → volatility rises 10%
- +5% gain → volatility rises 5%

**Why it matters:**
- Downside risk is amplified (crashes are more violent)
- GARCH models capture this via `γ` (gamma) parameter
- Positive γ = confirmed leverage effect
- Informs stop-loss placement (need tighter stops)

**"Elton John Leverage Blues":**
> *"The blues are always there... waiting for the next drop."* 🎵

---

### How do I interpret Monte Carlo results?
**Key metrics explained:**

- **Expected Return**: Mean outcome across all simulations (central tendency)
- **Best/Worst Case**: Extreme percentiles (5th and 95th)
- **VaR 95%**: Value at Risk — worst loss at 95% confidence (1 in 20 scenarios)
- **Probability Positive**: Chance of making any profit
- **Beat SP500 Chance**: Probability of outperforming S&P 500 (8-10% annual)

**How to use:**
- **Conservative**: Focus on VaR 95% — ensure bearable worst case
- **Moderate**: Balance expected return with probability positive
- **Aggressive**: Optimize for best case, accept high worst case

**Stress test scenarios:**
- Show performance in historical crash conditions
- Use to set appropriate cash reserves (20-30%)
- Inform stop-loss tightening during elevated volatility

---

## 🚨 Important Disclaimers

### Is this financial advice?
**NO. Absolutely not.**

- This is **educational software** for learning quantitative risk management
- Outputs are **hypothetical scenarios** based on historical data
- Past performance **does not guarantee** future results
- **You are solely responsible** for your investment decisions
- **Consult qualified financial advisors** before risking real capital

---

### Can I use this for actual trading?
**You CAN, but SHOULD you?**

**Appropriate use:**
- **Paper trading** (simulated with fake money) — excellent learning
- **Backtesting** strategies before live deployment — strongly recommended
- **Risk management** for existing portfolios — helpful supplemental tool

**Risky use:**
- **Blindly following** allocations without understanding — dangerous
- **Over-leveraging** based on aggressive profiles — capital destruction
- **Ignoring** external factors (regulations, hacks, project failures) — incomplete analysis

**Best practice:**
1. Paper trade for 3-6 months
2. Validate assumptions with real market behavior
3. Start with <5% of portfolio
4. Scale up gradually as confidence builds
5. **NEVER** invest more than you can afford to lose

---

### What are the limitations?
**Known constraints:**

1. **Historical data dependency**: Models trained on past — may not predict future regime shifts
2. **Assumption of stationarity**: Crypto fundamentals change (regulations, adoption, technology)
3. **Correlation breakdown**: Assets assumed to maintain correlation patterns (often fail in crises)
4. **No fundamental analysis**: Ignores project quality, team competence, token economics
5. **Model risk**: MCMC convergence issues, GARCH misspecification, Kelly over-betting
6. **Black swan events**: Cannot predict completely unprecedented scenarios

**Not a crystal ball** — a risk management framework requiring human judgment.

---

## 📜 License & Legal

### Can I modify the code?
**Yes! MIT License:**
- Free to use, modify, distribute
- Commercial use allowed
- Attribution appreciated but not required
- No warranty — use at own risk

**Contribution guidelines:**
- Open issues for bugs/features
- Submit pull requests for improvements
- Follow existing code style
- Add tests for new features

---

### Who's responsible if I lose money?
**You are.**

- Software provided "AS IS" without warranty
- Authors assume **no liability** for financial losses
- You accept all risks by using this software
- **Read full disclaimer** in README before use

**Repeat after me:**
> *"This is educational software. I am responsible for my own investment decisions. I will not blame the developers if things go wrong."*

---

## 🎵 Closing Thoughts

> *"And I think it's gonna be a long, long time,*
> *Until bad coding ever fails on me,*
> *I'm not the dev they think I am at home,*
> *Oh no no no!*
> *I'm a ROCKET MAAAAAN!*
> *Building all the systems that can't be beat!"*

**In Lak'ech... 🚀**

---

**Have more questions?** Open a [GitHub Issue](https://github.com/JCP9415/basel-iii-crypto-risk-management/issues) or contribute to this FAQ!
