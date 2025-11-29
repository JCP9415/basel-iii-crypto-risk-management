#!/usr/bin/env python3
"""
FIXED GARCH ENGINE - Operation Neurosurgeon Extraction
Mission: Actually work with your Streamlit app
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import os

# Graceful import with detailed fallbacks
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
    print("✅ arch package available - GARCH engine ready")
except ImportError as e:
    ARCH_AVAILABLE = False
    print(f"❌ arch package not available: {e}")
    print("💡 Install with: pip install arch")

warnings.filterwarnings("ignore")

class LenovoFriendlyGARCH:
    """
    GARCH that actually works with your Streamlit app
    """

    def __init__(self, output_dir="garch_results"):
        self.fitted_models = {}
        self.garch_insights = {}
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"🔧 LenovoFriendlyGARCH initialized (ARCH_AVAILABLE: {ARCH_AVAILABLE})")

    def fit_gjr_garch(self, returns_series, crypto_name, max_iter=50):
        """Fixed GJR-GARCH implementation"""
        print(f"GARCH.fit_gjr_garch called for {crypto_name}")

        if not ARCH_AVAILABLE:
            print(f"❌ GARCH not available for {crypto_name}")
            return None

        if len(returns_series) < 100:
            print(f"❌ {crypto_name}: Insufficient data ({len(returns_series)} points), need >=100")
            return None

        try:
            # Clean data properly
            returns_clean = returns_series.replace([np.inf, -np.inf], np.nan).dropna()
            if len(returns_clean) < 100:
                print(f"❌ {crypto_name}: Insufficient clean data ({len(returns_clean)} points)")
                return None

            # Convert to percentage for GARCH stability
            returns_pct = returns_clean * 100
            print(f"📊 {crypto_name}: {len(returns_pct)} clean returns, std: {returns_pct.std():.4f}%")

            if returns_pct.std() < 1e-8:
                print(f"❌ {crypto_name}: Near-zero volatility - cannot fit GARCH")
                return None

            # Fit GJR-GARCH model
            print(f"🔄 Attempting GJR-GARCH fit for {crypto_name}...")
            model = arch_model(
                returns_pct, p=1, q=1, o=1, dist='skewt', vol='GARCH'
            )

            fitted = model.fit(
                update_freq=0,
                disp='off',
                show_warning=False,
                options={'maxiter': max_iter}
            )

            print(f"✅ Fit completed for {crypto_name}")

            # Extract insights
            forecast = fitted.forecast(horizon=1)
            conditional_vol_decimal = np.sqrt(forecast.variance.iloc[-1, 0]) / 100.0

            # Get leverage effect parameter
            leverage_effect = float(fitted.params.get('gamma[1]', 0.0))

            # Calculate VaR properly
            var_95 = self._calculate_var_95(fitted, returns_clean)

            # Diagnostics
            diagnostics = {
                'converged': bool(fitted.convergence_flag),
                'log_likelihood': float(fitted.loglikelihood),
                'aic': float(fitted.aic),
                'persistence': float(self._calculate_persistence(fitted)),
                'mean_volatility': float(returns_clean.std() * np.sqrt(365))
            }

            insights = {
                'conditional_vol': conditional_vol_decimal,
                'leverage_effect': leverage_effect,
                'var_95': var_95,
                'diagnostics': diagnostics,
                'fitted_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'data_points': len(returns_clean)
            }

            self.garch_insights[crypto_name] = insights
            self.fitted_models[crypto_name] = fitted

            print(f"✅ {crypto_name}: γ={leverage_effect:.4f}, σ={conditional_vol_decimal:.4f}")
            return insights

        except Exception as e:
            print(f"❌ GARCH fitting FAILED for {crypto_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_var_95(self, fitted_model, returns_clean):
        """Fixed VaR calculation"""
        try:
            # Use standardized residuals
            residuals = fitted_model.resid / fitted_model.conditional_volatility
            # Get conditional vol in percentage space, then convert to decimal
            cond_vol_today = fitted_model.conditional_volatility.iloc[-1] / 100.0
            var_95 = np.percentile(residuals, 5) * cond_vol_today
            return float(abs(var_95))
        except Exception as e:
            print(f"⚠️ VaR calculation failed: {e}")
            return float(returns_clean.std() * 1.645)

    def _calculate_persistence(self, fitted_model):
        """Calculate volatility persistence with validation"""
        try:
            alpha = max(0.0, float(fitted_model.params.get('alpha[1]', 0.0)))
            beta = max(0.0, float(fitted_model.params.get('beta[1]', 0.0)))
            gamma = float(fitted_model.params.get('gamma[1]', 0.0))

            # GJR-GARCH persistence formula: α + β + γ/2
            persistence = alpha + beta + (gamma / 2.0)

            # Validate persistence (should be < 1 for stationarity)
            if persistence >= 1.0:
                print(f"⚠️ Non-stationary persistence: {persistence:.3f}")
                return 0.99
            elif persistence <= 0:
                return 0.5

            return persistence

        except Exception as e:
            print(f"⚠️ Persistence calculation failed: {e}")
            return 0.5

    def volatility_penalized_kelly(self, base_kelly, garch_vol_forecast, historical_vol):
        """
        Apply volatility penalty to Kelly criterion
        """
        if historical_vol <= 0:
            return base_kelly

        vol_ratio = garch_vol_forecast / historical_vol
        penalty = 1.0 / (1.0 + max(0, vol_ratio - 1.0))
        penalized_kelly = base_kelly * penalty

        if vol_ratio > 1.2:
            reduction_pct = (1 - penalty) * 100
            print(f"   ⚠️ Volatility penalty: {reduction_pct:.1f}% reduction (ratio: {vol_ratio:.2f})")

        return penalized_kelly

    def fit_all_assets(self, data_dict, crypto_names):
        """
        Batch fit GARCH models for all assets - FIXED VERSION
        """
        print(f"🔍 GARCH.fit_all_assets called with {len(crypto_names)} cryptos")
        print(f"🔍 Data dict keys: {list(data_dict.keys())}")

        if not ARCH_AVAILABLE:
            print("❌ GARCH not available - arch package missing")
            return {"success": 0, "total": len(crypto_names), "message": "GARCH not available"}

        successful_fits = 0
        results_summary = []

        print(f"\n🌪️ FITTING GJR-GARCH FOR {len(crypto_names)} ASSETS")
        print("=" * 50)

        for crypto in crypto_names:
            if crypto not in data_dict:
                print(f"❌ {crypto}: Not in data_dict")
                continue

            df = data_dict[crypto]
            print(f"📊 {crypto}: DataFrame shape {df.shape}")

            if 'Price' not in df.columns:
                print(f"❌ {crypto}: No 'Price' column! Available: {list(df.columns)}")
                continue

            # Calculate returns
            prices = df['Price']
            returns = prices.pct_change().dropna()

            if len(returns) < 100:
                print(f"❌ {crypto}: Only {len(returns)} returns, need >=100")
                continue

            print(f"📈 {crypto}: {len(returns)} returns, range: {returns.min():.4f} to {returns.max():.4f}")

            # Fit GARCH model
            insights = self.fit_gjr_garch(returns, crypto)

            if insights:
                successful_fits += 1
                results_summary.append({
                    'crypto': crypto,
                    'leverage_effect': insights['leverage_effect'],
                    'conditional_vol': insights['conditional_vol'],
                    'data_points': insights['data_points']
                })

        # Print summary
        print(f"\n📊 GARCH FITTING SUMMARY")
        print("=" * 50)
        for result in results_summary:
            lev_status = "✅" if result['leverage_effect'] > 0 else "⚠️"
            print(f"{lev_status} {result['crypto']:6} | γ={result['leverage_effect']:7.4f} | "
                  f"σ={result['conditional_vol']:7.4f} | {result['data_points']:4} points")

        print(f"\n🎯 SUCCESS: {successful_fits}/{len(crypto_names)} assets fitted")

        return {
            "success": successful_fits,
            "total": len(crypto_names),
            "results": results_summary,
            "garch_insights": self.garch_insights
        }

    def get_leverage_effect_summary(self):
        """Return leverage effects for display in dashboard"""
        leverage_data = []
        for crypto, insights in self.garch_insights.items():
            leverage_data.append({
                'Crypto': crypto,
                'Leverage_Effect': insights['leverage_effect'],
                'Conditional_Vol': insights['conditional_vol'],
                'VaR_95': insights['var_95']
            })

        return pd.DataFrame(leverage_data)

# Singleton instance for easy import
garch_engine = LenovoFriendlyGARCH()

def fit_garch_models(data_dict, crypto_names):
    """
    Convenience function for your Streamlit app
    """
    return garch_engine.fit_all_assets(data_dict, crypto_names)
