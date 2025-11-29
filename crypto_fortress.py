#!/usr/bin/env python3
"""
CRYPTO RISK MANAGEMENT & CBQRA DASHBOARD - OPERATION FORTRESS v4.6
Mission: Eliminate guerrilla bugs with defensive programming + fail-safes + error recovery
New: Comprehensive error boundaries, state validation, graceful degradation
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import tempfile
import shutil
import datetime
import hashlib
import joblib
import arviz
from pathlib import Path as PPath
from itertools import combinations
from contextlib import contextmanager
from io import BytesIO
import traceback
import logging
import warnings
import zipfile

# Nuclear option – silence ALL Streamlit warnings forever
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")

def pre_session_cleanup():
    """
    🎯 SMARTER CLEANUP: Only cleans truly stale files, preserves active session work
    """
    try:
        # Check if we have active analysis that should be preserved
        has_active_analysis = (
            st.session_state.get('portfolio_state', {}).get('cbqra_completed', False) or
            st.session_state.get('analyzer') is not None or
            st.session_state.get('forecasts') is not None
        )

        # Check if cleanup already done for this session
        if 'cleanup_done' in st.session_state:
            if has_active_analysis:
                logger.debug("✅ Cleanup skipped - active analysis present")
            else:
                logger.debug("✅ Cleanup already done for this session")
            return

        output_dir = BASE_CONFIG['output_dir']
        if not os.path.exists(output_dir):
            logger.debug("✅ No output directory - nothing to clean")
            st.session_state['cleanup_done'] = True
            return

        # If we have active analysis, DO NOT CLEAN visualization files
        if has_active_analysis:
            logger.info("🔒 Active analysis detected - preserving visualizations")
            # Only clean temporary or clearly stale files
            zip_files = [f for f in os.listdir(output_dir) if f.endswith('.zip')]
            for zip_file in zip_files:
                try:
                    os.remove(os.path.join(output_dir, zip_file))
                    logger.info(f"🧹 Cleaned residual ZIP: {zip_file}")
                except:
                    pass
            st.session_state['cleanup_done'] = True
            return

        # Only clean PNG files if no active analysis exists
        png_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
        if png_files:
            logger.info(f"🧹 New session detected - cleaning {len(png_files)} old visualizations")
            for file in png_files:
                file_path = os.path.join(output_dir, file)
                try:
                    os.remove(file_path)
                    logger.info(f"🧹 Cleaned residual visualization: {file}")
                except Exception as e:
                    logger.warning(f"Could not remove {file}: {e}")

        # Clean old pairwise comparisons (only if no active analysis)
        old_pairwise = [f for f in os.listdir(output_dir)
                       if f.startswith('pairwise_') and f.endswith('.png')]
        for f in old_pairwise:
            try:
                os.remove(os.path.join(output_dir, f))
                logger.info(f"🧹 Cleaned pairwise: {f}")
            except Exception as e:
                logger.warning(f"Could not remove pairwise {f}: {e}")

        # Mark cleanup as done for this session
        st.session_state['cleanup_done'] = True
        logger.info("✅ Pre-session cleanup completed")

    except Exception as e:
        logger.error(f"Pre-session cleanup failed: {e}")
        # Even if cleanup fails, mark it as done to prevent repeated attempts
        st.session_state['cleanup_done'] = True
# === ENHANCED LOGGING SETUP (FIXED REPETITION) ===
def setup_logging():
    """Setup logging with proper handlers to prevent repetition"""
    logger = logging.getLogger(__name__)

    # Clear any existing handlers to prevent duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(logging.INFO)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    # Create formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(funcName)s: %(message)s')

    # File handler
    file_handler = logging.FileHandler('crypto_dashboard_debug.log', mode='w')  # 'w' overwrites each session
    file_handler.setFormatter(formatter)

    # Stream handler (console)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

# Replace your existing logging setup
logger = setup_logging()

def cleanup_previous_logs():
    """Ensure clean log file at startup"""
    try:
        log_file = 'crypto_dashboard_debug.log'
        if os.path.exists(log_file):
            # Truncate the log file instead of deleting it to avoid permission issues
            with open(log_file, 'w') as f:
                f.write("")
            logger.info("🧹 Previous log file cleared")
    except Exception as e:
        print(f"Log cleanup warning: {e}")

# Call this right after setup_logging()
cleanup_previous_logs()

def is_skeleton_state(state_dict):
    """
    🎯 Distinguish between skeleton profiles and fully analyzed states
    """
    try:
        if not state_dict:
            return True

        # Check if this is just a skeleton (default initialization)
        portfolio_state = state_dict.get('portfolio_state', {})

        # A skeleton state has no real analysis data
        if (portfolio_state.get('allocations') is None and
            portfolio_state.get('cbqra_completed') is False and
            portfolio_state.get('last_updated') is None and
            state_dict.get('analyzer') is None and
            state_dict.get('forecasts') is None):
            return True

        # Check if we have meaningful data
        if (portfolio_state.get('cbqra_completed') is True or
            state_dict.get('analyzer') is not None or
            state_dict.get('forecasts') is not None):
            return False

        return True
    except Exception as e:
        logger.error(f"Error checking skeleton state: {e}")
        return True

# Enhanced thaw state functionality
def enhanced_thaw_state():
    """
    🧠 Smarter state loading that distinguishes between skeletons and full states
    """
    if not os.path.exists('crypto_brain_freeze.pkl'):
        logger.info("❌ No saved state found - skipping thaw")
        st.warning("⚠️ No saved analysis found. Run CBQRA first to create a save point.")
        return False

    try:
        brain_packet = joblib.load('crypto_brain_freeze.pkl')

        # Check if this is just a skeleton state
        if is_skeleton_state(brain_packet):
            logger.info("🩻 Skipping skeleton state - no meaningful data")
            st.info("💡 Saved state contains only initialization data. Run CBQRA to create a full analysis.")
            return False

        logger.info("✅ Loading full analysis state from frozen brain")

        # Restore the full state (your existing restoration code)
        keys_to_restore = [
            'analyzer', 'correlation_matrix', 'forecasts', 'uploaded_file_names',
            'monte_carlo_cache', 'last_allocations', 'garch_insights',
            'garch_fitted_models', 'volatility_engine'
        ]

        for key in keys_to_restore:
            if brain_packet.get(key) is not None:
                st.session_state[key] = brain_packet.get(key)

        # Special handling for GARCH models
        if brain_packet.get('garch_fitted_models') and GARCH_AVAILABLE:
            try:
                garch_engine.fitted_models = brain_packet['garch_fitted_models'].copy()
                st.info(f"✅ Restored {len(garch_engine.fitted_models)} GARCH models to engine")
            except Exception as e:
                st.warning(f"⚠️ GARCH model restoration partial: {e}")

        # Careful merge for portfolio_state
        saved_pstate = brain_packet.get('portfolio_state', {})
        if saved_pstate:
            safe_keys = ['allocations', 'last_updated', 'cbqra_completed',
                        'risk_tolerance', 'locked_profile', 'cbqra_running']
            for k in safe_keys:
                if k in saved_pstate:
                    st.session_state['portfolio_state'][k] = saved_pstate[k]

        st.success("✅ Brain restored! Full analysis state loaded.")
        logger.info("✅ Full brain state loaded successfully")
        return True

    except Exception as e:
        logger.error(f"Brain load failed: {e}")
        st.error(f"❌ Load failed: {e}")
        return False

# === DEFENSIVE ERROR BOUNDARIES ===
def safe_execute(func, default=None, error_msg="Operation failed"):
    """
    Universal error boundary - catches exceptions and returns safe defaults
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{error_msg}: {str(e)}\n{traceback.format_exc()}")
            return default
    return wrapper

def validate_session_state():
    """
    🎯 SNIPER UNIT: Validates critical session state integrity
    Fixes corruption before it cascades
    """
    try:
        # Check portfolio_state structure
        if 'portfolio_state' not in st.session_state:
            logger.warning("portfolio_state missing - rebuilding")
            st.session_state['portfolio_state'] = {
                'allocations': None,
                'last_updated': None,
                'cbqra_completed': False,
                'risk_tolerance': 'moderate',
                'cbqra_running': False,
                'locked_profile': None
            }

        pstate = st.session_state['portfolio_state']

        # Validate types
        if not isinstance(pstate, dict):
            logger.error(f"portfolio_state corrupted - type: {type(pstate)}")
            raise ValueError("portfolio_state is not a dict")

        # Ensure all required keys exist
        required_keys = ['allocations', 'last_updated', 'cbqra_completed',
                        'risk_tolerance', 'cbqra_running', 'locked_profile']
        for key in required_keys:
            if key not in pstate:
                logger.warning(f"Missing key '{key}' in portfolio_state - adding default")
                defaults = {
                    'allocations': None,
                    'last_updated': None,
                    'cbqra_completed': False,
                    'risk_tolerance': 'moderate',
                    'cbqra_running': False,
                    'locked_profile': None
                }
                pstate[key] = defaults.get(key)

        # Validate allocations structure if present
        if pstate['allocations'] is not None:
            if not isinstance(pstate['allocations'], dict):
                logger.error(f"allocations corrupted - type: {type(pstate['allocations'])}")
                pstate['allocations'] = None
            else:
                # Validate allocation values
                for crypto, alloc in list(pstate['allocations'].items()):
                    if not isinstance(alloc, (int, float)) or alloc < 0 or alloc > 1:
                        logger.error(f"Invalid allocation for {crypto}: {alloc}")
                        del pstate['allocations'][crypto]

        # Validate boolean flags
        for bool_key in ['cbqra_completed', 'cbqra_running']:
            if not isinstance(pstate[bool_key], bool):
                logger.warning(f"{bool_key} corrupted - resetting to False")
                pstate[bool_key] = False

        # Validate risk_tolerance
        valid_profiles = ['conservative', 'moderate', 'aggressive']
        if pstate['risk_tolerance'] not in valid_profiles:
            logger.warning(f"Invalid risk_tolerance: {pstate['risk_tolerance']} - defaulting to moderate")
            pstate['risk_tolerance'] = 'moderate'

        return True

    except Exception as e:
        logger.critical(f"Session state validation failed: {e}")
        # Nuclear option - rebuild from scratch
        st.session_state['portfolio_state'] = {
            'allocations': None,
            'last_updated': None,
            'cbqra_completed': False,
            'risk_tolerance': 'moderate',
            'cbqra_running': False,
            'locked_profile': None
        }
        return False

def safe_dict_get(d, key, default=None, expected_type=None):
    """
    🎯 SNIPER: Safe dictionary access with type validation
    """
    try:
        value = d.get(key, default)
        if expected_type and value is not None:
            if not isinstance(value, expected_type):
                logger.warning(f"Type mismatch for key '{key}': expected {expected_type}, got {type(value)}")
                return default
        return value
    except Exception as e:
        logger.error(f"Error accessing dict key '{key}': {e}")
        return default

def safe_dataframe_access(df, key, default=None):
    """
    🎯 SNIPER: Safe DataFrame column/index access
    """
    try:
        if df is None:
            return default
        if not isinstance(df, pd.DataFrame):
            logger.error(f"Expected DataFrame, got {type(df)}")
            return default
        if key in df.columns:
            return df[key]
        elif key in df.index:
            return df.loc[key]
        else:
            logger.warning(f"Key '{key}' not found in DataFrame")
            return default
    except Exception as e:
        logger.error(f"DataFrame access error for key '{key}': {e}")
        return default

def safe_file_operation(filepath, operation='read', data=None, default=None):
    """
    🎯 SNIPER: All file operations with error recovery
    """
    try:
        if operation == 'read':
            if not os.path.exists(filepath):
                logger.warning(f"File not found: {filepath}")
                return default
            with open(filepath, 'rb') as f:
                return f.read()

        elif operation == 'write':
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                f.write(data)
            return True

        elif operation == 'delete':
            if os.path.exists(filepath):
                os.remove(filepath)
            return True

        elif operation == 'exists':
            return os.path.exists(filepath)

    except Exception as e:
        logger.error(f"File operation '{operation}' failed for {filepath}: {e}")
        return default

def display_error_boundary(error, context=""):
    """
    User-friendly error display with logging
    """
    logger.error(f"Error in {context}: {error}\n{traceback.format_exc()}")
    st.error(f"❌ An error occurred: {str(error)}")
    with st.expander("🔍 Technical Details (for debugging)"):
        st.code(traceback.format_exc())
        st.info("This error has been logged. Check crypto_dashboard_debug.log for details.")

# === GARCH VISUALIZATION FUNCTIONS ===
def quick_garch_vol_plot(crypto, fitted_model, data_dict):
    """HARDENED version"""
    try:
        if not isinstance(crypto, str) or not crypto:
            raise ValueError(f"Invalid crypto: {crypto}")
        if fitted_model is None:
            raise ValueError("fitted_model is None")
        if not isinstance(data_dict, dict) or crypto not in data_dict:
            raise ValueError(f"{crypto} not in data_dict")

        fig, ax = plt.subplots(figsize=(14, 6))
        vol = fitted_model.conditional_volatility.iloc[-500:] * np.sqrt(365) * 100
        dates = data_dict[crypto]['Date'].iloc[-500:]
        ax.plot(dates, vol, color='#cc0000', linewidth=2.5, label='GARCH Conditional Vol')
        ax.set_title(f"{crypto} – Same, same until it ISN'T", fontsize=16)
        ax.set_ylabel("Annualized Volatility (%)")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"quick_garch_vol_plot failed for {crypto}: {e}")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.text(0.5, 0.5, f"Error generating plot for {crypto}",
                ha='center', va='center', fontsize=14)
        return fig

def plot_garch_forecast(crypto, fitted_model, data_dict, forecast_days=30):
    """HARDENED: Plot GARCH volatility forecast"""
    try:
        if not isinstance(crypto, str) or not crypto:
            raise ValueError(f"Invalid crypto: {crypto}")
        if fitted_model is None:
            raise ValueError("fitted_model is None")
        if not isinstance(data_dict, dict) or crypto not in data_dict:
            raise ValueError(f"{crypto} not in data_dict")

        fig, ax = plt.subplots(figsize=(14, 7))

        # Historical volatility
        hist_vol = fitted_model.conditional_volatility.iloc[-500:] * np.sqrt(365) * 100
        dates = data_dict[crypto]['Date'].iloc[-500:]

        ax.plot(dates, hist_vol, color='#0066cc', linewidth=2,
                label='Historical Conditional Vol', alpha=0.7)

        # Forecast
        forecast = fitted_model.forecast(horizon=forecast_days)
        forecast_vol = np.sqrt(forecast.variance.values[-1, :]) * np.sqrt(365) * 100

        forecast_dates = pd.date_range(dates.iloc[-1], periods=forecast_days+1, freq='D')[1:]
        ax.plot(forecast_dates, forecast_vol, color='#cc0000', linewidth=2.5,
                label='GARCH Forecast', linestyle='--')

        # Confidence bands
        ax.fill_between(forecast_dates, forecast_vol * 0.8, forecast_vol * 1.2,
                         alpha=0.2, color='red')

        ax.set_title(f"{crypto} – GARCH Volatility Forecast", fontsize=16, pad=20)
        ax.set_ylabel("Annualized Volatility (%)", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        return fig
    except Exception as e:
        logger.error(f"plot_garch_forecast failed for {crypto}: {e}")
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.text(0.5, 0.5, f"Error generating forecast for {crypto}",
                ha='center', va='center', fontsize=14)
        return fig

# === GRACEFUL IMPORTS WITH FALLBACKS (continued) ===
try:
    from multi_crypto_bqr import MultiCryptoBQRAnalysis
    BQR_AVAILABLE = True
except ImportError:
    MultiCryptoBQRAnalysis = None
    BQR_AVAILABLE = False
    logger.warning("BQR module not available")

try:
    from garch_engine import garch_engine, fit_garch_models
    GARCH_AVAILABLE = True
    logger.info("✅ GARCH Engine: Available")
except ImportError:
    GARCH_AVAILABLE = False
    logger.warning("❌ GARCH Engine: Not available")

try:
    from advanced_visualizations import AdvancedCryptoVisualizations
    VIZ_AVAILABLE = True
except ImportError:
    AdvancedCryptoVisualizations = None
    VIZ_AVAILABLE = False
    logger.warning("Advanced visualizations not available")

try:
    from monte_carlo_simulator import CryptoMonteCarlo
    MONTE_CARLO_AVAILABLE = True
except ImportError:
    CryptoMonteCarlo = None
    MONTE_CARLO_AVAILABLE = False
    logger.warning("Monte Carlo module not available")

try:
    from glossary import GLOSSARY
    if not isinstance(GLOSSARY, dict) or len(GLOSSARY) == 0:
        raise ImportError("Empty glossary")
except ImportError:
    GLOSSARY = {
        "Sharpe Ratio": "Risk-adjusted return measure (higher is better)",
        "Volatility": "Standard deviation of returns (measure of risk)",
        "Max Drawdown": "Maximum peak-to-trough decline",
        "Value at Risk (VaR)": "Worst-case loss at a given confidence level",
        "Bayesian Quantile Regression (BQR)": "Advanced statistical method for forecasting",
        "Monte Carlo Simulation": "Random sampling technique for forecasting",
        "Kelly Criterion": "Optimal bet sizing formula"
    }
    logger.info("Using default glossary")

try:
    from thefuzz import fuzz, process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    logger.warning("Fuzzy search not available")

# === ENHANCED GARCH DIAGNOSTICS ===
if GARCH_AVAILABLE:
    try:
        from garch_engine import garch_engine, fit_garch_models
        logger.info(f"✅ garch_engine type: {type(garch_engine)}")
        logger.info(f"✅ fit_garch_models type: {type(fit_garch_models)}")
    except ImportError as e:
        logger.error(f"❌ garch_engine import failed: {e}")
        GARCH_AVAILABLE = False

# === STARTUP DIAGNOSTICS ===
print("\n" + "="*60)
print("🔧 CRYPTO DASHBOARD - OPERATION FORTRESS v4.6")
print("="*60)
print(f"✅ BQR Analysis: {BQR_AVAILABLE}")
print(f"✅ Visualizations: {VIZ_AVAILABLE}")
print(f"✅ Monte Carlo: {MONTE_CARLO_AVAILABLE}")
print(f"✅ Glossary: {len(GLOSSARY)} terms")
print(f"✅ Fuzzy Search: {FUZZY_AVAILABLE}")
print(f"✅ GARCH Engine: {GARCH_AVAILABLE}")
print("="*60 + "\n")

# === CONFIGURATION ===
PROFILE_SEEDS = {'conservative': 42, 'moderate': 123, 'aggressive': 789}

RISK_THRESHOLDS = {
    'max_correlation': 0.90,
    'volatility_multiplier': 2.5,
    'speculative_allocation_max': 0.15,
    'daily_loss_limit': 0.40,
    'flash_crash_assets': 3,
    'flash_crash_drop': 0.15,
    'rebalance_drift': 0.05
}

SPECULATIVE_PATTERNS = ['DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'MEME']

BASE_CONFIG = {
    'quantiles': [0.05, 0.5, 0.95],
    'forecast_days': 30,
    'rolling_window': 30,
    'output_dir': 'crypto_analysis_results',
    'dpi': 150,
    'mcmc_samples': 1000,
    'mcmc_tune': 1000,
    'mcmc_target_accept': 0.95,
    'mcmc_cores': 4
}

# === CALL PRE-SESSION CLEANUP HERE (AFTER BASE_CONFIG IS DEFINED) ===
pre_session_cleanup()

# === DATA STRUCTURES ===
crypto_data = {
    "XRP": {"volatility": 0.89, "correlation": 0.644, "sharpe": 1.2, "max_drawdown": 45, "beta": 1.1, "expected_return": 0.0},
    # ... rest of your crypto_data ...
}
risk_profiles = {
    "conservative": {"max_position": 0.15, "max_portfolio_vol": 0.50, "stop_loss": 0.10, "take_profit": 0.25, "kelly_scale": 0.65},
    "moderate": {"max_position": 0.25, "max_portfolio_vol": 0.75, "stop_loss": 0.15, "take_profit": 0.40, "kelly_scale": 1.00},
    "aggressive": {"max_position": 0.35, "max_portfolio_vol": 1.00, "stop_loss": 0.20, "take_profit": 0.60, "kelly_scale": 1.35}
}

# === HELPER FUNCTIONS ===
def get_file_hash(file):
    """
    HARDENED: Get MD5 hash with error recovery
    """
    try:
        file.seek(0)
        file_hash = hashlib.md5(file.read()).hexdigest()
        file.seek(0)
        return file_hash
    except Exception as e:
        logger.error(f"File hash calculation failed: {e}")
        return f"error_{time.time()}"

def is_analysis_running():
    """
    HARDENED: Safe check with validation
    """
    try:
        validate_session_state()
        return st.session_state.get('portfolio_state', {}).get('cbqra_running', False)
    except Exception as e:
        logger.error(f"is_analysis_running check failed: {e}")
        return False

def is_speculative_asset(crypto_name):
    """
    HARDENED: Safe speculative check
    """
    try:
        if not isinstance(crypto_name, str):
            return False
        return any(pattern in crypto_name.upper() for pattern in SPECULATIVE_PATTERNS)
    except Exception as e:
        logger.error(f"Speculative check failed for {crypto_name}: {e}")
        return False

def fuzzy_search_glossary(search_term, glossary_dict, min_score=70, max_results=10):
    """
    HARDENED: Robust fuzzy search with fallback
    """
    try:
        if not search_term or len(search_term.strip()) < 2:
            return []

        if FUZZY_AVAILABLE:
            matches = process.extract(search_term, list(glossary_dict.keys()),
                                    limit=max_results, scorer=fuzz.partial_ratio)
            return [(m[0], m[1], glossary_dict[m[0]]) for m in matches if m[1] >= min_score]
        else:
            results = []
            for term, definition in glossary_dict.items():
                if search_term.lower() in term.lower():
                    results.append((term, 100, definition))
            return results[:max_results]
    except Exception as e:
        logger.error(f"Glossary search error: {e}")
        return []

def calculate_position_size(crypto, risk_tolerance):
    """
    HARDENED: Calculate position size with comprehensive validation
    """
    try:
        # Validate inputs
        if not isinstance(crypto, str) or not crypto:
            logger.error(f"Invalid crypto name: {crypto}")
            return 0.0

        valid_profiles = ['conservative', 'moderate', 'aggressive']
        if risk_tolerance not in valid_profiles:
            logger.warning(f"Invalid risk_tolerance: {risk_tolerance} - using moderate")
            risk_tolerance = 'moderate'

        # Initialize crypto_data if missing
        if crypto not in crypto_data:
            logger.info(f"Adding default data for {crypto}")
            crypto_data[crypto] = {
                "volatility": 1.0,
                "correlation": 0.5,
                "sharpe": 0.5,
                "max_drawdown": 50,
                "beta": 1.0,
                "expected_return": 0.0,
                "speculative": is_speculative_asset(crypto)
            }

        data = crypto_data[crypto]
        profile = risk_profiles[risk_tolerance]

        # Safe Kelly calculation
        expected_return = safe_dict_get(data, "expected_return", 0.0, (int, float))
        volatility = safe_dict_get(data, "volatility", 1.0, (int, float))

        if volatility <= 0:
            logger.warning(f"Invalid volatility for {crypto}: {volatility}")
            volatility = 1.0

        kelly_fraction_raw = 0.25  # Default
        if expected_return > 0:
            kelly_fraction_raw = max(0, min(1, expected_return / (volatility ** 2)))

        kelly_scale = safe_dict_get(profile, "kelly_scale", 1.0, (int, float))
        kelly_fraction_scaled = kelly_fraction_raw * kelly_scale

        # Safe adjustments
        correlation = safe_dict_get(data, "correlation", 0.5, (int, float))
        corr_adjustment = max(0.05, 1 - correlation)
        vol_adjustment = (1.0 / max(0.01, volatility)) * corr_adjustment

        max_position = safe_dict_get(profile, "max_position", 0.25, (int, float))
        base_allocation = max_position * vol_adjustment * kelly_fraction_scaled

        # Speculative cap
        if safe_dict_get(data, "speculative", False, bool):
            base_allocation = min(base_allocation, RISK_THRESHOLDS['speculative_allocation_max'])

        final_allocation = min(base_allocation, max_position)

        # GARCH penalty with validation
        if (st.session_state.get('volatility_engine') == 'GARCH' and
            st.session_state.get('garch_insights') and
            isinstance(st.session_state.get('garch_insights'), dict) and
            crypto in st.session_state['garch_insights']):

            try:
                garch_data = st.session_state['garch_insights'][crypto]
                garch_vol = safe_dict_get(garch_data, 'conditional_vol', volatility, (int, float))
                historical_vol = volatility

                if GARCH_AVAILABLE and hasattr(garch_engine, 'volatility_penalized_kelly'):
                    final_allocation = garch_engine.volatility_penalized_kelly(
                        final_allocation,
                        garch_vol,
                        historical_vol
                    )
            except Exception as e:
                logger.warning(f"GARCH penalty failed for {crypto}: {e}")

        # Final sanity check
        if not isinstance(final_allocation, (int, float)) or final_allocation < 0:
            logger.error(f"Invalid final allocation for {crypto}: {final_allocation}")
            return 0.0

        return max(0.0, min(1.0, final_allocation))

    except Exception as e:
        logger.error(f"Position size calculation failed for {crypto}: {e}\n{traceback.format_exc()}")
        return 0.0

def check_risk_violations(allocations, correlation_matrix=None, current_cryptos=None):
    """
    HARDENED: Risk violation checks with extensive validation
    """
    violations = []

    try:
        # Input validation
        if not allocations or not isinstance(allocations, dict):
            logger.warning("Invalid allocations passed to risk check")
            return violations

        if not current_cryptos or not isinstance(current_cryptos, (list, tuple)):
            logger.warning("Invalid current_cryptos passed to risk check")
            return violations

        # Correlation check with validation
        if correlation_matrix is not None:
            try:
                if not isinstance(correlation_matrix, (np.ndarray, pd.DataFrame)):
                    logger.warning(f"Invalid correlation matrix type: {type(correlation_matrix)}")
                else:
                    corr_df = pd.DataFrame(correlation_matrix, index=current_cryptos, columns=current_cryptos)
                    max_corr = 0
                    corr_pair = None

                    for i, crypto1 in enumerate(current_cryptos):
                        for j, crypto2 in enumerate(current_cryptos):
                            if i < j:
                                try:
                                    corr_val = abs(corr_df.loc[crypto1, crypto2])
                                    if pd.notna(corr_val) and corr_val > max_corr:
                                        max_corr = corr_val
                                        corr_pair = (crypto1, crypto2)
                                except Exception as e:
                                    logger.debug(f"Correlation access failed for {crypto1}-{crypto2}: {e}")
                                    continue

                    if max_corr > RISK_THRESHOLDS['max_correlation'] and corr_pair:
                        violations.append({
                            'type': 'HIGH_CORRELATION',
                            'severity': 'WARNING',
                            'message': f"⚠️ Correlation between {corr_pair[0]} and {corr_pair[1]} is {max_corr:.2f}",
                            'recommendation': "Reduce combined exposure by 20%"
                        })
            except Exception as e:
                logger.error(f"Correlation check failed: {e}")

        # Speculative asset checks
        for crypto, allocation in allocations.items():
            try:
                if not isinstance(allocation, (int, float)):
                    logger.warning(f"Invalid allocation type for {crypto}: {type(allocation)}")
                    continue

                if is_speculative_asset(crypto) and allocation > RISK_THRESHOLDS['speculative_allocation_max']:
                    violations.append({
                        'type': 'SPECULATIVE_OVERWEIGHT',
                        'severity': 'CRITICAL',
                        'message': f"🚨 {crypto} allocation is {allocation*100:.1f}% (max: 15%)",
                        'recommendation': f"Reduce {crypto} to ≤15% with 25% trailing stop-loss"
                    })
            except Exception as e:
                logger.error(f"Speculative check failed for {crypto}: {e}")

    except Exception as e:
        logger.error(f"Risk violation check failed: {e}\n{traceback.format_exc()}")

    return violations

def display_risk_warnings(violations):
    """
    HARDENED: Display risk warnings
    """
    try:
        if not violations:
            return

        if 'dismissed_warnings' not in st.session_state:
            st.session_state['dismissed_warnings'] = set()

        critical = [v for v in violations if v.get('severity') == 'CRITICAL']
        warnings = [v for v in violations if v.get('severity') == 'WARNING']

        for v in critical:
            try:
                v_id = f"{v.get('type', 'UNKNOWN')}_{v.get('message', '')}"
                if v_id not in st.session_state['dismissed_warnings']:
                    st.error(f"🚨 **{v.get('message', 'Critical warning')}**")
                    st.warning(f"**Action Required**: {v.get('recommendation', 'Review allocation')}")
            except Exception as e:
                logger.error(f"Error displaying critical warning: {e}")

        for v in warnings:
            try:
                v_id = f"{v.get('type', 'UNKNOWN')}_{v.get('message', '')}"
                if v_id not in st.session_state['dismissed_warnings']:
                    with st.expander(f"⚠️ {v.get('type', 'WARNING')}", expanded=True):
                        st.warning(v.get('message', 'Warning'))
                        st.info(f"**Recommendation**: {v.get('recommendation', 'Review allocation')}")
            except Exception as e:
                logger.error(f"Error displaying warning: {e}")

    except Exception as e:
        logger.error(f"display_risk_warnings failed: {e}")

def nuclear_flush():
    """
    ENHANCED: Nuclear flush with comprehensive cleanup and validation
    """
    logger.info("🚨 NUCLEAR FLUSH INITIATED")

    try:
        # 1. Clean temp directory
        if 'temp_dir' in st.session_state:
            temp_dir = st.session_state['temp_dir']
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    logger.info(f"✅ Nuked temp directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Temp cleanup warning: {e}")

        # 2. Annihilate all session state
        keys_to_delete = list(st.session_state.keys())
        for key in keys_to_delete:
            try:
                del st.session_state[key]
            except Exception as e:
                logger.debug(f"Could not delete key {key}: {e}")

        logger.info("✅ NUCLEAR FLUSH COMPLETED")
        return True

    except Exception as e:
        logger.critical(f"Nuclear flush failed: {e}\n{traceback.format_exc()}")
        return False

def create_zip_archive(directory: str):
    """Lightning-fast ZIP creation - no progress needed since files are small"""
    try:
        if not os.path.exists(directory):
            return None, None

        # Quick file scan - crypto vis are typically small (<1MB total)
        files = [f for f in os.listdir(directory)
                 if f.lower().endswith(('.png', '.csv', '.html', '.txt', '.pdf', '.log'))
                 and os.path.isfile(os.path.join(directory, f))]

        if not files:
            return None, None

        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:  # Max compression
            for file in files:
                file_path = os.path.join(directory, file)
                zipf.write(file_path, arcname=file)

        buffer.seek(0)

        # Keep the branding!
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
        risk_profile = st.session_state['portfolio_state'].get('risk_tolerance', 'unknown')
        zip_filename = f"CRYPTO_FORTRESS_{risk_profile.upper()}_{timestamp}.zip"

        logger.info(f"🍟 Jumbo Pack served: {zip_filename} ({len(files)} files)")
        return buffer.getvalue(), zip_filename

    except Exception as e:
        logger.error(f"Jumbo Pack failed: {e}")
        return None, None
# === SESSION STATE INITIALIZATION ===
required_states = {
    'portfolio_state': {
        'allocations': None,
        'last_updated': None,
        'cbqra_completed': False,
        'risk_tolerance': 'moderate',
        'cbqra_running': False,
        'locked_profile': None
    },
    'cleanup_done': False,  # ← ADDED THIS TO PREVENT UNNECESSARY CLEANUP
    'forecasts': None,
    'analyzer': None,
    'correlation_matrix': None,
    'uploader_key': 0,
    'monte_carlo_cache': None,
    'monte_carlo_toggle': False,
    'backtest_cache': None,
    'dismissed_warnings': set(),
    'warning_confirmation_count': {},
    'last_allocations': None,
    'uploaded_file_names': None,
    'uploaded_file_hashes': None,
    'sidebar_locked': False,
    'use_uploaded': 'Use Default Files',
    'uploaded_files': None,
    'crypto_names_from_upload': [],
    'garch_engine': None,
    'garch_insights': None,
    'garch_fitted_models': None,
    'volatility_engine': 'BQR'
}

# Initialize session state FIRST
for key, default_value in required_states.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# THEN do enhanced session tracking
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = f"session_{int(time.time())}"
    logger.info(f"🆕 New session started: {st.session_state['session_id']}")
    # Run cleanup ONCE at session start
    pre_session_cleanup()
    st.session_state['portfolio_state']['cbqra_completed'] = False
    st.session_state['portfolio_state']['cbqra_running'] = False

# Continue with safe_cbqra_wrapper definition
def safe_cbqra_wrapper(config, risk_profile):
    """
    ENHANCED: CBQRA wrapper with comprehensive error recovery
    """
    try:
        validate_session_state()
        st.session_state['portfolio_state']['cbqra_running'] = True
        st.session_state['portfolio_state']['cbqra_completed'] = False

        logger.info(f"Starting CBQRA analysis with {risk_profile} profile")
        result = run_cbqra_analysis(config, risk_profile)

        st.session_state['portfolio_state']['cbqra_running'] = False

        if result is not None:
            st.session_state['portfolio_state']['cbqra_completed'] = True
            logger.info("CBQRA analysis completed successfully")
        else:
            st.session_state['portfolio_state']['cbqra_completed'] = False
            logger.warning("CBQRA analysis returned None")

        return result

    except Exception as e:
        logger.error(f"CBQRA wrapper failed: {e}\n{traceback.format_exc()}")
        st.session_state['portfolio_state']['cbqra_running'] = False
        st.session_state['portfolio_state']['cbqra_completed'] = False
        st.error(f"❌ CBQRA failed: {e}")
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        return None

    finally:
        # Absolute failsafe
        if st.session_state['portfolio_state'].get('cbqra_running'):
            st.session_state['portfolio_state']['cbqra_running'] = False
            logger.warning("Forced cbqra_running to False in finally block")
def run_cbqra_analysis(config, risk_profile):
    """
    HARDENED: Main CBQRA analysis engine
    """
    try:
        if not BQR_AVAILABLE:
            st.error("❌ CBQRA unavailable - required modules not found")
            return None

        output_dir = config['output_dir']
        PPath(output_dir).mkdir(parents=True, exist_ok=True)

        # Clean old pairwise comparisons
        try:
            old_pairwise = [f for f in os.listdir(output_dir)
                           if f.startswith('pairwise_') and f.endswith('.png')]
            if old_pairwise:
                st.info(f"🗑️ Cleaning {len(old_pairwise)} old pairwise comparison(s)...")
                for f in old_pairwise:
                    safe_file_operation(os.path.join(output_dir, f), 'delete')
        except Exception as e:
            st.warning(f"⚠️ Couldn't clean old pairwise files: {e}")

        csv_files = [item['file'] for item in config['crypto_data']]
        crypto_names = [item['name'] for item in config['crypto_data']]

        # Validate files exist
        missing = [f for f in csv_files if not os.path.exists(f)]
        if missing:
            st.error(f"❌ Missing files: {', '.join(missing)}")
            return None

        # Initialize analyzer
        analyzer = MultiCryptoBQRAnalysis(
            csv_files=csv_files,
            crypto_names=crypto_names,
            quantiles=config['quantiles']
        )

        st.info("🚀 Active MCMC sampling (check terminal)")
        analyzer.run_full_analysis(
            samples=config['mcmc_samples'],
            tune=config['mcmc_tune'],
            cores=config['mcmc_cores']
        )

        st.session_state['correlation_matrix'] = analyzer.correlation_matrix
        st.session_state['analyzer'] = analyzer

        # Generate visualizations
        if VIZ_AVAILABLE:
            try:
                advanced_viz = AdvancedCryptoVisualizations(analyzer)
                advanced_viz.generate_all_advanced_visualizations()
                st.success("✅ Advanced visualizations generated")
            except Exception as e:
                st.warning(f"Visualizations partially failed: {e}")
                logger.error(f"Visualization error: {e}")

        # Update metrics
        metrics_file = os.path.join(output_dir, 'performance_metrics_multi.csv')
        if os.path.exists(metrics_file):
            try:
                metrics_df = pd.read_csv(metrics_file)
                for _, row in metrics_df.iterrows():
                    crypto = row['Crypto']
                    if crypto not in crypto_data:
                        crypto_data[crypto] = {
                            "volatility": 1.0,
                            "correlation": 0.5,
                            "sharpe": 0.5,
                            "max_drawdown": 50,
                            "beta": 1.0,
                            "expected_return": 0.0,
                            "speculative": is_speculative_asset(crypto)
                        }

                    if 'Ann. Volatility (%)' in metrics_df.columns:
                        vol = float(row['Ann. Volatility (%)']) / 100.0
                        if vol > 0:
                            crypto_data[crypto]['volatility'] = vol

                    if 'Sharpe Ratio' in metrics_df.columns:
                        crypto_data[crypto]['sharpe'] = float(row['Sharpe Ratio'])

                    if 'Max Drawdown (%)' in metrics_df.columns:
                        crypto_data[crypto]['max_drawdown'] = abs(float(row['Max Drawdown (%)']))
            except Exception as e:
                st.warning(f"Metrics update failed: {e}")
                logger.error(f"Metrics update error: {e}")

        # Generate forecasts
        forecasts = {}
        for crypto in crypto_names:
            try:
                last_date = analyzer.data_dict[crypto]['Date'].max()
                forecast_rows = analyzer.pred_df[
                    (analyzer.pred_df['Crypto'] == crypto) &
                    (analyzer.pred_df['Date'] == last_date)
                ]

                if not forecast_rows.empty:
                    prices = analyzer.data_dict[crypto]['Price']
                    daily_returns = prices.pct_change().dropna()
                    hist_annual = daily_returns.mean() * 365

                    b50 = analyzer.trace_dict[f"{crypto}_q0.5"].posterior["beta"].values.mean().item()
                    bqr_annual = b50 * 365
                    blended = 0.7 * hist_annual + 0.3 * bqr_annual

                    forecasts[crypto] = {
                        'Q0.05 (%)': forecast_rows[forecast_rows['Quantile'] == 0.05]['Estimate'].iloc[0],
                        'Q0.5 (%)': forecast_rows[forecast_rows['Quantile'] == 0.5]['Estimate'].iloc[0],
                        'Q0.95 (%)': forecast_rows[forecast_rows['Quantile'] == 0.95]['Estimate'].iloc[0]
                    }
                    crypto_data[crypto]['expected_return'] = blended
            except Exception as e:
                st.warning(f"Forecast failed for {crypto}: {e}")
                logger.error(f"Forecast error for {crypto}: {e}")

        st.session_state['forecasts'] = forecasts

        # Calculate allocations
        new_allocations = {c: calculate_position_size(c, risk_profile) for c in crypto_names}
        total = sum(new_allocations.values())
        if total > 0:
            new_allocations = {c: alloc / total for c, alloc in new_allocations.items()}

        st.session_state['last_allocations'] = new_allocations.copy()
        st.session_state['portfolio_state'].update({
            'allocations': new_allocations,
            'last_updated': pd.Timestamp.now(),
            'locked_profile': risk_profile
        })

        # Clear caches
        st.session_state['monte_carlo_cache'] = None
        st.session_state['backtest_cache'] = None
        if 'monte_carlo_toggle' in st.session_state:
            st.session_state['monte_carlo_toggle'] = False

        st.success(f"✅ CBQRA completed with {risk_profile.upper()} profile")
        return True

    except Exception as e:
        logger.error(f"CBQRA analysis failed: {e}\n{traceback.format_exc()}")
        st.error(f"❌ CBQRA failed: {e}")
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        return None

def parse_backtest_period(period_str, data_dict):
    """
    HARDENED: Convert period string to dates
    """
    try:
        if not isinstance(data_dict, dict) or not data_dict:
            logger.error("Invalid data_dict for backtest period parsing")
            return None, None

        end_date = min(data_dict[c]['Date'].max() for c in data_dict)

        period_map = {
            "Last 30 Days": 30,
            "Last 90 Days": 90,
            "Last 180 Days": 180,
            "Last Year": 365
        }

        if period_str in period_map:
            start_date = end_date - pd.Timedelta(days=period_map[period_str])
        else:
            start_date = max(data_dict[c]['Date'].min() for c in data_dict)

        return start_date, end_date
    except Exception as e:
        logger.error(f"parse_backtest_period failed: {e}")
        return None, None
def get_rebalance_dates(start_date, end_date, frequency):
    """
    HARDENED: Generate rebalancing dates
    """
    try:
        if start_date is None or end_date is None:
            return []

        if frequency == "Daily":
            return pd.date_range(start_date, end_date, freq='D')
        elif frequency == "Weekly":
            return pd.date_range(start_date, end_date, freq='W-MON')
        else:
            return pd.date_range(start_date, end_date, freq='MS')
    except Exception as e:
        logger.error(f"get_rebalance_dates failed: {e}")
        return []
## 📦 **BLOCK 4/8: BACKTEST ENGINE + STREAMLIT CONFIG**
def run_portfolio_backtest(analyzer, allocations, initial_capital, start_date, end_date, rebalance_freq, risk_profile):
    """
    HARDENED: Run portfolio backtest with extensive validation
    """
    try:
        # Input validation
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if start_date is None or end_date is None:
            raise ValueError("Invalid date range")
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
        if not isinstance(allocations, dict) or not allocations:
            raise ValueError("Invalid allocations")

        price_data = {}
        date_ranges = {}

        # Gather price data
        for crypto in allocations.keys():
            if crypto not in analyzer.data_dict:
                raise ValueError(f"Crypto {crypto} not found")

            df = analyzer.data_dict[crypto].copy()
            if 'Price' not in df.columns:
                raise ValueError(f"Price column not found for {crypto}")

            df = df[df['Price'] > 0]
            df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

            if len(df) < 2:
                raise ValueError(f"Insufficient data for {crypto}")

            price_data[crypto] = df[['Date', 'Price']].set_index('Date')
            date_ranges[crypto] = (df['Date'].min(), df['Date'].max())

        # Find common date range
        common_start = max(dr[0] for dr in date_ranges.values())
        common_end = min(dr[1] for dr in date_ranges.values())

        if common_start >= common_end:
            raise ValueError("Insufficient date overlap")

        # Align prices
        aligned_prices = pd.DataFrame({crypto: price_data[crypto]['Price'] for crypto in allocations.keys()})
        aligned_prices = aligned_prices.ffill(limit=3).dropna()

        if len(aligned_prices) < 2:
            raise ValueError("Insufficient aligned data")

        # Run backtest
        portfolio_values = []
        holdings = {}
        cash = initial_capital

        rebalance_dates = get_rebalance_dates(aligned_prices.index[0], aligned_prices.index[-1], rebalance_freq)
        rebalance_dates = [d for d in rebalance_dates if d in aligned_prices.index]
        if not rebalance_dates:
            rebalance_dates = [aligned_prices.index[0]]

        for date in aligned_prices.index:
            if date in rebalance_dates:
                # Liquidate holdings
                for crypto in holdings:
                    cash += holdings[crypto] * aligned_prices.loc[date, crypto]
                holdings = {}

                # Rebalance
                for crypto, allocation in allocations.items():
                    target_value = cash * allocation
                    price = aligned_prices.loc[date, crypto]
                    if price > 0:
                        shares = target_value / price
                        holdings[crypto] = shares
                        cash -= target_value

                if cash < -0.01:
                    cash = 0

            # Calculate portfolio value
            portfolio_value = cash
            for crypto, shares in holdings.items():
                price = aligned_prices.loc[date, crypto]
                portfolio_value += shares * price

            portfolio_values.append({'Date': date, 'Value': portfolio_value, 'Cash': cash})

        results_df = pd.DataFrame(portfolio_values).dropna(subset=['Value'])

        if len(results_df) < 2:
            raise ValueError("Insufficient portfolio data")

        # Calculate metrics
        returns = results_df['Value'].pct_change().dropna()
        final_value = results_df['Value'].iloc[-1]
        total_return = (final_value / initial_capital - 1.0) * 100

        days = (results_df['Date'].iloc[-1] - results_df['Date'].iloc[0]).days
        if days < 1:
            raise ValueError("Insufficient time period")

        ann_return = ((final_value / initial_capital) ** (365.25 / days) - 1) * 100
        volatility = returns.std() * np.sqrt(365) if len(returns) > 1 else 0
        sharpe = (ann_return / 100) / volatility if volatility > 0 else 0

        cummax = results_df['Value'].expanding().max()
        drawdown = (results_df['Value'] - cummax) / cummax
        max_drawdown = drawdown.min() * 100

        win_rate = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0

        return {
            'results_df': results_df,
            'total_return': total_return,
            'ann_return': ann_return,
            'volatility': volatility * 100,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'final_value': final_value
        }

    except Exception as e:
        logger.error(f"Backtest failed: {e}\n{traceback.format_exc()}")
        raise

def save_visualization_to_disk(fig, filename, output_dir='crypto_analysis_results'):
    """
    HARDENED: Save matplotlib figure
    """
    try:
        PPath(output_dir).mkdir(parents=True, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save visualization {filename}: {e}")
        return None

# === STREAMLIT APP CONFIGURATION ===
st.set_page_config(
    page_title="Crypto Risk Manager Pro v4.6",
    layout="wide",
    page_icon="📊"
)

# CRITICAL: Validate session state on every page load
validate_session_state()

# === SESSION STATE INITIALIZATION ===
required_states = {
    'portfolio_state': {
        'allocations': None,
        'last_updated': None,
        'cbqra_completed': False,
        'risk_tolerance': 'moderate',
        'cbqra_running': False,
        'locked_profile': None
    },
    'forecasts': None,
    'analyzer': None,
    'correlation_matrix': None,
    'uploader_key': 0,
    'monte_carlo_cache': None,
    'monte_carlo_toggle': False,
    'backtest_cache': None,
    'dismissed_warnings': set(),
    'warning_confirmation_count': {},
    'last_allocations': None,
    'uploaded_file_names': None,
    'uploaded_file_hashes': None,
    'sidebar_locked': False,
    'use_uploaded': 'Use Default Files',
    'uploaded_files': None,
    'crypto_names_from_upload': [],
    'garch_engine': None,
    'garch_insights': None,
    'garch_fitted_models': None,
    'volatility_engine': 'BQR'

}
# Add this to your session state initialization
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(time.time())
    logger.info(f"🆕 New session started: {st.session_state['session_id']}")

for key, default_value in required_states.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# === ENHANCED SIDEBAR WITH LOCKDOWN ===
with st.sidebar:
    try:
        # ENHANCED LOCKDOWN CHECK
        is_locked = is_analysis_running()

        if is_locked:
            st.error("🔒 SIDEBAR LOCKED")
            st.warning("⚙️ MCMC analysis in progress")
            st.info("Sidebar controls disabled to prevent reload")

            # Emergency stop only
            if st.button("🛑 EMERGENCY STOP", type="primary"):
                st.session_state['portfolio_state']['cbqra_running'] = False
                st.session_state['portfolio_state']['cbqra_completed'] = False
                st.warning("⚠️ Analysis interrupted!")
                st.rerun()
        else:
            # NORMAL SIDEBAR OPERATIONS
            st.markdown("---")
            st.header("📚 Smart Glossary")

            if GLOSSARY:
                st.caption(f"📖 {len(GLOSSARY)} terms available")
                search_term = st.text_input(
                    "🔍 Search glossary:",
                    placeholder="e.g., sharpe ratio, volatility...",
                    key="sidebar_glossary_search"
                )

                if search_term:
                    matches = fuzzy_search_glossary(search_term, GLOSSARY)
                    if matches:
                        st.success(f"🎯 Found {len(matches)} matches:")
                        for term, score, definition in matches:
                            emoji = "🟢" if score >= 90 else "🟡" if score >= 80 else "🟠"
                            with st.expander(f"{emoji} {term} ({score}%)", expanded=(score >= 90)):
                                st.markdown(f"**Definition:** {definition}")
                    else:
                        st.warning(f"No matches for '{search_term}'")
                else:
                    selected = st.selectbox("Browse terms:", [""] + sorted(GLOSSARY.keys()))
                    if selected:
                        st.markdown(f"**{selected}:** {GLOSSARY[selected]}")

            # FILE UPLOAD SECTION
            st.markdown("---")
            st.header("📂 Data Source")

            use_uploaded = st.radio(
                "Choose data source:",
                ["Use Default Files", "Upload CSV Files"],
                index=0 if st.session_state['use_uploaded'] == "Use Default Files" else 1
            )

            st.session_state['use_uploaded'] = use_uploaded

            uploaded_files = None
            crypto_names_from_upload = []

            if use_uploaded == "Upload CSV Files":
                st.info("Upload CSV files with Date and Price columns")
                uploaded_files = st.file_uploader(
                    "Select CSV files",
                    type=['csv'],
                    accept_multiple_files=True,
                    key=f"file_uploader_{st.session_state.get('uploader_key', 0)}"
                )

                st.session_state['uploaded_files'] = uploaded_files

                if uploaded_files:
                    current_file_names = sorted([f.name for f in uploaded_files])
                    current_file_hashes = {f.name: get_file_hash(f) for f in uploaded_files}

                    names_changed = st.session_state.get('uploaded_file_names') != current_file_names
                    hashes_changed = st.session_state.get('uploaded_file_hashes') != current_file_hashes

                    if names_changed or hashes_changed:
                        if names_changed:
                            st.info("📄 New files detected - clearing previous analysis")
                        if hashes_changed and not names_changed:
                            st.info("📄 File content changed - clearing previous analysis")

                        st.session_state['uploaded_file_names'] = current_file_names
                        st.session_state['uploaded_file_hashes'] = current_file_hashes

                        # Clear analysis but preserve file upload state
                        st.session_state['portfolio_state']['cbqra_completed'] = False
                        st.session_state['portfolio_state']['cbqra_running'] = False
                        st.session_state['forecasts'] = None
                        st.session_state['analyzer'] = None
                        st.session_state['correlation_matrix'] = None
                        st.session_state['monte_carlo_cache'] = None
                        st.session_state['backtest_cache'] = None

                    unique_files = {}
                    for file in uploaded_files:
                        ticker = file.name.split('_')[0].split('.')[0].upper()
                        if ticker not in unique_files:
                            unique_files[ticker] = file

                    crypto_names_from_upload = list(unique_files.keys())
                    st.session_state['crypto_names_from_upload'] = crypto_names_from_upload
                    st.success(f"✅ {len(crypto_names_from_upload)} files loaded")
                    for crypto in crypto_names_from_upload:
                        st.write(f"• {crypto}")
                else:
                    st.warning("⏳ No files uploaded")
                    st.session_state['crypto_names_from_upload'] = []
            else:
                st.info("Using default dataset (XRP, XLM, XMR, TRX, DOGE)")
                st.success("✅ 5 default files loaded")
                st.session_state['crypto_names_from_upload'] = []

            # VOLATILITY ENGINE TOGGLE
            st.markdown("---")
            st.subheader("🌪️ Volatility Engine")

            vol_engine = st.radio(
                "Choose Engine:",
                ["BQR (Current)", "GJR-GARCH (Experimental)"],
                index=0 if st.session_state.get('volatility_engine', 'BQR') == 'BQR' else 1,
                help="GARCH captures volatility clustering for dynamic risk adjustment"
            )

            if vol_engine == "BQR (Current)":
                st.session_state['volatility_engine'] = 'BQR'
            else:
                st.session_state['volatility_engine'] = 'GARCH'

        # TIME MACHINE (PERSISTENCE LAYER)
        st.markdown("---")
        st.subheader("🔧 System Control")

        with st.expander("💾 Time Machine (Save/Load Analysis)", expanded=True):
            if is_locked:
                st.warning("🔒 Time Machine locked during MCMC")
                st.info("⚙️ Wait for analysis to complete")
            else:
                st.info("Save the 'Brain' of the app to disk so you don't have to re-calculate.")

                col_save, col_load = st.columns(2)

                with col_save:
                    if st.button("💾 Freeze State", help="Saves BQR models to disk", disabled=is_locked):
                        try:
                            brain_packet = {
                                'analyzer': st.session_state.get('analyzer'),
                                'correlation_matrix': st.session_state.get('correlation_matrix'),
                                'forecasts': st.session_state.get('forecasts'),
                                'portfolio_state': st.session_state.get('portfolio_state'),
                                'uploaded_file_names': st.session_state.get('uploaded_file_names'),
                                'monte_carlo_cache': st.session_state.get('monte_carlo_cache'),
                                'last_allocations': st.session_state.get('last_allocations'),
                                'garch_insights': st.session_state.get('garch_insights'),
                                'garch_fitted_models': st.session_state.get('garch_fitted_models'),
                                'volatility_engine': st.session_state.get('volatility_engine')
                            }

                            joblib.dump(brain_packet, 'crypto_brain_freeze.pkl', compress=3)
                            st.success("✅ State frozen! Safe to close app.")
                            logger.info("Brain state saved successfully")
                        except Exception as e:
                            st.error(f"❌ Save failed: {e}")
                            logger.error(f"Brain save failed: {e}")

                with col_load:
                    if st.button("📂 Thaw State", help="Loads previous analysis instantly", disabled=is_locked):
                        if enhanced_thaw_state():
                            st.rerun()  # Only rerun if we actually loaded a meaningful state


        # NUCLEAR FLUSH WITH LOCKDOWN
        if is_locked:
            st.error("🔒 NUCLEAR FLUSH DISABLED")
            st.warning("⚙️ MCMC in progress - use Emergency Stop if needed")
            if st.button("🗑️ NUCLEAR FLUSH", type="secondary", disabled=True, help="Locked during MCMC"):
                pass
        else:
            if st.button("🗑️ NUCLEAR FLUSH", type="secondary", help="Complete system reset"):
                if nuclear_flush():
                    st.success("✅ NUCLEAR FLUSH COMPLETE!")
                    st.info("Page will reload in 1 second...")
                    time.sleep(1)
                    st.rerun()

        with st.expander("🎯 Selective Reset", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Clear Analysis", key="clear_analysis"):
                    st.session_state['portfolio_state']['cbqra_completed'] = False
                    st.session_state['portfolio_state']['cbqra_running'] = False
                    st.session_state['forecasts'] = None
                    st.session_state['analyzer'] = None
                    st.success("✅ Cleared!")
                    st.rerun()

            with col2:
                if st.button("Reset Warnings", key="reset_warnings"):
                    st.session_state['dismissed_warnings'] = set()
                    st.success("✅ Reset!")
                    st.rerun()

            if st.button("Reset Monte Carlo", key="reset_mc"):
                if 'monte_carlo_toggle' in st.session_state:
                    st.session_state['monte_carlo_toggle'] = False
                st.session_state['monte_carlo_cache'] = None
                st.success("✅ MC Reset!")
                st.rerun()

            if st.button("Clear Uploads", key="clear_uploads"):
                if 'temp_dir' in st.session_state:
                    temp_dir = st.session_state['temp_dir']
                    try:
                        if os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir, ignore_errors=True)
                    except:
                        pass
                    del st.session_state['temp_dir']

                st.session_state['uploaded_file_names'] = None
                st.session_state['uploaded_file_hashes'] = None
                st.session_state['uploader_key'] = st.session_state.get('uploader_key', 0) + 1
                st.success("✅ Uploads cleared!")
                st.rerun()

    except Exception as e:
        logger.error(f"Sidebar error: {e}\n{traceback.format_exc()}")
        st.error("⚠️ Sidebar error - check logs")

## 📦 **BLOCK 5/8: CONFIG SETUP + TAB 1 (RISK DASHBOARD) START**
# === DYNAMIC CONFIG (continued) ===
try:
    use_uploaded = st.session_state.get('use_uploaded', 'Use Default Files')
    uploaded_files = st.session_state.get('uploaded_files', None)
    crypto_names_from_upload = st.session_state.get('crypto_names_from_upload', [])

    if use_uploaded == "Upload CSV Files" and uploaded_files:
        if 'temp_dir' not in st.session_state:
            st.session_state['temp_dir'] = tempfile.mkdtemp()

        temp_dir = st.session_state['temp_dir']
        crypto_data_list = []

        for file in uploaded_files:
            ticker = file.name.split('_')[0].split('.')[0].upper()
            temp_path = os.path.join(temp_dir, file.name)
            with open(temp_path, 'wb') as f:
                f.write(file.getbuffer())
            crypto_data_list.append({'file': temp_path, 'name': ticker})

        CONFIG = {**BASE_CONFIG, 'crypto_data': crypto_data_list}
    else:
        CONFIG = {
            **BASE_CONFIG,
            'crypto_data': [
                {'file': 'xrp_2017-09-13_2025-10-14.csv', 'name': 'XRP'},
                {'file': 'xlm_2017-09-13_2025-10-14.csv', 'name': 'XLM'},
                {'file': 'xmr_2017-09-13_2025-10-14.csv', 'name': 'XMR'},
                {'file': 'trx_2017-09-13_2025-10-14.csv', 'name': 'TRX'},
                {'file': 'doge_2017-09-13_2025-10-14.csv', 'name': 'DOGE'},
            ]
        }
except Exception as e:
    logger.error(f"Config setup failed: {e}")
    CONFIG = {
        **BASE_CONFIG,
        'crypto_data': [
            {'file': 'xrp_2017-09-13_2025-10-14.csv', 'name': 'XRP'},
            {'file': 'xlm_2017-09-13_2025-10-14.csv', 'name': 'XLM'},
            {'file': 'xmr_2017-09-13_2025-10-14.csv', 'name': 'XMR'},
            {'file': 'trx_2017-09-13_2025-10-14.csv', 'name': 'TRX'},
            {'file': 'doge_2017-09-13_2025-10-14.csv', 'name': 'DOGE'},
        ]
    }

# === MAIN TABS ===
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Risk Dashboard", "🧠 CBQRA", "📈 Backtesting", "🌪️ GJR-GARCH"])

# === TAB 1: RISK DASHBOARD ===
with tab1:
    try:
        st.header("📊 Portfolio Risk Dashboard")
        st.info("💡 **Pro Tip**: Check the Smart Glossary in the sidebar for term definitions")

        if st.checkbox("📘 Show Learning Center"):
            with st.expander("🎓 Crypto Quant Education Hub", expanded=True):
                search_term = st.text_input("🔍 Search learning center:", key="edu_search")

                if search_term:
                    matches = fuzzy_search_glossary(search_term, GLOSSARY, min_score=60)
                    if matches:
                        st.success(f"🎯 Found {len(matches)} relevant concepts:")
                        for term, score, definition in matches:
                            with st.expander(f"📖 {term} (relevance: {score}%)", expanded=False):
                                st.markdown(f"**Definition:** {definition}")
                    else:
                        st.info("Try different keywords")

                categories = {
                    "📈 Risk & Return Metrics": ["Sharpe Ratio", "Max Drawdown", "Volatility", "Value at Risk (VaR)"],
                    "🔮 Simulation & Forecasting": ["Monte Carlo Simulation", "Bayesian Quantile Regression (BQR)"],
                    "⚖️ Portfolio Construction": ["Kelly Criterion"],
                }

                for category, terms in categories.items():
                    st.markdown(f"### {category}")
                    for term in terms:
                        if term in GLOSSARY:
                            st.markdown(f"**{term}**: {GLOSSARY[term]}")
                    st.markdown("---")

        st.subheader("💰 Portfolio Configuration")
        portfolio_value = st.number_input("Portfolio Value ($)", value=10000.0, min_value=100.0, step=1000.0)

        pstate = st.session_state['portfolio_state']

        st.subheader("🎯 Risk Tolerance Selection")

        if pstate.get('cbqra_completed') and pstate.get('locked_profile'):
            st.warning(f"🔒 **Analysis locked to {pstate['locked_profile'].upper()} profile**. Re-run CBQRA to change.")

        risk_tolerance = st.radio(
            "Select your risk profile:",
            ["conservative", "moderate", "aggressive"],
            index=["conservative", "moderate", "aggressive"].index(pstate.get('risk_tolerance', 'moderate')),
            help="""
            **Conservative**: Max 15% per position, 65% Kelly scaling, tight stops (10%)
            **Moderate**: Max 25% per position, 100% Kelly scaling, standard stops (15%)
            **Aggressive**: Max 35% per position, 135% Kelly scaling, wide stops (20%)
            """
        )

        if risk_tolerance != pstate.get('risk_tolerance'):
            st.session_state['portfolio_state']['risk_tolerance'] = risk_tolerance
            st.session_state['monte_carlo_cache'] = None
            st.session_state['backtest_cache'] = None

            if pstate.get('cbqra_completed') and pstate.get('locked_profile') == risk_tolerance:
                current_cryptos = [item['name'] for item in CONFIG['crypto_data']]
                new_allocations = {crypto: calculate_position_size(crypto, risk_tolerance) for crypto in current_cryptos}
                total = sum(new_allocations.values())
                if total > 0:
                    new_allocations = {c: alloc / total for c, alloc in new_allocations.items()}
                st.session_state['portfolio_state']['allocations'] = new_allocations
                st.success(f"🔄 Allocations updated for {risk_tolerance.upper()} profile")

        current_cryptos = [item['name'] for item in CONFIG['crypto_data']]

        if pstate['allocations'] is None and current_cryptos:
            allocations = {crypto: calculate_position_size(crypto, risk_tolerance) for crypto in current_cryptos}
            total = sum(allocations.values())
            if total > 0:
                allocations = {crypto: alloc / total for crypto, alloc in allocations.items()}
        else:
            allocations = pstate['allocations'] if pstate['allocations'] else {}

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Risk Profile", risk_tolerance.upper())
        with col2:
            st.metric("Assets in Portfolio", f"{len(current_cryptos)}")

        if pstate.get('cbqra_completed'):
            locked_profile = pstate.get('locked_profile', 'unknown')
            if locked_profile == risk_tolerance:
                st.success(f"✅ Dashboard synced with **{risk_tolerance.upper()}** analysis")
            else:
                st.error(f"❌ **MISMATCH**: Showing {risk_tolerance.upper()}, analysis used {locked_profile.upper()}")
        else:
            st.info(f"📊 Ready for analysis with **{risk_tolerance.upper()}** profile")

        if current_cryptos and allocations:
            portfolio_metrics = {
                "weighted_vol": sum(allocations.get(crypto, 0) * crypto_data.get(crypto, {}).get("volatility", 1.0) for crypto in current_cryptos),
                "weighted_sharpe": sum(allocations.get(crypto, 0) * crypto_data.get(crypto, {}).get("sharpe", 0.5) for crypto in current_cryptos),
                "max_drawdown": max([allocations.get(crypto, 0) * crypto_data.get(crypto, {}).get("max_drawdown", 50) for crypto in current_cryptos] + [0])
            }

            st.subheader("📈 Portfolio Health Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Expected Volatility", f"{portfolio_metrics['weighted_vol'] * 100:.1f}%")
            col2.metric("Sharpe Ratio", f"{portfolio_metrics['weighted_sharpe']:.2f}")
            col3.metric("Worst Case Drawdown", f"{portfolio_metrics['max_drawdown']:.1f}%")

            st.subheader("💼 Recommended Positions")
            for crypto in current_cryptos:
                allocation = allocations.get(crypto, 0)
                position_value = portfolio_value * allocation
                profile = risk_profiles[risk_tolerance]
                stop_loss = position_value * (1 - profile["stop_loss"])
                take_profit = position_value * (1 + profile["take_profit"])

                label = f"{crypto}: {allocation*100:.1f}% (${position_value:,.0f})"
                if is_speculative_asset(crypto):
                    label = f"⚠️ {label} [SPECULATIVE]"

                with st.expander(label):
                    col_sl, col_tp = st.columns(2)
                    with col_sl:
                        st.markdown(f"🔴 **Stop Loss:** ${stop_loss:,.0f}")
                    with col_tp:
                        st.markdown(f"🟢 **Take Profit:** ${take_profit:,.0f}")

                    if is_speculative_asset(crypto):
                        st.warning(f"⚠️ **{crypto} is high-risk**. Use 25% trailing stop-loss.")

            violations = check_risk_violations(allocations, st.session_state.get('correlation_matrix'), current_cryptos)

            if violations:
                st.markdown("---")
                st.subheader("🚨 Active Risk Monitoring")
                display_risk_warnings(violations)

        if st.session_state['forecasts']:
            st.subheader("🔮 BQR Trend Forecasts")
            forecast_df = pd.DataFrame(st.session_state['forecasts']).T
            st.dataframe(forecast_df.style.format("{:.2f}"), width='stretch')
        else:
            st.info("📈 Run CBQRA analysis to get Bayesian trend forecasts")

        profile_match = pstate.get('locked_profile') == pstate.get('risk_tolerance') if pstate.get('cbqra_completed') else True

        # MONTE CARLO SECTION
        st.markdown("---")
        st.subheader("🎲 Monte Carlo Simulations")

        if not pstate.get('cbqra_completed'):
            st.warning("⏳ **Monte Carlo functionality disabled**")
            st.info("📊 To enable Monte Carlo simulations, you must first run CBQRA analysis in the 'CBQRA' tab.")

            with st.expander("📖 What Monte Carlo Simulations Provide"):
                st.markdown("""
                Once CBQRA analysis is complete, Monte Carlo simulations will allow you to:

                - **Project portfolio performance** using thousands of random scenarios
                - **Visualize probability distributions** of potential outcomes
                - **Calculate risk metrics** like Value at Risk (VaR) and Conditional VaR
                - **Run stress tests** simulating market crash scenarios
                - **Estimate probability** of beating market benchmarks

                **Configuration options:**
                - Simulation count: 500 / 1000 / 2000 paths
                - Time horizons: 3 months / 6 months / 1 year / 2 years
                - Risk-adjusted random seed based on your profile
                """)

        elif not profile_match:
            st.error("❌ **Cannot run Monte Carlo with mismatched profiles!**")
            st.warning(f"Current: **{risk_tolerance.upper()}** | Analysis: **{pstate.get('locked_profile', 'unknown').upper()}**")
            st.info("Please re-run CBQRA with the current risk profile to enable Monte Carlo simulations.")

        elif not MONTE_CARLO_AVAILABLE:
            st.error("❌ **Monte Carlo module not found**")
            st.info("The 'monte_carlo_simulator.py' module is required for this functionality.")

        else:
            st.success(f"✅ Monte Carlo ready | Profile: **{risk_tolerance.upper()}**")

            if st.checkbox("Show Monte Carlo Projections", key="monte_carlo_toggle"):
                st.info("💡 Monte Carlo simulations project potential portfolio performance using random sampling")

                col1, col2 = st.columns(2)
                with col1:
                    n_simulations = st.selectbox("Number of Simulations", [500, 1000, 2000], index=1)
                with col2:
                    time_horizon = st.selectbox("Time Horizon", ["3 months", "6 months", "1 year", "2 years"], index=2)

                days_map = {"3 months": 90, "6 months": 180, "1 year": 365, "2 years": 730}
                days = days_map[time_horizon]

                # Cache validation
                cache_valid = True
                if st.session_state.get('monte_carlo_cache'):
                    cached_config = st.session_state['monte_carlo_cache'].get('config', {})
                    current_config_check = {
                        'portfolio_value': portfolio_value,
                        'risk_tolerance': risk_tolerance,
                        'n_simulations': n_simulations,
                        'time_horizon': time_horizon
                    }

                    for key, value in current_config_check.items():
                        if cached_config.get(key) != value:
                            cache_valid = False
                            st.info(f"📊 Configuration changed ({key}) - cache will refresh")
                            break

                    if not cache_valid:
                        st.session_state['monte_carlo_cache'] = None

                if st.button("🚀 Run Monte Carlo Simulation", type="primary"):
                    with st.spinner(f"Running {n_simulations} Monte Carlo simulations..."):
                        try:
                            if st.session_state['analyzer'] is None:
                                st.error("❌ Analyzer not found. Run CBQRA first.")
                            else:
                                current_crypto_data = {}
                                for crypto in allocations.keys():
                                    if crypto in crypto_data:
                                        current_crypto_data[crypto] = crypto_data[crypto]
                                    else:
                                        current_crypto_data[crypto] = {
                                            "volatility": 1.0,
                                            "correlation": 0.5,
                                            "sharpe": 0.5,
                                            "max_drawdown": 50,
                                            "beta": 1.0,
                                            "expected_return": 0.0
                                        }

                                profile_seed = PROFILE_SEEDS.get(risk_tolerance, 42)
                                np.random.seed(profile_seed)

                                mc = CryptoMonteCarlo(
                                    st.session_state['analyzer'],
                                    current_crypto_data,
                                    precomputed_correlation=st.session_state['correlation_matrix']
                                )

                                portfolio_paths, asset_paths = mc.simulate_portfolio_paths(
                                    allocations=allocations,
                                    initial_capital=portfolio_value,
                                    days=days,
                                    n_simulations=n_simulations
                                )

                                metrics, final_values = mc.generate_metrics(portfolio_paths, portfolio_value)

                                st.session_state['monte_carlo_cache'] = {
                                    'portfolio_paths': portfolio_paths,
                                    'asset_paths': asset_paths,
                                    'metrics': metrics,
                                    'final_values': final_values,
                                    'mc_instance': mc,
                                    'config': {
                                        'n_simulations': n_simulations,
                                        'time_horizon': time_horizon,
                                        'days': days,
                                        'risk_tolerance': risk_tolerance,
                                        'portfolio_value': portfolio_value
                                    }
                                }

                                st.success(f"✅ Simulation completed: {n_simulations} paths over {days} days")
                                logger.info(f"Monte Carlo completed: {n_simulations} sims, {days} days")
                                st.rerun()

                        except Exception as e:
                            display_error_boundary(e, "Monte Carlo Simulation")

                # Display cached results
                if st.session_state.get('monte_carlo_cache'):
                    cache = st.session_state['monte_carlo_cache']
                    metrics = cache['metrics']

                    st.markdown("---")
                    st.subheader("📊 Monte Carlo Results")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Expected Return", f"{metrics['expected_return']:.1f}%")
                    col2.metric("Best Case", f"{metrics['best_case']:.1f}%")
                    col3.metric("Worst Case", f"{metrics['worst_case']:.1f}%")
                    col4.metric("95% VaR", f"{metrics['var_95']:.1f}%")

                    col5, col6, col7 = st.columns(3)
                    col5.metric("Probability Positive", f"{metrics['probability_positive']:.1f}%")
                    col6.metric("Beat SP500 Chance", f"{metrics['probability_beating_sp500']:.1f}%")
                    col7.metric("Median Final Value", f"${metrics['median_final_value']:,.0f}")

                    st.subheader("📈 Simulation Visualization")
                    try:
                        mc = cache['mc_instance']
                        fig = mc.plot_monte_carlo_results(
                            cache['portfolio_paths'],
                            cache['config']['portfolio_value'],
                            metrics
                        )
                        st.pyplot(fig)
                        st.caption("💡 To enlarge: Right-click → 'Open image in new tab' → [Enter]")
                        plt.close(fig)

                        config = cache['config']
                        mc_filename = f"monte_carlo_{config['risk_tolerance']}_{config['time_horizon'].replace(' ', '_')}_{config['n_simulations']}sims.png"
                        save_visualization_to_disk(fig, mc_filename)

                        with open(f"crypto_analysis_results/{mc_filename}", "rb") as f:
                            img_bytes = f.read()
                            st.download_button(
                                label="📥 Download Monte Carlo Chart",
                                data=img_bytes,
                                file_name=mc_filename,
                                mime="image/png",
                                key="dl_monte_carlo"
                            )
                    except Exception as e:
                        display_error_boundary(e, "Monte Carlo Plot Display")

                    st.subheader("📊 Risk Analysis")
                    col8, col9, col10 = st.columns(3)
                    col8.metric("Return Volatility", f"{metrics['return_std']:.1f}%")
                    col9.metric("Conditional VaR (CVaR)", f"{metrics['cvar_95']:.1f}%")
                    col10.metric("Final Value Std Dev", f"${metrics['final_value_std']:,.0f}")

                    # Stress testing
                    if st.checkbox("🔬 Include Stress Testing", key="stress_test_toggle"):
                        st.subheader("⚠️ Stress Test Scenarios")

                        st.info("""
                        **What is Stress Testing?**

                        Stress testing shows how your portfolio would perform during major market crashes.
                        These scenarios apply historical crisis drawdowns to your current Monte Carlo projections.

                        **Scenarios Based On:**
                        - **2008 Financial Crisis**: -50% market decline
                        - **2020 COVID Crash**: -35% sudden drop
                        - **Bear Market**: -20% sustained decline
                        - **Mild Correction**: -10% typical pullback
                        """)

                        stress_scenarios = {
                            "2008 Financial Crisis (-50%)": 0.5,
                            "2020 COVID Crash (-35%)": 0.65,
                            "Bear Market (-20%)": 0.8,
                            "Mild Correction (-10%)": 0.9
                        }

                        final_values = cache['final_values']
                        stress_results = []
                        for scenario, multiplier in stress_scenarios.items():
                            stressed_final = final_values * multiplier
                            stressed_return = (np.mean(stressed_final) / cache['config']['portfolio_value'] - 1) * 100
                            stressed_median = np.median(stressed_final)

                            stress_results.append({
                                "Scenario": scenario,
                                "Expected Return": f"{stressed_return:.2f}%",
                                "Median Final Value": f"${stressed_median:,.0f}",
                                "Worst Case": f"${np.min(stressed_final):,.0f}"
                            })

                        stress_df = pd.DataFrame(stress_results)
                        st.dataframe(stress_df, width='stretch')

                        st.markdown("---")
                        st.subheader("💡 Stress Test Interpretation")

                        worst_scenario = stress_results[0]
                        st.warning(f"""
                        **Key Takeaways:**
                        - In a 2008-style crisis, expect returns around **{worst_scenario['Expected Return']}**
                        - Your worst-case portfolio value could be **{worst_scenario['Worst Case']}**
                        - Use these scenarios to set appropriate stop-losses
                        - Consider keeping 20-30% cash reserves for crisis buying opportunities
                        """)

                        stress_csv = os.path.join('crypto_analysis_results', f'stress_test_{cache["config"]["risk_tolerance"]}.csv')
                        stress_df.to_csv(stress_csv, index=False)

                        with open(stress_csv, "rb") as f:
                            csv_bytes = f.read()
                            st.download_button(
                                label="📥 Download Stress Test Results",
                                data=csv_bytes,
                                file_name=f"stress_test_{cache['config']['risk_tolerance']}.csv",
                                mime="text/csv",
                                key="dl_stress_test"
                            )

    except Exception as e:
        display_error_boundary(e, "Risk Dashboard Tab")

# === TAB 2: CBQRA ANALYSIS ===
with tab2:
    try:
        st.header("🧠 CBQRA Engine")

        pstate = st.session_state['portfolio_state']

        if pstate.get('cbqra_running') and not pstate.get('cbqra_completed'):
            st.error("🚨 **ANALYSIS IN PROGRESS** - Don't close this tab!")
            st.warning("⚙️ **MCMC Sampling Running** - Check terminal for progress")

            with st.expander("📊 Monitoring Instructions", expanded=True):
                st.markdown("""
                ### Real-Time Progress Monitoring

                **Where to look:**
                - **Terminal/Console** where you ran `streamlit run v4.6.py`
                - **VS Code**: View → Terminal
                - **PyCharm**: Run/Debug console at bottom
                - **Streamlit.app**: click '<Manage app' at bottom right

                **Expected output:**
```
                Sampling 4 chains for 1,000 tune and 1,000 draw iterations...
                Progress | Draws | Divergences | Step size | Speed
                ████████ | 2000  | 0           | 0.559     | 79.54 draws/s
                **⏳ Expected duration:** 1-3 minutes per asset
                            **🔄 Page auto-updates when complete**
                            """)

        elif pstate.get('cbqra_completed'):
            st.success("✅ Analysis completed successfully!")

        output_dir = CONFIG['output_dir']
        st.subheader("📈 Analysis Visualizations")

        all_png_files = [f for f in os.listdir(output_dir) if f.endswith('.png')] if os.path.exists(output_dir) else []

        main_viz_files = []
        pairwise_viz_files = []

        main_viz = [
            ('correlation_matrix_heatmap.png', 'Correlation Matrix Heatmap'),
            ('rolling_correlation_heatmap.png', 'Rolling Correlation Heatmap'),
            ('volatility_comparison.png', 'Volatility Comparison'),
            ('performance_dashboard.png', 'Performance Dashboard'),
            ('return_distributions.png', 'Return Distributions'),
            ('cumulative_returns_comparison.png', 'Cumulative Returns'),
            ('risk_return_scatter.png', 'Risk-Return Scatter'),
            ('drawdown_comparison.png', 'Drawdown Comparison'),
            ('multi_crypto_correlation_matrix.png', 'Multi-Crypto Correlation'),
            ('multi_asset_summary.png', 'Multi-Asset Summary'),
            ('forecast_comparison.png', 'Forecast Comparison')
        ]

        for file, title in main_viz:
            if file in all_png_files:
                main_viz_files.append((file, title))

        pairwise_files = [f for f in all_png_files if f.startswith('pairwise_')]
        for pair_file in pairwise_files:
            crypto_pair = pair_file.replace('pairwise_', '').replace('.png', '').replace('_vs_', ' vs ')
            pairwise_viz_files.append((pair_file, crypto_pair))

        main_viz_files.sort()
        pairwise_viz_files.sort()
        # === ENHANCED JUMBO PACK WITH EDUCATIONAL CONTEXT ===
        if all_png_files:
            with st.expander("📦 Jumbo Pack - Download before Freeze and/or Exiting", expanded=False):
                st.markdown("""
                ### 🎯 Why download the Jumbo Pack before ending sessions?

                **Understanding Freeze/Thaw vs. Jumbo Pack:**

                - **Freeze & Thaw Procedure**: Saves essential data to restore critical post-CBQRA tests only
                - **Jumbo Pack**: Preserves ALL current session visualizations and analysis results

                **What Freeze/Thaw Saves:**
                ✅ Backtest configurations and results
                ✅ Monte Carlo simulation data
                ✅ GJR-GARCH model parameters and insights
                ✅ Portfolio allocations and risk settings
                ✅ Correlation matrices and forecast data

                **What Freeze/Thaw DOESN'T Save:**
                ❌ Main correlation heatmaps and rolling correlation visualizations
                ❌ Performance dashboards and return distribution charts
                ❌ Cumulative returns comparisons and risk-return scatter plots
                ❌ Drawdown analysis and pairwise asset comparisons
                ❌ Multi-asset summary and forecast comparison charts

                **The Privacy-Centric Design:**
                > *"During startup, all residual visualizations are treated as stale artifacts and purged before each new session via an aggressive `pre_session_cleanup` function. This preserves User Privacy and prevents subsequent users from viewing work they do not own."*

                **Bottom Line:** Freeze/Thaw saves expensive compute time for tests, but visualizations require fresh CBQRA per session.

                ### 🎯 Quick Guide: When to Download

                **🔴 MUST download Jumbo Pack if:**
                - You want to keep correlation heatmaps, performance dashboards, or any visualizations
                - You're about to end your session (visualizations auto-purge for privacy)
                - You need visualizations for presentations, reports, or documentation

                **✅ Can skip Jumbo Pack if:**
                - You only care about re-running tests (Monte Carlo, Backtest, GARCH)
                - You'll run fresh CBQRA next session anyway (generates new visualizations)

                ---

                ### 📊 Freeze/Thaw vs. Jumbo Pack

                | Feature      | Freeze/Thaw                               | Jumbo Pack         |
                |------------- |-------------------------------------------|--------------------|
                | **Purpose**  | Skip expensive CBQRA reruns               | Preserve artifacts |
                | **Saves**    | Test results, model params, allocations   | PNG/CSV/HTML files |
                | **Size**     | 50-200 MB                                 | 5-15 MB |
                | **Restores** | Monte Carlo, Backtest, GJR-GARCH instantly| N/A (one-time D/L) |
                | **Visualizations**|❌ Not included                       | ✅ All included     |

                ---

                ### 🔒 Why Visualizations Aren't in Freeze/Thaw

                **Privacy-Centric Design:**
                Visualizations are purged on session start to prevent data leakage between users.
                This ensures your analysis artifacts can't be viewed by subsequent sessions.

                **Performance Optimization:**
                Images are large but quick to regenerate (seconds). We optimize Freeze/Thaw for
                saving expensive computations (MCMC sampling = minutes), not cheap renders.

                ---

                ### 💡 Best Practice Workflow

                1️⃣ Run CBQRA (5 min)
                2️⃣ **Download Jumbo Pack immediately** ← Get visualizations
                3️⃣ Run Monte Carlo, Backtest, GARCH (3-5 min)
                4️⃣ Freeze State before exiting ← Save test results
                5️⃣ Next session: Thaw State → Instant test restore (skip CBQRA if not needed)

                **Time saved:** 5 minutes per session if you only need test results!
                """)
                zip_bytes, zip_filename = create_zip_archive(output_dir)

                if zip_bytes:
                    st.download_button(
                        label="💾 Download Jumbo Pack · Everything in one tiny zip (5–15 MB)",
                        data=zip_bytes,
                        file_name=zip_filename,
                        mime="application/zip",
                        type="primary",
                        help="Download ALL current session visualizations before ending session"
                    )
                    st.caption("🔐 Download now to preserve your work - these files will be purged on session end")

                    # Additional context about what's included
                    with st.expander("📋 What's included in the Jumbo Pack?"):
                        st.markdown("""
                        **Main Visualizations:**
                        - Correlation matrix heatmaps
                        - Rolling correlation analyses
                        - Volatility comparisons
                        - Performance dashboards
                        - Return distribution charts
                        - Cumulative returns comparisons
                        - Risk-return scatter plots
                        - Drawdown analysis charts
                        - Multi-asset summaries
                        - Forecast comparisons

                        **Pairwise Comparisons:**
                        - All asset-vs-asset correlation charts
                        - Side-by-side performance comparisons
                        - Relative strength analyses

                        **Supporting Data:**
                        - Performance metrics CSV files
                        - Configuration snapshots
                        - Analysis timestamps
                        """)
                else:
                    st.info("No analysis files found for Jumbo Pack")
        # === END: ENHANCED JUMBO PACK ===

        if main_viz_files:
            st.subheader("📊 Main Visualizations")
            st.success(f"✅ Found {len(main_viz_files)} main visualizations")

        cols = st.columns(2)
        for idx, (viz_file, viz_title) in enumerate(main_viz_files):
            file_path = os.path.join(output_dir, viz_file)
            with cols[idx % 2]:
                try:
                    st.image(file_path, caption=viz_title)
                    st.caption("💡 To enlarge: Right-click → 'Open image in new tab' → [Enter]")

                    with open(file_path, "rb") as f:
                        img_bytes = f.read()
                        st.download_button(
                            label=f"📥 Download {viz_title}",
                            data=img_bytes,
                            file_name=viz_file,
                            mime="image/png",
                            key=f"dl_main_{viz_file}"
                        )
                except Exception as e:
                    st.error(f"Error loading {viz_file}: {e}")
                    logger.error(f"Viz load error for {viz_file}: {e}")
        else:
            st.warning("⏳ No main visualizations found")

            if pairwise_viz_files:
                st.markdown("---")
                st.subheader("🔍 Pairwise Asset Comparisons")
                st.info(f"🎯 Found {len(pairwise_viz_files)} pairwise comparisons")

                pairwise_options = {title: (file, title) for file, title in pairwise_viz_files}
                selected_pairwise = st.selectbox(
                    "Select pairwise comparison to view:",
                    options=list(pairwise_options.keys()),
                    index=0,
                    key="pairwise_selector"
                )

                if selected_pairwise:
                    selected_file, selected_title = pairwise_options[selected_pairwise]
                    file_path = os.path.join(output_dir, selected_file)

                    try:
                        st.image(file_path, caption=f"Pairwise Comparison: {selected_title}")
                        st.caption("💡 To enlarge: Right-click → 'Open image in new tab' → [Enter]")

                        with open(file_path, "rb") as f:
                            img_bytes = f.read()
                            st.download_button(
                                label=f"📥 Download {selected_title} Comparison",
                                data=img_bytes,
                                file_name=selected_file,
                                mime="image/png",
                                key=f"dl_pairwise_{selected_file}"
                            )
                    except Exception as e:
                        st.error(f"Error loading {selected_file}: {e}")
                        logger.error(f"Pairwise viz load error: {e}")

                    with st.expander("📋 View All Pairwise Comparisons List", expanded=False):
                        st.write("All available pairwise comparisons:")
                        for file, title in pairwise_viz_files:
                            st.write(f"• {title}")
                else:
                    st.info("🔍 No pairwise comparisons found")

                metrics_file = os.path.join(output_dir, 'performance_metrics_multi.csv')
                if os.path.exists(metrics_file):
                    st.markdown("---")
                    st.subheader("📊 Performance Metrics")
                    metrics_df = pd.read_csv(metrics_file)
                    st.dataframe(metrics_df, width='stretch')

                    csv_data = metrics_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Metrics CSV",
                        csv_data,
                        "performance_metrics.csv",
                        "text/csv",
                        key="dl_metrics"
                    )

            if use_uploaded == "Upload CSV Files" and uploaded_files:
                st.info(f"📂 Using {len(crypto_names_from_upload)} uploaded files")
            elif use_uploaded == "Upload CSV Files" and not uploaded_files:
                st.warning("⚠️ No files uploaded yet")
            else:
                st.info("📂 Using default dataset (XRP, XLM, XMR, TRX, DOGE)")

            st.warning("⚠️ Total time required for completing Markov Chain Monte Carlo simulations depends solely on hardware capability and number of assets being analyzed.")

            current_risk = st.session_state['portfolio_state']['risk_tolerance']
            st.info(f"🎯 Current Risk Profile: **{current_risk.upper()}**")

            if pstate.get('cbqra_completed') and pstate.get('locked_profile') != current_risk:
                st.error(f"⚠️ **Profile mismatch**: Analysis used {pstate['locked_profile'].upper()}, current is {current_risk.upper()}")
                st.info("Running new analysis will use current profile and clear previous results.")

            can_run = True
            if use_uploaded == "Upload CSV Files" and not uploaded_files:
                can_run = False
                st.error("⚠️ Upload CSV files first")

            button_disabled = not can_run or (pstate.get('cbqra_running') and not pstate.get('cbqra_completed'))

            if st.button("🚀 Run CBQRA Analysis", type="primary", disabled=button_disabled):
                if not BQR_AVAILABLE:
                    st.error("❌ CBQRA unavailable - required modules not found")
                else:
                    st.session_state['portfolio_state']['cbqra_running'] = True
                    st.session_state['portfolio_state']['cbqra_completed'] = False
                    st.rerun()

            if pstate.get('cbqra_running') and not pstate.get('cbqra_completed'):
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i in range(30):
                    progress_bar.progress(i + 1)
                    status_text.text(f"⚙️ Initializing Bayesian models... {i+1}%")
                    time.sleep(0.05)

                status_text.text("🔬 Running MCMC chains...")

                start_time = datetime.datetime.now()

                with st.spinner("Running Bayesian simulations puts an extra load on your CPU. We recommend keeping non-critical, background processes to a minimum during active MCMC ops."):
                    results = safe_cbqra_wrapper(CONFIG, current_risk)

                elapsed = (datetime.datetime.now() - start_time).total_seconds()
                status_text.text(f"✅ MCMC completed in {elapsed:.1f} seconds")

                if results:
                    for i in range(30, 100):
                        progress_bar.progress(i + 1)
                        status_text.text(f"✅ Generating visualizations... {i+1}%")
                        time.sleep(0.01)

                progress_bar.empty()
                status_text.empty()

                if results:
                    st.success("✅ Analysis complete! Updating dashboard...")
                    st.rerun()
                else:
                    st.error("❌ Analysis failed. Check error messages above.")

    except Exception as e:
                        display_error_boundary(e, "CBQRA Tab")
# === TAB 3: BACKTESTING ===
with tab3:
    try:
        st.header("📆 Backtesting Results")
        pstate = st.session_state['portfolio_state']

        if not pstate['cbqra_completed']:
            st.info("📊 Run CBQRA analysis first to enable backtesting.")
            st.markdown(f"""
            ### What is Backtesting?

            Test your portfolio strategy against historical data to see how it would have performed.

            **Features:**
            - Historical performance simulation
            - Risk-adjusted returns analysis
            - Drawdown scenarios
            - Configurable rebalancing

            **Current profile**: **{pstate['risk_tolerance'].upper()}**
            """)
        else:
            profile_match = pstate.get('locked_profile') == pstate.get('risk_tolerance')

            if not profile_match:
                st.error("❌ **Cannot run backtest with mismatched profiles!**")
                st.warning(f"Current: **{pstate['risk_tolerance'].upper()}** | Analysis: **{pstate.get('locked_profile', 'unknown').upper()}**")
                st.info("Please re-run CBQRA or switch profiles in Dashboard tab.")
            else:
                st.success(f"Backtesting ready | Using **{pstate['risk_tolerance'].upper()}** profile")

                col1, col2, col3 = st.columns(3)
                with col1:
                    backtest_period = st.selectbox("Backtest Period",
                        ["Last 30 Days", "Last 90 Days", "Last 180 Days", "Last Year", "All Available Data"],
                        index=3,
                        help="Longer periods = more reliable statistics, but may include different market regimes")
                with col2:
                    rebalance_freq = st.selectbox("Rebalancing Frequency",
                        ["Daily", "Weekly", "Monthly"],
                        index=1)
                with col3:
                    initial_capital = st.number_input("Initial Capital ($)",
                        min_value=100.0, max_value=1000000.0, value=10000.0, step=1000.0)

                if st.button("Run Backtest", type="primary"):
                    with st.spinner("Running backtest..."):
                        try:
                            if st.session_state['analyzer'] is None:
                                st.error("❌ Analyzer not found. Run CBQRA first.")
                            else:
                                analyzer = st.session_state['analyzer']
                                current_allocations = pstate['allocations']

                                start_date, end_date = parse_backtest_period(backtest_period, analyzer.data_dict)
                                profile = risk_profiles[pstate['risk_tolerance']]

                                results = run_portfolio_backtest(
                                    analyzer, current_allocations, initial_capital,
                                    start_date, end_date, rebalance_freq, profile
                                )

                                if results:
                                    st.subheader(f"Backtest Results – {pstate['risk_tolerance'].upper()} Profile")

                                    col1, col2, col3, col4 = st.columns(4)
                                    col1.metric("Total Return", f"{results['total_return']:.2f}%",
                                              f"{results['ann_return']:.2f}% ann.")
                                    col2.metric("Sharpe Ratio", f"{results['sharpe']:.2f}")
                                    col3.metric("Max Drawdown", f"{results['max_drawdown']:.2f}%")
                                    col4.metric("Win Rate", f"{results['win_rate']:.1f}%")

                                    col5, col6 = st.columns(2)
                                    col5.metric("Initial Capital", f"${initial_capital:,.2f}")
                                    col6.metric("Final Value", f"${results['final_value']:,.2f}",
                                              f"${results['final_value'] - initial_capital:+,.2f}")

                                    st.subheader("Portfolio Value Over Time")
                                    fig, ax = plt.subplots(figsize=(12, 6))
                                    ax.plot(results['results_df']['Date'], results['results_df']['Value'],
                                           linewidth=2, color='#1f77b4')
                                    ax.axhline(initial_capital, color='red', linestyle='--',
                                              alpha=0.5, label='Initial Capital')
                                    ax.fill_between(results['results_df']['Date'],
                                                   results['results_df']['Value'],
                                                   initial_capital, alpha=0.3,
                                                   color='green' if results['total_return'] > 0 else 'red')
                                    ax.set_xlabel("Date")
                                    ax.set_ylabel("Portfolio Value ($)")
                                    ax.set_title(f"Portfolio Performance ({backtest_period}, {rebalance_freq} Rebalancing)")
                                    ax.legend()
                                    ax.grid(True, alpha=0.3)
                                    plt.xticks(rotation=45)
                                    plt.tight_layout()

                                    st.pyplot(fig)
                                    st.caption("💡 To enlarge: Right-click → 'Open image in new tab' → [Enter]")
                                    plt.close(fig)

                                    bt_filename = f"backtest_{pstate['risk_tolerance']}_{backtest_period.replace(' ', '_')}.png"
                                    save_visualization_to_disk(fig, bt_filename)

                                    with open(f"crypto_analysis_results/{bt_filename}", "rb") as f:
                                        img_bytes = f.read()
                                        st.download_button(
                                            label="📥 Download Backtest Chart",
                                            data=img_bytes,
                                            file_name=bt_filename,
                                            mime="image/png",
                                            key="dl_backtest"
                                        )

                                    st.subheader("Portfolio Allocation Used")
                                    alloc_df = pd.DataFrame([
                                        {"Asset": k, "Allocation": f"{v*100:.2f}%",
                                         "Value": f"${initial_capital*v:,.2f}"}
                                        for k, v in current_allocations.items()
                                    ])
                                    st.dataframe(alloc_df, width='stretch')

                                    actual_start = results['results_df']['Date'].iloc[0]
                                    actual_end = results['results_df']['Date'].iloc[-1]
                                    st.success(f"Backtest completed: {len(results['results_df'])} days from {actual_start.date()} to {actual_end.date()}")

                        except Exception as e:
                            display_error_boundary(e, "Backtesting")

    except Exception as e:
        display_error_boundary(e, "Backtesting Tab")
# === TAB 4: GJR-GARCH ANALYSIS ===
with tab4:
    try:
        st.header("🌪️ GJR-GARCH Volatility Analysis")

        st.info("""
        **GJR-GARCH Model Features:**
        - Captures volatility clustering and persistence
        - Models asymmetric leverage effects (bad news impacts volatility more than good news)
        - Provides dynamic conditional volatility forecasts
        - Enhances risk management with time-varying volatility estimates
        """)

        col1, col2 = st.columns(2)
        with col1:
            current_engine = st.session_state.get('volatility_engine', 'BQR')
            st.metric("Current Volatility Engine", current_engine)
        with col2:
            garch_available_status = "✅ Available" if GARCH_AVAILABLE else "❌ Not Available"
            st.metric("GARCH Engine Status", garch_available_status)

        st.subheader("🔄 GARCH Model Fitting")

        if not GARCH_AVAILABLE:
            st.error("❌ GARCH engine not available - install 'arch' package")
            st.info("""
            **To enable GARCH functionality:**
```bash
            pip install arch
```
            The 'arch' package provides GJR-GARCH model implementation.
            """)
        else:
            st.warning("⚠️ One-time fitting cost per asset - computationally intensive")

        with st.expander("🔧 GARCH Debug Info", expanded=False):
            st.write(f"GARCH_AVAILABLE: {GARCH_AVAILABLE}")
            st.write(f"Analyzer available: {st.session_state.get('analyzer') is not None}")
            if st.session_state.get('analyzer'):
                st.write(f"Crypto names: {st.session_state['analyzer'].crypto_names}")
                st.write(f"Data dict keys: {list(st.session_state['analyzer'].data_dict.keys())}")

        if st.button("🧪 Test GARCH Engine with Real Data", key="test_garch_tab4"):
            try:
                from garch_engine import garch_engine
                st.write(f"✅ garch_engine loaded: {type(garch_engine)}")

                if st.session_state.get('analyzer'):
                    crypto_names = st.session_state['analyzer'].crypto_names[:1]
                    data_dict = st.session_state['analyzer'].data_dict

                    if crypto_names and crypto_names[0] in data_dict:
                        crypto = crypto_names[0]
                        df = data_dict[crypto]
                        returns = df['Price'].pct_change().dropna()

                        st.write(f"Testing with {crypto}: {len(returns)} returns")
                        test_result = garch_engine.fit_gjr_garch(returns, f"TEST_{crypto}")

                        if test_result:
                            st.success(f"✅ GARCH test passed for {crypto}!")
                            st.json(test_result)
                        else:
                            st.error(f"❌ GARCH test failed for {crypto}")
                else:
                    np.random.seed(42)
                    n_points = 500
                    returns = np.random.normal(0, 0.02, n_points)
                    for i in range(10, n_points):
                        if abs(returns[i-1]) > 0.03:
                            returns[i] = np.random.normal(0, 0.04)

                    dummy_returns = pd.Series(returns)
                    test_result = garch_engine.fit_gjr_garch(dummy_returns, "SYNTH_ASSET")
                    st.write(f"🧪 Synthetic test result: {test_result}")

            except Exception as e:
                display_error_boundary(e, "GARCH Test")

        if st.button("🔄 Fit GARCH Models", key="fit_garch_tab4", type="primary"):
            if not GARCH_AVAILABLE:
                st.error("❌ GARCH engine not available - install 'arch' package")
                st.code("pip install arch", language="bash")
                st.info("The 'arch' package provides GJR-GARCH model implementation.")
            else:
                st.warning("⚠️ One-time fitting cost per asset - computationally intensive")
                with st.spinner("Fitting GJR-GARCH models - check terminal..."):
                    st.info("🔄 Starting GARCH fitting process...")

                    if not st.session_state.get('analyzer'):
                        st.error("❌ No analyzer found - run CBQRA first!")
                    else:
                        crypto_names = st.session_state['analyzer'].crypto_names
                        data_dict = st.session_state['analyzer'].data_dict

                        st.write(f"🔍 Fitting {len(crypto_names)} assets: {crypto_names}")

                        valid_cryptos = []
                        for crypto in crypto_names:
                            if crypto in data_dict:
                                df = data_dict[crypto]
                                if 'Price' in df.columns:
                                    returns = df['Price'].pct_change().dropna()
                                    if len(returns) >= 100:
                                        valid_cryptos.append(crypto)
                                        st.success(f"✅ {crypto}: {len(returns)} returns available")
                                    else:
                                        st.warning(f"⚠️ {crypto}: Only {len(returns)} returns (need 100+)")
                                else:
                                    st.error(f"❌ {crypto}: No 'Price' column")
                            else:
                                st.error(f"❌ {crypto} not in data_dict!")

                        if not valid_cryptos:
                            st.error("❌ No valid assets for GARCH fitting!")
                        else:
                            try:
                                st.info(f"🚀 Calling fit_garch_models for {len(valid_cryptos)} valid assets...")
                                results = fit_garch_models(data_dict, valid_cryptos)

                                if results and results['success'] > 0:
                                    st.session_state['garch_insights'] = results['garch_insights']
                                    st.session_state['garch_fitted_models'] = garch_engine.fitted_models.copy()
                                    st.success(f"✅ {results['success']} GARCH models fitted and cached!")
                                    logger.info(f"GARCH fitting success: {results['success']} models")
                                    st.rerun()
                                else:
                                    st.error(f"❌ GARCH fitting failed - success: {results.get('success', 0)}")
                                    if results:
                                        st.write(f"Details: {results}")

                            except Exception as e:
                                st.error(f"💥 GARCH batch fitting crashed: {e}")
                                logger.error(f"GARCH batch fit error: {e}\n{traceback.format_exc()}")

                                st.warning("🔄 Attempting individual asset recovery...")
                                recovery_results = {'garch_insights': {}, 'success': 0}

                                progress_bar = st.progress(0)
                                for idx, crypto in enumerate(valid_cryptos):
                                    try:
                                        progress_bar.progress((idx + 1) / len(valid_cryptos))
                                        st.info(f"Fitting {crypto}...")

                                        single_result = fit_garch_models(data_dict, [crypto])

                                        if single_result and single_result['success'] > 0:
                                            recovery_results['garch_insights'].update(single_result['garch_insights'])
                                            recovery_results['success'] += 1
                                            st.success(f"✅ {crypto} fitted successfully")
                                        else:
                                            st.warning(f"⚠️ {crypto} fit failed")

                                    except Exception as crypto_error:
                                        st.warning(f"⚠️ {crypto} crashed: {crypto_error}")
                                        logger.error(f"Individual GARCH fit error for {crypto}: {crypto_error}")
                                        continue

                                progress_bar.empty()

                                if recovery_results['success'] > 0:
                                    st.session_state['garch_insights'] = recovery_results['garch_insights']
                                    st.session_state['garch_fitted_models'] = garch_engine.fitted_models.copy()
                                    st.success(f"🎯 RECOVERY SUCCESS: {recovery_results['success']}/{len(valid_cryptos)} models fitted")
                                    logger.info(f"GARCH recovery: {recovery_results['success']} models")
                                    st.rerun()
                                else:
                                    st.error("💀 All recovery attempts failed")

        if (st.session_state.get('volatility_engine') == 'GARCH' and
            st.session_state.get('garch_insights')):

            st.markdown("---")
            st.subheader("📊 GARCH Volatility Insights")

            leverage_data = []
            for crypto, insights in st.session_state['garch_insights'].items():
                leverage_effect = insights.get('leverage_effect', 0)
                status = "✅ Confirmed" if leverage_effect > 0 else "⚠️ Not detected"

                leverage_data.append({
                    'Crypto': crypto,
                    'Leverage Effect': f"{leverage_effect:.4f}",
                    'Status': status,
                    'Conditional Vol': f"{insights.get('conditional_vol', 0):.4f}",
                    'VaR 95%': f"{insights.get('var_95', 0):.4f}"
                })

            if leverage_data:
                leverage_df = pd.DataFrame(leverage_data)
                st.dataframe(leverage_df, width='stretch')

                confirmed_count = sum(1 for item in leverage_data if "Confirmed" in item['Status'])
                if confirmed_count > 0:
                    st.success(f"🎵 Leverage effect confirmed for {confirmed_count} assets - 'Down moves hit harder!'")

            st.subheader("📈 GARCH Model Diagnostics")
            st.info("GARCH models provide conditional volatility estimates that adapt to market conditions")

        if st.button("Generate Quick GARCH Volatility Charts", key="quick_garch", type="primary"):
            if not hasattr(garch_engine, 'fitted_models') or not garch_engine.fitted_models:
                st.info("No models fitted yet – click 'Fit GARCH Models' first")
            else:
                for crypto, model in garch_engine.fitted_models.items():
                    if model is None:
                        continue
                    with st.expander(f"{crypto} – Same, same until it ISN'T", expanded=True):
                        try:
                            fig = quick_garch_vol_plot(crypto, model, st.session_state['analyzer'].data_dict)
                            st.pyplot(fig)

                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
                            buf.seek(0)
                            st.download_button(
                                f"Download {crypto} Volatility Chart",
                                data=buf,
                                file_name=f"GARCH_{crypto}_SameSameUntilItIsnt.png",
                                mime="image/png",
                                key=f"dl_garch_quick_{crypto}"
                            )
                            plt.close(fig)
                        except Exception as e:
                            st.error(f"Error generating chart for {crypto}: {e}")
                            logger.error(f"Quick GARCH plot error for {crypto}: {e}")

                st.success("Quick GARCH charts deployed!")

        st.markdown("---")
        st.subheader("⚡ Volatility Forecast Comparison: BQR vs GJR-GARCH")

        if not st.session_state.get('garch_insights'):
            st.info("Run 'Fit GARCH Models' first to enable the showdown")
        else:
            rows = []
            for crypto in st.session_state['analyzer'].crypto_names:
                if crypto not in st.session_state['garch_insights']:
                    continue

                prices = st.session_state['analyzer'].data_dict[crypto]['Price']
                ret = prices.pct_change().dropna()
                hist_vol_30d = ret.rolling(30).std().iloc[-1] * np.sqrt(365)

                try:
                    trace05 = st.session_state['analyzer'].trace_dict[f"{crypto}_q0.05"]
                    trace95 = st.session_state['analyzer'].trace_dict[f"{crypto}_q0.95"]
                    beta05 = trace05.posterior["beta"].mean().item()
                    beta95 = trace95.posterior["beta"].mean().item()
                    bqr_vol = (beta95 - beta05) * np.sqrt(365)
                except:
                    bqr_vol = hist_vol_30d

                garch_cond_vol = st.session_state['garch_insights'][crypto]['conditional_vol'] * np.sqrt(365)
                gamma = st.session_state['garch_insights'][crypto]['leverage_effect']

                rows.append({
                    "Asset": crypto,
                    "Hist_Vol": hist_vol_30d,
                    "BQR_Vol": bqr_vol,
                    "GARCH_Vol": garch_cond_vol,
                    "GARCH_minus_BQR_pp": (garch_cond_vol - bqr_vol) * 100,
                    "Leverage_γ": gamma
                })

            if rows:
                df = pd.DataFrame(rows)

                df["Winner"] = np.where(
                    abs(df["GARCH_Vol"] - df["Hist_Vol"]) < abs(df["BQR_Vol"] - df["Hist_Vol"]),
                    "GARCH", "BQR"
                )

                df_display = df.copy()
                df_display["Historical Vol"] = (df_display["Hist_Vol"] * 100).round(1).astype(str) + "%"
                df_display["BQR Risk Spread"] = (df_display["BQR_Vol"] * 100).round(1).astype(str) + "%"
                df_display["GARCH Conditional Vol"] = (df_display["GARCH_Vol"] * 100).round(1).astype(str) + "%"
                df_display["GARCH − BQR"] = (df_display["GARCH_minus_BQR_pp"].round(1).map("{:+.1f}".format) + " pp")
                df_display["Leverage γ"] = df_display["Leverage_γ"].round(4)
                df_display = df_display[["Asset", "Historical Vol", "BQR Risk Spread", "GARCH Conditional Vol",
                                         "GARCH − BQR", "Leverage γ", "Winner"]]

                def highlight_winner(s):
                    return ['color: #cc0000; font-weight: bold' if v == 'GARCH' else 'color: #0066cc; font-weight: bold' if v == 'BQR' else '' for v in s]

                st.dataframe(
                    df_display.style.apply(highlight_winner, subset=["Winner"])
                                    .set_properties(**{"text-align": "center"}),
                    width='stretch'
                )

                fig, ax = plt.subplots(figsize=(14, 8))
                x = np.arange(len(df))
                width = 0.25

                ax.bar(x - width, df["Hist_Vol"]*100, width, label="Historical (30d)", color="gray", alpha=0.7)
                ax.bar(x,         df["BQR_Vol"]*100, width, label="BQR Risk Spread", color="#0066cc", alpha=0.8)
                ax.bar(x + width, df["GARCH_Vol"]*100, width, label="GARCH Conditional Vol", color="#cc0000", alpha=0.9)

                ax.set_ylabel("Annualized Volatility (%)", fontsize=13)
                ax.set_title("Volatility Showdown – Who Called the Storm?", fontsize=16, pad=20)
                ax.set_xticks(x)
                ax.set_xticklabels(df["Asset"])
                ax.legend(fontsize=12)
                ax.grid(axis='y', alpha=0.3)

                for i, row in df.iterrows():
                    winner_vol = max(row["BQR_Vol"], row["GARCH_Vol"]) * 100
                    ax.text(i, winner_vol + 15, "WIN" if row["Winner"] == "GARCH" else "WIN",
                            ha='center', va='bottom', fontweight='bold', fontsize=12,
                            color='#cc0000' if row["Winner"] == "GARCH" else '#0066cc')

                st.pyplot(fig)
                plt.close(fig)

                garch_wins = (df["Winner"] == "GARCH").sum()
                total = len(df)
                st.success(f"GARCH wins {garch_wins}/{total} assets – especially where leverage effect is strong (γ > 0.05)")

        st.subheader("⚖️ GARCH-Adjusted Position Sizes")

        if st.session_state.get('garch_insights'):
            current_cryptos = [item['name'] for item in CONFIG['crypto_data']]
            risk_tolerance = st.session_state['portfolio_state']['risk_tolerance']
            allocations = st.session_state['portfolio_state'].get('allocations', {})

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Standard Kelly Allocation**")
                standard_allocs = {c: calculate_position_size(c, risk_tolerance)
                                  for c in current_cryptos}
                st.dataframe(pd.DataFrame.from_dict(standard_allocs, orient='index',
                                                   columns=['Allocation']))

            with col2:
                st.markdown("**GARCH-Penalized Allocation** 🌪️")
                st.dataframe(pd.DataFrame.from_dict(allocations, orient='index',
                                                   columns=['Allocation']))

            st.info("""
            💡 **GARCH Penalty Applied**: Positions are reduced when GARCH detects
            elevated conditional volatility compared to historical levels.
            """)
## 📦 **BLOCK 8/8: TAB 4 (GJR-GARCH) FINALE + FOOTER**
        # Current Volatility Regime
# Current Volatility Regime
        st.subheader("📊 Current Volatility Regime")

        if st.session_state.get('garch_insights'):
            regime_data = []
            for crypto, insights in st.session_state['garch_insights'].items():
                current_vol = insights['conditional_vol'] * np.sqrt(365) * 100

                # Define regimes based on annualized vol
                if current_vol < 50:
                    regime = "😴 Low Vol"
                    color = "green"
                elif current_vol < 80:
                    regime = "⚠️ Normal"
                    color = "blue"
                elif current_vol < 120:
                    regime = "🔥 Elevated"
                    color = "orange"
                else:
                    regime = "🌪️ EXTREME"
                    color = "red"

                regime_data.append({
                    'Asset': crypto,
                    'Current Vol': f"{current_vol:.1f}%",
                    'Regime': regime,
                    'Status': color
                })

            regime_df = pd.DataFrame(regime_data)

            for _, row in regime_df.iterrows():
                col1, col2, col3 = st.columns([2, 2, 3])
                with col1:
                    st.markdown(f"**{row['Asset']}**")
                with col2:
                    st.metric("Volatility", row['Current Vol'])
                with col3:
                    st.markdown(row['Regime'])

        # Elton John Leverage Blues
        st.subheader("🎵 The Elton John Leverage Blues")
        if st.session_state.get('garch_insights'):
            leverage_data = []
            for crypto, insights in st.session_state['garch_insights'].items():
                γ = insights['leverage_effect']
                leverage_data.append({
                    'Asset': crypto,
                    'γ (Leverage)': γ,
                    'Impact': f"{(1 + γ):.2f}x worse on down days",
                    'Status': '🔥 STRONG' if γ > 0.08 else '⚠️ MODERATE' if γ > 0.04 else '✓ MILD'
                })

            df = pd.DataFrame(leverage_data).sort_values('γ (Leverage)', ascending=False)

            # Highlight the bluest assets
            st.markdown("### 🎸 Who's Got the Blues?")
            st.dataframe(df.style.background_gradient(subset=['γ (Leverage)'], cmap='Reds'))

            # Victory message
            strong_leverage = df[df['γ (Leverage)'] > 0.08].shape[0]
            if strong_leverage > 0:
                st.success(f"""
                🎵 **{strong_leverage} assets are singing the blues!**

                Bad news hits **{df['Impact'].iloc[0]}** on these assets.
                That's the leverage effect in action — downside shocks linger longer.
                """)

            # One final flourish
            st.markdown("---")
            st.caption("""
            > *"And I think it's gonna be a long, long time…*
            > *Till volatility comes down to earth again…"*
            > — **Elton γ John**, 2025
            """)

        if st.checkbox("📈 Show GARCH Volatility Forecasts", key="garch_forecast_toggle"):
            forecast_days = st.slider("Forecast Horizon (days)", 7, 90, 30)

            # Use cached models if available - WITH PROPER VALIDATION
            fitted_models = st.session_state.get('garch_fitted_models', {})
            if not fitted_models and GARCH_AVAILABLE:
                fitted_models = getattr(garch_engine, 'fitted_models', {})

            # Ensure fitted_models is a dictionary
            if not isinstance(fitted_models, dict):
                fitted_models = {}
                logger.warning("fitted_models was not a dict - reset to empty")

            current_cryptos = [item['name'] for item in CONFIG['crypto_data']]

            for crypto in current_cryptos:
                # SAFE CHECK: Verify crypto exists in fitted_models and model is not None
                if (fitted_models and
                    isinstance(fitted_models, dict) and
                    crypto in fitted_models and
                    fitted_models[crypto] is not None):

                    model = fitted_models[crypto]
                    with st.expander(f"{crypto} Forecast", expanded=True):
                        try:
                            fig = plot_garch_forecast(crypto, model,
                                                     st.session_state['analyzer'].data_dict,
                                                     forecast_days)
                            st.pyplot(fig)
                            st.caption("💡 To enlarge: Right-click → 'Open image in new tab' → [Enter]")
                            plt.close(fig)
                        except Exception as e:
                            st.error(f"Error generating forecast for {crypto}: {e}")
                else:
                    # Only show warning if we expected models but none found
                    if fitted_models and st.session_state.get('garch_insights'):
                        st.info(f"📊 No GARCH model available for {crypto} - run 'Fit GARCH Models' first")
        # Crisis Detection System
        st.subheader("🚨 Crisis Detection System")

        if st.session_state.get('garch_insights'):
            crisis_threshold = 100  # 100% annualized vol

            crisis_assets = []
            for crypto, insights in st.session_state['garch_insights'].items():
                vol = insights['conditional_vol'] * np.sqrt(365) * 100
                if vol > crisis_threshold:
                    crisis_assets.append((crypto, vol))

            if crisis_assets:
                st.error(f"⚠️ **CRISIS MODE**: {len(crisis_assets)} assets in extreme volatility")
                for asset, vol in crisis_assets:
                    st.warning(f"🌪️ **{asset}**: {vol:.1f}% annualized volatility")
                    st.info(f"**Recommendation**: Reduce {asset} exposure by 50% or implement tight stop-losses")
            else:
                st.success("✅ No crisis-level volatility detected")

        # GARCH Educational Content
        with st.expander("🎓 GJR-GARCH Model Explanation", expanded=False):
            st.markdown("""
            ### What is GJR-GARCH?

            **GJR-GARCH** (Glosten-Jagannathan-Runkle GARCH) is an advanced volatility model that captures:

            - **Volatility Clustering**: Large price moves tend to cluster together
            - **Leverage Effect**: Negative returns increase volatility more than positive returns
            - **Persistence**: Volatility shocks persist over time

            ### Model Equation:
```
            σ²ₜ = ω + α⋅ε²ₜ₋₁ + γ⋅Iₜ₋₁⋅ε²ₜ₋₁ + β⋅σ²ₜ₋₁
```

            Where:
            - `σ²ₜ` = conditional variance (volatility squared)
            - `εₜ₋₁` = previous period's shock/innovation
            - `Iₜ₋₁` = indicator (1 if εₜ₋₁ < 0, else 0)
            - `γ` = leverage effect parameter

            ### Why Use GJR-GARCH for Crypto?
            - Cryptocurrencies exhibit strong volatility clustering
            - Leverage effects are pronounced during crashes
            - Provides dynamic, time-varying risk estimates
            - Enhances VaR calculations and position sizing
            """)

    except Exception as e:
        display_error_boundary(e, "GJR-GARCH Tab")

# === FOOTER ===
st.markdown("---")
st.markdown("""
🚀 **Crypto Risk Manager Pro - Operation Fortress v4.6** |
Enhanced robustness + comprehensive error recovery + state validation + Time Machine |
All systems operational 🟢

---
**Superior Quantitative Methodology, Minimal Compute Requirements:**
- Bayesian Quantile Regression (BQR) for tail risk modeling
- GJR-GARCH for volatility clustering and leverage effects
- Monte Carlo simulation for scenario analysis and stress testing
- Kelly Criterion for optimal position sizing
- Dynamic correlation analysis for portfolio construction
- **Time Machine persistence layer for instant analysis restoration**
- **Comprehensive error boundaries with graceful degradation**
- Tested on Lenovo 110S BR11, dual core Atom processor, 2GB RAM

*Built for rigorous crypto risk management and portfolio optimization*

---
**🎸 Parting Shot:**

> *"And I think it's gonna be a long-long time,*
> *Until this dashboard ever fails on me,*
> *I'm not the dev they think I am at home,*
> *Oh no no no!*
> *I'm a ROCKET MAAAAAN!*
> *Rocket MAAAAN, building all the systems that can't be beat!"*

> *...Oh sweet freedom whispers in my ear...*

> *In Lak'ech...🚀*
""")
