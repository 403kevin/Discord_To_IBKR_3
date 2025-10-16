"""
services/vix_checker.py

Author: 403-Forbidden
Purpose: Fetches real-time VIX and volatility metrics from TradingView (no API key needed).
         Provides intelligent volatility regime filtering for trade execution.
"""

import requests
import logging
from datetime import datetime, timedelta


class VIXChecker:
    """
    Fetches real-time VIX data from TradingView's public endpoints.
    Much more reliable and faster than Yahoo Finance.
    No API key required.
    """
    
    def __init__(self, config):
        self.config = config
        self.cached_vix = None
        self.cache_time = None
        self.cached_metrics = None
        self.metrics_cache_time = None
        self.session = requests.Session()
        
        # Set user agent to avoid blocks
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        
        logging.info("VIXChecker initialized with TradingView data source")
    
    def get_current_vix(self):
        """
        Fetches current VIX from TradingView with intelligent caching.
        
        Returns:
            float: Current VIX value, or None if fetch fails
        """
        # Check if cache is still valid
        if self.cached_vix and self.cache_time:
            seconds_since_cache = (datetime.now() - self.cache_time).seconds
            cache_duration = self.config.vix_filter.get('cache_duration', 300)
            
            if seconds_since_cache < cache_duration:
                logging.debug(f"Using cached VIX: {self.cached_vix:.2f} ({seconds_since_cache}s old)")
                return self.cached_vix
        
        # Fetch fresh VIX data
        try:
            url = "https://scanner.tradingview.com/symbol"
            
            payload = {
                "symbols": {
                    "tickers": ["CBOE:VIX"],
                    "query": {"types": []}
                },
                "columns": ["close", "volume", "change"]
            }
            
            response = self.session.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get('data') or len(data['data']) == 0:
                logging.error("TradingView returned empty VIX data")
                return self._handle_fetch_failure()
            
            current_vix = data['data'][0]['d'][0]  # Close price
            
            # Update cache
            self.cached_vix = current_vix
            self.cache_time = datetime.now()
            
            logging.info(f"✅ Fetched VIX from TradingView: {current_vix:.2f}")
            return current_vix
            
        except requests.exceptions.Timeout:
            logging.error("TradingView request timed out")
            return self._handle_fetch_failure()
            
        except requests.exceptions.RequestException as e:
            logging.error(f"TradingView request failed: {e}")
            return self._handle_fetch_failure()
            
        except (KeyError, IndexError, ValueError) as e:
            logging.error(f"Failed to parse TradingView VIX response: {e}")
            return self._handle_fetch_failure()
    
    def _handle_fetch_failure(self):
        """
        Handles VIX fetch failures with intelligent fallback logic.
        
        Returns:
            float or None: Cached VIX if recent enough, otherwise None
        """
        # Fallback: if we have cached data less than 1 hour old, use it
        if self.cached_vix and self.cache_time:
            age_seconds = (datetime.now() - self.cache_time).seconds
            
            if age_seconds < 3600:  # 1 hour
                logging.warning(f"⚠️ Using stale VIX cache ({age_seconds}s old): {self.cached_vix:.2f}")
                return self.cached_vix
        
        # If all else fails, return None and let caller handle it
        logging.error("❌ No valid VIX data available (fetch failed and no cache)")
        return None
    
    def should_trade(self):
        """
        Determines if trading should be allowed based on VIX levels.
        
        Returns:
            bool: True if VIX is in acceptable range, False otherwise
        """
        if not self.config.vix_filter.get('enabled', False):
            return True
        
        vix = self.get_current_vix()
        
        # If VIX fetch failed, decide based on fail_open setting
        if vix is None:
            fail_open = self.config.vix_filter.get('fail_open', True)
            
            if fail_open:
                logging.warning("⚠️ VIX fetch failed - ALLOWING trade (fail_open=True)")
                return True
            else:
                logging.warning("⚠️ VIX fetch failed - BLOCKING trade (fail_open=False)")
                return False
        
        # Check if VIX is in acceptable range
        vix_max = self.config.vix_filter.get('vix_max', 30)
        vix_min = self.config.vix_filter.get('vix_min', 12)
        
        in_range = vix_min <= vix <= vix_max
        
        if not in_range:
            if vix > vix_max:
                logging.info(f"❌ VIX TOO HIGH: {vix:.2f} > {vix_max} (market panic - skipping trade)")
            else:
                logging.info(f"❌ VIX TOO LOW: {vix:.2f} < {vix_min} (market too calm - skipping trade)")
        else:
            logging.debug(f"✅ VIX in acceptable range: {vix:.2f} (range: {vix_min}-{vix_max})")
        
        return in_range
    
    def get_additional_metrics(self):
        """
        ADVANCED: Fetches additional volatility metrics from TradingView.
        Includes VVIX (vol of vol), VIX Mid-Term, and CBOE Skew Index.
        
        Returns:
            dict: Volatility metrics, or None if fetch fails
        """
        # Check metrics cache
        if self.cached_metrics and self.metrics_cache_time:
            seconds_since_cache = (datetime.now() - self.metrics_cache_time).seconds
            cache_duration = self.config.vix_filter.get('cache_duration', 300)
            
            if seconds_since_cache < cache_duration:
                logging.debug(f"Using cached volatility metrics ({seconds_since_cache}s old)")
                return self.cached_metrics
        
        try:
            url = "https://scanner.tradingview.com/symbol"
            
            payload = {
                "symbols": {
                    "tickers": [
                        "CBOE:VIX",      # VIX
                        "CBOE:VVIX",     # VIX of VIX (volatility of volatility)
                        "CBOE:VIXM",     # VIX Mid-Term (3-month forward)
                        "CBOE:SKEW"      # CBOE Skew Index (tail risk)
                    ],
                    "query": {"types": []}
                },
                "columns": ["close", "change"]
            }
            
            response = self.session.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get('data') or len(data['data']) < 4:
                logging.error("TradingView returned incomplete metrics data")
                return None
            
            metrics = {
                "VIX": data['data'][0]['d'][0],
                "VIX_change": data['data'][0]['d'][1],
                "VVIX": data['data'][1]['d'][0],
                "VVIX_change": data['data'][1]['d'][1],
                "VIXM": data['data'][2]['d'][0],
                "VIXM_change": data['data'][2]['d'][1],
                "SKEW": data['data'][3]['d'][0],
                "SKEW_change": data['data'][3]['d'][1],
                
                # Derived metrics
                "VIX_term_structure": data['data'][2]['d'][0] - data['data'][0]['d'][0],  # VIXM - VIX
                "contango": data['data'][2]['d'][0] > data['data'][0]['d'][0]  # True if in contango
            }
            
            # Update cache
            self.cached_metrics = metrics
            self.metrics_cache_time = datetime.now()
            
            logging.info(f"✅ Fetched volatility metrics - VIX: {metrics['VIX']:.2f}, "
                        f"VVIX: {metrics['VVIX']:.2f}, SKEW: {metrics['SKEW']:.2f}")
            
            return metrics
            
        except Exception as e:
            logging.error(f"Failed to fetch additional metrics: {e}")
            return None
    
    def should_trade_advanced(self):
        """
        ADVANCED: Multi-metric volatility regime check.
        Uses VIX, VVIX, SKEW, and term structure for sophisticated filtering.
        
        Returns:
            bool: True if all volatility metrics are favorable for trading
        """
        if not self.config.vix_filter.get('advanced_metrics', {}).get('enabled', False):
            return self.should_trade()  # Fall back to basic VIX check
        
        metrics = self.get_additional_metrics()
        
        if not metrics:
            fail_open = self.config.vix_filter.get('fail_open', True)
            if fail_open:
                logging.warning("⚠️ Metrics fetch failed - ALLOWING trade (fail_open=True)")
                return True
            else:
                logging.warning("⚠️ Metrics fetch failed - BLOCKING trade (fail_open=False)")
                return False
        
        # Check 1: Basic VIX range
        vix_max = self.config.vix_filter.get('vix_max', 30)
        vix_min = self.config.vix_filter.get('vix_min', 12)
        
        if not (vix_min <= metrics['VIX'] <= vix_max):
            logging.info(f"❌ VIX out of range: {metrics['VIX']:.2f} (range: {vix_min}-{vix_max})")
            return False
        
        # Check 2: VVIX (volatility of volatility)
        vvix_max = self.config.vix_filter['advanced_metrics'].get('vvix_max', 999)
        if metrics['VVIX'] > vvix_max:
            logging.info(f"❌ VVIX too high (unstable volatility): {metrics['VVIX']:.2f} > {vvix_max}")
            return False
        
        # Check 3: SKEW (tail risk / crash protection)
        skew_min = self.config.vix_filter['advanced_metrics'].get('skew_min', 0)
        if metrics['SKEW'] < skew_min:
            logging.info(f"❌ SKEW too low (elevated crash risk): {metrics['SKEW']:.2f} < {skew_min}")
            return False
        
        # Check 4: VIX Term Structure (backwardation = fear)
        if self.config.vix_filter['advanced_metrics'].get('avoid_backwardation', False):
            if not metrics['contango']:
                logging.info(f"❌ VIX in backwardation (fear spike): VIXM={metrics['VIXM']:.2f} < VIX={metrics['VIX']:.2f}")
                return False
        
        # All checks passed
        logging.info(f"✅ VOL REGIME CHECK PASSED - VIX: {metrics['VIX']:.2f}, "
                    f"VVIX: {metrics['VVIX']:.2f}, SKEW: {metrics['SKEW']:.2f}, "
                    f"Term Structure: {'Contango' if metrics['contango'] else 'Backwardation'}")
        return True
    
    def get_regime_summary(self):
        """
        Returns a human-readable summary of current volatility regime.
        Useful for logging/debugging.
        
        Returns:
            str: Description of current market volatility state
        """
        metrics = self.get_additional_metrics()
        
        if not metrics:
            return "⚠️ Unable to determine volatility regime (data unavailable)"
        
        vix = metrics['VIX']
        
        # Classify VIX level
        if vix < 12:
            vix_regime = "VERY LOW (complacent)"
        elif vix < 16:
            vix_regime = "LOW (calm)"
        elif vix < 20:
            vix_regime = "NORMAL (stable)"
        elif vix < 25:
            vix_regime = "ELEVATED (cautious)"
        elif vix < 30:
            vix_regime = "HIGH (fearful)"
        else:
            vix_regime = "VERY HIGH (panic)"
        
        # Classify term structure
        structure = "Contango (normal)" if metrics['contango'] else "Backwardation (fear)"
        
        summary = (f"📊 VOLATILITY REGIME:\n"
                  f"   VIX: {vix:.2f} ({vix_regime})\n"
                  f"   VVIX: {metrics['VVIX']:.2f}\n"
                  f"   SKEW: {metrics['SKEW']:.2f}\n"
                  f"   Term Structure: {structure}")
        
        return summary
