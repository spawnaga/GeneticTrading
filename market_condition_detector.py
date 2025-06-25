
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class MarketConditionDetector:
    """
    Detects market conditions to inform adaptive training decisions
    """
    
    def __init__(self, lookback_window: int = 100):
        self.lookback_window = lookback_window
        self.price_history = deque(maxlen=lookback_window)
        self.volume_history = deque(maxlen=lookback_window)
        self.return_history = deque(maxlen=lookback_window)
        
        # Market regime indicators
        self.current_regime = "normal"
        self.volatility_threshold = 2.0  # Multiple of historical volatility
        self.trend_threshold = 0.7      # Trend strength threshold
        
    def update(self, price: float, volume: float) -> Dict:
        """
        Update market condition with new price/volume data
        """
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        if len(self.price_history) >= 2:
            ret = (price - self.price_history[-2]) / self.price_history[-2]
            self.return_history.append(ret)
        
        return self.analyze_conditions()
    
    def analyze_conditions(self) -> Dict:
        """
        Analyze current market conditions
        """
        if len(self.price_history) < 20:
            return {"regime": "insufficient_data", "confidence": 0.0}
        
        conditions = {}
        
        # 1. Volatility Analysis
        conditions.update(self._analyze_volatility())
        
        # 2. Trend Analysis
        conditions.update(self._analyze_trend())
        
        # 3. Volume Analysis
        conditions.update(self._analyze_volume())
        
        # 4. Market Microstructure
        conditions.update(self._analyze_microstructure())
        
        # 5. Overall Regime Classification
        regime, confidence = self._classify_regime(conditions)
        conditions["regime"] = regime
        conditions["confidence"] = confidence
        
        return conditions
    
    def _analyze_volatility(self) -> Dict:
        """
        Analyze volatility patterns
        """
        if len(self.return_history) < 10:
            return {"volatility_regime": "unknown", "volatility_percentile": 50}
        
        returns = np.array(list(self.return_history))
        
        # Current volatility (last 10 periods)
        recent_vol = np.std(returns[-10:])
        
        # Historical volatility
        hist_vol = np.std(returns)
        
        # Volatility ratio
        vol_ratio = recent_vol / (hist_vol + 1e-8)
        
        # Percentile of current volatility
        vol_percentile = np.percentile(
            [np.std(returns[i:i+10]) for i in range(len(returns)-10)],
            50
        ) if len(returns) >= 20 else 50
        
        # Classify volatility regime
        if vol_ratio > self.volatility_threshold:
            vol_regime = "high_volatility"
        elif vol_ratio < 0.5:
            vol_regime = "low_volatility"
        else:
            vol_regime = "normal_volatility"
        
        return {
            "volatility_regime": vol_regime,
            "volatility_ratio": vol_ratio,
            "volatility_percentile": vol_percentile,
            "recent_volatility": recent_vol,
            "historical_volatility": hist_vol
        }
    
    def _analyze_trend(self) -> Dict:
        """
        Analyze trend patterns
        """
        if len(self.price_history) < 20:
            return {"trend_regime": "unknown", "trend_strength": 0}
        
        prices = np.array(list(self.price_history))
        
        # Linear regression for trend
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)
        
        # R-squared for trend strength
        y_pred = slope * x + intercept
        ss_res = np.sum((prices - y_pred) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-8))
        
        # Trend direction and strength
        trend_strength = r_squared
        price_change_pct = (prices[-1] - prices[0]) / prices[0]
        
        # Classify trend
        if trend_strength > self.trend_threshold:
            if price_change_pct > 0.02:  # 2% or more
                trend_regime = "strong_uptrend"
            elif price_change_pct < -0.02:
                trend_regime = "strong_downtrend"
            else:
                trend_regime = "sideways"
        else:
            trend_regime = "choppy"
        
        return {
            "trend_regime": trend_regime,
            "trend_strength": trend_strength,
            "trend_slope": slope,
            "price_change_pct": price_change_pct
        }
    
    def _analyze_volume(self) -> Dict:
        """
        Analyze volume patterns
        """
        if len(self.volume_history) < 10:
            return {"volume_regime": "unknown", "volume_trend": "flat"}
        
        volumes = np.array(list(self.volume_history))
        
        # Volume trend
        recent_avg_vol = np.mean(volumes[-5:])
        hist_avg_vol = np.mean(volumes[:-5])
        
        vol_change = (recent_avg_vol - hist_avg_vol) / (hist_avg_vol + 1e-8)
        
        if vol_change > 0.3:
            volume_trend = "increasing"
            volume_regime = "high_volume" if recent_avg_vol > np.percentile(volumes, 75) else "normal_volume"
        elif vol_change < -0.3:
            volume_trend = "decreasing"
            volume_regime = "low_volume" if recent_avg_vol < np.percentile(volumes, 25) else "normal_volume"
        else:
            volume_trend = "stable"
            volume_regime = "normal_volume"
        
        return {
            "volume_regime": volume_regime,
            "volume_trend": volume_trend,
            "volume_change_pct": vol_change,
            "recent_volume": recent_avg_vol,
            "historical_volume": hist_avg_vol
        }
    
    def _analyze_microstructure(self) -> Dict:
        """
        Analyze market microstructure patterns
        """
        if len(self.return_history) < 20:
            return {"microstructure_regime": "unknown"}
        
        returns = np.array(list(self.return_history))
        
        # Serial correlation (momentum vs mean reversion)
        if len(returns) >= 2:
            serial_corr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        else:
            serial_corr = 0
        
        # Return clustering (volatility clustering)
        abs_returns = np.abs(returns)
        if len(abs_returns) >= 2:
            vol_clustering = np.corrcoef(abs_returns[:-1], abs_returns[1:])[0, 1]
        else:
            vol_clustering = 0
        
        # Classify microstructure
        if serial_corr > 0.3:
            microstructure_regime = "momentum"
        elif serial_corr < -0.3:
            microstructure_regime = "mean_reversion"
        else:
            microstructure_regime = "random_walk"
        
        return {
            "microstructure_regime": microstructure_regime,
            "serial_correlation": serial_corr,
            "volatility_clustering": vol_clustering
        }
    
    def _classify_regime(self, conditions: Dict) -> Tuple[str, float]:
        """
        Classify overall market regime with confidence
        """
        regime_scores = {
            "trending": 0,
            "volatile": 0,
            "quiet": 0,
            "crisis": 0,
            "normal": 0
        }
        
        # Trending market indicators
        if conditions.get("trend_regime") in ["strong_uptrend", "strong_downtrend"]:
            regime_scores["trending"] += 3
        if conditions.get("volume_regime") == "high_volume":
            regime_scores["trending"] += 1
        if conditions.get("microstructure_regime") == "momentum":
            regime_scores["trending"] += 2
        
        # Volatile market indicators
        if conditions.get("volatility_regime") == "high_volatility":
            regime_scores["volatile"] += 3
        if conditions.get("trend_regime") == "choppy":
            regime_scores["volatile"] += 2
        if conditions.get("volatility_ratio", 1) > 2:
            regime_scores["volatile"] += 2
        
        # Quiet market indicators
        if conditions.get("volatility_regime") == "low_volatility":
            regime_scores["quiet"] += 3
        if conditions.get("volume_regime") == "low_volume":
            regime_scores["quiet"] += 2
        if conditions.get("trend_regime") == "sideways":
            regime_scores["quiet"] += 1
        
        # Crisis indicators
        if conditions.get("volatility_ratio", 1) > 3:
            regime_scores["crisis"] += 4
        if conditions.get("volume_regime") == "high_volume" and conditions.get("volatility_regime") == "high_volatility":
            regime_scores["crisis"] += 2
        
        # Normal market (default)
        regime_scores["normal"] = 2  # Base score
        
        # Select regime with highest score
        best_regime = max(regime_scores, key=regime_scores.get)
        max_score = regime_scores[best_regime]
        total_possible = 8  # Maximum possible score
        confidence = min(max_score / total_possible, 1.0)
        
        return best_regime, confidence
    
    def get_training_recommendations(self, conditions: Dict) -> Dict:
        """
        Get training recommendations based on market conditions
        """
        regime = conditions.get("regime", "normal")
        
        recommendations = {
            "preferred_method": "GA",  # Default
            "population_size": 20,
            "generations": 20,
            "ppo_updates": 100,
            "mutation_rate": 0.4,
            "learning_rate_multiplier": 1.0,
            "reason": "default"
        }
        
        if regime == "trending":
            # Trending markets: prefer PPO for exploitation
            recommendations.update({
                "preferred_method": "PPO",
                "ppo_updates": 150,
                "learning_rate_multiplier": 1.2,
                "reason": "trending_market_exploitation"
            })
            
        elif regime == "volatile":
            # Volatile markets: prefer GA for exploration
            recommendations.update({
                "preferred_method": "GA",
                "population_size": 30,
                "generations": 25,
                "mutation_rate": 0.6,
                "reason": "volatile_market_exploration"
            })
            
        elif regime == "quiet":
            # Quiet markets: balanced approach
            recommendations.update({
                "preferred_method": "PPO",
                "ppo_updates": 80,
                "learning_rate_multiplier": 0.8,
                "reason": "quiet_market_refinement"
            })
            
        elif regime == "crisis":
            # Crisis: aggressive exploration
            recommendations.update({
                "preferred_method": "GA",
                "population_size": 40,
                "generations": 35,
                "mutation_rate": 0.8,
                "reason": "crisis_mode_adaptation"
            })
        
        return recommendations
