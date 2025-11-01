"""
ASX Stock Predictor - Advanced Technical Analysis System
Estimates stock prices for tomorrow, week, and month using multiple technical indicators
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ASXStockPredictor:
    """
    Comprehensive stock predictor using multiple technical analysis methods:
    - Fibonacci retracements and extensions
    - Breakout pattern detection
    - Reversal pattern detection
    - Elliott Wave structure
    - Fair Value Gaps
    - Candlestick pattern recognition
    - Heikin Ashi trend signals
    - Renko trend direction
    - Harmonic pattern detection
    - Support and Resistance levels
    """
    
    # Configuration constants
    CONSOLIDATION_THRESHOLD = 0.05  # 5% range for consolidation detection
    PREDICTION_WEIGHTS = {
        'linear_regression': 0.25,
        'ema_momentum': 0.25,
        'mean_reversion': 0.20,
        'rsi_based': 0.15,
        'pattern_based': 0.15
    }
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize predictor with OHLCV data
        
        Args:
            df: DataFrame with columns: Open, High, Low, Close, Volume
        """
        self.df = df.copy()
        self.df.columns = [col.capitalize() for col in self.df.columns]
        self._validate_data()
        self._calculate_indicators()
        
    def _validate_data(self):
        """Validate input data has required columns"""
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if len(self.df) < 50:
            raise ValueError("Need at least 50 periods of data for accurate predictions")
    
    def _calculate_indicators(self):
        """Calculate all technical indicators"""
        # Moving Averages
        self.df['EMA_9'] = self.df['Close'].ewm(span=9, adjust=False).mean()
        self.df['EMA_21'] = self.df['Close'].ewm(span=21, adjust=False).mean()
        self.df['SMA_50'] = self.df['Close'].rolling(window=50).mean()
        self.df['SMA_200'] = self.df['Close'].rolling(window=200).mean()
        
        # RSI
        self.df['RSI'] = self._calculate_rsi(self.df['Close'], 14)
        
        # MACD
        macd_data = self._calculate_macd(self.df['Close'])
        self.df['MACD'] = macd_data['macd']
        self.df['MACD_Signal'] = macd_data['signal']
        self.df['MACD_Hist'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = self._calculate_bollinger_bands(self.df['Close'])
        self.df['BB_Upper'] = bb_data['upper']
        self.df['BB_Middle'] = bb_data['middle']
        self.df['BB_Lower'] = bb_data['lower']
        
        # ATR
        self.df['ATR'] = self._calculate_atr(self.df, 14)
        
        # ADX
        adx_data = self._calculate_adx(self.df, 14)
        self.df['ADX'] = adx_data['adx']
        self.df['DI_Plus'] = adx_data['di_plus']
        self.df['DI_Minus'] = adx_data['di_minus']
        
        # Stochastic
        stoch_data = self._calculate_stochastic(self.df)
        self.df['Stoch_K'] = stoch_data['k']
        self.df['Stoch_D'] = stoch_data['d']
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate MACD indicators"""
        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return {'macd': macd, 'signal': signal, 'histogram': histogram}
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return {'upper': upper, 'middle': middle, 'lower': lower}
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
        """Calculate ADX and Directional Indicators"""
        high_diff = df['High'].diff()
        low_diff = -df['Low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        atr = self._calculate_atr(df, period)
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return {'adx': adx, 'di_plus': plus_di, 'di_minus': minus_di}
    
    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14, smooth: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator"""
        low_min = df['Low'].rolling(window=period).min()
        high_max = df['High'].rolling(window=period).max()
        k = 100 * (df['Close'] - low_min) / (high_max - low_min)
        d = k.rolling(window=smooth).mean()
        return {'k': k, 'd': d}
    
    # === FIBONACCI RETRACEMENTS AND EXTENSIONS ===
    
    def calculate_fibonacci_levels(self, lookback: int = 50) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement and extension levels
        
        Args:
            lookback: Number of periods to look back for swing high/low
            
        Returns:
            Dictionary with Fibonacci levels
        """
        recent_data = self.df.tail(lookback)
        swing_high = recent_data['High'].max()
        swing_low = recent_data['Low'].min()
        diff = swing_high - swing_low
        
        # Fibonacci ratios
        fib_levels = {
            'swing_high': swing_high,
            'swing_low': swing_low,
            'fib_0.236': swing_high - (diff * 0.236),
            'fib_0.382': swing_high - (diff * 0.382),
            'fib_0.500': swing_high - (diff * 0.500),
            'fib_0.618': swing_high - (diff * 0.618),
            'fib_0.786': swing_high - (diff * 0.786),
            'fib_1.000': swing_low,
            # Extensions
            'fib_1.272': swing_high + (diff * 0.272),
            'fib_1.618': swing_high + (diff * 0.618),
            'fib_2.618': swing_high + (diff * 1.618),
        }
        
        return fib_levels
    
    # === BREAKOUT PATTERN DETECTION ===
    
    def detect_breakout_patterns(self, consolidation_period: int = 20) -> Dict[str, any]:
        """
        Detect breakout patterns from consolidation
        
        Args:
            consolidation_period: Number of periods to check for consolidation
            
        Returns:
            Dictionary with breakout information
        """
        recent = self.df.tail(consolidation_period)
        current_price = self.df['Close'].iloc[-1]
        
        # Calculate consolidation range
        high_range = recent['High'].max()
        low_range = recent['Low'].min()
        range_size = high_range - low_range
        avg_price = recent['Close'].mean()
        
        # Check if we're in consolidation (tight range)
        volatility = range_size / avg_price
        is_consolidating = volatility < self.CONSOLIDATION_THRESHOLD
        
        # Detect breakout
        breakout_up = current_price > high_range and is_consolidating
        breakout_down = current_price < low_range and is_consolidating
        
        # Volume confirmation
        avg_volume = recent['Volume'].mean()
        current_volume = self.df['Volume'].iloc[-1]
        volume_surge = current_volume > avg_volume * 1.5
        
        return {
            'is_consolidating': is_consolidating,
            'breakout_up': breakout_up,
            'breakout_down': breakout_down,
            'volume_confirmed': volume_surge,
            'consolidation_range': (low_range, high_range),
            'breakout_strength': 'Strong' if volume_surge else 'Weak'
        }
    
    # === REVERSAL PATTERN DETECTION ===
    
    def detect_reversal_patterns(self) -> Dict[str, bool]:
        """
        Detect common reversal patterns
        
        Returns:
            Dictionary with detected reversal patterns
        """
        patterns = {}
        
        # Get recent candles
        if len(self.df) < 5:
            return patterns
        
        last_5 = self.df.tail(5)
        c = last_5['Close'].values
        o = last_5['Open'].values
        h = last_5['High'].values
        l = last_5['Low'].values
        
        # Double Top (bearish reversal)
        if len(self.df) >= 20:
            recent_highs = self.df['High'].tail(20)
            peaks = recent_highs.nlargest(2)
            if len(peaks) >= 2:
                diff = abs(peaks.iloc[0] - peaks.iloc[1]) / peaks.iloc[0]
                patterns['double_top'] = diff < 0.02 and c[-1] < peaks.min() * 0.98
        
        # Double Bottom (bullish reversal)
        if len(self.df) >= 20:
            recent_lows = self.df['Low'].tail(20)
            troughs = recent_lows.nsmallest(2)
            if len(troughs) >= 2:
                diff = abs(troughs.iloc[0] - troughs.iloc[1]) / troughs.iloc[0]
                patterns['double_bottom'] = diff < 0.02 and c[-1] > troughs.max() * 1.02
        
        # Head and Shoulders (bearish)
        if len(last_5) >= 5:
            patterns['head_and_shoulders'] = (
                h[1] < h[2] and h[3] < h[2] and  # Middle high is highest
                abs(h[1] - h[3]) / h[1] < 0.03 and  # Shoulders similar
                c[-1] < min(l[1], l[3])  # Breakdown below neckline
            )
        
        # Inverse Head and Shoulders (bullish)
        if len(last_5) >= 5:
            patterns['inverse_head_shoulders'] = (
                l[1] > l[2] and l[3] > l[2] and  # Middle low is lowest
                abs(l[1] - l[3]) / l[1] < 0.03 and  # Shoulders similar
                c[-1] > max(h[1], h[3])  # Breakout above neckline
            )
        
        # Rising/Falling Wedge patterns
        if len(self.df) >= 10:
            recent_10 = self.df.tail(10)
            highs_slope = np.polyfit(range(10), recent_10['High'].values, 1)[0]
            lows_slope = np.polyfit(range(10), recent_10['Low'].values, 1)[0]
            
            # Rising Wedge (bearish) - both slopes up but converging
            patterns['rising_wedge'] = (highs_slope > 0 and lows_slope > 0 and 
                                       lows_slope > highs_slope)
            
            # Falling Wedge (bullish) - both slopes down but converging
            patterns['falling_wedge'] = (highs_slope < 0 and lows_slope < 0 and 
                                        highs_slope > lows_slope)
        
        return patterns
    
    # === ELLIOTT WAVE STRUCTURE ===
    
    def analyze_elliott_wave(self, lookback: int = 50) -> Dict[str, any]:
        """
        Simplified Elliott Wave analysis
        
        Args:
            lookback: Number of periods to analyze
            
        Returns:
            Dictionary with wave structure information
        """
        recent = self.df.tail(lookback)
        prices = recent['Close'].values
        
        # Find local peaks and troughs
        peaks = []
        troughs = []
        
        for i in range(2, len(prices) - 2):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                if prices[i] > prices[i-2] and prices[i] > prices[i+2]:
                    peaks.append((i, prices[i]))
            
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                if prices[i] < prices[i-2] and prices[i] < prices[i+2]:
                    troughs.append((i, prices[i]))
        
        # Determine wave structure (simplified)
        wave_count = len(peaks) + len(troughs)
        
        # Trend direction
        if len(peaks) >= 2 and len(troughs) >= 2:
            recent_peaks_rising = peaks[-1][1] > peaks[-2][1] if len(peaks) >= 2 else False
            recent_troughs_rising = troughs[-1][1] > troughs[-2][1] if len(troughs) >= 2 else False
            
            if recent_peaks_rising and recent_troughs_rising:
                wave_trend = 'Impulsive Up (Waves 1-5)'
            elif not recent_peaks_rising and not recent_troughs_rising:
                wave_trend = 'Impulsive Down (Waves 1-5)'
            else:
                wave_trend = 'Corrective (Waves A-B-C)'
        else:
            wave_trend = 'Insufficient data'
        
        return {
            'wave_trend': wave_trend,
            'wave_count': wave_count,
            'peaks_count': len(peaks),
            'troughs_count': len(troughs),
            'current_phase': 'Wave 5' if wave_count >= 9 else f'Wave {(wave_count % 5) + 1}'
        }
    
    # === FAIR VALUE GAPS ===
    
    def detect_fair_value_gaps(self, min_gap_percent: float = 0.5) -> List[Dict[str, any]]:
        """
        Detect Fair Value Gaps (FVG) - imbalances in price action
        
        Args:
            min_gap_percent: Minimum gap size as percentage
            
        Returns:
            List of detected fair value gaps
        """
        gaps = []
        
        if len(self.df) < 3:
            return gaps
        
        for i in range(2, len(self.df)):
            # Bullish FVG: Gap between candle i-2 high and candle i low
            if self.df['Low'].iloc[i] > self.df['High'].iloc[i-2]:
                gap_size = self.df['Low'].iloc[i] - self.df['High'].iloc[i-2]
                gap_percent = (gap_size / self.df['Close'].iloc[i-1]) * 100
                
                if gap_percent >= min_gap_percent:
                    gaps.append({
                        'type': 'Bullish FVG',
                        'index': i,
                        'gap_low': self.df['High'].iloc[i-2],
                        'gap_high': self.df['Low'].iloc[i],
                        'gap_size': gap_size,
                        'gap_percent': gap_percent
                    })
            
            # Bearish FVG: Gap between candle i-2 low and candle i high
            elif self.df['High'].iloc[i] < self.df['Low'].iloc[i-2]:
                gap_size = self.df['Low'].iloc[i-2] - self.df['High'].iloc[i]
                gap_percent = (gap_size / self.df['Close'].iloc[i-1]) * 100
                
                if gap_percent >= min_gap_percent:
                    gaps.append({
                        'type': 'Bearish FVG',
                        'index': i,
                        'gap_high': self.df['Low'].iloc[i-2],
                        'gap_low': self.df['High'].iloc[i],
                        'gap_size': gap_size,
                        'gap_percent': gap_percent
                    })
        
        # Return only recent gaps (last 5)
        return gaps[-5:] if len(gaps) > 5 else gaps
    
    # === CANDLESTICK PATTERNS ===
    
    def recognize_candlestick_patterns(self) -> Dict[str, bool]:
        """
        Recognize common candlestick patterns
        
        Returns:
            Dictionary with detected candlestick patterns
        """
        patterns = {}
        
        if len(self.df) < 3:
            return patterns
        
        # Get last 3 candles
        c = self.df['Close'].tail(3).values
        o = self.df['Open'].tail(3).values
        h = self.df['High'].tail(3).values
        l = self.df['Low'].tail(3).values
        
        # Current candle
        curr_body = abs(c[-1] - o[-1])
        curr_range = h[-1] - l[-1]
        
        # Previous candle
        prev_body = abs(c[-2] - o[-2])
        prev_range = h[-2] - l[-2]
        
        # Doji - small body, long wicks
        patterns['doji'] = curr_body < (curr_range * 0.1) and curr_range > 0
        
        # Hammer (bullish) - small body at top, long lower wick
        patterns['hammer'] = (
            c[-1] > o[-1] and
            (h[-1] - c[-1]) < curr_body * 0.3 and
            (o[-1] - l[-1]) > curr_body * 2
        )
        
        # Shooting Star (bearish) - small body at bottom, long upper wick
        patterns['shooting_star'] = (
            c[-1] < o[-1] and
            (c[-1] - l[-1]) < curr_body * 0.3 and
            (h[-1] - o[-1]) > curr_body * 2
        )
        
        # Engulfing patterns
        patterns['bullish_engulfing'] = (
            c[-2] < o[-2] and  # Previous red
            c[-1] > o[-1] and  # Current green
            o[-1] <= c[-2] and  # Opens at or below prev close
            c[-1] >= o[-2]  # Closes at or above prev open
        )
        
        patterns['bearish_engulfing'] = (
            c[-2] > o[-2] and  # Previous green
            c[-1] < o[-1] and  # Current red
            o[-1] >= c[-2] and  # Opens at or above prev close
            c[-1] <= o[-2]  # Closes at or below prev open
        )
        
        # Morning Star (bullish) - 3 candle pattern
        if len(c) >= 3:
            patterns['morning_star'] = (
                c[-3] < o[-3] and  # First red
                abs(c[-2] - o[-2]) < prev_body * 0.3 and  # Second small body
                c[-1] > o[-1] and  # Third green
                c[-1] > (o[-3] + c[-3]) / 2  # Third closes above midpoint of first
            )
            
            # Evening Star (bearish)
            patterns['evening_star'] = (
                c[-3] > o[-3] and  # First green
                abs(c[-2] - o[-2]) < prev_body * 0.3 and  # Second small body
                c[-1] < o[-1] and  # Third red
                c[-1] < (o[-3] + c[-3]) / 2  # Third closes below midpoint of first
            )
        
        return patterns
    
    # === HEIKIN ASHI ===
    
    def calculate_heikin_ashi(self) -> pd.DataFrame:
        """
        Calculate Heikin Ashi candles
        
        Returns:
            DataFrame with Heikin Ashi OHLC
        """
        ha_df = pd.DataFrame(index=self.df.index)
        
        ha_df['Close'] = (self.df['Open'] + self.df['High'] + 
                         self.df['Low'] + self.df['Close']) / 4
        
        ha_df['Open'] = 0.0
        ha_df.loc[ha_df.index[0], 'Open'] = self.df['Open'].iloc[0]
        
        for i in range(1, len(ha_df)):
            ha_df['Open'].iloc[i] = (ha_df['Open'].iloc[i-1] + ha_df['Close'].iloc[i-1]) / 2
        
        ha_df['High'] = ha_df[['Open', 'Close']].join(self.df['High']).max(axis=1)
        ha_df['Low'] = ha_df[['Open', 'Close']].join(self.df['Low']).min(axis=1)
        
        return ha_df
    
    def get_heikin_ashi_signals(self) -> Dict[str, any]:
        """
        Get trend signals from Heikin Ashi
        
        Returns:
            Dictionary with HA trend signals
        """
        ha_df = self.calculate_heikin_ashi()
        
        # Get last 5 candles
        recent_ha = ha_df.tail(5)
        
        # Count consecutive green/red candles
        green_count = sum(recent_ha['Close'] > recent_ha['Open'])
        red_count = sum(recent_ha['Close'] < recent_ha['Open'])
        
        # Trend strength
        if green_count >= 4:
            trend = 'Strong Uptrend'
        elif green_count >= 3:
            trend = 'Uptrend'
        elif red_count >= 4:
            trend = 'Strong Downtrend'
        elif red_count >= 3:
            trend = 'Downtrend'
        else:
            trend = 'Neutral'
        
        # Check for trend reversal
        last_ha = recent_ha.iloc[-1]
        prev_ha = recent_ha.iloc[-2]
        
        reversal = False
        if (prev_ha['Close'] < prev_ha['Open'] and 
            last_ha['Close'] > last_ha['Open']):
            reversal = 'Bullish Reversal'
        elif (prev_ha['Close'] > prev_ha['Open'] and 
              last_ha['Close'] < last_ha['Open']):
            reversal = 'Bearish Reversal'
        
        return {
            'trend': trend,
            'green_candles': green_count,
            'red_candles': red_count,
            'reversal': reversal
        }
    
    # === RENKO ===
    
    def calculate_renko(self, brick_size: Optional[float] = None) -> List[Dict[str, any]]:
        """
        Calculate Renko bricks
        
        Args:
            brick_size: Size of each brick (if None, uses ATR)
            
        Returns:
            List of Renko bricks
        """
        if brick_size is None:
            brick_size = self.df['ATR'].iloc[-1] * 0.5
        
        bricks = []
        current_price = self.df['Close'].iloc[0]
        
        for idx, row in self.df.iterrows():
            close = row['Close']
            
            # Calculate how many bricks to create
            diff = close - current_price
            num_bricks = int(abs(diff) / brick_size)
            
            if num_bricks > 0:
                direction = 1 if diff > 0 else -1
                
                for _ in range(num_bricks):
                    brick_close = current_price + (brick_size * direction)
                    bricks.append({
                        'open': current_price,
                        'close': brick_close,
                        'direction': 'up' if direction > 0 else 'down',
                        'timestamp': idx
                    })
                    current_price = brick_close
        
        return bricks
    
    def get_renko_trend(self) -> Dict[str, any]:
        """
        Get trend direction from Renko
        
        Returns:
            Dictionary with Renko trend information
        """
        bricks = self.calculate_renko()
        
        if len(bricks) < 5:
            return {'trend': 'Insufficient data', 'strength': 0}
        
        # Get last 10 bricks
        recent_bricks = bricks[-10:] if len(bricks) >= 10 else bricks
        
        up_count = sum(1 for b in recent_bricks if b['direction'] == 'up')
        down_count = sum(1 for b in recent_bricks if b['direction'] == 'down')
        
        if up_count > down_count * 1.5:
            trend = 'Strong Uptrend'
            strength = up_count / len(recent_bricks) * 100
        elif up_count > down_count:
            trend = 'Uptrend'
            strength = up_count / len(recent_bricks) * 100
        elif down_count > up_count * 1.5:
            trend = 'Strong Downtrend'
            strength = down_count / len(recent_bricks) * 100
        elif down_count > up_count:
            trend = 'Downtrend'
            strength = down_count / len(recent_bricks) * 100
        else:
            trend = 'Neutral'
            strength = 50
        
        return {
            'trend': trend,
            'strength': strength,
            'up_bricks': up_count,
            'down_bricks': down_count,
            'total_bricks': len(recent_bricks)
        }
    
    # === HARMONIC PATTERNS ===
    
    def detect_harmonic_patterns(self) -> Dict[str, any]:
        """
        Detect harmonic patterns (Gartley, Butterfly, Bat, Crab)
        
        Returns:
            Dictionary with detected harmonic patterns
        """
        if len(self.df) < 50:
            return {'patterns': [], 'note': 'Insufficient data'}
        
        patterns = []
        
        # Find significant swing points
        prices = self.df['Close'].tail(50).values
        highs = self.df['High'].tail(50).values
        lows = self.df['Low'].tail(50).values
        
        # Find peaks and troughs
        swing_points = []
        for i in range(5, len(prices) - 5):
            if highs[i] == max(highs[i-5:i+6]):
                swing_points.append(('high', i, highs[i]))
            elif lows[i] == min(lows[i-5:i+6]):
                swing_points.append(('low', i, lows[i]))
        
        # Need at least 5 swing points for pattern (XABCD)
        if len(swing_points) >= 5:
            # Get last 5 swing points
            X, A, B, C, D = swing_points[-5:]
            
            # Calculate Fibonacci ratios
            XA = abs(A[2] - X[2])
            AB = abs(B[2] - A[2])
            BC = abs(C[2] - B[2])
            CD = abs(D[2] - C[2])
            
            if XA > 0:
                AB_XA = AB / XA
                BC_AB = BC / AB if AB > 0 else 0
                CD_BC = CD / BC if BC > 0 else 0
                
                # Gartley Pattern (0.618, 0.382-0.886, 1.272-1.618)
                if (0.58 <= AB_XA <= 0.68 and 
                    0.35 <= BC_AB <= 0.90 and 
                    1.20 <= CD_BC <= 1.70):
                    patterns.append({
                        'type': 'Gartley',
                        'direction': 'Bullish' if D[0] == 'low' else 'Bearish',
                        'completion': D[2]
                    })
                
                # Butterfly Pattern (0.786, 0.382-0.886, 1.618-2.618)
                elif (0.75 <= AB_XA <= 0.82 and 
                      0.35 <= BC_AB <= 0.90 and 
                      1.50 <= CD_BC <= 2.70):
                    patterns.append({
                        'type': 'Butterfly',
                        'direction': 'Bullish' if D[0] == 'low' else 'Bearish',
                        'completion': D[2]
                    })
                
                # Bat Pattern (0.382-0.50, 0.382-0.886, 1.618-2.618)
                elif (0.35 <= AB_XA <= 0.52 and 
                      0.35 <= BC_AB <= 0.90 and 
                      1.50 <= CD_BC <= 2.70):
                    patterns.append({
                        'type': 'Bat',
                        'direction': 'Bullish' if D[0] == 'low' else 'Bearish',
                        'completion': D[2]
                    })
                
                # Crab Pattern (0.382-0.618, 0.382-0.886, 2.618-3.618)
                elif (0.35 <= AB_XA <= 0.65 and 
                      0.35 <= BC_AB <= 0.90 and 
                      2.50 <= CD_BC <= 3.70):
                    patterns.append({
                        'type': 'Crab',
                        'direction': 'Bullish' if D[0] == 'low' else 'Bearish',
                        'completion': D[2]
                    })
        
        return {
            'patterns_detected': len(patterns),
            'patterns': patterns,
            'swing_points_found': len(swing_points)
        }
    
    # === SUPPORT AND RESISTANCE ===
    
    def calculate_support_resistance(self, lookback: int = 50, num_levels: int = 3) -> Dict[str, List[float]]:
        """
        Calculate support and resistance levels
        
        Args:
            lookback: Number of periods to analyze
            num_levels: Number of S/R levels to identify
            
        Returns:
            Dictionary with support and resistance levels
        """
        recent = self.df.tail(lookback)
        
        # Find local peaks (resistance)
        resistance_levels = []
        for i in range(5, len(recent) - 5):
            if recent['High'].iloc[i] == recent['High'].iloc[i-5:i+6].max():
                resistance_levels.append(recent['High'].iloc[i])
        
        # Find local troughs (support)
        support_levels = []
        for i in range(5, len(recent) - 5):
            if recent['Low'].iloc[i] == recent['Low'].iloc[i-5:i+6].min():
                support_levels.append(recent['Low'].iloc[i])
        
        # Cluster nearby levels
        def cluster_levels(levels, threshold=0.02):
            if not levels:
                return []
            levels = sorted(levels)
            clusters = []
            current_cluster = [levels[0]]
            
            for level in levels[1:]:
                if abs(level - current_cluster[-1]) / current_cluster[-1] < threshold:
                    current_cluster.append(level)
                else:
                    clusters.append(np.mean(current_cluster))
                    current_cluster = [level]
            
            clusters.append(np.mean(current_cluster))
            return clusters
        
        resistance_clustered = cluster_levels(resistance_levels)
        support_clustered = cluster_levels(support_levels)
        
        # Get top levels closest to current price
        current_price = self.df['Close'].iloc[-1]
        
        resistance_sorted = sorted([r for r in resistance_clustered if r > current_price])[:num_levels]
        support_sorted = sorted([s for s in support_clustered if s < current_price], reverse=True)[:num_levels]
        
        return {
            'resistance': resistance_sorted,
            'support': support_sorted,
            'current_price': current_price
        }
    
    # === PRICE PREDICTION ===
    
    def predict_prices(self) -> Dict[str, Dict[str, float]]:
        """
        Predict prices for tomorrow, week, and month using multiple methods
        
        Returns:
            Dictionary with predictions for different timeframes
        """
        current_price = self.df['Close'].iloc[-1]
        atr = self.df['ATR'].iloc[-1]
        
        # === Method 1: Linear Regression ===
        X = np.arange(len(self.df)).reshape(-1, 1)
        y = self.df['Close'].values
        
        # Use last 50 periods for short-term trend
        X_recent = X[-50:]
        y_recent = y[-50:]
        slope, intercept = np.polyfit(X_recent.flatten(), y_recent, 1)
        
        lr_tomorrow = slope * len(self.df) + intercept
        lr_week = slope * (len(self.df) + 5) + intercept
        lr_month = slope * (len(self.df) + 22) + intercept
        
        # === Method 2: EMA Projection ===
        ema_9 = self.df['EMA_9'].iloc[-1]
        ema_21 = self.df['EMA_21'].iloc[-1]
        ema_momentum = ema_9 - ema_21
        
        ema_tomorrow = current_price + ema_momentum * 0.3
        ema_week = current_price + ema_momentum * 1.5
        ema_month = current_price + ema_momentum * 6
        
        # === Method 3: Mean Reversion (Bollinger Bands) ===
        bb_middle = self.df['BB_Middle'].iloc[-1]
        bb_upper = self.df['BB_Upper'].iloc[-1]
        bb_lower = self.df['BB_Lower'].iloc[-1]
        
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
        
        if bb_position > 0.8:  # Near upper band - expect reversion down
            mr_factor = -0.3
        elif bb_position < 0.2:  # Near lower band - expect reversion up
            mr_factor = 0.3
        else:
            mr_factor = 0
        
        mr_tomorrow = current_price + (bb_middle - current_price) * 0.2 + atr * mr_factor
        mr_week = current_price + (bb_middle - current_price) * 0.5 + atr * mr_factor * 2
        mr_month = bb_middle
        
        # === Method 4: RSI-based ===
        rsi = self.df['RSI'].iloc[-1]
        
        if rsi < 30:  # Oversold - expect bounce
            rsi_factor = 0.02
        elif rsi > 70:  # Overbought - expect pullback
            rsi_factor = -0.02
        else:
            rsi_factor = 0
        
        rsi_tomorrow = current_price * (1 + rsi_factor)
        rsi_week = current_price * (1 + rsi_factor * 3)
        rsi_month = current_price * (1 + rsi_factor * 8)
        
        # === Method 5: Technical Pattern Based ===
        patterns = self.recognize_candlestick_patterns()
        reversal_patterns = self.detect_reversal_patterns()
        
        bullish_score = sum([
            patterns.get('hammer', False),
            patterns.get('bullish_engulfing', False),
            patterns.get('morning_star', False),
            reversal_patterns.get('double_bottom', False),
            reversal_patterns.get('inverse_head_shoulders', False),
            reversal_patterns.get('falling_wedge', False)
        ])
        
        bearish_score = sum([
            patterns.get('shooting_star', False),
            patterns.get('bearish_engulfing', False),
            patterns.get('evening_star', False),
            reversal_patterns.get('double_top', False),
            reversal_patterns.get('head_and_shoulders', False),
            reversal_patterns.get('rising_wedge', False)
        ])
        
        pattern_factor = (bullish_score - bearish_score) * 0.01
        
        pattern_tomorrow = current_price * (1 + pattern_factor)
        pattern_week = current_price * (1 + pattern_factor * 3)
        pattern_month = current_price * (1 + pattern_factor * 8)
        
        # === Combine Predictions (Weighted Average) ===
        def combine_predictions(p1, p2, p3, p4, p5):
            return (p1 * self.PREDICTION_WEIGHTS['linear_regression'] + 
                   p2 * self.PREDICTION_WEIGHTS['ema_momentum'] + 
                   p3 * self.PREDICTION_WEIGHTS['mean_reversion'] + 
                   p4 * self.PREDICTION_WEIGHTS['rsi_based'] + 
                   p5 * self.PREDICTION_WEIGHTS['pattern_based'])
        
        tomorrow_pred = combine_predictions(lr_tomorrow, ema_tomorrow, mr_tomorrow, 
                                           rsi_tomorrow, pattern_tomorrow)
        week_pred = combine_predictions(lr_week, ema_week, mr_week, 
                                       rsi_week, pattern_week)
        month_pred = combine_predictions(lr_month, ema_month, mr_month, 
                                        rsi_month, pattern_month)
        
        # Calculate confidence based on method agreement
        def calc_confidence(predictions):
            std = np.std(predictions)
            mean = np.mean(predictions)
            cv = (std / mean) if mean != 0 else 1
            confidence = max(0, min(100, 100 - (cv * 200)))
            return confidence
        
        tomorrow_conf = calc_confidence([lr_tomorrow, ema_tomorrow, mr_tomorrow, 
                                        rsi_tomorrow, pattern_tomorrow])
        week_conf = calc_confidence([lr_week, ema_week, mr_week, 
                                    rsi_week, pattern_week])
        month_conf = calc_confidence([lr_month, ema_month, mr_month, 
                                     rsi_month, pattern_month])
        
        # Add support/resistance influence
        sr_levels = self.calculate_support_resistance()
        
        def adjust_for_sr(prediction, sr_levels):
            resistance = sr_levels['resistance']
            support = sr_levels['support']
            
            # Check if prediction near resistance
            for r in resistance:
                if abs(prediction - r) / r < 0.02:
                    prediction = r * 0.98  # Pull back slightly
            
            # Check if prediction near support
            for s in support:
                if abs(prediction - s) / s < 0.02:
                    prediction = s * 1.02  # Bounce slightly
            
            return prediction
        
        tomorrow_pred = adjust_for_sr(tomorrow_pred, sr_levels)
        week_pred = adjust_for_sr(week_pred, sr_levels)
        month_pred = adjust_for_sr(month_pred, sr_levels)
        
        # Calculate ranges using ATR
        tomorrow_range = atr * 0.5
        week_range = atr * 1.5
        month_range = atr * 3
        
        return {
            'current': {
                'price': current_price,
                'date': self.df.index[-1]
            },
            'tomorrow': {
                'prediction': tomorrow_pred,
                'change': tomorrow_pred - current_price,
                'change_percent': ((tomorrow_pred - current_price) / current_price) * 100,
                'range_low': tomorrow_pred - tomorrow_range,
                'range_high': tomorrow_pred + tomorrow_range,
                'confidence': tomorrow_conf
            },
            'week': {
                'prediction': week_pred,
                'change': week_pred - current_price,
                'change_percent': ((week_pred - current_price) / current_price) * 100,
                'range_low': week_pred - week_range,
                'range_high': week_pred + week_range,
                'confidence': week_conf
            },
            'month': {
                'prediction': month_pred,
                'change': month_pred - current_price,
                'change_percent': ((month_pred - current_price) / current_price) * 100,
                'range_low': month_pred - month_range,
                'range_high': month_pred + month_range,
                'confidence': month_conf
            }
        }
    
    # === COMPREHENSIVE ANALYSIS ===
    
    def get_comprehensive_analysis(self) -> Dict[str, any]:
        """
        Get comprehensive analysis combining all methods
        
        Returns:
            Dictionary with complete analysis
        """
        analysis = {
            'predictions': self.predict_prices(),
            'fibonacci': self.calculate_fibonacci_levels(),
            'breakout': self.detect_breakout_patterns(),
            'reversal_patterns': self.detect_reversal_patterns(),
            'elliott_wave': self.analyze_elliott_wave(),
            'fair_value_gaps': self.detect_fair_value_gaps(),
            'candlestick_patterns': self.recognize_candlestick_patterns(),
            'heikin_ashi': self.get_heikin_ashi_signals(),
            'renko': self.get_renko_trend(),
            'harmonic_patterns': self.detect_harmonic_patterns(),
            'support_resistance': self.calculate_support_resistance(),
            'indicators': {
                'rsi': self.df['RSI'].iloc[-1],
                'macd': self.df['MACD'].iloc[-1],
                'macd_signal': self.df['MACD_Signal'].iloc[-1],
                'adx': self.df['ADX'].iloc[-1],
                'atr': self.df['ATR'].iloc[-1],
                'stoch_k': self.df['Stoch_K'].iloc[-1],
                'stoch_d': self.df['Stoch_D'].iloc[-1]
            }
        }
        
        # Overall signal
        predictions = analysis['predictions']
        
        bullish_signals = sum([
            predictions['tomorrow']['change_percent'] > 0,
            predictions['week']['change_percent'] > 0,
            predictions['month']['change_percent'] > 0,
            analysis['renko']['trend'] in ['Uptrend', 'Strong Uptrend'],
            analysis['heikin_ashi']['trend'] in ['Uptrend', 'Strong Uptrend'],
            analysis['indicators']['rsi'] < 50,
            analysis['indicators']['macd'] > analysis['indicators']['macd_signal']
        ])
        
        bearish_signals = sum([
            predictions['tomorrow']['change_percent'] < 0,
            predictions['week']['change_percent'] < 0,
            predictions['month']['change_percent'] < 0,
            analysis['renko']['trend'] in ['Downtrend', 'Strong Downtrend'],
            analysis['heikin_ashi']['trend'] in ['Downtrend', 'Strong Downtrend'],
            analysis['indicators']['rsi'] > 50,
            analysis['indicators']['macd'] < analysis['indicators']['macd_signal']
        ])
        
        if bullish_signals > bearish_signals + 2:
            overall_signal = 'Strong Buy'
        elif bullish_signals > bearish_signals:
            overall_signal = 'Buy'
        elif bearish_signals > bullish_signals + 2:
            overall_signal = 'Strong Sell'
        elif bearish_signals > bullish_signals:
            overall_signal = 'Sell'
        else:
            overall_signal = 'Neutral'
        
        analysis['overall_signal'] = overall_signal
        analysis['bullish_signals'] = bullish_signals
        analysis['bearish_signals'] = bearish_signals
        
        return analysis
