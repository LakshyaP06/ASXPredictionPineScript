# ASX Stock Predictor - Advanced Technical Analysis System

A comprehensive Python-based stock prediction system for ASX (Australian Securities Exchange) stocks that uses multiple technical analysis methods to estimate future prices.

## Features

This advanced predictor implements the following technical analysis methods:

### 📊 Technical Analysis Methods

1. **Fibonacci Retracements and Extensions**
   - Calculates key retracement levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)
   - Extension levels for target projections (127.2%, 161.8%, 261.8%)

2. **Breakout Pattern Detection**
   - Identifies consolidation periods
   - Detects breakout directions (up/down)
   - Volume confirmation for breakout strength

3. **Reversal Pattern Detection**
   - Double Top/Bottom
   - Head and Shoulders
   - Inverse Head and Shoulders
   - Rising/Falling Wedge patterns

4. **Elliott Wave Structure**
   - Wave count analysis
   - Identifies impulsive and corrective waves
   - Current wave phase determination

5. **Fair Value Gaps (FVG)**
   - Detects price imbalances
   - Bullish and bearish gap identification
   - Gap size and percentage calculation

6. **Candlestick Pattern Recognition**
   - Doji, Hammer, Shooting Star
   - Bullish/Bearish Engulfing
   - Morning Star, Evening Star
   - And more...

7. **Heikin Ashi Trend Signals**
   - Smoothed trend identification
   - Reversal signal detection
   - Trend strength measurement

8. **Renko Trend Direction**
   - Noise-filtered trend analysis
   - Brick-based trend strength
   - Directional momentum

9. **Harmonic Pattern Detection**
   - Gartley, Butterfly, Bat, Crab patterns
   - Fibonacci ratio validation
   - Pattern completion levels

10. **Support and Resistance Levels**
    - Dynamic S/R level identification
    - Level clustering algorithm
    - Distance from current price

### 🎯 Predictions

The system provides price predictions for three timeframes:

- **Tomorrow** - Next trading day forecast
- **Week** - 5 trading days ahead
- **Month** - 22 trading days ahead

Each prediction includes:
- Predicted price
- Expected change ($ and %)
- Price range (high/low estimates)
- Confidence score (0-100%)

### 📈 Prediction Methods

The system combines multiple forecasting methods:

1. **Linear Regression** - Trend-based projection
2. **EMA Momentum** - Moving average momentum
3. **Mean Reversion** - Bollinger Bands-based
4. **RSI-based** - Overbought/oversold adjustments
5. **Pattern-based** - Technical pattern signals

All methods are weighted and combined for final predictions.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Jupyter Notebook

### Setup

1. Clone the repository:
```bash
git clone https://github.com/LakshyaP06/ASXPredictionPineScript.git
cd ASXPredictionPineScript
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Open `ASX_Stock_Predictor.ipynb`

## Usage

### Basic Usage

```python
import pandas as pd
import yfinance as yf
from asx_stock_predictor import ASXStockPredictor

# Download stock data
ticker = "BHP.AX"  # BHP Group
df = yf.download(ticker, period="1y")

# Create predictor
predictor = ASXStockPredictor(df)

# Get predictions
predictions = predictor.predict_prices()

# Display results
print(f"Tomorrow's prediction: ${predictions['tomorrow']['prediction']:.2f}")
print(f"Expected change: {predictions['tomorrow']['change_percent']:.2f}%")
print(f"Confidence: {predictions['tomorrow']['confidence']:.0f}%")
```

### Comprehensive Analysis

```python
# Get full analysis
analysis = predictor.get_comprehensive_analysis()

# Access different components
fibonacci = analysis['fibonacci']
support_resistance = analysis['support_resistance']
patterns = analysis['candlestick_patterns']
overall_signal = analysis['overall_signal']

print(f"Overall Signal: {overall_signal}")
```

### Using the Jupyter Notebook

The included notebook (`ASX_Stock_Predictor.ipynb`) provides:

- Step-by-step examples
- Visualization of predictions and patterns
- Multiple stock comparison
- Detailed analysis reports
- Export functionality

Simply change the `TICKER` variable to analyze different stocks:

```python
TICKER = "CBA.AX"  # Commonwealth Bank
PERIOD = "1y"       # 1 year of data
```

## Supported ASX Stocks

The predictor works with any ASX-listed stock. Common examples:

- **BHP.AX** - BHP Group
- **CBA.AX** - Commonwealth Bank
- **NAB.AX** - National Australia Bank
- **WBC.AX** - Westpac Banking
- **ANZ.AX** - ANZ Banking
- **CSL.AX** - CSL Limited
- **WES.AX** - Wesfarmers
- **RIO.AX** - Rio Tinto
- **WOW.AX** - Woolworths
- **TLS.AX** - Telstra

## Technical Indicators

The system calculates and uses the following indicators:

- **Moving Averages**: EMA (9, 21), SMA (50, 200)
- **RSI** (Relative Strength Index)
- **MACD** (Moving Average Convergence Divergence)
- **Bollinger Bands**
- **ATR** (Average True Range)
- **ADX** (Average Directional Index)
- **Stochastic Oscillator**
- **Volume Analysis**

## API Reference

### ASXStockPredictor Class

#### Methods

##### `__init__(df: pd.DataFrame)`
Initialize the predictor with OHLCV data.

##### `predict_prices() -> Dict`
Generate price predictions for tomorrow, week, and month.

##### `calculate_fibonacci_levels(lookback: int = 50) -> Dict`
Calculate Fibonacci retracement and extension levels.

##### `detect_breakout_patterns(consolidation_period: int = 20) -> Dict`
Detect breakout patterns from consolidation.

##### `detect_reversal_patterns() -> Dict`
Identify common reversal patterns.

##### `analyze_elliott_wave(lookback: int = 50) -> Dict`
Perform Elliott Wave analysis.

##### `detect_fair_value_gaps(min_gap_percent: float = 0.5) -> List`
Find fair value gaps in price action.

##### `recognize_candlestick_patterns() -> Dict`
Recognize candlestick patterns.

##### `get_heikin_ashi_signals() -> Dict`
Get Heikin Ashi trend signals.

##### `get_renko_trend() -> Dict`
Analyze Renko trend direction.

##### `detect_harmonic_patterns() -> Dict`
Detect harmonic patterns (Gartley, Butterfly, etc.).

##### `calculate_support_resistance(lookback: int = 50, num_levels: int = 3) -> Dict`
Calculate support and resistance levels.

##### `get_comprehensive_analysis() -> Dict`
Get complete analysis combining all methods.

## Output Format

### Prediction Output

```python
{
    'current': {
        'price': 45.23,
        'date': '2025-10-30'
    },
    'tomorrow': {
        'prediction': 45.67,
        'change': 0.44,
        'change_percent': 0.97,
        'range_low': 45.20,
        'range_high': 46.14,
        'confidence': 72
    },
    'week': { ... },
    'month': { ... }
}
```

### Analysis Output

The comprehensive analysis includes:
- Predictions (tomorrow, week, month)
- Fibonacci levels
- Breakout patterns
- Reversal patterns
- Elliott Wave analysis
- Fair value gaps
- Candlestick patterns
- Heikin Ashi signals
- Renko trend
- Harmonic patterns
- Support/Resistance levels
- Technical indicators
- Overall signal (Strong Buy, Buy, Neutral, Sell, Strong Sell)

## Examples

### Example 1: Quick Prediction

```python
import yfinance as yf
from asx_stock_predictor import ASXStockPredictor

# Get data and predict
df = yf.download("CBA.AX", period="6mo")
predictor = ASXStockPredictor(df)
predictions = predictor.predict_prices()

print(f"Tomorrow: ${predictions['tomorrow']['prediction']:.2f}")
print(f"Week: ${predictions['week']['prediction']:.2f}")
print(f"Month: ${predictions['month']['prediction']:.2f}")
```

### Example 2: Pattern Detection

```python
# Detect patterns
candles = predictor.recognize_candlestick_patterns()
reversals = predictor.detect_reversal_patterns()
harmonics = predictor.detect_harmonic_patterns()

# Print detected patterns
for pattern, detected in candles.items():
    if detected:
        print(f"Found: {pattern}")
```

### Example 3: Support/Resistance

```python
# Get S/R levels
sr = predictor.calculate_support_resistance()

print(f"Current: ${sr['current_price']:.2f}")
print(f"Resistance: {sr['resistance']}")
print(f"Support: {sr['support']}")
```

## Visualizations

The Jupyter notebook includes visualizations for:

- Price history with predictions
- Expected price changes
- Confidence scores
- Technical indicators
- Fibonacci levels
- Support/Resistance levels
- Multi-stock comparison

## Accuracy and Limitations

### Strengths
- Combines multiple proven technical analysis methods
- Provides confidence scores
- Adapts to different market conditions
- Identifies multiple pattern types

### Limitations
- Based on historical data only
- Cannot predict unforeseen events (news, earnings, etc.)
- Works best in trending markets
- Requires sufficient historical data (minimum 50 periods)

### Tips for Best Results
- Use with stocks that have good liquidity
- Combine with fundamental analysis
- Consider confidence scores
- Use multiple timeframes
- Monitor news and events

## Troubleshooting

### "Missing required columns" error
Ensure your DataFrame has: Open, High, Low, Close, Volume

### "Need at least 50 periods" error
Download more historical data (use `period="1y"` or longer)

### Import errors
Install all requirements: `pip install -r requirements.txt`

### Yahoo Finance data issues
Try using a different period or check internet connection

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the Mozilla Public License 2.0 - see the LICENSE file for details.

## Disclaimer

**IMPORTANT:** This predictor is for educational and research purposes only.

⚠️ **NOT FINANCIAL ADVICE**

- Past performance does not guarantee future results
- The stock market involves risk
- You may lose money
- Always conduct your own research
- Consider consulting with a licensed financial advisor

The predictions are based on technical analysis and historical patterns, which may not accurately predict future price movements. Use at your own risk.

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review the Jupyter notebook examples

## Acknowledgments

- Technical analysis methods based on established trading principles
- Data sourced via yfinance (Yahoo Finance API)
- Inspired by professional trading systems

## Version History

- **v1.0.0** - Initial release
  - Complete technical analysis suite
  - Multi-timeframe predictions
  - Jupyter notebook interface
  - Comprehensive documentation

---

**Made for research purposes. Always research before you invest.**
