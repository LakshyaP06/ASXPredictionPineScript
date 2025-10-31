# ASX Stock Prediction - Advanced Technical Analysis

This repository contains two implementations of advanced stock prediction for ASX (Australian Securities Exchange) stocks:

1. **PineScript Version** (`predictionscript.pine`) - For TradingView
2. **Python Version** (`asx_stock_predictor.py`) - For Jupyter Notebook and standalone use

## 🐍 Python Implementation (NEW!)

A comprehensive Python-based stock predictor that works in Jupyter Notebook with advanced technical analysis methods:

### **Technical Analysis Methods:**
- ✅ **Fibonacci Retracements and Extensions** - Key support/resistance levels
- ✅ **Breakout Pattern Detection** - Consolidation and breakout identification
- ✅ **Reversal Pattern Detection** - Double tops/bottoms, head & shoulders, wedges
- ✅ **Elliott Wave Structure** - Wave counting and phase identification
- ✅ **Fair Value Gaps** - Price imbalance detection
- ✅ **Candlestick Pattern Recognition** - Doji, hammers, engulfing, stars
- ✅ **Heikin Ashi Trend Signals** - Smoothed trend analysis
- ✅ **Renko Trend Direction** - Noise-filtered trends
- ✅ **Harmonic Pattern Detection** - Gartley, Butterfly, Bat, Crab
- ✅ **Support and Resistance Levels** - Dynamic S/R identification

### **Price Predictions:**
The system provides accurate price estimates for:
- 📈 **Tomorrow** - Next trading day forecast
- 📊 **One Week** - 5 trading days ahead
- 📅 **One Month** - 22 trading days ahead

Each prediction includes confidence scores and price ranges!

### **Quick Start (Python):**

```bash
# Install dependencies
pip install -r requirements.txt

# Run example
python example_usage.py

# Or use Jupyter Notebook
jupyter notebook ASX_Stock_Predictor.ipynb
```

### **Usage Example:**

```python
import yfinance as yf
from asx_stock_predictor import ASXStockPredictor

# Download stock data
df = yf.download("BHP.AX", period="1y")

# Create predictor
predictor = ASXStockPredictor(df)

# Get predictions
predictions = predictor.predict_prices()

print(f"Tomorrow: ${predictions['tomorrow']['prediction']:.2f}")
print(f"Week: ${predictions['week']['prediction']:.2f}")
print(f"Month: ${predictions['month']['prediction']:.2f}")

# Get comprehensive analysis
analysis = predictor.get_comprehensive_analysis()
print(f"Overall Signal: {analysis['overall_signal']}")
```

See [USAGE_GUIDE.md](USAGE_GUIDE.md) for complete documentation.

---

## 📊 PineScript Implementation (Original)

### **Features:**

### **Technical Indicators:**
- **Multiple EMAs/SMAs** - Short-term (9, 21) and long-term (50, 200) moving averages
- **RSI** - Relative Strength Index for overbought/oversold conditions
- **MACD** - Moving Average Convergence Divergence for momentum
- **Bollinger Bands** - Volatility and price range analysis
- **ATR** - Average True Range for stop-loss placement
- **Stochastic Oscillator** - Momentum indicator
- **ADX** - Trend strength measurement
- **Volume Analysis** - Confirms price movements

### **Prediction System:**
- **Scoring System** - Combines multiple indicators (0-100 score)
- **Strong Buy/Sell Signals** - When multiple conditions align
- **Trend Confirmation** - Uses ADX to validate trend strength
- **Support/Resistance** - Automatically identifies key levels

### **Visual Features:**
- **Color-coded signals** - Easy to spot buy/sell opportunities
- **Real-time dashboard** - Shows all key metrics at a glance
- **Background coloring** - Indicates overall market sentiment
- **Stop-loss levels** - ATR-based risk management

### **ASX-Specific Considerations:**
- Works with ASX trading hours and volatility patterns
- Pivot point calculations suitable for daily timeframes
- Volume analysis calibrated for Australian market conditions

### **How to Use (PineScript):**
1. Copy the script into TradingView Pine Editor
2. Apply to any ASX stock chart
3. Adjust parameters based on stock volatility and your trading style
4. Set up alerts for buy/sell signals
5. Use stop-loss levels for risk management

---

## 📋 Files in This Repository

- `asx_stock_predictor.py` - Main Python predictor class with all technical analysis methods
- `ASX_Stock_Predictor.ipynb` - Interactive Jupyter notebook with examples and visualizations
- `example_usage.py` - Simple Python script demonstrating basic usage
- `requirements.txt` - Python package dependencies
- `USAGE_GUIDE.md` - Complete documentation and API reference
- `predictionscript.pine` - Original TradingView PineScript indicator

## 🎓 Supported ASX Stocks

Works with any ASX-listed stock, including:
- BHP.AX, CBA.AX, NAB.AX, WBC.AX, ANZ.AX
- CSL.AX, WES.AX, RIO.AX, WOW.AX, TLS.AX
- And many more!

## ⚠️ Disclaimer

**IMPORTANT:** This predictor is for educational and research purposes only.

- ❌ **NOT FINANCIAL ADVICE**
- Past performance does not guarantee future results
- The stock market involves risk and you may lose money
- Always conduct your own research before making investment decisions
- Consider consulting with a licensed financial advisor

## 📝 License

This project is subject to the terms of the Mozilla Public License 2.0.

---

**NOTE: For research purposes only, research before you buy stocks.**
