# ASX Stock Prediction - Advanced Technical Analysis with Machine Learning

This repository contains advanced stock prediction implementations for ASX (Australian Securities Exchange) stocks:

1. **PineScript Version** (`predictionscript.pine`) - For TradingView
2. **Python Version** (`asx_stock_predictor.py`) - Traditional technical analysis
3. **ML-Enhanced Version** (`ml_asx_predictor.py`) - 🆕 **Machine Learning with TensorFlow**

## 🤖 ML-Enhanced Implementation (NEW!)

A state-of-the-art ML prediction system combining **Deep Learning (LSTM)** with traditional technical analysis:

### **Machine Learning Features:**
- 🧠 **LSTM Neural Networks** - Deep learning for sequence prediction
- ✅ **K-Fold Cross-Validation** - Robust model evaluation with 5-fold CV
- 📊 **Feature Engineering** - 40+ features from technical indicators
- 🎯 **Hybrid Predictions** - Ensemble of ML (60%) + Technical Analysis (40%)
- 📈 **Backtesting Framework** - Validate against historical data
- 🔬 **Performance Metrics** - MAE, RMSE, MAPE, R² scores

### **Technical Analysis Methods (Integrated as Features):**
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

### **Multi-Horizon Predictions:**
The ML system provides predictions for:
- 📈 **Tomorrow** - Next trading day forecast with ML confidence
- 📊 **One Week** - 5 trading days ahead
- 📅 **One Month** - 22 trading days ahead

Each prediction includes:
- ML prediction, TA prediction, and hybrid (combined) prediction
- Confidence scores and price ranges
- Model performance metrics (MAPE, MAE, RMSE)

### **Quick Start (ML Version):**

```bash
# Install dependencies (includes TensorFlow)
pip install -r requirements.txt

# Run ML test on specified stocks
python test_ml_predictor.py

# Or test single stock quickly
python test_ml_predictor.py --single

# Or use Jupyter Notebook
jupyter notebook ML_ASX_Predictor.ipynb
```

### **Usage Example (ML):**

```python
import yfinance as yf
from ml_asx_predictor import MLASXPredictor

# Download stock data (need 2 years for ML training)
df = yf.download("FLT.AX", period="2y")

# Create ML-enhanced predictor
predictor = MLASXPredictor(df, enable_ml=True)

# Train ML models with K-fold cross-validation
training_results = predictor.train_ml_models(epochs=30)

# Get hybrid predictions (ML + TA)
predictions = predictor.predict_prices_hybrid()

print(f"Tomorrow: ${predictions['tomorrow']['prediction']:.2f}")
print(f"  ML: ${predictions['tomorrow']['ml_prediction']:.2f}")
print(f"  TA: ${predictions['tomorrow']['ta_prediction']:.2f}")
print(f"  Confidence: {predictions['tomorrow']['confidence']:.0f}%")

# Backtest against historical data
backtest = predictor.backtest_predictions(test_days=20)
print(f"MAPE: {backtest['metrics']['tomorrow']['mape']:.2f}%")
```

### **Tested Stocks:**
The system has been validated on these ASX stocks:
- AEE.AX, BTL.AX, BTR.AX, CXL.AX, DKM.AX, FLT.AX
- GNG.AX, HGEN.AX, HLO.AX, IMM.AX, IVV.AX, NDQ.AX
- NGI.AX, RAD.AX, SNT.AX, URNM.AX

Plus trained on 20+ major ASX stocks for data diversity.

---

## 🐍 Traditional Python Implementation

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

### Machine Learning Files
- `ml_asx_predictor.py` - ML-enhanced predictor with TensorFlow LSTM networks
- `ML_ASX_Predictor.ipynb` - Interactive ML notebook with training and backtesting
- `test_ml_predictor.py` - Testing script for specified ASX stocks with K-fold validation

### Traditional Technical Analysis Files
- `asx_stock_predictor.py` - Base predictor class with all technical analysis methods
- `ASX_Stock_Predictor.ipynb` - Interactive Jupyter notebook with TA examples and visualizations
- `example_usage.py` - Simple Python script demonstrating basic TA usage

### Documentation and Configuration
- `requirements.txt` - Python package dependencies (includes TensorFlow and scikit-learn)
- `USAGE_GUIDE.md` - Complete documentation and API reference
- `predictionscript.pine` - Original TradingView PineScript indicator
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
