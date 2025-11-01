# ML-Enhanced ASX Stock Predictor - Technical Documentation

## Overview

This document provides technical details about the Machine Learning integration for the ASX Stock Predictor system.

## Architecture

### 1. ML Model Structure

The system uses **LSTM (Long Short-Term Memory)** neural networks for time series prediction:

```
Input Layer (60 timesteps × 40+ features)
    ↓
LSTM Layer (128 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer (64 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer (32 units, return_sequences=False)
    ↓
Dropout (0.2)
    ↓
Dense Layer (16 units, ReLU activation)
    ↓
Output Layer (1 unit, linear activation)
```

### 2. Feature Engineering

The system extracts 40+ features from technical analysis methods:

#### Price Features
- Open, High, Low, Close, Volume
- Returns (simple and log)

#### Moving Averages
- EMA (9, 21)
- SMA (50, 200)
- Moving average crossovers

#### Technical Indicators
- RSI (Relative Strength Index)
- MACD (MACD line, signal line, histogram)
- Bollinger Bands (upper, middle, lower, position, width)
- ATR (Average True Range)
- ADX (Average Directional Index)
- Directional Indicators (DI+, DI-)
- Stochastic Oscillator (K, D)

#### Pattern Features
- Candlestick patterns (binary encoding)
- Volume ratios
- Price momentum (5, 10, 20 periods)
- Volatility measures

## Training Process

### 1. K-Fold Cross-Validation

The system uses **5-fold cross-validation** for robust model evaluation:

```
Training Data Split:
┌─────────────────────────────────────┐
│  Fold 1  │  Fold 2  │  Fold 3  │  Fold 4  │  Fold 5  │
├─────────────────────────────────────┤
│  Train   │  Train   │  Train   │  Train   │   Val    │  (Iteration 1)
│  Train   │  Train   │  Train   │   Val    │  Train   │  (Iteration 2)
│  Train   │  Train   │   Val    │  Train   │  Train   │  (Iteration 3)
│  Train   │   Val    │  Train   │  Train   │  Train   │  (Iteration 4)
│   Val    │  Train   │  Train   │  Train   │  Train   │  (Iteration 5)
└─────────────────────────────────────┘
```

For each fold:
1. Split data into training and validation sets
2. Train LSTM model on training data
3. Evaluate on validation data
4. Record performance metrics

**Best model** (lowest validation MAE) is selected from all folds.

### 2. Training Parameters

Default training configuration:
- **Epochs**: 30 (with early stopping)
- **Batch Size**: 32
- **Validation Split**: 20% within each fold
- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Mean Squared Error (MSE)

Callbacks:
- **Early Stopping**: Patience=10, restores best weights
- **Learning Rate Reduction**: Factor=0.5, patience=5

### 3. Data Preprocessing

**Sequence Creation:**
- Lookback window: 60 days
- Prediction horizons: 1 day (tomorrow), 5 days (week), 22 days (month)

**Normalization:**
- StandardScaler applied to all features
- Separate scalers for each prediction horizon
- Fitted on training data, applied to validation/test data

## Prediction Process

### 1. ML Predictions

For each horizon (tomorrow, week, month):
1. Extract last 60 days of features
2. Normalize using trained scaler
3. Feed through LSTM model
4. Output predicted price

### 2. Hybrid Predictions

Combines ML and Technical Analysis:

```
Hybrid Price = (ML_Price × 0.6) + (TA_Price × 0.4)
```

Where:
- **ML_Price**: LSTM neural network prediction
- **TA_Price**: Traditional technical analysis prediction (5 methods combined)

### 3. Confidence Scoring

Confidence is based on:
1. Agreement between prediction methods
2. Volatility (ATR)
3. Trend strength (ADX)
4. Volume confirmation
5. Price consistency

Score ranges from 0-100%, where:
- 70-100%: High confidence
- 50-70%: Medium confidence
- 0-50%: Low confidence

## Performance Metrics

The system reports multiple metrics for evaluation:

### 1. Training Metrics (K-Fold)

For each horizon, averaged across folds:
- **MAE** (Mean Absolute Error): Average absolute difference
- **MSE** (Mean Squared Error): Average squared difference
- **RMSE** (Root Mean Squared Error): Square root of MSE
- **R²** (R-squared): Proportion of variance explained
- **MAPE** (Mean Absolute Percentage Error): Average percentage error

### 2. Backtest Metrics

Tested on last N days (default 20):
- Prediction accuracy on historical data
- Error distribution
- Maximum and minimum errors

## Testing Framework

### Test Stocks

Primary test set (16 stocks):
```
AEE.AX, BTL.AX, BTR.AX, CXL.AX, DKM.AX, FLT.AX,
GNG.AX, HGEN.AX, HLO.AX, IMM.AX, IVV.AX, NDQ.AX,
NGI.AX, RAD.AX, SNT.AX, URNM.AX
```

Additional training stocks (24 stocks):
```
BHP.AX, CBA.AX, CSL.AX, NAB.AX, WBC.AX, ANZ.AX,
WES.AX, RIO.AX, WOW.AX, TLS.AX, MQG.AX, GMG.AX,
TCL.AX, QBE.AX, SCG.AX, STO.AX, ORG.AX, COL.AX,
WDS.AX, S32.AX, ALL.AX, AMP.AX, IAG.AX, SUN.AX
```

### Test Execution

**Full Test (All Stocks):**
```bash
python test_ml_predictor.py
```

**Single Stock Test:**
```bash
python test_ml_predictor.py --single
```

**Quick Test (First 3 stocks, TA only):**
```bash
python test_ml_predictor.py --quick
```

## Code Structure

### Main Classes

**MLASXPredictor** (extends ASXStockPredictor):
- Inherits all technical analysis methods
- Adds ML training and prediction capabilities
- Provides hybrid prediction functionality

### Key Methods

**Training:**
- `train_ml_models()`: K-fold training with validation
- `_prepare_ml_features()`: Feature engineering
- `_create_sequences()`: Sequence generation for LSTM
- `_build_lstm_model()`: Model architecture definition

**Prediction:**
- `predict_with_ml()`: Pure ML predictions
- `predict_prices_hybrid()`: Combined ML + TA predictions
- `backtest_predictions()`: Historical validation

**Analysis:**
- `get_comprehensive_analysis_ml()`: Full analysis with ML

## Usage Examples

### Basic ML Training

```python
from ml_asx_predictor import MLASXPredictor
import yfinance as yf

# Download data (need 2 years for ML)
df = yf.download("FLT.AX", period="2y")

# Create ML predictor
predictor = MLASXPredictor(df, enable_ml=True)

# Train models with K-fold
results = predictor.train_ml_models(
    epochs=30,
    batch_size=32
)

# Get hybrid predictions
predictions = predictor.predict_prices_hybrid()

print(f"Tomorrow: ${predictions['tomorrow']['prediction']:.2f}")
print(f"ML: ${predictions['tomorrow']['ml_prediction']:.2f}")
print(f"TA: ${predictions['tomorrow']['ta_prediction']:.2f}")
```

### Backtesting

```python
# Backtest on last 20 days
backtest = predictor.backtest_predictions(test_days=20)

# View metrics
for horizon, metrics in backtest['metrics'].items():
    print(f"{horizon}:")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  MAE: ${metrics['mae']:.2f}")
```

### Multiple Stock Testing

```python
from test_ml_predictor import test_multiple_stocks

# Test specified stocks
tickers = ['AEE.AX', 'BTL.AX', 'FLT.AX']
results = test_multiple_stocks(tickers, train_ml=True)
```

## Requirements

### Python Packages

```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
tensorflow>=2.10.0
scikit-learn>=1.0.0
yfinance>=0.2.18
matplotlib>=3.4.0
seaborn>=0.11.0
```

### Hardware Recommendations

**Minimum:**
- CPU: 2+ cores
- RAM: 8GB
- Training time: ~5-10 minutes per stock

**Recommended:**
- CPU: 4+ cores or GPU
- RAM: 16GB+
- GPU: CUDA-compatible for faster training

## Limitations and Considerations

### 1. Data Requirements

- Minimum 200 days of historical data
- Optimal: 2+ years for robust training
- Missing or sparse data may reduce accuracy

### 2. Market Conditions

- Trained on historical patterns
- May not predict unprecedented events
- Works best in trending markets

### 3. Computational Cost

- Training takes 5-10 minutes per stock
- K-fold validation adds ~5x overhead
- Consider GPU for large-scale testing

### 4. Prediction Accuracy

- MAPE typically 2-5% for tomorrow
- Accuracy decreases for longer horizons
- Always consider confidence scores

## Best Practices

### 1. Data Quality

- Use clean, complete data
- Check for gaps and anomalies
- Validate ticker symbols

### 2. Model Training

- Use default parameters initially
- Increase epochs for better accuracy (with early stopping)
- Monitor validation loss during training

### 3. Prediction Usage

- Check confidence scores
- Consider both ML and TA predictions
- Use backtesting results for context
- Combine with fundamental analysis

### 4. Risk Management

- Never rely solely on predictions
- Use appropriate position sizing
- Set stop-loss levels
- Diversify portfolio

## Troubleshooting

### TensorFlow Installation Issues

**Problem:** TensorFlow fails to install

**Solutions:**
- Use Python 3.8-3.11
- Try: `pip install tensorflow==2.10.0`
- For Apple Silicon: `pip install tensorflow-macos`

### Memory Errors

**Problem:** Out of memory during training

**Solutions:**
- Reduce batch size (try 16 or 8)
- Reduce lookback window
- Close other applications

### Poor Prediction Accuracy

**Problem:** High MAPE or low R²

**Solutions:**
- Increase training data (use 3+ years)
- Increase epochs (try 50-100)
- Check data quality
- Try different stocks (higher liquidity)

## Future Enhancements

Potential improvements:
- Attention mechanisms
- Transformer architectures
- Multi-stock training (transfer learning)
- Real-time prediction updates
- Hyperparameter optimization
- Ensemble of different architectures

## References

- TensorFlow: https://www.tensorflow.org/
- Scikit-learn: https://scikit-learn.org/
- LSTM Networks: https://www.bioinf.jku.at/publications/older/2604.pdf
- Time Series Cross-Validation: https://robjhyndman.com/hyndsight/tscv/

## License

This project is subject to the terms of the Mozilla Public License 2.0.

## Disclaimer

⚠️ **For research and educational purposes only. Not financial advice.**

Machine learning predictions are based on historical patterns and may not accurately predict future prices. Always conduct thorough research and consider consulting a licensed financial advisor before making investment decisions.
