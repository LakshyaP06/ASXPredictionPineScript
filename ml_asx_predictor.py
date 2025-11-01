"""
ASX Stock Predictor with Machine Learning - TensorFlow Integration
Combines technical analysis with deep learning for enhanced predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ML imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    from sklearn.model_selection import KFold, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    HAS_ML = True
    KerasModel = keras.Model
except ImportError:
    HAS_ML = False
    KerasModel = Any  # Fallback type
    print("Warning: TensorFlow not installed. ML features disabled.")

from asx_stock_predictor import ASXStockPredictor


class MLASXPredictor(ASXStockPredictor):
    """
    Enhanced ASX Stock Predictor with TensorFlow Machine Learning
    
    Combines traditional technical analysis with deep learning:
    - LSTM networks for sequence prediction
    - Feature engineering from technical indicators
    - K-fold cross-validation for robust model evaluation
    - Ensemble predictions combining ML and technical analysis
    - Backtesting against historical data
    """
    
    # ML Configuration
    ML_LOOKBACK_WINDOW = 60  # Days to look back for sequence prediction
    ML_PREDICTION_HORIZONS = {
        'tomorrow': 1,
        'week': 5,
        'month': 22
    }
    N_FOLDS = 5  # For K-fold cross-validation
    
    def __init__(self, df: pd.DataFrame, enable_ml: bool = True):
        """
        Initialize ML-enhanced predictor
        
        Args:
            df: DataFrame with OHLCV data
            enable_ml: Whether to enable ML predictions
        """
        super().__init__(df)
        self.enable_ml = enable_ml and HAS_ML
        self.ml_models = {}
        self.scalers = {}
        self.ml_trained = False
        
        if not HAS_ML and enable_ml:
            print("Warning: TensorFlow not available. Install with: pip install tensorflow scikit-learn")
            self.enable_ml = False
    
    def _prepare_ml_features(self) -> pd.DataFrame:
        """
        Prepare feature matrix from technical indicators
        
        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame(index=self.df.index)
        
        # Price features
        features['close'] = self.df['Close']
        features['open'] = self.df['Open']
        features['high'] = self.df['High']
        features['low'] = self.df['Low']
        features['volume'] = self.df['Volume']
        
        # Returns
        features['returns'] = self.df['Close'].pct_change()
        features['log_returns'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        
        # Moving averages
        features['ema_9'] = self.df['EMA_9']
        features['ema_21'] = self.df['EMA_21']
        features['sma_50'] = self.df['SMA_50']
        features['sma_200'] = self.df['SMA_200']
        
        # Moving average crossovers
        features['ema_cross'] = (self.df['EMA_9'] - self.df['EMA_21']) / self.df['Close']
        features['price_to_sma50'] = (self.df['Close'] - self.df['SMA_50']) / self.df['SMA_50']
        features['price_to_sma200'] = (self.df['Close'] - self.df['SMA_200']) / self.df['SMA_200']
        
        # Technical indicators
        features['rsi'] = self.df['RSI']
        features['macd'] = self.df['MACD']
        features['macd_signal'] = self.df['MACD_Signal']
        features['macd_hist'] = self.df['MACD_Hist']
        
        # Bollinger Bands
        features['bb_upper'] = self.df['BB_Upper']
        features['bb_middle'] = self.df['BB_Middle']
        features['bb_lower'] = self.df['BB_Lower']
        features['bb_position'] = (self.df['Close'] - self.df['BB_Lower']) / (self.df['BB_Upper'] - self.df['BB_Lower'])
        features['bb_width'] = (self.df['BB_Upper'] - self.df['BB_Lower']) / self.df['BB_Middle']
        
        # Volatility
        features['atr'] = self.df['ATR']
        features['atr_percent'] = self.df['ATR'] / self.df['Close']
        
        # Momentum indicators
        features['adx'] = self.df['ADX']
        features['di_plus'] = self.df['DI_Plus']
        features['di_minus'] = self.df['DI_Minus']
        features['stoch_k'] = self.df['Stoch_K']
        features['stoch_d'] = self.df['Stoch_D']
        
        # Pattern signals (numerical encoding)
        patterns = self.recognize_candlestick_patterns()
        for pattern_name, detected in patterns.items():
            features[f'pattern_{pattern_name}'] = 1.0 if detected else 0.0
        
        # Volume indicators
        features['volume_ma'] = self.df['Volume'].rolling(window=20).mean()
        features['volume_ratio'] = self.df['Volume'] / features['volume_ma']
        
        # Price momentum
        features['momentum_5'] = self.df['Close'].pct_change(5)
        features['momentum_10'] = self.df['Close'].pct_change(10)
        features['momentum_20'] = self.df['Close'].pct_change(20)
        
        # Volatility features
        features['volatility_10'] = self.df['Close'].rolling(window=10).std()
        features['volatility_30'] = self.df['Close'].rolling(window=30).std()
        
        # High-Low range
        features['hl_ratio'] = (self.df['High'] - self.df['Low']) / self.df['Close']
        
        # Drop NaN values
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        return features
    
    def _create_sequences(self, features: pd.DataFrame, target: pd.Series, 
                         lookback: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        
        Args:
            features: Feature DataFrame
            target: Target series (prices to predict)
            lookback: Number of timesteps to look back
            horizon: Number of timesteps to predict ahead
            
        Returns:
            Tuple of (X, y) arrays
        """
        X, y = [], []
        
        for i in range(lookback, len(features) - horizon + 1):
            X.append(features.iloc[i-lookback:i].values)
            y.append(target.iloc[i+horizon-1])
        
        return np.array(X), np.array(y)
    
    def _build_lstm_model(self, input_shape: Tuple[int, int], 
                         learning_rate: float = 0.001) -> KerasModel:
        """
        Build LSTM model architecture
        
        Args:
            input_shape: (lookback, n_features)
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_ml_models(self, test_stocks: Optional[List[str]] = None,
                       epochs: int = 50, batch_size: int = 32,
                       validation_split: float = 0.2) -> Dict[str, Dict]:
        """
        Train ML models with K-fold cross-validation
        
        Args:
            test_stocks: List of stock tickers to test against (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            
        Returns:
            Dictionary with training results and metrics
        """
        if not self.enable_ml:
            return {'error': 'ML not enabled or TensorFlow not available'}
        
        print("Preparing features for ML training...")
        features = self._prepare_ml_features()
        
        # Prepare target (price to predict)
        target = self.df['Close']
        
        results = {}
        
        # Train model for each prediction horizon
        for horizon_name, horizon_days in self.ML_PREDICTION_HORIZONS.items():
            print(f"\n{'='*60}")
            print(f"Training model for {horizon_name} ({horizon_days} days ahead)")
            print('='*60)
            
            # Create sequences
            X, y = self._create_sequences(features, target, 
                                         self.ML_LOOKBACK_WINDOW, horizon_days)
            
            if len(X) < 100:
                print(f"Warning: Not enough data for {horizon_name} (need 100+, have {len(X)})")
                continue
            
            # Initialize scaler
            scaler = StandardScaler()
            
            # Reshape for scaling
            n_samples, n_timesteps, n_features = X.shape
            X_reshaped = X.reshape(-1, n_features)
            X_scaled = scaler.fit_transform(X_reshaped)
            X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
            
            # Store scaler
            self.scalers[horizon_name] = scaler
            
            # K-fold cross-validation
            kfold = KFold(n_splits=self.N_FOLDS, shuffle=False)
            fold_results = []
            
            print(f"\nPerforming {self.N_FOLDS}-fold cross-validation...")
            
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
                print(f"\nFold {fold_idx + 1}/{self.N_FOLDS}")
                
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Build model
                model = self._build_lstm_model((n_timesteps, n_features))
                
                # Callbacks
                early_stop = callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
                
                reduce_lr = callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                )
                
                # Train
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop, reduce_lr],
                    verbose=0
                )
                
                # Evaluate
                y_pred = model.predict(X_val, verbose=0).flatten()
                
                mse = mean_squared_error(y_val, y_pred)
                mae = mean_absolute_error(y_val, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_val, y_pred)
                
                # Calculate MAPE (Mean Absolute Percentage Error)
                mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
                
                fold_results.append({
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'mape': mape,
                    'model': model,
                    'history': history.history
                })
                
                print(f"  MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
            
            # Select best model (lowest validation MAE)
            best_fold = min(fold_results, key=lambda x: x['mae'])
            self.ml_models[horizon_name] = best_fold['model']
            
            # Calculate average metrics
            avg_metrics = {
                'mse': np.mean([f['mse'] for f in fold_results]),
                'mae': np.mean([f['mae'] for f in fold_results]),
                'rmse': np.mean([f['rmse'] for f in fold_results]),
                'r2': np.mean([f['r2'] for f in fold_results]),
                'mape': np.mean([f['mape'] for f in fold_results]),
                'std_mse': np.std([f['mse'] for f in fold_results]),
                'std_mae': np.std([f['mae'] for f in fold_results])
            }
            
            results[horizon_name] = {
                'metrics': avg_metrics,
                'fold_results': fold_results,
                'best_fold': fold_results.index(best_fold)
            }
            
            print(f"\n{horizon_name.upper()} - Average Metrics:")
            print(f"  MSE: {avg_metrics['mse']:.4f} ± {avg_metrics['std_mse']:.4f}")
            print(f"  MAE: {avg_metrics['mae']:.4f} ± {avg_metrics['std_mae']:.4f}")
            print(f"  RMSE: {avg_metrics['rmse']:.4f}")
            print(f"  R²: {avg_metrics['r2']:.4f}")
            print(f"  MAPE: {avg_metrics['mape']:.2f}%")
        
        self.ml_trained = True
        
        print(f"\n{'='*60}")
        print("ML Model Training Complete!")
        print('='*60)
        
        return results
    
    def predict_with_ml(self) -> Dict[str, Dict]:
        """
        Make predictions using trained ML models
        
        Returns:
            Dictionary with ML predictions for each horizon
        """
        if not self.enable_ml:
            return {'error': 'ML not enabled'}
        
        if not self.ml_trained:
            return {'error': 'Models not trained. Call train_ml_models() first'}
        
        features = self._prepare_ml_features()
        predictions = {}
        
        current_price = self.df['Close'].iloc[-1]
        
        for horizon_name, horizon_days in self.ML_PREDICTION_HORIZONS.items():
            if horizon_name not in self.ml_models:
                continue
            
            model = self.ml_models[horizon_name]
            scaler = self.scalers[horizon_name]
            
            # Get last sequence
            X_last = features.iloc[-self.ML_LOOKBACK_WINDOW:].values
            X_last = X_last.reshape(1, self.ML_LOOKBACK_WINDOW, -1)
            
            # Scale
            n_features = X_last.shape[2]
            X_last_reshaped = X_last.reshape(-1, n_features)
            X_last_scaled = scaler.transform(X_last_reshaped)
            X_last_scaled = X_last_scaled.reshape(1, self.ML_LOOKBACK_WINDOW, n_features)
            
            # Predict
            pred_price = model.predict(X_last_scaled, verbose=0)[0][0]
            
            predictions[horizon_name] = {
                'prediction': float(pred_price),
                'current': float(current_price),
                'change': float(pred_price - current_price),
                'change_percent': float((pred_price - current_price) / current_price * 100)
            }
        
        return predictions
    
    def predict_prices_hybrid(self) -> Dict[str, Dict]:
        """
        Hybrid predictions combining ML and technical analysis
        
        Returns:
            Dictionary with combined predictions
        """
        # Get traditional predictions
        traditional_pred = super().predict_prices()
        
        if not self.enable_ml or not self.ml_trained:
            return traditional_pred
        
        # Get ML predictions
        ml_pred = self.predict_with_ml()
        
        if 'error' in ml_pred:
            return traditional_pred
        
        # Combine predictions (weighted average)
        ml_weight = 0.6  # Give more weight to ML
        ta_weight = 0.4  # Traditional technical analysis
        
        hybrid_predictions = {'current': traditional_pred['current']}
        
        for horizon in ['tomorrow', 'week', 'month']:
            if horizon in ml_pred and horizon in traditional_pred:
                ml_price = ml_pred[horizon]['prediction']
                ta_price = traditional_pred[horizon]['prediction']
                
                # Weighted average
                combined_price = ml_price * ml_weight + ta_price * ta_weight
                
                current = traditional_pred['current']['price']
                
                hybrid_predictions[horizon] = {
                    'prediction': combined_price,
                    'ml_prediction': ml_price,
                    'ta_prediction': ta_price,
                    'change': combined_price - current,
                    'change_percent': (combined_price - current) / current * 100,
                    'range_low': traditional_pred[horizon]['range_low'],
                    'range_high': traditional_pred[horizon]['range_high'],
                    'confidence': traditional_pred[horizon]['confidence'],
                    'ml_weight': ml_weight,
                    'ta_weight': ta_weight
                }
        
        return hybrid_predictions
    
    def backtest_predictions(self, test_days: int = 30) -> Dict[str, any]:
        """
        Backtest predictions against historical data
        
        Args:
            test_days: Number of days to test
            
        Returns:
            Dictionary with backtest results
        """
        if not self.enable_ml or not self.ml_trained:
            return {'error': 'ML not enabled or models not trained'}
        
        if len(self.df) < self.ML_LOOKBACK_WINDOW + test_days + 22:
            return {'error': 'Not enough data for backtesting'}
        
        print(f"\nBacktesting predictions over last {test_days} days...")
        
        results = {
            'tomorrow': {'predictions': [], 'actuals': [], 'errors': []},
            'week': {'predictions': [], 'actuals': [], 'errors': []},
            'month': {'predictions': [], 'actuals': [], 'errors': []}
        }
        
        # Test on last test_days
        for i in range(test_days):
            test_idx = -(test_days - i + 22)  # Ensure we have future data
            
            if test_idx >= -22:
                break
            
            # Create temporary predictor with data up to test point
            temp_df = self.df.iloc[:test_idx].copy()
            
            if len(temp_df) < self.ML_LOOKBACK_WINDOW:
                continue
            
            # Make predictions
            temp_predictor = MLASXPredictor(temp_df, enable_ml=True)
            temp_predictor.ml_models = self.ml_models
            temp_predictor.scalers = self.scalers
            temp_predictor.ml_trained = True
            
            predictions = temp_predictor.predict_with_ml()
            
            # Get actual values
            for horizon_name, horizon_days in self.ML_PREDICTION_HORIZONS.items():
                if test_idx + horizon_days < 0:
                    actual_price = self.df['Close'].iloc[test_idx + horizon_days]
                    pred_price = predictions[horizon_name]['prediction']
                    
                    error = abs(actual_price - pred_price) / actual_price * 100
                    
                    results[horizon_name]['predictions'].append(pred_price)
                    results[horizon_name]['actuals'].append(actual_price)
                    results[horizon_name]['errors'].append(error)
        
        # Calculate metrics
        metrics = {}
        for horizon in ['tomorrow', 'week', 'month']:
            if len(results[horizon]['predictions']) > 0:
                preds = np.array(results[horizon]['predictions'])
                actuals = np.array(results[horizon]['actuals'])
                errors = np.array(results[horizon]['errors'])
                
                metrics[horizon] = {
                    'mae': mean_absolute_error(actuals, preds),
                    'mse': mean_squared_error(actuals, preds),
                    'rmse': np.sqrt(mean_squared_error(actuals, preds)),
                    'mape': np.mean(errors),
                    'max_error': np.max(errors),
                    'min_error': np.min(errors),
                    'n_predictions': len(preds)
                }
        
        return {
            'results': results,
            'metrics': metrics
        }
    
    def get_comprehensive_analysis_ml(self) -> Dict[str, any]:
        """
        Get comprehensive analysis including ML predictions
        
        Returns:
            Dictionary with complete analysis
        """
        # Get traditional analysis
        analysis = super().get_comprehensive_analysis()
        
        # Add ML predictions if available
        if self.enable_ml and self.ml_trained:
            analysis['ml_predictions'] = self.predict_with_ml()
            analysis['hybrid_predictions'] = self.predict_prices_hybrid()
            analysis['ml_enabled'] = True
        else:
            analysis['ml_enabled'] = False
        
        return analysis
