#!/usr/bin/env python3
"""
ML ASX Stock Predictor - Testing Script
Tests ML predictions against specified ASX stocks with K-fold validation
"""

import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from ml_asx_predictor import MLASXPredictor


# Stocks to test against
TEST_STOCKS = [
    'AEE.AX', 'BTL.AX', 'BTR.AX', 'CXL.AX', 'DKM.AX', 'FLT.AX',
    'GNG.AX', 'HGEN.AX', 'HLO.AX', 'IMM.AX', 'IVV.AX', 'NDQ.AX',
    'NGI.AX', 'RAD.AX', 'SNT.AX', 'URNM.AX'
]

# Additional ASX stocks for training data diversity
TRAINING_STOCKS = [
    'BHP.AX', 'CBA.AX', 'CSL.AX', 'NAB.AX', 'WBC.AX', 'ANZ.AX',
    'WES.AX', 'RIO.AX', 'WOW.AX', 'TLS.AX', 'MQG.AX', 'GMG.AX',
    'TCL.AX', 'QBE.AX', 'SCG.AX', 'STO.AX', 'ORG.AX', 'COL.AX',
    'WDS.AX', 'S32.AX', 'ALL.AX', 'AMP.AX', 'IAG.AX', 'SUN.AX'
]


def download_stock_data(ticker: str, period: str = '2y') -> pd.DataFrame:
    """Download stock data with error handling"""
    try:
        print(f"  Downloading {ticker}...", end=' ')
        df = yf.download(ticker, period=period, progress=False)
        if len(df) > 50:
            print(f"✓ ({len(df)} days)")
            return df
        else:
            print(f"✗ (insufficient data: {len(df)} days)")
            return None
    except Exception as e:
        print(f"✗ (error: {str(e)})")
        return None


def test_single_stock(ticker: str, train_ml: bool = True) -> dict:
    """
    Test ML predictor on a single stock
    
    Args:
        ticker: Stock ticker symbol
        train_ml: Whether to train ML models
        
    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*70}")
    print(f"Testing {ticker}")
    print('='*70)
    
    # Download data
    df = download_stock_data(ticker, period='2y')
    
    if df is None or len(df) < 200:
        return {
            'ticker': ticker,
            'status': 'failed',
            'error': 'Insufficient data'
        }
    
    try:
        # Create predictor
        print(f"\nInitializing ML predictor...")
        predictor = MLASXPredictor(df, enable_ml=True)
        
        current_price = df['Close'].iloc[-1]
        print(f"Current price: ${current_price:.2f}")
        
        results = {
            'ticker': ticker,
            'current_price': current_price,
            'data_points': len(df),
            'date_range': f"{df.index[0].date()} to {df.index[-1].date()}"
        }
        
        if train_ml:
            # Train ML models with K-fold
            print(f"\nTraining ML models with K-fold cross-validation...")
            training_results = predictor.train_ml_models(
                epochs=30,
                batch_size=32,
                validation_split=0.2
            )
            
            results['training'] = training_results
            
            # Make ML predictions
            print(f"\nGenerating ML predictions...")
            ml_predictions = predictor.predict_with_ml()
            results['ml_predictions'] = ml_predictions
            
            # Make hybrid predictions
            print(f"\nGenerating hybrid predictions...")
            hybrid_predictions = predictor.predict_prices_hybrid()
            results['hybrid_predictions'] = hybrid_predictions
            
            # Backtest
            print(f"\nBacktesting predictions...")
            backtest_results = predictor.backtest_predictions(test_days=20)
            results['backtest'] = backtest_results
            
            # Display results
            print(f"\n{'='*70}")
            print(f"RESULTS FOR {ticker}")
            print('='*70)
            
            print(f"\n📊 HYBRID PREDICTIONS (ML + Technical Analysis):")
            for horizon in ['tomorrow', 'week', 'month']:
                if horizon in hybrid_predictions:
                    pred = hybrid_predictions[horizon]
                    print(f"\n{horizon.upper()}:")
                    print(f"  Combined: ${pred['prediction']:.2f} ({pred['change_percent']:+.2f}%)")
                    print(f"  ML: ${pred['ml_prediction']:.2f}")
                    print(f"  TA: ${pred['ta_prediction']:.2f}")
                    print(f"  Confidence: {pred['confidence']:.0f}%")
            
            if 'metrics' in backtest_results:
                print(f"\n📈 BACKTEST METRICS:")
                for horizon, metrics in backtest_results['metrics'].items():
                    print(f"\n{horizon.upper()}:")
                    print(f"  MAPE: {metrics['mape']:.2f}%")
                    print(f"  MAE: ${metrics['mae']:.2f}")
                    print(f"  RMSE: ${metrics['rmse']:.2f}")
                    print(f"  Predictions tested: {metrics['n_predictions']}")
            
            results['status'] = 'success'
        
        else:
            # Just technical analysis
            analysis = predictor.get_comprehensive_analysis()
            results['ta_analysis'] = analysis
            results['status'] = 'success_ta_only'
        
        return results
        
    except Exception as e:
        print(f"\n✗ Error testing {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'ticker': ticker,
            'status': 'error',
            'error': str(e)
        }


def test_multiple_stocks(tickers: list, train_ml: bool = True, 
                        save_results: bool = True) -> dict:
    """
    Test ML predictor on multiple stocks
    
    Args:
        tickers: List of stock ticker symbols
        train_ml: Whether to train ML models for each stock
        save_results: Whether to save results to file
        
    Returns:
        Dictionary with all results
    """
    print(f"\n{'='*70}")
    print(f"TESTING ML ASX PREDICTOR ON {len(tickers)} STOCKS")
    print('='*70)
    print(f"\nTesting stocks: {', '.join(tickers)}")
    
    all_results = {
        'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'stocks_tested': len(tickers),
        'results': {}
    }
    
    successful = 0
    failed = 0
    
    for idx, ticker in enumerate(tickers, 1):
        print(f"\n\n[{idx}/{len(tickers)}] Testing {ticker}...")
        
        result = test_single_stock(ticker, train_ml=train_ml)
        all_results['results'][ticker] = result
        
        if result['status'] == 'success':
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n\n{'='*70}")
    print(f"TESTING COMPLETE")
    print('='*70)
    print(f"Total stocks tested: {len(tickers)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    # Create summary report
    summary = []
    for ticker, result in all_results['results'].items():
        if result['status'] == 'success' and 'hybrid_predictions' in result:
            hybrid = result['hybrid_predictions']
            
            if 'tomorrow' in hybrid:
                summary.append({
                    'Ticker': ticker,
                    'Current': f"${result['current_price']:.2f}",
                    'Tomorrow': f"${hybrid['tomorrow']['prediction']:.2f}",
                    'Change_1D_%': f"{hybrid['tomorrow']['change_percent']:+.2f}%",
                    'Week': f"${hybrid['week']['prediction']:.2f}",
                    'Change_1W_%': f"{hybrid['week']['change_percent']:+.2f}%",
                    'Month': f"${hybrid['month']['prediction']:.2f}",
                    'Change_1M_%': f"{hybrid['month']['change_percent']:+.2f}%",
                    'Status': '✓'
                })
            else:
                summary.append({
                    'Ticker': ticker,
                    'Status': '✗ No predictions'
                })
        else:
            summary.append({
                'Ticker': ticker,
                'Status': f"✗ {result.get('error', 'Failed')}"
            })
    
    if summary:
        summary_df = pd.DataFrame(summary)
        print(f"\n{'='*70}")
        print("SUMMARY TABLE")
        print('='*70)
        print(summary_df.to_string(index=False))
    
    # Save results
    if save_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"ml_test_results_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write(f"ML ASX PREDICTOR TEST RESULTS\n")
            f.write(f"{'='*70}\n")
            f.write(f"Test Date: {all_results['test_date']}\n")
            f.write(f"Stocks Tested: {len(tickers)}\n")
            f.write(f"Successful: {successful}\n")
            f.write(f"Failed: {failed}\n\n")
            
            if summary:
                f.write(f"{'='*70}\n")
                f.write("SUMMARY TABLE\n")
                f.write(f"{'='*70}\n")
                f.write(summary_df.to_string(index=False))
                f.write("\n\n")
            
            f.write(f"{'='*70}\n")
            f.write("DETAILED RESULTS\n")
            f.write(f"{'='*70}\n\n")
            
            for ticker, result in all_results['results'].items():
                f.write(f"\n{ticker}:\n")
                f.write(f"  Status: {result['status']}\n")
                
                if 'hybrid_predictions' in result:
                    f.write(f"  Current Price: ${result['current_price']:.2f}\n")
                    
                    hybrid = result['hybrid_predictions']
                    for horizon in ['tomorrow', 'week', 'month']:
                        if horizon in hybrid:
                            pred = hybrid[horizon]
                            f.write(f"  {horizon.upper()}: ${pred['prediction']:.2f} ({pred['change_percent']:+.2f}%)\n")
                    
                    if 'backtest' in result and 'metrics' in result['backtest']:
                        f.write(f"\n  Backtest Metrics:\n")
                        for horizon, metrics in result['backtest']['metrics'].items():
                            f.write(f"    {horizon}: MAPE {metrics['mape']:.2f}%, MAE ${metrics['mae']:.2f}\n")
                
                elif 'error' in result:
                    f.write(f"  Error: {result['error']}\n")
        
        print(f"\nResults saved to: {filename}")
    
    return all_results


def main():
    """Main execution function"""
    print(f"\n{'='*70}")
    print("ML ASX STOCK PREDICTOR - TESTING FRAMEWORK")
    print('='*70)
    print("\nThis script tests the ML-enhanced predictor on specified ASX stocks")
    print("with K-fold cross-validation and backtesting.")
    
    # Check if user wants quick test or full test
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        print("\n🚀 Running QUICK TEST (first 3 stocks, TA only)...")
        test_stocks = TEST_STOCKS[:3]
        train_ml = False
    elif len(sys.argv) > 1 and sys.argv[1] == '--single':
        print("\n🎯 Running SINGLE STOCK TEST with full ML training...")
        test_stocks = [TEST_STOCKS[0]]
        train_ml = True
    else:
        print("\n🔬 Running FULL TEST on all specified stocks...")
        test_stocks = TEST_STOCKS
        train_ml = True
    
    print(f"\nStocks to test: {test_stocks}")
    print(f"ML Training: {'Enabled' if train_ml else 'Disabled'}")
    
    # Run tests
    results = test_multiple_stocks(test_stocks, train_ml=train_ml)
    
    print(f"\n{'='*70}")
    print("✅ Testing complete!")
    print('='*70)
    
    return results


if __name__ == "__main__":
    main()
