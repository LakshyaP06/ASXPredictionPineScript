#!/usr/bin/env python3
"""
Simple example demonstrating ASX Stock Predictor usage
"""

import sys
import pandas as pd
import yfinance as yf
from asx_stock_predictor import ASXStockPredictor


def main():
    # Configuration
    ticker = "BHP.AX"  # BHP Group
    period = "1y"
    
    print(f"ASX Stock Predictor - Example")
    print("=" * 60)
    print(f"\nAnalyzing: {ticker}")
    print(f"Period: {period}")
    
    # Download data
    print(f"\nDownloading data...")
    df = yf.download(ticker, period=period, progress=False)
    
    if len(df) < 50:
        print("Error: Insufficient data. Need at least 50 periods.")
        sys.exit(1)
    
    print(f"Data loaded: {len(df)} periods")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
    # Create predictor
    print(f"\nInitializing predictor...")
    predictor = ASXStockPredictor(df)
    
    # Get predictions
    print(f"\nGenerating predictions...")
    predictions = predictor.predict_prices()
    
    # Display results
    current = predictions['current']
    tomorrow = predictions['tomorrow']
    week = predictions['week']
    month = predictions['month']
    
    print("\n" + "=" * 60)
    print("PRICE PREDICTIONS")
    print("=" * 60)
    print(f"\nCurrent Price: ${current['price']:.2f}")
    
    print(f"\n📈 TOMORROW:")
    print(f"   Price: ${tomorrow['prediction']:.2f}")
    print(f"   Change: ${tomorrow['change']:.2f} ({tomorrow['change_percent']:.2f}%)")
    print(f"   Range: ${tomorrow['range_low']:.2f} - ${tomorrow['range_high']:.2f}")
    print(f"   Confidence: {tomorrow['confidence']:.0f}%")
    
    print(f"\n📊 ONE WEEK:")
    print(f"   Price: ${week['prediction']:.2f}")
    print(f"   Change: ${week['change']:.2f} ({week['change_percent']:.2f}%)")
    print(f"   Range: ${week['range_low']:.2f} - ${week['range_high']:.2f}")
    print(f"   Confidence: {week['confidence']:.0f}%")
    
    print(f"\n📅 ONE MONTH:")
    print(f"   Price: ${month['prediction']:.2f}")
    print(f"   Change: ${month['change']:.2f} ({month['change_percent']:.2f}%)")
    print(f"   Range: ${month['range_low']:.2f} - ${month['range_high']:.2f}")
    print(f"   Confidence: {month['confidence']:.0f}%")
    
    # Get comprehensive analysis
    print("\n" + "=" * 60)
    print("TECHNICAL ANALYSIS")
    print("=" * 60)
    
    analysis = predictor.get_comprehensive_analysis()
    
    print(f"\n🎯 Overall Signal: {analysis['overall_signal']}")
    print(f"   Bullish Signals: {analysis['bullish_signals']}")
    print(f"   Bearish Signals: {analysis['bearish_signals']}")
    
    # Display key indicators
    ind = analysis['indicators']
    print(f"\n📊 Key Indicators:")
    print(f"   RSI: {ind['rsi']:.2f}")
    print(f"   MACD: {ind['macd']:.4f}")
    print(f"   ADX: {ind['adx']:.2f}")
    print(f"   ATR: ${ind['atr']:.2f}")
    
    # Display trends
    print(f"\n📈 Trend Analysis:")
    print(f"   Heikin Ashi: {analysis['heikin_ashi']['trend']}")
    print(f"   Renko: {analysis['renko']['trend']}")
    print(f"   Elliott Wave: {analysis['elliott_wave']['wave_trend']}")
    
    # Display patterns
    print(f"\n🔍 Patterns Detected:")
    candle_patterns = [k for k, v in analysis['candlestick_patterns'].items() if v]
    reversal_patterns = [k for k, v in analysis['reversal_patterns'].items() if v]
    
    if candle_patterns:
        print(f"   Candlestick: {', '.join(p.replace('_', ' ').title() for p in candle_patterns)}")
    if reversal_patterns:
        print(f"   Reversal: {', '.join(p.replace('_', ' ').title() for p in reversal_patterns)}")
    if analysis['harmonic_patterns']['patterns']:
        print(f"   Harmonic: {', '.join(p['type'] for p in analysis['harmonic_patterns']['patterns'])}")
    if not (candle_patterns or reversal_patterns or analysis['harmonic_patterns']['patterns']):
        print(f"   No significant patterns detected")
    
    # Support and Resistance
    sr = analysis['support_resistance']
    print(f"\n📍 Support & Resistance:")
    if sr['resistance']:
        print(f"   Resistance: {', '.join(f'${r:.2f}' for r in sr['resistance'])}")
    if sr['support']:
        print(f"   Support: {', '.join(f'${s:.2f}' for s in sr['support'])}")
    
    # Fibonacci levels
    fib = analysis['fibonacci']
    print(f"\n🔢 Key Fibonacci Levels:")
    print(f"   Swing High: ${fib['swing_high']:.2f}")
    print(f"   0.618 (Key): ${fib['fib_0.618']:.2f}")
    print(f"   0.500: ${fib['fib_0.500']:.2f}")
    print(f"   Swing Low: ${fib['swing_low']:.2f}")
    
    print("\n" + "=" * 60)
    print("\n✅ Analysis complete!")
    print("\n⚠️  DISCLAIMER: For research purposes only. Not financial advice.")
    print("=" * 60)


if __name__ == "__main__":
    main()
