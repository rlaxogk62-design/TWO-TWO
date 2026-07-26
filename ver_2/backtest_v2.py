import pandas as pd
import numpy as np
import ta
import ccxt
import joblib
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
import os

from model_training_v2 import fetch_binance_futures_data, feature_engineering, create_labels

def run_backtest():
    print("==========================================")
    print(">>> ver_2 백테스트 시뮬레이션 시작")
    print("==========================================")
    
    # 1. 데이터 수집 & 피처 생성
    df = fetch_binance_futures_data(total_candles=15000)
    df = feature_engineering(df)
    df = create_labels(df)
    
    features = ['Returns', 'Body_Size', 'Upper_Shadow', 'Lower_Shadow', 
                'RSI_14', 'ATR_Ratio', 'Close_vs_SMA20', 'BB_Width', 'BB_Pos', 'Close_vs_SMA20_1H']
    
    X = df[features]
    y = df['Target']
    
    # Train / Test (Out-of-sample 최근 20% 백테스트)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    df_test = df.iloc[split_idx:].copy()
    
    # 모델 학습
    sw_train = compute_sample_weight(class_weight='balanced', y=y_train)
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='multi:softprob',
        n_jobs=-1
    )
    model.fit(X_train, y_train, sample_weight=sw_train)
    
    # 테스트 예측 (확률)
    probs = model.predict_proba(X_test)
    df_test['Prob_Short'] = probs[:, 0]
    df_test['Prob_Neutral'] = probs[:, 1]
    df_test['Prob_Long'] = probs[:, 2]
    
    # 시뮬레이션 조건 (사용자 ver1 조건 반영: 레버리지 25x, 진입 비율 25%, 수수료 왕복 0.08%)
    LEVERAGE = 25
    POS_RATIO = 0.25
    FEE_RATE = 0.0008 # 0.08% Taker 왕복 수수료
    ENTRY_THRESHOLD = 0.45 # 확률 0.45 이상일 때 진입
    
    initial_balance = 1000.0
    balance = initial_balance
    position = 0  # 1: Long, -1: Short, 0: None
    entry_price = 0.0
    
    trades = []
    equity_curve = [initial_balance]
    
    prices = df_test['Close'].values
    p_long = df_test['Prob_Long'].values
    p_short = df_test['Prob_Short'].values
    times = df_test.index
    
    for i in range(len(df_test) - 1):
        curr_price = prices[i]
        next_price = prices[i+1]
        
        # 포지션 종료 조건 (약 1시간/4캔들 경과 혹은 반대 신호)
        if position == 1:
            # Long 청산
            if p_short[i] > ENTRY_THRESHOLD or p_long[i] < 0.35:
                pnl_pct = (curr_price - entry_price) / entry_price
                trade_pnl = balance * POS_RATIO * LEVERAGE * pnl_pct
                trade_fee = balance * POS_RATIO * LEVERAGE * FEE_RATE
                net_pnl = trade_pnl - trade_fee
                balance += net_pnl
                trades.append({'type': 'LONG', 'pnl_pct': pnl_pct * LEVERAGE * 100, 'net_pnl': net_pnl})
                position = 0
                
        elif position == -1:
            # Short 청산
            if p_long[i] > ENTRY_THRESHOLD or p_short[i] < 0.35:
                pnl_pct = (entry_price - curr_price) / entry_price
                trade_pnl = balance * POS_RATIO * LEVERAGE * pnl_pct
                trade_fee = balance * POS_RATIO * LEVERAGE * FEE_RATE
                net_pnl = trade_pnl - trade_fee
                balance += net_pnl
                trades.append({'type': 'SHORT', 'pnl_pct': pnl_pct * LEVERAGE * 100, 'net_pnl': net_pnl})
                position = 0
                
        # 포지션 진입 조건
        if position == 0:
            if p_long[i] > ENTRY_THRESHOLD and p_long[i] > p_short[i]:
                position = 1
                entry_price = curr_price
            elif p_short[i] > ENTRY_THRESHOLD and p_short[i] > p_long[i]:
                position = -1
                entry_price = curr_price
                
        equity_curve.append(balance)
        
    # 결과 집계
    df_trades = pd.DataFrame(trades)
    total_trades = len(df_trades)
    
    if total_trades > 0:
        win_trades = len(df_trades[df_trades['net_pnl'] > 0])
        win_rate = (win_trades / total_trades) * 100
        total_return = ((balance - initial_balance) / initial_balance) * 100
        
        # MDD 계산
        eq_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(eq_arr)
        drawdown = (eq_arr - peak) / peak
        mdd = np.min(drawdown) * 100
    else:
        win_rate = 0
        total_return = 0
        mdd = 0
        
    print("\n==========================================")
    print("📊 [ver_2 모델 Out-of-Sample 백테스트 최종 결과]")
    print("==========================================")
    print(f"테스트 기간 캔들 수: {len(df_test)}개 (약 {len(df_test)*15/60/24:.1f}일)")
    print(f"초기 자본금: ${initial_balance:,.2f}")
    print(f"최종 자본금: ${balance:,.2f}")
    print(f"총 수익률: {total_return:+.2f}%")
    print(f"총 거래 횟수: {total_trades}회")
    print(f"승률(Win Rate): {win_rate:.2f}%")
    print(f"최대 낙폭(MDD): {mdd:.2f}%")
    print("==========================================")

if __name__ == "__main__":
    run_backtest()
